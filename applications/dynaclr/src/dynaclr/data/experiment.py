"""Experiment registry for multi-experiment DynaCLR training.

Provides :class:`ExperimentRegistry` — a validated collection with channel
resolution, tau-range conversion, and Z-range auto-resolution, backed by
:class:`~viscy_data.collection.Collection`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from iohub.ngff import open_ome_zarr

from viscy_data.cell_index import read_cell_index
from viscy_data.collection import ChannelEntry, Collection, ExperimentEntry, load_collection

_logger = logging.getLogger(__name__)

__all__ = ["ExperimentRegistry"]


@dataclass
class ExperimentRegistry:
    """Validated collection of experiments with channel and Z resolution.

    On creation (``__post_init__``), the registry performs fail-fast validation:

    1. Experiments list must not be empty.
    2. Experiment names must be unique.
    3. ``interval_minutes`` must be positive for each experiment.
    4. ``perturbation_wells`` must not be empty for each experiment.
    5. ``data_path`` must point to an existing directory.
    6. Zarr metadata channel names must match ``channel_names``.

    After validation the registry computes:

    * ``z_ranges`` -- per-experiment ``(z_start, z_end)`` ranges.
    * ``scale_factors`` -- per-experiment ``(scale_z, scale_y, scale_x)``
      for physical-space normalization across microscopes.

    Parameters
    ----------
    collection : Collection
        Validated collection of experiment configurations.
    z_window : int or None
        Number of Z slices the model consumes.
    focus_channel : str or None
        Channel name to look up ``focus_slice`` metadata in plate zattrs.
    reference_pixel_size_xy_um : float or None
        Reference pixel size in XY (micrometers). None = no rescaling.
    reference_pixel_size_z_um : float or None
        Reference voxel size in Z (micrometers). None = no rescaling.
    """

    collection: Collection
    z_window: int | None = None
    focus_channel: str | None = None
    reference_pixel_size_xy_um: float | None = None
    reference_pixel_size_z_um: float | None = None
    z_ranges: dict[str, tuple[int, int]] = field(init=False)
    scale_factors: dict[str, tuple[float, float, float]] = field(init=False)

    # internal lookup
    _name_map: dict[str, ExperimentEntry] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:  # noqa: D105
        experiments = self.collection.experiments

        # 1. Empty check
        if not experiments:
            raise ValueError("Empty experiments list: at least one experiment is required.")

        # 2. Duplicate names
        names: list[str] = [e.name for e in experiments]
        seen: set[str] = set()
        for n in names:
            if n in seen:
                raise ValueError(f"Duplicate experiment name '{n}'. Each experiment must have a unique name.")
            seen.add(n)

        # Build name -> config map
        self._name_map = {e.name: e for e in experiments}

        # Per-experiment validations
        for exp in experiments:
            # 4. Negative interval
            if exp.interval_minutes <= 0:
                raise ValueError(
                    f"Experiment '{exp.name}': interval_minutes must be positive, got {exp.interval_minutes}."
                )

            # 5. Empty perturbation_wells
            if not exp.perturbation_wells:
                raise ValueError(f"Experiment '{exp.name}': perturbation_wells must not be empty.")

            # 6. data_path existence
            if not Path(exp.data_path).exists():
                raise ValueError(f"Experiment '{exp.name}': data_path does not exist: {exp.data_path}")

            # 7. Zarr channel validation — selected channels must exist in zarr
            with open_ome_zarr(exp.data_path, mode="r") as plate:
                first_position = next(iter(plate.positions()))[1]
                zarr_channels = list(first_position.channel_names)
            # Store the full zarr channel list for index resolution
            exp.channel_names = zarr_channels
            missing_channels = [ch.name for ch in exp.channels if ch.name not in zarr_channels]
            if missing_channels:
                raise ValueError(
                    f"Experiment '{exp.name}': channels {missing_channels} "
                    f"not found in zarr. Available: {zarr_channels}."
                )

        # Resolve per-experiment z_ranges
        self.z_ranges = self._resolve_z_ranges()

        # Validate pixel sizes and compute scale factors
        if self.reference_pixel_size_xy_um is not None or self.reference_pixel_size_z_um is not None:
            missing = [e.name for e in experiments if e.pixel_size_xy_um is None or e.pixel_size_z_um is None]
            if missing:
                raise ValueError(
                    f"reference_pixel_size set but experiments are missing pixel_size_xy_um/z_um: {missing}"
                )
        self.scale_factors = self._compute_scale_factors()

    @property
    def experiments(self) -> list[ExperimentEntry]:
        """Return the list of experiment entries."""
        return self.collection.experiments

    @property
    def source_channel_labels(self) -> list[str]:
        """Return unique marker labels across all experiments' channels."""
        seen: set[str] = set()
        labels: list[str] = []
        for exp in self.collection.experiments:
            for ch in exp.channels:
                if ch.marker not in seen:
                    labels.append(ch.marker)
                    seen.add(ch.marker)
        return labels

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_z_ranges(self) -> dict[str, tuple[int, int]]:
        """Resolve per-experiment Z ranges.

        For experiments with explicit ``z_range`` in zattrs, use it directly.
        Otherwise read ``focus_slice`` metadata from the plate-level zattrs
        and center a window of ``self.z_window`` slices around ``z_focus_mean``.
        """
        experiments = self.collection.experiments
        z_ranges: dict[str, tuple[int, int]] = {}

        for exp in experiments:
            # Auto-resolve from focus_slice zattrs
            focus_ch = self.focus_channel or (exp.channels[0].name if exp.channels else None)

            with open_ome_zarr(exp.data_path, mode="r") as plate:
                first_pos = next(iter(plate.positions()))[1]
                z_total = first_pos["0"].shape[2]

                if self.z_window is None:
                    # Use full Z
                    z_ranges[exp.name] = (0, z_total)
                    continue

                focus_data = plate.zattrs.get("focus_slice", {})
                ch_focus = focus_data.get(focus_ch, {}) if focus_ch else {}
                ds_stats = ch_focus.get("dataset_statistics", {})
                z_focus_mean = ds_stats.get("z_focus_mean")

            if z_focus_mean is None:
                # Default to center of Z stack
                z_center = z_total // 2
            else:
                z_center = int(round(z_focus_mean))

            z_half = self.z_window // 2
            z_start = max(0, z_center - z_half)
            z_end = min(z_total, z_start + self.z_window)
            z_start = max(0, z_end - self.z_window)

            z_ranges[exp.name] = (z_start, z_end)
            _logger.info(
                "Experiment '%s': z_range=(%d, %d), z_total=%d, z_window=%d",
                exp.name,
                z_start,
                z_end,
                z_total,
                self.z_window,
            )

        # Validate all z windows have the same size
        if z_ranges:
            window_sizes = {name: r[1] - r[0] for name, r in z_ranges.items()}
            unique_sizes = set(window_sizes.values())
            if len(unique_sizes) > 1:
                detail = ", ".join(f"'{n}': {s}" for n, s in window_sizes.items())
                raise ValueError(
                    f"All experiments must have the same z_window size, but found: {detail}. "
                    f"Adjust z_range values or ensure consistent z_window."
                )

        return z_ranges

    def _compute_scale_factors(self) -> dict[str, tuple[float, float, float]]:
        """Compute per-experiment scale factors for physical-space normalization.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            ``{exp_name: (scale_z, scale_y, scale_x)}`` where scale = experiment_um /
            reference_um.  When reference pixel size is 0.0, scale = 1.0 (no rescaling).
        """
        scale_factors: dict[str, tuple[float, float, float]] = {}
        for exp in self.collection.experiments:
            if (
                self.reference_pixel_size_xy_um is not None
                and self.reference_pixel_size_z_um is not None
                and exp.pixel_size_xy_um is not None
                and exp.pixel_size_z_um is not None
            ):
                scale_y = exp.pixel_size_xy_um / self.reference_pixel_size_xy_um
                scale_x = exp.pixel_size_xy_um / self.reference_pixel_size_xy_um
                scale_z = exp.pixel_size_z_um / self.reference_pixel_size_z_um
            else:
                scale_y = 1.0
                scale_x = 1.0
                scale_z = 1.0
            scale_factors[exp.name] = (scale_z, scale_y, scale_x)
        return scale_factors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def from_collection(
        cls,
        path: str | Path,
        z_window: int | None = None,
        focus_channel: str | None = None,
        reference_pixel_size_xy_um: float | None = None,
        reference_pixel_size_z_um: float | None = None,
    ) -> ExperimentRegistry:
        """Load experiments from a collection YAML file.

        Parameters
        ----------
        path : str | Path
            Path to the collection YAML.
        z_window : int or None
            Number of Z slices the model consumes.
        focus_channel : str or None
            Channel name for ``focus_slice`` lookup.
        reference_pixel_size_xy_um : float or None
            Reference pixel size in XY (micrometers). None = no rescaling.
        reference_pixel_size_z_um : float or None
            Reference voxel size in Z (micrometers). None = no rescaling.

        Returns
        -------
        ExperimentRegistry
            Validated registry of experiments.
        """
        collection = load_collection(path)
        return cls(
            collection=collection,
            z_window=z_window,
            focus_channel=focus_channel,
            reference_pixel_size_xy_um=reference_pixel_size_xy_um,
            reference_pixel_size_z_um=reference_pixel_size_z_um,
        )

    @classmethod
    def from_cell_index(
        cls,
        cell_index_path: str | Path,
        z_window: int | None = None,
        focus_channel: str | None = None,
        reference_pixel_size_xy_um: float | None = None,
        reference_pixel_size_z_um: float | None = None,
    ) -> ExperimentRegistry:
        """Build a registry from a flat cell index parquet and zarr metadata.

        Derives per-experiment channels from the parquet's ``marker`` and
        ``channel_name`` columns. No collection YAML needed — the parquet
        is the universal entry point.

        Parameters
        ----------
        cell_index_path : str | Path
            Path to the cell index parquet.
        z_window : int or None
            Number of Z slices the model consumes.
        focus_channel : str or None
            Channel name for ``focus_slice`` lookup.
        reference_pixel_size_xy_um : float or None
            Reference pixel size in XY (micrometers). None = no rescaling.
        reference_pixel_size_z_um : float or None
            Reference voxel size in Z (micrometers). None = no rescaling.

        Returns
        -------
        ExperimentRegistry
            Validated registry of experiments.
        """
        df = read_cell_index(cell_index_path)
        if df.empty:
            raise ValueError(f"Cell index is empty: {cell_index_path}")

        # Step 1: Read channel names per (store_path, well) from zarr.
        channel_names_cache: dict[tuple[str, str], list[str]] = {}
        store_cache: dict[str, object] = {}

        for store_path, group in df.groupby("store_path"):
            plate = open_ome_zarr(str(store_path), mode="r")
            store_cache[str(store_path)] = plate
            for well in group["well"].unique():
                # Find one position in this well
                well_str = str(well)
                for pos_path, pos in plate.positions():
                    if pos_path.startswith(well_str + "/"):
                        channel_names_cache[(str(store_path), well_str)] = list(pos.channel_names)
                        break

        # Close all opened stores
        for plate in store_cache.values():
            if hasattr(plate, "close"):
                plate.close()

        # Step 2: Derive per-experiment channels from flat (marker, channel_name) columns.
        exp_channels: dict[str, list[ChannelEntry]] = defaultdict(list)
        exp_seen: dict[str, set[str]] = defaultdict(set)
        for (exp_name, marker, ch_name), _ in df.groupby(["experiment", "marker", "channel_name"]):
            exp_name, marker, ch_name = str(exp_name), str(marker), str(ch_name)
            key = (ch_name, marker)
            if key not in exp_seen[exp_name]:
                exp_channels[exp_name].append(ChannelEntry(name=ch_name, marker=marker))
                exp_seen[exp_name].add(key)

        # Step 3: Build ExperimentEntry per experiment.
        experiments: list[ExperimentEntry] = []
        for exp_name, exp_group in df.groupby("experiment"):
            exp_name = str(exp_name)
            store_path = str(exp_group["store_path"].iloc[0])
            first_well = str(exp_group["well"].iloc[0])

            channel_names = channel_names_cache.get((store_path, first_well))
            if channel_names is None:
                raise ValueError(
                    f"Experiment '{exp_name}': could not read channel names from zarr "
                    f"(store_path={store_path}, well={first_well})."
                )

            # Derive perturbation_wells from parquet
            perturbation_wells: dict[str, list[str]] = defaultdict(list)
            seen_pairs: set[tuple[str, str]] = set()
            for _, row in exp_group[["perturbation", "well"]].drop_duplicates().iterrows():
                cond = str(row["perturbation"])
                well = str(row["well"])
                if (cond, well) not in seen_pairs:
                    perturbation_wells[cond].append(well)
                    seen_pairs.add((cond, well))

            marker = ""
            if "marker" in exp_group.columns:
                first_marker = exp_group["marker"].dropna()
                if not first_marker.empty:
                    marker = str(first_marker.iloc[0])

            organelle = ""
            if "organelle" in exp_group.columns:
                first_organelle = exp_group["organelle"].dropna()
                if not first_organelle.empty:
                    organelle = str(first_organelle.iloc[0])

            if "interval_minutes" not in exp_group.columns or exp_group["interval_minutes"].dropna().empty:
                raise ValueError(
                    f"Experiment '{exp_name}': cell index parquet missing 'interval_minutes'. "
                    "Rebuild the parquet with `build-cell-index`."
                )
            interval_minutes = float(exp_group["interval_minutes"].dropna().iloc[0])

            pixel_size_xy_um = None
            if "pixel_size_xy_um" in exp_group.columns:
                ps = exp_group["pixel_size_xy_um"].dropna()
                if not ps.empty:
                    pixel_size_xy_um = float(ps.iloc[0])

            pixel_size_z_um = None
            if "pixel_size_z_um" in exp_group.columns:
                ps = exp_group["pixel_size_z_um"].dropna()
                if not ps.empty:
                    pixel_size_z_um = float(ps.iloc[0])

            experiments.append(
                ExperimentEntry(
                    name=exp_name,
                    data_path=store_path,
                    tracks_path="",
                    channels=exp_channels.get(exp_name, []),
                    channel_names=channel_names,
                    perturbation_wells=dict(perturbation_wells),
                    interval_minutes=interval_minutes,
                    marker=marker,
                    organelle=organelle,
                    pixel_size_xy_um=pixel_size_xy_um,
                    pixel_size_z_um=pixel_size_z_um,
                )
            )

        collection = Collection(
            name=Path(cell_index_path).stem,
            description=f"Auto-generated from cell index: {cell_index_path}",
            experiments=experiments,
        )

        return cls(
            collection=collection,
            z_window=z_window,
            focus_channel=focus_channel,
            reference_pixel_size_xy_um=reference_pixel_size_xy_um,
            reference_pixel_size_z_um=reference_pixel_size_z_um,
        )

    def subset(self, experiment_names: list[str]) -> ExperimentRegistry:
        """Create a new registry with a subset of experiments.

        Parameters
        ----------
        experiment_names : list[str]
            Experiment names to include.

        Returns
        -------
        ExperimentRegistry
            New registry with only the specified experiments.
        """
        subset_experiments = [e for e in self.collection.experiments if e.name in experiment_names]
        subset_collection = Collection(
            name=self.collection.name,
            description=self.collection.description,
            provenance=self.collection.provenance,
            experiments=subset_experiments,
            fov_records=self.collection.fov_records,
        )
        return ExperimentRegistry(
            collection=subset_collection,
            z_window=self.z_window,
            focus_channel=self.focus_channel,
            reference_pixel_size_xy_um=self.reference_pixel_size_xy_um,
            reference_pixel_size_z_um=self.reference_pixel_size_z_um,
        )

    def tau_range_frames(
        self,
        experiment_name: str,
        tau_range_hours: tuple[float, float],
    ) -> tuple[int, int]:
        """Convert a tau range from hours to frames for a given experiment.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment whose ``interval_minutes`` is used.
        tau_range_hours : tuple[float, float]
            ``(min_hours, max_hours)`` range.

        Returns
        -------
        tuple[int, int]
            ``(min_frames, max_frames)`` after conversion.
        """
        exp = self.get_experiment(experiment_name)
        min_frames = round(tau_range_hours[0] * 60 / exp.interval_minutes)
        max_frames = round(tau_range_hours[1] * 60 / exp.interval_minutes)

        if min_frames >= max_frames:
            _logger.warning(
                "Experiment '%s': tau_range_hours=%s yields fewer than 2 valid frames (min=%d, max=%d).",
                experiment_name,
                tau_range_hours,
                min_frames,
                max_frames,
            )

        return (min_frames, max_frames)

    def get_experiment(self, name: str) -> ExperimentEntry:
        """Look up an experiment by name.

        Parameters
        ----------
        name : str
            Experiment name.

        Returns
        -------
        ExperimentEntry

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        try:
            return self._name_map[name]
        except KeyError:
            raise KeyError(f"Experiment '{name}' not found in registry. Available: {list(self._name_map.keys())}")
