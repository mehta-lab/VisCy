"""Experiment registry for multi-experiment DynaCLR training.

Provides :class:`ExperimentRegistry` — a validated collection with channel
resolution, tau-range conversion, and Z-range auto-resolution, backed by
:class:`~viscy_data.collection.Collection`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from iohub.ngff import open_ome_zarr

from viscy_data.collection import Collection, ExperimentEntry, SourceChannel, load_collection

_logger = logging.getLogger(__name__)

__all__ = ["ExperimentRegistry"]


@dataclass
class ExperimentRegistry:
    """Validated collection of experiments with channel and Z resolution.

    On creation (``__post_init__``), the registry performs fail-fast validation:

    1. Experiments list must not be empty.
    2. Experiment names must be unique.
    3. Source channel mappings must reference valid channel names (experiments
       may omit a source channel — not every experiment needs every channel).
    4. ``interval_minutes`` must be positive for each experiment.
    5. ``condition_wells`` must not be empty for each experiment.
    6. ``data_path`` must point to an existing directory.
    7. Zarr metadata channel names must match ``channel_names``.

    After validation the registry computes:

    * ``num_source_channels`` -- common count of source channels.
    * ``channel_maps`` -- per-experiment mapping of source position to zarr
      channel index.
    * ``z_ranges`` -- per-experiment ``(z_start, z_end)`` ranges.

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
    num_source_channels: int = field(init=False)
    channel_maps: dict[str, dict[int, int]] = field(init=False)
    norm_meta_key_maps: dict[str, dict[str, str]] = field(init=False)
    z_ranges: dict[str, tuple[int, int]] = field(init=False)
    scale_factors: dict[str, tuple[float, float, float]] = field(init=False)

    # internal lookup
    _name_map: dict[str, ExperimentEntry] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
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

            # 5. Empty condition_wells
            if not exp.condition_wells:
                raise ValueError(f"Experiment '{exp.name}': condition_wells must not be empty.")

            # 6. data_path existence
            if not Path(exp.data_path).exists():
                raise ValueError(f"Experiment '{exp.name}': data_path does not exist: {exp.data_path}")

            # 7. Zarr channel validation
            with open_ome_zarr(exp.data_path, mode="r") as plate:
                first_position = next(iter(plate.positions()))[1]
                zarr_channels = list(first_position.channel_names)
            if zarr_channels != exp.channel_names:
                raise ValueError(
                    f"Experiment '{exp.name}': channel_names mismatch. "
                    f"Expected (from config): {exp.channel_names}, "
                    f"got (from zarr): {zarr_channels}."
                )

        # Compute channel_maps from source_channels
        # Experiments may not have all source channels — skip missing ones.
        source_channels = self.collection.source_channels
        self.channel_maps = {}
        for exp in experiments:
            self.channel_maps[exp.name] = {
                i: exp.channel_names.index(sc.per_experiment[exp.name])
                for i, sc in enumerate(source_channels)
                if exp.name in sc.per_experiment
            }

        # Build norm_meta key maps: zarr channel name -> source label
        self.norm_meta_key_maps = {}
        for exp in experiments:
            self.norm_meta_key_maps[exp.name] = {
                sc.per_experiment[exp.name]: sc.label for sc in source_channels if exp.name in sc.per_experiment
            }

        # Validate consistent source channel count
        self.num_source_channels = len(source_channels)

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
        """Return the list of source channel labels."""
        return [sc.label for sc in self.collection.source_channels]

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
            first_sc = self.collection.source_channels[0] if self.collection.source_channels else None
            focus_ch = self.focus_channel or (first_sc.per_experiment.get(exp.name) if first_sc else None)

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
        name_set = set(experiment_names)
        subset_source_channels = [
            SourceChannel(
                label=sc.label,
                per_experiment={k: v for k, v in sc.per_experiment.items() if k in name_set},
            )
            for sc in self.collection.source_channels
        ]
        subset_collection = Collection(
            name=self.collection.name,
            description=self.collection.description,
            provenance=self.collection.provenance,
            source_channels=subset_source_channels,
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
