"""Unified cell observation index across multiple experiments.

Provides :class:`MultiExperimentIndex` which builds a flat DataFrame
(``self.tracks``) from all experiments in an :class:`ExperimentRegistry`,
with one row per cell observation per timepoint, enriched with experiment
metadata, lineage links, and border-clamped centroids.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from iohub.ngff import Plate, Position, open_ome_zarr

from dynaclr.data.experiment import ExperimentRegistry
from viscy_data.cell_index import read_cell_index

_logger = logging.getLogger(__name__)

__all__ = ["MultiExperimentIndex"]


def _load_experiment_fovs(
    exp_name: str,
    data_path: str,
    tracks_path: str,
    condition_wells: dict[str, list[str]],
    marker: str,
    organelle: str,
    microscope: str,
    start_hpi: float,
    interval_minutes: float,
    fluorescence_channel: str,
    include_wells: list[str] | None,
    exclude_fovs: list[str] | None,
) -> list[pd.DataFrame]:
    """Load all FOV track DataFrames for one experiment (no Position objects).

    Module-level for ProcessPoolExecutor picklability.

    Parameters
    ----------
    exp_name : str
        Experiment name.
    data_path : str
        Path to the OME-Zarr plate store.
    tracks_path : str
        Root directory of tracking CSVs.
    condition_wells : dict[str, list[str]]
        Mapping of condition label to list of well names.
    marker : str
        Marker name.
    organelle : str
        Organelle name.
    microscope : str
        Microscope identifier.
    start_hpi : float
        Hours post perturbation at t=0.
    interval_minutes : float
        Minutes per frame.
    fluorescence_channel : str
        Fluorescence channel name for this experiment.
    include_wells : list[str] | None
        If provided, only include these wells.
    exclude_fovs : list[str] | None
        If provided, exclude these FOVs.

    Returns
    -------
    list[pd.DataFrame]
        One DataFrame per FOV with store_path/fov_name but no position column
        (resolved later by _resolve_positions_and_dims).
    """
    registered_wells: set[str] = set()
    for wells in condition_wells.values():
        registered_wells.update(wells)

    fov_dfs: list[pd.DataFrame] = []

    with open_ome_zarr(data_path, mode="r") as plate:
        for _pos_path, position in plate.positions():
            fov_name = position.zgroup.name.strip("/")
            parts = fov_name.split("/")
            well_name = "/".join(parts[:2])

            if well_name not in registered_wells:
                continue
            if include_wells is not None and well_name not in include_wells:
                continue
            if exclude_fovs is not None and fov_name in exclude_fovs:
                continue

            # Resolve condition from condition_wells
            condition = None
            for condition_label, wells in condition_wells.items():
                if well_name in wells:
                    condition = condition_label
                    break
            if condition is None:
                raise ValueError(
                    f"Well '{well_name}' not found in condition_wells mapping "
                    f"for experiment '{exp_name}'. Available wells: {dict(condition_wells)}"
                )

            # Read tracking CSV
            tracks_dir = Path(tracks_path) / fov_name
            csv_files = list(tracks_dir.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No tracking CSV in {tracks_dir}")
            if len(csv_files) > 1:
                raise ValueError(f"Expected exactly one tracking CSV in {tracks_dir}, found: {csv_files}")
            tracks_df = pd.read_csv(csv_files[0])

            # Enrich columns
            tracks_df["store_path"] = data_path
            tracks_df["experiment"] = exp_name
            tracks_df["condition"] = condition
            tracks_df["marker"] = marker
            tracks_df["organelle"] = organelle
            tracks_df["microscope"] = microscope
            tracks_df["well_name"] = well_name
            tracks_df["fov_name"] = fov_name
            tracks_df["global_track_id"] = exp_name + "_" + fov_name + "_" + tracks_df["track_id"].astype(str)
            tracks_df["hours_post_perturbation"] = start_hpi + tracks_df["t"] * interval_minutes / 60.0
            tracks_df["fluorescence_channel"] = fluorescence_channel

            fov_dfs.append(tracks_df)

    return fov_dfs


class MultiExperimentIndex:
    """Unified cell observation index across multiple experiments.

    Builds a flat DataFrame (``self.tracks``) with one row per cell observation
    per timepoint, enriched with experiment metadata, lineage links, and
    border-clamped centroids.  When *tau_range_hours* is provided, also
    computes ``valid_anchors`` -- the subset of rows that have at least one
    temporal positive (same lineage) at any tau in the configured range.

    Parameters
    ----------
    registry : ExperimentRegistry
        Validated collection of experiment configurations.  Must have
        resolved ``z_ranges`` (per-experiment Z slices).
    yx_patch_size : tuple[int, int]
        Patch size (height, width) used for border clamping.
    tau_range_hours : tuple[float, float]
        ``(min_hours, max_hours)`` converted to frames per experiment
        via :meth:`ExperimentRegistry.tau_range_frames`.
    include_wells : list[str] | None
        If provided, only include positions from these wells (e.g. ``["A/1"]``).
    exclude_fovs : list[str] | None
        If provided, exclude these FOVs by name (e.g. ``["A/1/0"]``).
    cell_index_path : str | Path | None
        Optional path to a pre-built cell index parquet (from
        ``build_timelapse_cell_index``).  When provided, tracks are loaded
        from the parquet instead of traversing every zarr store and CSV,
        dramatically speeding up startup.
    num_workers : int
        Number of parallel processes for loading experiments. Default 1
        (sequential). When > 1, dispatches one process per experiment via
        ``ProcessPoolExecutor``. Ignored when *cell_index_path* is provided.
    """

    def __init__(
        self,
        registry: ExperimentRegistry,
        yx_patch_size: tuple[int, int],
        tau_range_hours: tuple[float, float] = (0.5, 2.0),
        include_wells: list[str] | None = None,
        exclude_fovs: list[str] | None = None,
        cell_index_path: str | Path | None = None,
        num_workers: int = 1,
    ) -> None:
        self.registry = registry
        self.yx_patch_size = yx_patch_size
        self._store_cache: dict[str, Plate] = {}

        # Merge collection-level exclude_fovs with runtime exclude_fovs
        collection_excludes: set[str] = set()
        for exp in registry.experiments:
            collection_excludes.update(exp.exclude_fovs)
        if exclude_fovs is not None:
            all_exclude_fovs = list(collection_excludes | set(exclude_fovs))
        elif collection_excludes:
            all_exclude_fovs = list(collection_excludes)
        else:
            all_exclude_fovs = None

        if cell_index_path is not None:
            _logger.info("Loading cell index from parquet: %s", cell_index_path)
            tracks = read_cell_index(cell_index_path)
            tracks = self._align_parquet_columns(tracks)
            if include_wells is not None:
                tracks = tracks[tracks["well_name"].isin(include_wells)].copy()
            if all_exclude_fovs is not None:
                tracks = tracks[~tracks["fov_name"].isin(all_exclude_fovs)].copy()
            tracks = self._filter_to_registry_experiments(tracks)
            positions, tracks = self._resolve_positions_and_dims(tracks)
            self.positions = positions
            # lineage_id already present from build step — skip _reconstruct_lineage
        else:
            all_tracks = self._load_all_experiments(
                include_wells=include_wells, exclude_fovs=all_exclude_fovs, num_workers=num_workers
            )
            tracks = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame()
            tracks = self._reconstruct_lineage(tracks)
            positions, tracks = self._resolve_positions_and_dims(tracks)
            self.positions = positions

        tracks = self._clamp_borders(tracks)
        self.tracks = tracks.reset_index(drop=True)
        self.valid_anchors = self._compute_valid_anchors(tau_range_hours)

    # ------- internal methods -------

    def _load_all_experiments(
        self,
        include_wells: list[str] | None,
        exclude_fovs: list[str] | None,
        num_workers: int,
    ) -> list[pd.DataFrame]:
        """Load enriched track DataFrames for every experiment.

        Parameters
        ----------
        include_wells : list[str] | None
            If provided, only include these wells.
        exclude_fovs : list[str] | None
            If provided, exclude these FOVs.
        num_workers : int
            Number of parallel processes. 1 = sequential.

        Returns
        -------
        list[pd.DataFrame]
            All per-FOV DataFrames (no Position objects; resolved later).
        """
        source_channels = self.registry.collection.source_channels

        job_args = []
        for exp in self.registry.experiments:
            fluorescence_ch = source_channels[1].per_experiment.get(exp.name, "") if len(source_channels) > 1 else ""
            job_args.append(
                (
                    exp.name,
                    str(exp.data_path),
                    str(exp.tracks_path),
                    dict(exp.condition_wells),
                    exp.marker,
                    exp.organelle,
                    exp.microscope,
                    exp.start_hpi,
                    exp.interval_minutes,
                    fluorescence_ch,
                    include_wells,
                    exclude_fovs,
                )
            )

        if num_workers == 1:
            results = []
            for args in job_args:
                _logger.info("Building cell index for experiment: %s", args[0])
                results.append(_load_experiment_fovs(*args))
        else:
            results = [None] * len(job_args)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_load_experiment_fovs, *args): (i, args[0]) for i, args in enumerate(job_args)
                }
                for future in as_completed(futures):
                    idx, exp_name = futures[future]
                    _logger.info("Finished loading experiment: %s", exp_name)
                    results[idx] = future.result()

        all_tracks = [df for fov_dfs in results for df in fov_dfs]
        _logger.info(
            "Cell index built: %d FOVs across %d experiments",
            len(all_tracks),
            len(self.registry.experiments),
        )
        return all_tracks

    @staticmethod
    def _align_parquet_columns(tracks: pd.DataFrame) -> pd.DataFrame:
        """Rename parquet columns to match runtime expectations.

        The cell index parquet uses ``fov``, ``well``, ``channel_name``
        while the runtime code expects ``fov_name``, ``well_name``,
        ``fluorescence_channel``.
        """
        tracks = tracks.rename(columns={"fov": "fov_name", "well": "well_name", "channel_name": "fluorescence_channel"})
        if "microscope" not in tracks.columns:
            tracks["microscope"] = ""
        return tracks

    def _filter_to_registry_experiments(self, tracks: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows whose experiment is present in the registry."""
        registry_names = {exp.name for exp in self.registry.experiments}
        return tracks[tracks["experiment"].isin(registry_names)].copy()

    def _resolve_positions_and_dims(self, tracks: pd.DataFrame) -> tuple[list[Position], pd.DataFrame]:
        """Open zarr stores for unique (store_path, fov_name) pairs.

        Attaches ``position``, ``_img_height``, ``_img_width`` columns to
        *tracks* and returns the list of resolved Position objects.
        """
        all_positions: list[Position] = []
        pos_lookup: dict[tuple[str, str], Position] = {}
        dim_lookup: dict[tuple[str, str], tuple[int, int]] = {}

        if tracks.empty:
            tracks["position"] = pd.Series(dtype=object)
            tracks["_img_height"] = pd.Series(dtype=int)
            tracks["_img_width"] = pd.Series(dtype=int)
            return all_positions, tracks

        for (store_path, fov_name), _group in tracks.groupby(["store_path", "fov_name"]):
            if store_path not in self._store_cache:
                self._store_cache[store_path] = open_ome_zarr(store_path, mode="r")
            plate = self._store_cache[store_path]
            position = plate[fov_name]
            pos_lookup[(store_path, fov_name)] = position
            image = position["0"]
            dim_lookup[(store_path, fov_name)] = (image.height, image.width)
            all_positions.append(position)

        tracks["position"] = [pos_lookup[(sp, fn)] for sp, fn in zip(tracks["store_path"], tracks["fov_name"])]
        tracks["_img_height"] = [dim_lookup[(sp, fn)][0] for sp, fn in zip(tracks["store_path"], tracks["fov_name"])]
        tracks["_img_width"] = [dim_lookup[(sp, fn)][1] for sp, fn in zip(tracks["store_path"], tracks["fov_name"])]

        return all_positions, tracks

    @staticmethod
    def _reconstruct_lineage(tracks: pd.DataFrame) -> pd.DataFrame:
        """Add lineage_id column linking daughters to root ancestor.

        Each track's ``lineage_id`` is set to the ``global_track_id`` of
        its root ancestor. Tracks without a ``parent_track_id`` (or whose
        parent is not present in the data) are their own root.
        """
        if tracks.empty:
            tracks["lineage_id"] = pd.Series(dtype=str)
            return tracks

        # Default: each track is its own lineage
        tracks["lineage_id"] = tracks["global_track_id"].copy()

        if "parent_track_id" not in tracks.columns:
            return tracks

        # Build parent->child mapping per experiment+fov and propagate lineage
        for (exp, fov), group in tracks.groupby(["experiment", "fov_name"]):
            # Map track_id -> global_track_id within this FOV
            tid_to_gtid: dict[int, str] = dict(zip(group["track_id"], group["global_track_id"]))

            # Build parent graph: child_gtid -> parent_gtid
            parent_map: dict[str, str] = {}
            for _, row in group.drop_duplicates("track_id").iterrows():
                ptid = row.get("parent_track_id")
                if pd.notna(ptid) and int(ptid) in tid_to_gtid:
                    parent_map[row["global_track_id"]] = tid_to_gtid[int(ptid)]

            # Chase to root for each track
            def _find_root(gtid: str) -> str:
                visited: set[str] = set()
                current = gtid
                while current in parent_map and current not in visited:
                    visited.add(current)
                    current = parent_map[current]
                return current

            mask = (tracks["experiment"] == exp) & (tracks["fov_name"] == fov)
            for gtid in group["global_track_id"].unique():
                root = _find_root(gtid)
                tracks.loc[mask & (tracks["global_track_id"] == gtid), "lineage_id"] = root

        return tracks

    def _clamp_borders(self, tracks: pd.DataFrame) -> pd.DataFrame:
        """Clamp centroids inward instead of excluding border cells.

        Cells whose centroids are completely outside the image boundary
        (``y < 0``, ``y >= height``, ``x < 0``, ``x >= width``) are excluded.
        All other cells have their centroids clamped to ensure valid patch
        extraction: ``y_clamp`` and ``x_clamp`` are at least ``half_patch``
        from the edges.
        """
        if tracks.empty:
            return tracks

        y_half = self.yx_patch_size[0] // 2
        x_half = self.yx_patch_size[1] // 2

        # Exclude cells completely outside image
        valid = (
            (tracks["y"] >= 0)
            & (tracks["y"] < tracks["_img_height"])
            & (tracks["x"] >= 0)
            & (tracks["x"] < tracks["_img_width"])
        )
        tracks = tracks[valid].copy()

        # Clamp inward
        tracks["y_clamp"] = np.clip(
            tracks["y"].values,
            y_half,
            (tracks["_img_height"] - y_half).values,
        )
        tracks["x_clamp"] = np.clip(
            tracks["x"].values,
            x_half,
            (tracks["_img_width"] - x_half).values,
        )

        # Drop internal columns
        tracks = tracks.drop(columns=["_img_height", "_img_width"])

        return tracks

    def _compute_valid_anchors(self, tau_range_hours: tuple[float, float]) -> pd.DataFrame:
        """Return the subset of ``self.tracks`` that are valid training anchors.

        An anchor is valid when there exists at least one tau in the
        per-experiment frame range such that another row with the **same
        lineage_id** and ``t == anchor_t + tau`` is present in the tracks.

        Parameters
        ----------
        tau_range_hours : tuple[float, float]
            ``(min_hours, max_hours)`` used with each experiment's
            ``interval_minutes`` for frame conversion.

        Returns
        -------
        pd.DataFrame
            Subset of ``self.tracks`` with reset index.
        """
        if self.tracks.empty:
            return self.tracks.copy()

        valid_mask = pd.Series(False, index=self.tracks.index)

        for exp in self.registry.experiments:
            min_f, max_f = self.registry.tau_range_frames(exp.name, tau_range_hours)
            exp_mask = self.tracks["experiment"] == exp.name
            exp_tracks = self.tracks[exp_mask]

            # Build set of (lineage_id, t) pairs for O(1) lookup
            lineage_timepoints: set[tuple[str, int]] = set(zip(exp_tracks["lineage_id"], exp_tracks["t"]))

            for idx, row in exp_tracks.iterrows():
                for tau in range(min_f, max_f + 1):
                    if tau == 0:
                        continue  # anchor cannot be its own positive
                    if (row["lineage_id"], row["t"] + tau) in lineage_timepoints:
                        valid_mask[idx] = True
                        break

        return self.tracks[valid_mask].reset_index(drop=True)

    # ------- public properties / methods -------

    @property
    def experiment_groups(self) -> dict[str, np.ndarray]:
        """Group ``self.tracks`` row indices by experiment name.

        Returns
        -------
        dict[str, np.ndarray]
            ``{experiment_name: array_of_row_indices}``.
        """
        return {name: group.index.to_numpy() for name, group in self.tracks.groupby("experiment")}

    @property
    def condition_groups(self) -> dict[str, np.ndarray]:
        """Group ``self.tracks`` row indices by condition label.

        Returns
        -------
        dict[str, np.ndarray]
            ``{condition_label: array_of_row_indices}``.
        """
        return {name: group.index.to_numpy() for name, group in self.tracks.groupby("condition")}

    def summary(self) -> str:
        """Return a human-readable overview of the index.

        Returns
        -------
        str
            Multi-line string with experiment counts, observation counts,
            anchor counts, and per-experiment condition breakdowns.
        """
        lines = [
            f"MultiExperimentIndex: {len(self.registry.experiments)} experiments, "
            f"{len(self.tracks)} total observations, "
            f"{len(self.valid_anchors)} valid anchors"
        ]
        for exp in self.registry.experiments:
            exp_tracks = self.tracks[self.tracks["experiment"] == exp.name]
            exp_anchors = self.valid_anchors[self.valid_anchors["experiment"] == exp.name]
            cond_counts = exp_tracks.groupby("condition").size()
            cond_str = ", ".join(f"{c}({n})" for c, n in cond_counts.items())
            lines.append(
                f"  {exp.name}: {len(exp_tracks)} observations, {len(exp_anchors)} anchors, conditions: {cond_str}"
            )
        return "\n".join(lines)
