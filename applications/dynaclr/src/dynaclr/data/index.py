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
from iohub.ngff import Plate, open_ome_zarr

from dynaclr.data.experiment import ExperimentRegistry
from viscy_data.cell_index import read_cell_index

_logger = logging.getLogger(__name__)

__all__ = ["MultiExperimentIndex"]


def _load_experiment_fovs(
    exp_name: str,
    data_path: str,
    tracks_path: str,
    perturbation_wells: dict[str, list[str]],
    marker: str,
    organelle: str,
    microscope: str,
    start_hpi: float,
    interval_minutes: float,
    channel_name: str,
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
    perturbation_wells : dict[str, list[str]]
        Mapping of perturbation label to list of well names.
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
    channel_name : str
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
    for wells in perturbation_wells.values():
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

            # Resolve perturbation from perturbation_wells
            perturbation = None
            for perturbation_label, wells in perturbation_wells.items():
                if well_name in wells:
                    perturbation = perturbation_label
                    break
            if perturbation is None:
                raise ValueError(
                    f"Well '{well_name}' not found in perturbation_wells mapping "
                    f"for experiment '{exp_name}'. Available wells: {dict(perturbation_wells)}"
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
            tracks_df["perturbation"] = perturbation
            tracks_df["marker"] = marker
            tracks_df["organelle"] = organelle
            tracks_df["microscope"] = microscope
            tracks_df["well_name"] = well_name
            tracks_df["fov_name"] = fov_name
            tracks_df["global_track_id"] = exp_name + "_" + fov_name + "_" + tracks_df["track_id"].astype(str)
            tracks_df["cell_id"] = (
                exp_name + "_" + fov_name + "_" + tracks_df["track_id"].astype(str) + "_" + tracks_df["t"].astype(str)
            )
            tracks_df["hours_post_perturbation"] = start_hpi + tracks_df["t"] * interval_minutes / 60.0
            tracks_df["channel_name"] = channel_name

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
    max_border_shift : int
        Maximum pixels to shift a patch center inward for border cells.
        ``-1`` (default) uses quarter of the patch size — border cells get
        a slightly off-center patch but are not excluded.
        ``0`` excludes any cell whose patch would extend beyond the FOV.
        Tiled datasets (OPS) rarely need this; untiled infectomics data does.
    """

    def __init__(
        self,
        registry: ExperimentRegistry,
        yx_patch_size: tuple[int, int],
        tau_range_hours: tuple[float, float] = (0.5, 2.0),
        include_wells: list[str] | None = None,
        exclude_fovs: list[str] | None = None,
        cell_index_path: str | Path | None = None,
        cell_index_df: pd.DataFrame | None = None,
        num_workers: int = 1,
        positive_cell_source: str = "lookup",
        positive_match_columns: list[str] | None = None,
        max_border_shift: int = -1,
        fit: bool = True,
    ) -> None:
        self.registry = registry
        self.yx_patch_size = yx_patch_size
        self.tau_range_hours = tau_range_hours
        # max_border_shift: max pixels to shift patch center for border cells.
        # -1 (default) = half the patch size. 0 = no clamping, exclude border cells.
        if max_border_shift < 0:
            max_border_shift = max(yx_patch_size[0] // 4, yx_patch_size[1] // 4)
        self.max_border_shift = max_border_shift
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

        if cell_index_df is not None or cell_index_path is not None:
            if cell_index_df is not None:
                _logger.info("Using pre-loaded cell index DataFrame (%d rows)", len(cell_index_df))
                tracks = self._align_parquet_columns(cell_index_df.copy())
            else:
                _logger.info("Loading cell index from parquet: %s", cell_index_path)
                tracks = read_cell_index(cell_index_path)
                tracks = self._align_parquet_columns(tracks)
            if include_wells is not None:
                tracks = tracks[tracks["well_name"].isin(include_wells)].copy()
            if all_exclude_fovs is not None:
                tracks = tracks[~tracks["fov_name"].isin(all_exclude_fovs)].copy()
            tracks = self._filter_to_registry_experiments(tracks)
            tracks = self._resolve_dims(tracks)
            # lineage_id already present from build step — skip _reconstruct_lineage
            # Empty frames already filtered at parquet build time — skip _filter_empty_frames
        else:
            all_tracks = self._load_all_experiments(
                include_wells=include_wells, exclude_fovs=all_exclude_fovs, num_workers=num_workers
            )
            tracks = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame()
            tracks = self._reconstruct_lineage(tracks)
            tracks = self._resolve_dims(tracks)

        tracks = self._clamp_borders(tracks)
        self.tracks = tracks.reset_index(drop=True)
        if fit:
            self.valid_anchors = self._compute_valid_anchors(
                tau_range_hours,
                positive_cell_source=positive_cell_source,
                positive_match_columns=positive_match_columns,
            )
            if self.valid_anchors.empty and not self.tracks.empty:
                raise ValueError(
                    f"No valid anchors found from {len(self.tracks)} tracks. "
                    f"positive_cell_source={positive_cell_source!r}, "
                    f"positive_match_columns={positive_match_columns!r}, "
                    f"tau_range_hours={tau_range_hours}. "
                    "Check that tracks have matching positives under these settings."
                )
        else:
            self.valid_anchors = self.tracks

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
        job_args = []
        for exp in self.registry.experiments:
            fluorescence_ch = exp.channels[1].name if len(exp.channels) > 1 else ""
            job_args.append(
                (
                    exp.name,
                    str(exp.data_path),
                    str(exp.tracks_path),
                    dict(exp.perturbation_wells),
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
        ``channel_name``.
        """
        tracks = tracks.rename(columns={"fov": "fov_name", "well": "well_name"})
        # Parquet stores fov as just the position index (e.g. "0");
        # reconstruct the full position path (e.g. "A/1/0") to match
        # the CSV path and exclude_fovs format.
        if "fov_name" in tracks.columns and "well_name" in tracks.columns:
            needs_prefix = ~tracks["fov_name"].str.contains("/")
            if needs_prefix.any():
                tracks.loc[needs_prefix, "fov_name"] = (
                    tracks.loc[needs_prefix, "well_name"] + "/" + tracks.loc[needs_prefix, "fov_name"]
                )
        if "microscope" not in tracks.columns:
            tracks["microscope"] = ""
        return tracks

    def _filter_to_registry_experiments(self, tracks: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows whose experiment is present in the registry."""
        registry_names = {exp.name for exp in self.registry.experiments}
        return tracks[tracks["experiment"].isin(registry_names)].copy()

    def _resolve_dims(self, tracks: pd.DataFrame) -> pd.DataFrame:
        """Attach image dimensions to tracks for border clamping.

        When the parquet has ``Y_shape`` / ``X_shape`` columns (built with the
        latest ``build_timelapse_cell_index``), reads dimensions directly — no
        zarr opens needed.  Falls back to opening stores when the columns are
        missing (old parquets).
        """
        if tracks.empty:
            tracks["_img_height"] = pd.Series(dtype=int)
            tracks["_img_width"] = pd.Series(dtype=int)
            return tracks

        if "Y_shape" in tracks.columns and "X_shape" in tracks.columns:
            tracks["_img_height"] = tracks["Y_shape"]
            tracks["_img_width"] = tracks["X_shape"]
            return tracks

        _logger.warning(
            "Parquet missing Y_shape/X_shape columns. Falling back to opening "
            "zarr stores for image dimensions. Rebuild the parquet with "
            "`build-cell-index` for faster startup."
        )
        dim_lookup: dict[tuple[str, str], tuple[int, int]] = {}
        for (store_path, well_name, fov_name), _group in tracks.groupby(["store_path", "well_name", "fov_name"]):
            if store_path not in self._store_cache:
                self._store_cache[store_path] = open_ome_zarr(store_path, mode="r")
            plate = self._store_cache[store_path]
            if "/" in fov_name:
                position_path = fov_name
            else:
                position_path = f"{well_name}/{fov_name}"
            position = plate[position_path]
            image = position["0"]
            dim_lookup[(store_path, fov_name)] = (image.height, image.width)

        tracks["_img_height"] = [dim_lookup[(sp, fn)][0] for sp, fn in zip(tracks["store_path"], tracks["fov_name"])]
        tracks["_img_width"] = [dim_lookup[(sp, fn)][1] for sp, fn in zip(tracks["store_path"], tracks["fov_name"])]
        return tracks

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
        """Handle cells near the image edge for patch extraction.

        Behavior controlled by ``self.max_border_shift``:

        - When > 0 (default ``half_patch``): cells near the edge have their
          patch center shifted inward so the patch is fully within the FOV.
          Cells requiring a shift larger than ``max_border_shift`` are dropped.
          This keeps most border cells with a slightly off-center patch.
        - When 0: any cell whose patch would extend beyond the FOV is excluded.
          No clamping — all patches are perfectly centered on their cell.
        """
        if tracks.empty:
            return tracks

        y_half = self.yx_patch_size[0] // 2
        x_half = self.yx_patch_size[1] // 2
        max_shift = self.max_border_shift

        # Exclude cells completely outside image
        valid = (
            (tracks["y"] >= 0)
            & (tracks["y"] < tracks["_img_height"])
            & (tracks["x"] >= 0)
            & (tracks["x"] < tracks["_img_width"])
        )
        n_before = len(tracks)
        tracks = tracks[valid].copy()

        if max_shift > 0:
            # Clamp patch center inward so patch is fully within FOV
            tracks["y_clamp"] = np.clip(
                tracks["y"].to_numpy(),
                y_half,
                (tracks["_img_height"] - y_half).to_numpy(),
            )
            tracks["x_clamp"] = np.clip(
                tracks["x"].to_numpy(),
                x_half,
                (tracks["_img_width"] - x_half).to_numpy(),
            )

            # Drop cells where the shift exceeds max_border_shift
            y_shift = np.abs(tracks["y_clamp"].to_numpy() - tracks["y"].to_numpy())
            x_shift = np.abs(tracks["x_clamp"].to_numpy() - tracks["x"].to_numpy())
            keep = (y_shift <= max_shift) & (x_shift <= max_shift)
            tracks = tracks[keep].copy()
        else:
            # No clamping — exclude any cell whose patch would go out of bounds
            in_bounds = (
                (tracks["y"] >= y_half)
                & (tracks["y"] <= tracks["_img_height"] - y_half)
                & (tracks["x"] >= x_half)
                & (tracks["x"] <= tracks["_img_width"] - x_half)
            )
            tracks = tracks[in_bounds].copy()
            tracks["y_clamp"] = tracks["y"]
            tracks["x_clamp"] = tracks["x"]

        n_dropped = n_before - len(tracks)
        if n_dropped > 0:
            _logger.info("Excluded %d border cells (%.1f%%)", n_dropped, 100 * n_dropped / n_before)

        tracks = tracks.drop(columns=["_img_height", "_img_width"])

        return tracks

    def _compute_valid_anchors(
        self,
        tau_range_hours: tuple[float, float],
        positive_cell_source: str = "lookup",
        positive_match_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return the subset of ``self.tracks`` that are valid training anchors.

        When ``positive_cell_source="self"`` or ``positive_match_columns`` does
        not include ``"lineage_id"``, all tracks are valid anchors (no tau
        filtering needed).  Tau filtering only runs for the temporal case where
        ``positive_cell_source="lookup"`` and ``"lineage_id"`` is in
        ``positive_match_columns`` (or ``positive_match_columns`` is ``None``,
        which defaults to lineage-based temporal matching).

        Parameters
        ----------
        tau_range_hours : tuple[float, float]
            ``(min_hours, max_hours)`` used with each experiment's
            ``interval_minutes`` for frame conversion.
        positive_cell_source : str
            ``"self"`` or ``"lookup"``. When ``"self"``, all tracks are valid.
        positive_match_columns : list[str] | None
            Columns used for positive lookup. When ``None`` or contains
            ``"lineage_id"``, tau filtering is applied.

        Returns
        -------
        pd.DataFrame
            Subset of ``self.tracks`` with reset index.
        """
        if self.tracks.empty:
            return self.tracks.copy()

        # Self mode: all tracks are valid anchors (augmentation creates two views).
        if positive_cell_source == "self":
            return self.tracks.reset_index(drop=True)

        # Non-temporal column-match mode: keep only rows whose match-key
        # group has ≥2 members so _find_column_match_positive can exclude
        # the anchor itself and still find a candidate.
        if positive_match_columns is not None and "lineage_id" not in positive_match_columns:
            group_sizes = self.tracks.groupby(list(positive_match_columns)).transform("size")
            return self.tracks[group_sizes >= 2].reset_index(drop=True)

        # Temporal mode: keep only anchors that have a positive at t+tau.
        # For each experiment, check whether (lineage_id, t+tau) exists
        # for any tau in [min_f, max_f] (excluding 0).
        valid_mask = np.zeros(len(self.tracks), dtype=bool)

        for exp in self.registry.experiments:
            min_f, max_f = self.registry.tau_range_frames(exp.name, tau_range_hours)
            exp_mask = self.tracks["experiment"] == exp.name
            exp_df = self.tracks.loc[exp_mask, ["lineage_id", "t"]]
            if exp_df.empty:
                continue

            taus = [tau for tau in range(min_f, max_f + 1) if tau != 0]

            # Unique (lineage_id, t) pairs as a MultiIndex for O(1) isin checks.
            existing = exp_df[["lineage_id", "t"]].drop_duplicates()
            existing_mi = pd.MultiIndex.from_frame(existing)

            # For each unique anchor (lid, t), check if (lid, t+tau) exists for any tau.
            # Iterate over ~15 tau values instead of millions of cells.
            found_any = np.zeros(len(existing), dtype=bool)
            for tau in taus:
                targets = pd.MultiIndex.from_arrays([existing["lineage_id"].to_numpy(), existing["t"].to_numpy() + tau])
                found_any |= targets.isin(existing_mi)

            # Map valid unique pairs back to all rows in the experiment.
            valid_pairs_mi = pd.MultiIndex.from_frame(existing[found_any])
            row_keys = pd.MultiIndex.from_frame(exp_df[["lineage_id", "t"]])
            valid_mask[exp_mask.to_numpy()] = row_keys.isin(valid_pairs_mi)

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
        return {name: group.index.to_numpy() for name, group in self.tracks.groupby("perturbation")}

    def clone_with_subset(
        self,
        tracks_subset: pd.DataFrame,
        positive_cell_source: str = "lookup",
        positive_match_columns: list[str] | None = None,
        max_border_shift: int = -1,
        precomputed_valid_anchors: pd.DataFrame | None = None,
    ) -> "MultiExperimentIndex":
        """Create a shallow copy with a different tracks DataFrame.

        Reuses the parent's registry, positions, and store cache so no
        zarr stores are re-opened.  Recomputes ``valid_anchors`` unless
        ``precomputed_valid_anchors`` is provided.

        Parameters
        ----------
        tracks_subset : pd.DataFrame
            Subset of ``self.tracks`` (will be reset-indexed).
        positive_cell_source : str
            Forwarded to ``_compute_valid_anchors``.
        positive_match_columns : list[str] | None
            Forwarded to ``_compute_valid_anchors``.
        max_border_shift : int
            Forwarded to ``self.max_border_shift``. -1 inherits from parent.
        precomputed_valid_anchors : pd.DataFrame | None
            When provided, skip recomputing valid anchors. Pass the already-
            filtered valid_anchors subset for this tracks_subset. Avoids
            redundant O(N * tau_range) computation in FOV split mode.
        """
        clone = object.__new__(MultiExperimentIndex)
        clone.registry = self.registry
        clone.yx_patch_size = self.yx_patch_size
        clone.tau_range_hours = self.tau_range_hours
        clone._store_cache = self._store_cache
        clone.max_border_shift = self.max_border_shift if max_border_shift < 0 else max_border_shift
        clone.tracks = tracks_subset.reset_index(drop=True)
        if precomputed_valid_anchors is not None:
            clone.valid_anchors = precomputed_valid_anchors.reset_index(drop=True)
        else:
            clone.valid_anchors = clone._compute_valid_anchors(
                tau_range_hours=self.tau_range_hours,
                positive_cell_source=positive_cell_source,
                positive_match_columns=positive_match_columns,
            )
        if clone.valid_anchors.empty and not clone.tracks.empty:
            raise ValueError(
                f"No valid anchors found from {len(clone.tracks)} tracks in subset. "
                f"positive_cell_source={positive_cell_source!r}, "
                f"positive_match_columns={positive_match_columns!r}. "
                "Check that the subset has matching positives under these settings."
            )
        return clone

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
            cond_counts = exp_tracks.groupby("perturbation").size()
            cond_str = ", ".join(f"{c}({n})" for c, n in cond_counts.items())
            lines.append(
                f"  {exp.name}: {len(exp_tracks)} observations, {len(exp_anchors)} anchors, conditions: {cond_str}"
            )
        return "\n".join(lines)
