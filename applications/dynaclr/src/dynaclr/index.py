"""Unified cell observation index across multiple experiments.

Provides :class:`MultiExperimentIndex` which builds a flat DataFrame
(``self.tracks``) from all experiments in an :class:`ExperimentRegistry`,
with one row per cell observation per timepoint, enriched with experiment
metadata, lineage links, and border-clamped centroids.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from iohub.ngff import Position, open_ome_zarr

from dynaclr.experiment import ExperimentRegistry

_logger = logging.getLogger(__name__)

__all__ = ["MultiExperimentIndex"]


class MultiExperimentIndex:
    """Unified cell observation index across multiple experiments.

    Builds a flat DataFrame (``self.tracks``) with one row per cell observation
    per timepoint, enriched with experiment metadata, lineage links, and
    border-clamped centroids.

    Parameters
    ----------
    registry : ExperimentRegistry
        Validated collection of experiment configurations.
    z_range : slice
        Z-slice range for data loading.
    yx_patch_size : tuple[int, int]
        Patch size (height, width) used for border clamping.
    include_wells : list[str] | None
        If provided, only include positions from these wells (e.g. ``["A/1"]``).
    exclude_fovs : list[str] | None
        If provided, exclude these FOVs by name (e.g. ``["A/1/0"]``).
    """

    def __init__(
        self,
        registry: ExperimentRegistry,
        z_range: slice,
        yx_patch_size: tuple[int, int],
        include_wells: list[str] | None = None,
        exclude_fovs: list[str] | None = None,
    ) -> None:
        self.registry = registry
        self.z_range = z_range
        self.yx_patch_size = yx_patch_size

        positions, tracks_dfs = self._load_all_experiments(
            include_wells=include_wells, exclude_fovs=exclude_fovs
        )
        self.positions = positions
        tracks = (
            pd.concat(tracks_dfs, ignore_index=True)
            if tracks_dfs
            else pd.DataFrame()
        )
        tracks = self._reconstruct_lineage(tracks)
        tracks = self._clamp_borders(tracks)
        self.tracks = tracks.reset_index(drop=True)

    # ------- internal methods -------

    def _load_all_experiments(
        self,
        include_wells: list[str] | None,
        exclude_fovs: list[str] | None,
    ) -> tuple[list[Position], list[pd.DataFrame]]:
        """Load positions and enriched tracks for every experiment."""
        all_positions: list[Position] = []
        all_tracks: list[pd.DataFrame] = []

        for exp in self.registry.experiments:
            plate = open_ome_zarr(exp.data_path, mode="r")
            for _pos_path, position in plate.positions():
                fov_name = position.zgroup.name.strip("/")
                # well_name is the first two path components (e.g. "A/1")
                parts = fov_name.split("/")
                well_name = "/".join(parts[:2])

                if include_wells is not None and well_name not in include_wells:
                    continue
                if exclude_fovs is not None and fov_name in exclude_fovs:
                    continue

                # Resolve condition from experiment's condition_wells
                condition = self._resolve_condition(exp, well_name)

                # Read tracking CSV
                tracks_dir = Path(exp.tracks_path) / fov_name
                csv_files = list(tracks_dir.glob("*.csv"))
                if not csv_files:
                    _logger.warning(
                        "No tracking CSV in %s, skipping", tracks_dir
                    )
                    continue
                tracks_df = pd.read_csv(csv_files[0])

                # Enrich columns
                tracks_df["experiment"] = exp.name
                tracks_df["condition"] = condition
                tracks_df["well_name"] = well_name
                tracks_df["fov_name"] = fov_name
                tracks_df["global_track_id"] = (
                    exp.name
                    + "_"
                    + fov_name
                    + "_"
                    + tracks_df["track_id"].astype(str)
                )
                tracks_df["hours_post_infection"] = (
                    exp.start_hpi + tracks_df["t"] * exp.interval_minutes / 60.0
                )
                fluorescence_ch = (
                    exp.source_channel[1]
                    if len(exp.source_channel) > 1
                    else ""
                )
                tracks_df["fluorescence_channel"] = fluorescence_ch
                tracks_df["position"] = [position] * len(tracks_df)

                # Store image dims for border clamping
                image = position["0"]
                tracks_df["_img_height"] = image.height
                tracks_df["_img_width"] = image.width

                all_positions.append(position)
                all_tracks.append(tracks_df)

        return all_positions, all_tracks

    @staticmethod
    def _resolve_condition(exp, well_name: str) -> str:
        """Map well_name to condition label from exp.condition_wells."""
        for condition_label, wells in exp.condition_wells.items():
            if well_name in wells:
                return condition_label
        return "unknown"

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
            tid_to_gtid: dict[int, str] = dict(
                zip(group["track_id"], group["global_track_id"])
            )

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
                tracks.loc[
                    mask & (tracks["global_track_id"] == gtid), "lineage_id"
                ] = root

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
