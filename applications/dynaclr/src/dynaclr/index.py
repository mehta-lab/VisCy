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
    border-clamped centroids.  When *tau_range_hours* is provided, also
    computes ``valid_anchors`` -- the subset of rows that have at least one
    temporal positive (same lineage) at any tau in the configured range.

    Parameters
    ----------
    registry : ExperimentRegistry
        Validated collection of experiment configurations.
    z_range : slice
        Z-slice range for data loading.
    yx_patch_size : tuple[int, int]
        Patch size (height, width) used for border clamping.
    tau_range_hours : tuple[float, float]
        ``(min_hours, max_hours)`` converted to frames per experiment
        via :meth:`ExperimentRegistry.tau_range_frames`.
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
        tau_range_hours: tuple[float, float] = (0.5, 2.0),
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
        self.valid_anchors = self._compute_valid_anchors(tau_range_hours)

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

    def _compute_valid_anchors(
        self, tau_range_hours: tuple[float, float]
    ) -> pd.DataFrame:
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
            min_f, max_f = self.registry.tau_range_frames(
                exp.name, tau_range_hours
            )
            exp_mask = self.tracks["experiment"] == exp.name
            exp_tracks = self.tracks[exp_mask]

            # Build set of (lineage_id, t) pairs for O(1) lookup
            lineage_timepoints: set[tuple[str, int]] = set(
                zip(exp_tracks["lineage_id"], exp_tracks["t"])
            )

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
        return {
            name: group.index.to_numpy()
            for name, group in self.tracks.groupby("experiment")
        }

    @property
    def condition_groups(self) -> dict[str, np.ndarray]:
        """Group ``self.tracks`` row indices by condition label.

        Returns
        -------
        dict[str, np.ndarray]
            ``{condition_label: array_of_row_indices}``.
        """
        return {
            name: group.index.to_numpy()
            for name, group in self.tracks.groupby("condition")
        }

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
            exp_anchors = self.valid_anchors[
                self.valid_anchors["experiment"] == exp.name
            ]
            cond_counts = exp_tracks.groupby("condition").size()
            cond_str = ", ".join(
                f"{c}({n})" for c, n in cond_counts.items()
            )
            lines.append(
                f"  {exp.name}: {len(exp_tracks)} observations, "
                f"{len(exp_anchors)} anchors, conditions: {cond_str}"
            )
        return "\n".join(lines)
