"""Track alignment and lineage-aware T_perturb assignment.

Provides functions to identify cell lineages from tracking data,
filter tracks by FOV pattern and length, and assign perturbation
onset times (T_perturb) using lineage-aware logic.

Ported from:
- dtw_clean:viscy/representation/pseudotime.py (identify_lineages, filter_tracks)
- .ed_planning/tmp/scripts/annotation_remodling.py (assign_infection_times)
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

_logger = logging.getLogger(__name__)


def identify_lineages(
    tracking_df: pd.DataFrame,
    return_both_branches: bool = False,
) -> list[tuple[str, list[int]]]:
    """Identify distinct lineages from cell tracking parent-child relationships.

    Builds a parent-child graph from (fov_name, track_id, parent_track_id)
    and traverses it to find connected lineage branches.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Tracking dataframe with columns: fov_name, track_id, parent_track_id.
    return_both_branches : bool
        If True, return both branches after division as separate lineages.
        If False, return only the first branch per root.

    Returns
    -------
    list[tuple[str, list[int]]]
        List of (fov_name, [track_ids]) per lineage branch.
    """
    all_lineages = []

    for fov_id, fov_df in tracking_df.groupby("fov_name"):
        # Create child-to-parent mapping
        child_to_parent = {}
        for track_id, track_group in fov_df.groupby("track_id"):
            parent_track_id = track_group.iloc[0]["parent_track_id"]
            if parent_track_id != -1:
                child_to_parent[track_id] = parent_track_id

        # Find root tracks (no parent or parent not in dataset)
        all_tracks = set(fov_df["track_id"].unique())
        root_tracks = set()
        for track_id in all_tracks:
            track_data = fov_df[fov_df["track_id"] == track_id]
            parent = track_data.iloc[0]["parent_track_id"]
            if parent == -1 or parent not in all_tracks:
                root_tracks.add(track_id)

        # Build parent-to-children mapping
        parent_to_children: dict[int, list[int]] = {}
        for child, parent in child_to_parent.items():
            parent_to_children.setdefault(parent, []).append(child)

        def _get_all_branches(track_id: int) -> list[list[int]]:
            """Recursively get all branches from a track."""
            branches = []
            current = [track_id]
            if track_id in parent_to_children:
                for child in parent_to_children[track_id]:
                    for branch in _get_all_branches(child):
                        branches.append(current + branch)
            else:
                branches.append(current)
            return branches

        for root_track in root_tracks:
            lineage_tracks = _get_all_branches(root_track)
            if return_both_branches:
                for branch in lineage_tracks:
                    all_lineages.append((fov_id, branch))
            else:
                all_lineages.append((fov_id, lineage_tracks[0]))

    return all_lineages


def filter_tracks(
    df: pd.DataFrame,
    fov_pattern: str | list[str] | None = None,
    min_timepoints: int = 1,
) -> pd.DataFrame:
    """Filter tracking data by FOV pattern and minimum track length.

    Parameters
    ----------
    df : pd.DataFrame
        Tracking dataframe with columns: fov_name, track_id, t.
    fov_pattern : str or list[str] or None
        Pattern(s) to match FOV names via str.contains (OR logic for lists).
        If None, no FOV filtering is applied.
    min_timepoints : int
        Minimum number of timepoints required per track.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    result = df.copy()

    # FOV filtering
    if fov_pattern is not None:
        patterns = [fov_pattern] if isinstance(fov_pattern, str) else fov_pattern
        fov_mask = pd.Series(False, index=result.index)
        for pattern in patterns:
            fov_mask |= result["fov_name"].astype(str).str.contains(
                pattern, regex=False
            )
        result = result[fov_mask].copy()
        if len(result) == 0:
            _logger.warning(f"No FOVs matched pattern(s): {patterns}")
            return result

    # Track length filtering
    if min_timepoints > 1:
        track_lengths = result.groupby(["fov_name", "track_id"]).size()
        valid_tracks = track_lengths[track_lengths >= min_timepoints].index
        result = (
            result.set_index(["fov_name", "track_id"])
            .loc[valid_tracks]
            .reset_index()
        )

    return result


def assign_t_perturb(
    df: pd.DataFrame,
    frame_interval_minutes: float,
    source: Literal["annotation", "prediction"] = "annotation",
    infection_col: str = "infection_state",
    infected_value: str = "infected",
    min_track_timepoints: int = 3,
) -> pd.DataFrame:
    """Assign T_perturb via lineage-aware alignment.

    For each lineage (connected tracks via parent_track_id), finds the
    earliest frame annotated/predicted as infected and assigns that as
    T_perturb for all tracks in the lineage. Orphan tracks (not part of
    any lineage) are handled individually.

    Parameters
    ----------
    df : pd.DataFrame
        Tracking dataframe with columns: fov_name, track_id, t,
        parent_track_id, and the infection column.
    frame_interval_minutes : float
        Time interval between frames in minutes.
    source : {"annotation", "prediction"}
        Whether to read infection state from the annotation column directly
        or from a ``predicted_`` prefixed column.
    infection_col : str
        Column name for infection state.
    infected_value : str
        Value indicating infected state.
    min_track_timepoints : int
        Minimum track length after alignment; shorter tracks are dropped.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: t_perturb (int), t_relative_minutes (float).
        Tracks with no detected infection are dropped.
    """
    df = df.copy()

    # Ensure parent_track_id exists
    if "parent_track_id" not in df.columns:
        df["parent_track_id"] = -1

    # Determine which column to read infection from
    col = f"predicted_{infection_col}" if source == "prediction" else infection_col

    if col not in df.columns:
        raise KeyError(
            f"Column '{col}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )

    lineages = identify_lineages(df, return_both_branches=True)

    # Map (fov, track_id) → t_perturb
    track_to_tperturb: dict[tuple[str, int], int] = {}
    tracks_in_lineages: set[tuple[str, int]] = set()

    for fov_name, track_ids in lineages:
        lineage_rows = df[
            (df["fov_name"] == fov_name) & (df["track_id"].isin(track_ids))
        ]
        infected = lineage_rows[lineage_rows[col] == infected_value]
        if len(infected) == 0:
            continue
        t_perturb = int(infected["t"].min())
        for tid in track_ids:
            track_to_tperturb[(fov_name, tid)] = t_perturb
            tracks_in_lineages.add((fov_name, tid))

    n_lineage_tracks = len(tracks_in_lineages)

    # Handle orphan tracks (not in any lineage)
    n_orphan_tracks = 0
    for (fov_name, tid), group in df.groupby(["fov_name", "track_id"]):
        if (fov_name, tid) in tracks_in_lineages:
            continue
        infected = group[group[col] == infected_value]
        if len(infected) > 0:
            track_to_tperturb[(fov_name, tid)] = int(infected["t"].min())
            n_orphan_tracks += 1

    # Apply t_perturb
    df["t_perturb"] = df.apply(
        lambda row: track_to_tperturb.get((row["fov_name"], row["track_id"])),
        axis=1,
    )

    # Drop tracks without infection
    df = df.dropna(subset=["t_perturb"])

    # Filter short tracks
    if min_track_timepoints > 1:
        track_lengths = df.groupby(["fov_name", "track_id"]).size()
        valid_tracks = track_lengths[track_lengths >= min_track_timepoints].index
        df = df.set_index(["fov_name", "track_id"]).loc[valid_tracks].reset_index()

    df["t_perturb"] = df["t_perturb"].astype(int)
    df["t_relative_minutes"] = (df["t"] - df["t_perturb"]) * frame_interval_minutes

    _logger.info(
        f"Tracks with infection: {len(track_to_tperturb)} "
        f"(lineage: {n_lineage_tracks}, orphan: {n_orphan_tracks})"
    )

    return df


def align_tracks(
    df: pd.DataFrame,
    frame_interval_minutes: float,
    source: Literal["annotation", "prediction"] = "annotation",
    infection_col: str = "infection_state",
    infected_value: str = "infected",
    min_track_timepoints: int = 3,
    fov_pattern: str | list[str] | None = None,
) -> pd.DataFrame:
    """Convenience wrapper: filter_tracks + assign_t_perturb in one call.

    Parameters
    ----------
    df : pd.DataFrame
        Tracking dataframe.
    frame_interval_minutes : float
        Time interval between frames in minutes.
    source : {"annotation", "prediction"}
        Infection state source.
    infection_col : str
        Column name for infection state.
    infected_value : str
        Value indicating infected state.
    min_track_timepoints : int
        Minimum track length after alignment.
    fov_pattern : str or list[str] or None
        FOV pattern for filtering. None skips FOV filtering.

    Returns
    -------
    pd.DataFrame
        Aligned dataframe with t_perturb and t_relative_minutes columns.
    """
    filtered = filter_tracks(df, fov_pattern=fov_pattern, min_timepoints=1)
    return assign_t_perturb(
        filtered,
        frame_interval_minutes=frame_interval_minutes,
        source=source,
        infection_col=infection_col,
        infected_value=infected_value,
        min_track_timepoints=min_track_timepoints,
    )
