"""Lineage reconnection and cohort tagging for Stage 0.

Reuses the library function ``identify_lineages`` from
``dynaclr.pseudotime.alignment`` and adds the per-lineage classifiers
needed by the new DAG: ``divides`` (none/pre/during/post relative to
``t_zero``) and ``cohort`` (productive/bystander/abortive/mock).

The cohort logic is operational and LC-derived per discussion §3.2:
- productive — lineage has manual ``t_key_event``
- bystander — lineage in infected well, LC says uninfected for ≥80% of frames
- abortive — lineage in infected well, LC shows brief positive run < ``min_run`` then no sustained rise
- mock — lineage from uninfected wells (well_pattern in candidates.yaml)
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from dynaclr.pseudotime.alignment import identify_lineages

_logger = logging.getLogger(__name__)


ORPHAN_LINEAGE_ID = ""


def _format_lineage_id(dataset_id: str, fov_name: str, root_tid: int, leaf_tid: int) -> str:
    """Build a content-stable lineage id from its branch endpoints.

    The id is unique across the whole pipeline as long as
    ``(dataset_id, fov_name, root_track_id, leaf_track_id)`` is unique.
    Stable across re-runs regardless of cohort composition or row order.
    """
    return f"{dataset_id}|{fov_name}|root{int(root_tid)}|leaf{int(leaf_tid)}"


def assign_lineage_ids(
    df: pd.DataFrame,
    return_both_branches: bool = True,
) -> pd.DataFrame:
    """Add a content-stable ``lineage_id`` column.

    Calls ``identify_lineages()`` per ``dataset_id`` group; for each branch
    ``[root_track_id, ..., leaf_track_id]`` builds the deterministic id
    ``"{dataset_id}|{fov_name}|root{root_tid}|leaf{leaf_tid}"`` and assigns
    every track in the branch to it. When ``return_both_branches=True`` and
    a mother track appears in multiple branches, the first branch wins (so
    the mother shares an id with one branch only — same convention as
    before).

    Tracks that do not match any lineage get ``lineage_id = ""``
    (the orphan sentinel).

    Parameters
    ----------
    df : pd.DataFrame
        Tracking dataframe with columns: dataset_id, fov_name, track_id,
        parent_track_id.
    return_both_branches : bool
        If True (recommended), keep both daughters as separate lineages
        sharing the mother. If False, only the first daughter survives.

    Returns
    -------
    pd.DataFrame
        Input with a string ``lineage_id`` column added.
    """
    required = {"dataset_id", "fov_name", "track_id", "parent_track_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"assign_lineage_ids missing required columns: {sorted(missing)}")

    if df.empty:
        out = df.copy()
        out["lineage_id"] = pd.Series(dtype=str)
        return out

    track_to_lineage: dict[tuple[str, str, int], str] = {}

    for dataset_id, ds_group in df.groupby("dataset_id", sort=False):
        lineages = identify_lineages(ds_group, return_both_branches=return_both_branches)
        for fov_name, track_ids in lineages:
            if not track_ids:
                continue
            root_tid = int(track_ids[0])
            leaf_tid = int(track_ids[-1])
            lineage_id = _format_lineage_id(str(dataset_id), str(fov_name), root_tid, leaf_tid)
            for tid in track_ids:
                key = (str(dataset_id), str(fov_name), int(tid))
                # When return_both_branches=True a mother can appear in multiple
                # branches; the first branch wins.
                if key not in track_to_lineage:
                    track_to_lineage[key] = lineage_id

    df = df.copy()
    df["lineage_id"] = df.apply(
        lambda row: track_to_lineage.get(
            (str(row["dataset_id"]), str(row["fov_name"]), int(row["track_id"])),
            ORPHAN_LINEAGE_ID,
        ),
        axis=1,
    )

    n_orphan_tracks = df[df["lineage_id"] == ORPHAN_LINEAGE_ID].groupby(["dataset_id", "fov_name", "track_id"]).ngroups
    n_total_tracks = df.groupby(["dataset_id", "fov_name", "track_id"]).ngroups
    if n_orphan_tracks:
        _logger.info(f"{n_orphan_tracks}/{n_total_tracks} tracks did not match any lineage")

    return df


DivideRegime = Literal["none", "pre", "during", "post"]


def classify_divides(
    df: pd.DataFrame,
    t_zero: int,
    k_pre_frames: int,
    k_post_frames: int,
) -> DivideRegime:
    """Classify a lineage's division timing relative to ``t_zero``.

    A division is detected when the lineage contains more than one
    track and the daughter's earliest frame falls inside the lineage.

    Parameters
    ----------
    df : pd.DataFrame
        Rows for a single lineage (one ``lineage_id``).
    t_zero : int
        Anchor frame (manual ``t_key_event`` or LC first-positive).
    k_pre_frames, k_post_frames : int
        Transition sub-window half-widths in frames.

    Returns
    -------
    str
        One of ``"none"``, ``"pre"``, ``"during"``, ``"post"``.
    """
    track_ids = df["track_id"].unique()
    if len(track_ids) < 2:
        return "none"

    # Division frame = earliest frame of the youngest daughter (the track
    # whose first frame is largest among non-root tracks).
    track_starts = df.groupby("track_id")["t"].min().sort_values()
    if len(track_starts) < 2:
        return "none"

    division_frame = int(track_starts.iloc[1])
    transition_lo = t_zero - k_pre_frames
    transition_hi = t_zero + k_post_frames

    if division_frame < transition_lo:
        return "pre"
    if division_frame <= transition_hi:
        return "during"
    return "post"


CohortLabel = Literal["productive", "bystander", "abortive", "unannotated_productive", "mock"]


def tag_cohort_for_lineage(
    df: pd.DataFrame,
    well_is_uninfected: bool,
    has_manual_t_zero: bool,
    lc_predictions: pd.Series | None = None,
    min_run: int = 3,
    bystander_uninfected_fraction: float = 0.8,
) -> CohortLabel:
    """Assign a cohort label to a lineage based on operational rules.

    Per discussion §3.2 and the locked plan:

    - Productive: lineage has a manual ``t_key_event``.
    - Bystander: in an infected well, LC says uninfected for at least
      ``bystander_uninfected_fraction`` of frames AND has no sustained
      positive run.
    - Abortive: in an infected well, LC shows at least one positive frame
      but no sustained run of ``min_run`` consecutive positives.
    - Unannotated productive: in an infected well, LC shows a sustained
      positive run but the lineage has no manual anchor. These are the
      LC's "extra" calls that the manual annotation missed; downstream
      stages may treat them as a separate cohort or exclude them.
    - Mock: in an uninfected well.

    Parameters
    ----------
    df : pd.DataFrame
        Lineage rows.
    well_is_uninfected : bool
        Whether the lineage's well is in the mock cohort.
    has_manual_t_zero : bool
        Whether the lineage has a manual anchor.
    lc_predictions : pd.Series or None
        LC ``predicted_infection_state`` per frame for this lineage
        (sorted by ``t``). ``None`` falls back to ``bystander`` for any
        non-mock, non-productive lineage.
    min_run : int
        Minimum consecutive positives to count as a sustained rise.
    bystander_uninfected_fraction : float
        Fraction of frames a bystander must spend negative (default 0.8).
    """
    if well_is_uninfected:
        return "mock"
    if has_manual_t_zero:
        return "productive"

    if lc_predictions is None:
        return "bystander"

    pos = (lc_predictions == "infected").astype(int).to_numpy()
    if pos.sum() == 0:
        return "bystander"

    if _has_run(pos, min_run):
        # LC found a sustained run but no manual anchor: an unannotated
        # productive candidate. Tag distinctly so downstream stages can
        # decide to include, exclude, or compare against the manual set.
        return "unannotated_productive"

    fraction_negative = 1.0 - (pos.sum() / len(pos))
    if fraction_negative >= bystander_uninfected_fraction:
        return "bystander"

    return "abortive"


def _has_run(arr, min_run: int) -> bool:
    """Return True if ``arr`` contains at least ``min_run`` consecutive 1s."""
    if min_run <= 0:
        return arr.sum() > 0
    run = 0
    for v in arr:
        if v:
            run += 1
            if run >= min_run:
                return True
        else:
            run = 0
    return False
