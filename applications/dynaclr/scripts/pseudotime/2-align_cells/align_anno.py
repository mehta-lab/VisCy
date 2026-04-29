"""Path A-anno alignment: real-time shift on human ``infection_state``.

Reads the productive-cohort CSV from Stage 0, anchors each lineage at
``t_zero = first frame where infection_state == "infected"``, and writes
a per-frame alignment parquet with the unified schema. No DTW, no
template, no warping.

For bystander, abortive, and mock cohorts: no per-cell anchor — these
cohorts pass through with ``t_zero = NaN`` and ``t_rel_minutes = NaN``,
to be used as null distributions in Stage 3 readouts.

Path A reuses :func:`dynaclr.pseudotime.alignment.assign_t_perturb` for
the per-lineage anchor logic; the rest of this script is parquet
plumbing.

Usage::

    cd applications/dynaclr/scripts/pseudotime/2-align_cells
    uv run python align_anno.py \
        --datasets ../../../configs/pseudotime/datasets.yaml \
        --config ../../../configs/pseudotime/candidates.yaml \
        --candidate-set zikv_productive_07_24
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATES_DIR = SCRIPT_DIR.parent / "0-select_candidates" / "candidates"
OUTPUT_DIR = SCRIPT_DIR / "A-anno" / "alignments"

sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils import load_stage_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)

# Unified Stage 2 parquet schema (per DAG §7.4). Path A populates the
# real-time columns; Path B-only columns are NaN/empty here.
PARQUET_COLUMNS = [
    # ids
    "dataset_id",
    "fov_name",
    "lineage_id",
    "track_id",
    "t",
    # cohort + lineage tags from Stage 0
    "cohort",
    "divides",
    # anchor + real-time
    "t_zero",
    "t_rel_minutes",
    "track_path",
    # Path B-only (left empty for Path A)
    "pseudotime",
    "alignment_region",
    "t_rel_minutes_warped",
    "dtw_cost",
    "length_normalized_cost",
    "path_skew",
    "match_q_start",
    "match_q_end",
    "template_id",
]


def _frame_interval_lookup(config: dict) -> dict[str, float]:
    """Map dataset_id → frame_interval_minutes from the merged config."""
    return {d["dataset_id"]: float(d["frame_interval_minutes"]) for d in config["datasets"]}


def _t_zero_from_annotations(
    cohort_df: pd.DataFrame,
    anchor_label: str,
    anchor_positive: str,
) -> dict[int, int]:
    """First frame per lineage where ``anchor_label == anchor_positive``."""
    out: dict[int, int] = {}
    if cohort_df.empty:
        return out
    positive_rows = cohort_df[cohort_df[anchor_label] == anchor_positive]
    for lineage_id, g in positive_rows.groupby("lineage_id"):
        if int(lineage_id) < 0:
            continue
        out[int(lineage_id)] = int(g["t"].min())
    return out


def _align_cohort(
    cohort_df: pd.DataFrame,
    cohort: str,
    t_zero_lookup: dict[int, int],
    frame_intervals: dict[str, float],
) -> pd.DataFrame:
    """Add ``t_zero``, ``t_rel_minutes``, ``track_path`` to a cohort frame.

    Lineages without an anchor in ``t_zero_lookup`` (bystander, abortive,
    mock) get NaN for ``t_zero`` and ``t_rel_minutes``.
    """
    if cohort_df.empty:
        out = pd.DataFrame(columns=PARQUET_COLUMNS)
        return out

    out = cohort_df.copy()

    out["t_zero"] = out["lineage_id"].map(t_zero_lookup)
    frame_interval_per_row = out["dataset_id"].map(frame_intervals)
    if frame_interval_per_row.isna().any():
        unknown = sorted(set(out["dataset_id"][frame_interval_per_row.isna()]))
        raise KeyError(f"Frame interval missing for dataset(s) {unknown}; check datasets.yaml entries.")

    has_anchor = out["t_zero"].notna()
    out["t_rel_minutes"] = np.where(
        has_anchor,
        (out["t"] - out["t_zero"].fillna(0)) * frame_interval_per_row,
        np.nan,
    )
    out["track_path"] = "A-anno"

    # Path B-only columns left empty.
    out["pseudotime"] = np.nan
    out["alignment_region"] = ""
    out["t_rel_minutes_warped"] = np.nan
    out["dtw_cost"] = np.nan
    out["length_normalized_cost"] = np.nan
    out["path_skew"] = np.nan
    out["match_q_start"] = pd.NA
    out["match_q_end"] = pd.NA
    out["template_id"] = ""

    for col in PARQUET_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[PARQUET_COLUMNS]


def main() -> None:
    """Write Path A-anno alignment parquet for one candidate set."""
    parser = argparse.ArgumentParser(description="Path A-anno alignment (annotation-anchored real-time shift).")
    parser.add_argument("--datasets", required=True, help="Path to datasets.yaml")
    parser.add_argument("--config", required=True, help="Path to candidates.yaml")
    parser.add_argument("--candidate-set", required=True, help="Candidate-set name")
    parser.add_argument(
        "--anchor-label",
        default="infection_state",
        help="Annotation column to anchor on (default: infection_state)",
    )
    parser.add_argument(
        "--anchor-positive",
        default="infected",
        help="Positive value of anchor column (default: infected)",
    )
    args = parser.parse_args()

    config = load_stage_config(args.datasets, args.config)
    frame_intervals = _frame_interval_lookup(config)

    cohorts = ["productive", "bystander", "abortive", "mock"]
    cohort_dfs: dict[str, pd.DataFrame] = {}
    for cohort in cohorts:
        path = CANDIDATES_DIR / f"{args.candidate_set}_{cohort}.csv"
        if path.exists():
            cohort_dfs[cohort] = pd.read_csv(path)
            _logger.info(f"Read {path} ({len(cohort_dfs[cohort])} rows)")
        else:
            _logger.warning(f"{path} not found; cohort '{cohort}' will be empty")
            cohort_dfs[cohort] = pd.DataFrame()

    productive_df = cohort_dfs["productive"]
    if productive_df.empty:
        raise RuntimeError(f"Productive cohort empty for {args.candidate_set!r}. Run select_candidates.py first.")

    t_zero_lookup = _t_zero_from_annotations(
        productive_df,
        anchor_label=args.anchor_label,
        anchor_positive=args.anchor_positive,
    )
    _logger.info(f"Computed t_zero for {len(t_zero_lookup)} productive lineages")

    aligned_parts = [_align_cohort(cohort_dfs[c], c, t_zero_lookup, frame_intervals) for c in cohorts]
    aligned = pd.concat(aligned_parts, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.candidate_set}.parquet"
    aligned.to_parquet(out_path, index=False)
    n_with_anchor = aligned["t_zero"].notna().sum()
    _logger.info(f"Wrote {out_path} ({len(aligned)} rows, {n_with_anchor} anchored)")


if __name__ == "__main__":
    main()
