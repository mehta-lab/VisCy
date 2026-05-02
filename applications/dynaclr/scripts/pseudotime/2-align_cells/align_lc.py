"""Path A-LC alignment: real-time shift on linear-classifier predictions.

Reads the productive-cohort CSV from Stage 0, pulls
``predicted_infection_state`` from the NS3 channel embedding zarr,
anchors each lineage at the first frame of a sustained positive run
(at least ``--min-run`` consecutive positives), and writes a per-frame
alignment parquet with the unified schema. No DTW, no template, no
warping.

The ``--min-run`` parameter (default 3 frames) prevents single-frame
LC flickers from defining ``t_zero``.

Bystander, abortive, and mock cohorts pass through with
``t_zero = NaN`` for use as null distributions in Stage 3.

Usage::

    cd applications/dynaclr/scripts/pseudotime/2-align_cells
    uv run python align_lc.py \
        --datasets ../../../configs/pseudotime/datasets.yaml \
        --config ../../../configs/pseudotime/candidates.yaml \
        --candidate-set zikv_productive_07_24 \
        --min-run 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATES_DIR = SCRIPT_DIR.parent / "0-select_candidates" / "candidates"
OUTPUT_DIR = SCRIPT_DIR / "A-LC" / "alignments"

sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils import load_stage_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)

# Same parquet schema as align_anno.py — track_path differs.
PARQUET_COLUMNS = [
    "dataset_id",
    "fov_name",
    "lineage_id",
    "track_id",
    "t",
    "cohort",
    "divides",
    "t_zero",
    "t_rel_minutes",
    "track_path",
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
    """Map dataset_id → frame_interval_minutes."""
    return {d["dataset_id"]: float(d["frame_interval_minutes"]) for d in config["datasets"]}


def _embedding_dir_lookup(config: dict) -> dict[str, Path]:
    """Map dataset_id → directory containing the NS3 channel embedding zarrs."""
    return {d["dataset_id"]: Path(d["pred_dir"]) for d in config["datasets"]}


def _load_lc_predictions(
    pred_dir: Path,
    dataset_id: str,
    pattern: str,
    pred_column: str,
) -> pd.DataFrame:
    """Read LC predictions per (fov_name, track_id, t) from a date-matched zarr.

    Filters glob matches to the dataset's date prefix (first three
    underscore-separated tokens) so multi-date pred_dirs pick the
    correct zarr.
    """
    date_prefix = "_".join(dataset_id.split("_")[:3])
    matches = [m for m in pred_dir.glob(pattern) if m.name.startswith(date_prefix)]
    if not matches:
        _logger.warning(f"[{dataset_id}] no embedding zarr matched {pattern} with prefix {date_prefix} in {pred_dir}")
        return pd.DataFrame(columns=["fov_name", "track_id", "t", pred_column])
    if len(matches) > 1:
        _logger.warning(
            f"[{dataset_id}] multiple zarrs matched {pattern} with prefix {date_prefix}: "
            f"{[m.name for m in matches]}; using first"
        )
    adata = ad.read_zarr(matches[0])
    adata.obs_names_make_unique()
    if pred_column not in adata.obs.columns:
        _logger.warning(f"{pred_column} not in {matches[0]} .obs")
        return pd.DataFrame(columns=["fov_name", "track_id", "t", pred_column])
    obs = adata.obs[["fov_name", "track_id", "t", pred_column]].copy()
    obs["fov_name"] = obs["fov_name"].astype(str)
    obs["track_id"] = obs["track_id"].astype(int)
    obs["t"] = obs["t"].astype(int)
    return obs


def _first_run_start(positive_mask: np.ndarray, min_run: int) -> int | None:
    """Return the index of the first frame entering a run of ≥ ``min_run`` 1s."""
    run = 0
    run_start = -1
    for i, v in enumerate(positive_mask):
        if v:
            if run == 0:
                run_start = i
            run += 1
            if run >= min_run:
                return run_start
        else:
            run = 0
            run_start = -1
    return None


def _t_zero_from_lc(
    productive_df: pd.DataFrame,
    lc_obs_by_dataset: dict[str, pd.DataFrame],
    pred_column: str,
    positive_value: str,
    min_run: int,
) -> dict[str, int]:
    """For each productive lineage, find the LC anchor frame.

    Joins productive cohort rows with LC predictions on
    ``(dataset_id, fov_name, track_id, t)``, then per lineage finds the
    first frame entering a run of at least ``min_run`` consecutive
    positive predictions.
    """
    out: dict[str, int] = {}
    if productive_df.empty:
        return out

    for lineage_id, g in productive_df.groupby("lineage_id"):
        if not lineage_id:
            continue
        ds_id = str(g["dataset_id"].iloc[0])
        if ds_id not in lc_obs_by_dataset or lc_obs_by_dataset[ds_id].empty:
            continue
        # Pull the LC predictions for this lineage's tracks.
        lc_df = lc_obs_by_dataset[ds_id]
        track_ids = set(g["track_id"].astype(int).unique())
        sub = lc_df[lc_df["track_id"].isin(track_ids)]
        if sub.empty:
            continue
        # Sort by t and find the first sustained positive run.
        sub = sub.sort_values("t")
        positive_mask = (sub[pred_column] == positive_value).to_numpy()
        run_start_idx = _first_run_start(positive_mask, min_run)
        if run_start_idx is None:
            continue
        out[str(lineage_id)] = int(sub["t"].iloc[run_start_idx])

    return out


def _align_cohort(
    cohort_df: pd.DataFrame,
    cohort: str,
    t_zero_lookup: dict[str, int],
    frame_intervals: dict[str, float],
) -> pd.DataFrame:
    """Add ``t_zero``, ``t_rel_minutes``, ``track_path`` columns to a cohort frame."""
    if cohort_df.empty:
        return pd.DataFrame(columns=PARQUET_COLUMNS)

    out = cohort_df.copy()
    out["t_zero"] = out["lineage_id"].map(t_zero_lookup)
    frame_interval_per_row = out["dataset_id"].map(frame_intervals)
    if frame_interval_per_row.isna().any():
        unknown = sorted(set(out["dataset_id"][frame_interval_per_row.isna()]))
        raise KeyError(f"Frame interval missing for dataset(s) {unknown}")

    has_anchor = out["t_zero"].notna()
    out["t_rel_minutes"] = np.where(
        has_anchor,
        (out["t"] - out["t_zero"].fillna(0)) * frame_interval_per_row,
        np.nan,
    )
    out["track_path"] = "A-LC"

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
    """Write Path A-LC alignment parquet for one candidate set."""
    parser = argparse.ArgumentParser(description="Path A-LC alignment (LC-anchored real-time shift).")
    parser.add_argument("--datasets", required=True, help="Path to datasets.yaml")
    parser.add_argument("--config", required=True, help="Path to candidates.yaml")
    parser.add_argument("--candidate-set", required=True, help="Candidate-set name")
    parser.add_argument("--min-run", type=int, default=3, help="Min consecutive positives for t_zero (default: 3)")
    parser.add_argument(
        "--pred-column",
        default="predicted_infection_state",
        help="LC prediction column in the embedding zarr's .obs",
    )
    parser.add_argument("--positive-value", default="infected", help="LC positive class label (default: infected)")
    args = parser.parse_args()

    config = load_stage_config(args.datasets, args.config)
    frame_intervals = _frame_interval_lookup(config)
    pred_dirs = _embedding_dir_lookup(config)
    sensor_pattern = config.get("embeddings", {}).get("sensor", "*_viral_sensor_*.zarr")

    cohorts = ["productive", "bystander", "abortive", "unannotated_productive", "mock"]
    cohort_dfs: dict[str, pd.DataFrame] = {}
    for cohort in cohorts:
        path = CANDIDATES_DIR / f"{args.candidate_set}_{cohort}.csv"
        if path.exists():
            cohort_dfs[cohort] = pd.read_csv(path)
            _logger.info(f"Read {path} ({len(cohort_dfs[cohort])} rows)")
        else:
            _logger.warning(f"{path} not found; cohort '{cohort}' empty")
            cohort_dfs[cohort] = pd.DataFrame()

    productive_df = cohort_dfs["productive"]
    if productive_df.empty:
        raise RuntimeError(f"Productive cohort empty for {args.candidate_set!r}.")

    # Load LC predictions per dataset present in the productive cohort.
    datasets_in_use = sorted(productive_df["dataset_id"].unique())
    lc_obs_by_dataset: dict[str, pd.DataFrame] = {}
    for ds_id in datasets_in_use:
        if ds_id not in pred_dirs:
            _logger.warning(f"dataset_id {ds_id!r} missing from datasets.yaml; LC anchor unavailable")
            continue
        lc_obs_by_dataset[ds_id] = _load_lc_predictions(
            pred_dirs[ds_id], dataset_id=ds_id, pattern=sensor_pattern, pred_column=args.pred_column
        )

    t_zero_lookup = _t_zero_from_lc(
        productive_df,
        lc_obs_by_dataset,
        pred_column=args.pred_column,
        positive_value=args.positive_value,
        min_run=args.min_run,
    )
    _logger.info(f"Computed LC t_zero for {len(t_zero_lookup)} productive lineages (min_run={args.min_run})")

    aligned_parts = [_align_cohort(cohort_dfs[c], c, t_zero_lookup, frame_intervals) for c in cohorts]
    aligned = pd.concat(aligned_parts, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.candidate_set}.parquet"
    aligned.to_parquet(out_path, index=False)
    n_with_anchor = aligned["t_zero"].notna().sum()
    _logger.info(f"Wrote {out_path} ({len(aligned)} rows, {n_with_anchor} anchored)")


if __name__ == "__main__":
    main()
