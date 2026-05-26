"""Stage 3 SEC61 readout: cosine distance from per-cell pre-baseline.

Reads a Stage 2 alignment parquet (Path A-anno, A-LC, or B), looks up
the matching SEC61 channel embeddings, computes per-cell cosine
distance from the pre-window baseline, aggregates across cells, and
emits a per-cohort population curve and per-cell timing metrics.

Per discussion §3.6: SEC61 dynamics are monotone and structural;
distance-from-baseline is the right scalar readout. FOV-stratified
mock null per discussion §3.7.

Usage::

    cd applications/dynaclr/scripts/pseudotime/3-organelle-remodeling
    uv run python readout_sec61.py \
        --datasets ../../../configs/pseudotime/datasets.yaml \
        --config ../../../configs/pseudotime/candidates.yaml \
        --candidate-set zikv_productive_07_24 \
        --track A-anno
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ALIGN_ROOT = SCRIPT_DIR.parent / "2-align_cells"

sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))
from readout_common import (  # noqa: E402
    fov_stratified_threshold,
    load_alignment_parquet,
    load_organelle_embeddings,
    per_cell_baseline_distance,
)
from utils import load_stage_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)

ORGANELLE = "sec61"
EMBEDDING_KEY = "organelle_sec61"


def main() -> None:
    """Compute per-cell SEC61 cosine-distance signal and per-FOV mock thresholds."""
    parser = argparse.ArgumentParser(description="Stage 3 SEC61 readout (cosine distance from baseline).")
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-set", required=True)
    parser.add_argument("--track", required=True, choices=["A-anno", "A-LC", "B"])
    parser.add_argument("--baseline-pre-min", type=float, default=-240.0)
    parser.add_argument("--baseline-pre-max", type=float, default=-60.0)
    args = parser.parse_args()

    config = load_stage_config(args.datasets, args.config)
    dataset_cfgs = {d["dataset_id"]: d for d in config["datasets"]}
    embedding_pattern = config.get("embeddings", {}).get(EMBEDDING_KEY)
    if embedding_pattern is None:
        raise KeyError(f"datasets.yaml embeddings.{EMBEDDING_KEY} not set")

    align_df = load_alignment_parquet(ALIGN_ROOT, args.track, args.candidate_set)
    _logger.info(f"Loaded {len(align_df)} alignment rows for track {args.track}")

    datasets_in_use = sorted(align_df["dataset_id"].unique())
    adata_by_dataset = load_organelle_embeddings(dataset_cfgs, datasets_in_use, embedding_pattern)
    if not adata_by_dataset:
        raise RuntimeError(f"No {ORGANELLE} embedding zarrs loaded for any dataset; check pattern + pred_dir")

    signal_df = per_cell_baseline_distance(
        align_df,
        adata_by_dataset,
        baseline_window_minutes=(args.baseline_pre_min, args.baseline_pre_max),
    )
    n_with_signal = int(signal_df["signal"].notna().sum())
    _logger.info(f"Computed cosine distance for {n_with_signal}/{len(signal_df)} rows")

    productive_signal = signal_df[signal_df["cohort"] == "productive"]
    mock_signal = signal_df[signal_df["cohort"] == "mock"]
    if mock_signal["signal"].notna().sum() == 0:
        _logger.warning("Mock cohort has no SEC61 signal; thresholds will fall back to global")
    threshold_df = fov_stratified_threshold(productive_signal, mock_signal, percentile=95.0)
    _logger.info(f"FOV-stratified mock 95th-percentile thresholds: {threshold_df['threshold'].describe().to_dict()}")

    out_dir = SCRIPT_DIR / args.track / ORGANELLE
    out_dir.mkdir(parents=True, exist_ok=True)
    signal_path = out_dir / f"{args.candidate_set}_signal.parquet"
    threshold_path = out_dir / f"{args.candidate_set}_threshold.csv"
    signal_df.to_parquet(signal_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    _logger.info(f"Wrote {signal_path} ({len(signal_df)} rows)")
    _logger.info(f"Wrote {threshold_path} ({len(threshold_df)} FOVs)")


if __name__ == "__main__":
    main()
