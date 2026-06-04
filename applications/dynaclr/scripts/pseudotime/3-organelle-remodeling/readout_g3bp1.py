"""Stage 3 G3BP1 readout: oscillation-aware metrics on the post-window.

Stress granules are transient phase-separated condensates that
assemble and disassemble on minute timescales (per discussion §3.6).
Distance-from-baseline is computed everywhere, but the headline metrics
are oscillation-aware: excursion count, dwell time above mock 95th
percentile, largest excursion amplitude and duration. Computed on the
real-time post-window only (not warped) — warping by NS3 dynamics
would average phase-shifted oscillations to a flat line.

Usage::

    cd applications/dynaclr/scripts/pseudotime/3-organelle-remodeling
    uv run python readout_g3bp1.py \
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
    oscillation_metrics_per_cell,
    per_cell_baseline_distance,
)
from utils import load_stage_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)

ORGANELLE = "g3bp1"
EMBEDDING_KEY = "organelle_g3bp1"


def main() -> None:
    """Compute G3BP1 cosine-distance signal + per-cell oscillation metrics."""
    parser = argparse.ArgumentParser(description="Stage 3 G3BP1 readout (oscillation-aware metrics).")
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-set", required=True)
    parser.add_argument("--track", required=True, choices=["A-anno", "A-LC", "B"])
    parser.add_argument("--baseline-pre-min", type=float, default=-240.0)
    parser.add_argument("--baseline-pre-max", type=float, default=-60.0)
    parser.add_argument("--post-min", type=float, default=180.0, help="Post-window start in minutes (default: 180)")
    parser.add_argument(
        "--post-max",
        type=float,
        default=540.0,
        help=(
            "Post-window end in minutes (default: 540 — extended for G3BP1 plateau "
            "per discussion §3.6 / biologist round 2)"
        ),
    )
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
        raise RuntimeError(f"No {ORGANELLE} embedding zarrs loaded; check pattern + pred_dir")

    signal_df = per_cell_baseline_distance(
        align_df,
        adata_by_dataset,
        baseline_window_minutes=(args.baseline_pre_min, args.baseline_pre_max),
    )

    productive_signal = signal_df[signal_df["cohort"] == "productive"]
    mock_signal = signal_df[signal_df["cohort"] == "mock"]
    if mock_signal["signal"].notna().sum() == 0:
        raise RuntimeError("Mock cohort has no G3BP1 signal; cannot compute oscillation thresholds")
    threshold_df = fov_stratified_threshold(productive_signal, mock_signal, percentile=95.0)

    osc_metrics = oscillation_metrics_per_cell(
        productive_signal,
        threshold_df,
        post_window_minutes=(args.post_min, args.post_max),
    )
    if not osc_metrics.empty:
        summary_cols = [
            "excursion_count",
            "dwell_time_minutes",
            "largest_excursion_amplitude",
            "largest_excursion_duration_minutes",
        ]
        _logger.info(
            f"Per-cell oscillation summary (n={len(osc_metrics)}):\n{osc_metrics[summary_cols].describe().to_string()}"
        )

    out_dir = SCRIPT_DIR / args.track / ORGANELLE
    out_dir.mkdir(parents=True, exist_ok=True)
    signal_path = out_dir / f"{args.candidate_set}_signal.parquet"
    threshold_path = out_dir / f"{args.candidate_set}_threshold.csv"
    osc_path = out_dir / f"{args.candidate_set}_oscillation_metrics.parquet"
    signal_df.to_parquet(signal_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    osc_metrics.to_parquet(osc_path, index=False)
    _logger.info(f"Wrote {signal_path} ({len(signal_df)} rows)")
    _logger.info(f"Wrote {threshold_path} ({len(threshold_df)} FOVs)")
    _logger.info(f"Wrote {osc_path} ({len(osc_metrics)} cells)")


if __name__ == "__main__":
    main()
