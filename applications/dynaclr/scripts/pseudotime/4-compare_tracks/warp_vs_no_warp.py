"""Stage 4 warp-vs-no-warp comparator (Path B only).

Per discussion §3.8 #10 / §4.5: this is the mandatory comparator that
forces the data to answer whether Path B's transition-window warp
sharpens organelle timing relative to using real-time alone. For each
(organelle, cohort) we compute the per-cell onset distribution under
two readout axes derived from Path B's parquet:

- ``t_rel_minutes`` (real-time relative to per-cell t_zero) — equivalent
  to Path A on the same cohort.
- ``t_rel_minutes_warped`` (back-projected real-time from DTW warp).

If the median onset times agree to within 25%, the warp is neutral and
Path B's warp propagation is kept for sharpness without distortion. If
they diverge by more than 25%, the warp is masking real organelle
timing — flag and investigate.

Usage::

    cd applications/dynaclr/scripts/pseudotime/4-compare_tracks
    uv run python warp_vs_no_warp.py \
        --datasets ../../../configs/pseudotime/datasets.yaml \
        --config ../../../configs/pseudotime/compare_tracks.yaml \
        --comparison zikv_07_24_full
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
READOUT_ROOT = SCRIPT_DIR.parent / "3-organelle-remodeling"
OUTPUT_DIR = SCRIPT_DIR / "comparisons"

sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils import load_stage_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def _per_cell_onset(signal_df: pd.DataFrame, threshold_df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """First time (real-time or warped) where signal exceeds per-FOV threshold."""
    threshold_lookup = dict(zip(threshold_df["fov_name"].astype(str), threshold_df["threshold"]))
    rows = []
    productive = signal_df[
        (signal_df["cohort"] == "productive") & signal_df["signal"].notna() & signal_df[time_col].notna()
    ]
    for (ds, fov, tid), g in productive.groupby(["dataset_id", "fov_name", "track_id"]):
        threshold = threshold_lookup.get(str(fov))
        if threshold is None:
            continue
        g = g.sort_values(time_col)
        above = g["signal"].to_numpy() > threshold
        if not above.any():
            continue
        first_idx = int(np.argmax(above))
        rows.append(
            {
                "dataset_id": ds,
                "fov_name": str(fov),
                "track_id": int(tid),
                "onset_minutes": float(g[time_col].iloc[first_idx]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Compare warped vs unwarped onset times under Path B."""
    parser = argparse.ArgumentParser(description="Stage 4 warp-vs-no-warp comparator (Path B only).")
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--comparison", required=True)
    args = parser.parse_args()

    config = load_stage_config(args.datasets, args.config)
    cmp_cfg = config["comparisons"][args.comparison]
    candidate_set = cmp_cfg["candidate_set"]
    organelles = cmp_cfg.get("organelles", ["sec61", "g3bp1", "phase"])
    threshold_fraction = float(cmp_cfg.get("methodological_threshold_fraction", 0.25))

    rows = []
    for organelle in organelles:
        signal_path = READOUT_ROOT / "B" / organelle / f"{candidate_set}_signal.parquet"
        threshold_path = READOUT_ROOT / "B" / organelle / f"{candidate_set}_threshold.csv"
        if not signal_path.exists():
            _logger.warning(f"Missing Path B signal for {organelle}; skipping")
            continue
        signal_df = pd.read_parquet(signal_path)
        threshold_df = pd.read_csv(threshold_path)

        unwarped = _per_cell_onset(signal_df, threshold_df, "t_rel_minutes")
        warped_col = "t_rel_minutes_warped" if "t_rel_minutes_warped" in signal_df.columns else "t_rel_minutes"
        warped = _per_cell_onset(signal_df, threshold_df, warped_col)

        unwarped_median = float(unwarped["onset_minutes"].median()) if not unwarped.empty else float("nan")
        warped_median = float(warped["onset_minutes"].median()) if not warped.empty else float("nan")
        diff = (
            float(abs(warped_median - unwarped_median))
            if not (np.isnan(warped_median) or np.isnan(unwarped_median))
            else float("nan")
        )
        ref = float(abs(unwarped_median)) if not np.isnan(unwarped_median) else float("nan")
        diverges = bool(not np.isnan(diff) and ref > 0 and diff / ref > threshold_fraction)
        rows.append(
            {
                "organelle": organelle,
                "n_unwarped": int(len(unwarped)),
                "n_warped": int(len(warped)),
                "median_unwarped_minutes": unwarped_median,
                "median_warped_minutes": warped_median,
                "absolute_diff_minutes": diff,
                "diverges": diverges,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.comparison}_warp_vs_no_warp.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    _logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
