"""Stage 4 claim (a'): Spearman ρ of phase onset vs fluorescent-marker onset.

Per discussion §2.1 / DAG §9.2: claim (a') succeeds iff per-cell phase
onset times correlate (ρ ≥ 0.20) with the matched fluorescent-marker
onset times across the productive cohort. Falsifier is per-organelle:
SEC61 carries weight; G3BP1 expected null is positive evidence for
fluorescence-and-phase complementarity.

Per-cell onset time = first frame where signal exceeds the per-FOV
mock 95th percentile. ρ computed via scipy.stats.spearmanr; p-value
via 1000-shuffle permutation null.

Usage::

    cd applications/dynaclr/scripts/pseudotime/4-compare_tracks
    uv run python compare_phase_to_fluor.py \
        --datasets ../../../configs/pseudotime/datasets.yaml \
        --config ../../../configs/pseudotime/compare_tracks.yaml \
        --comparison zikv_07_24_full \
        --organelle sec61 \
        --track A-anno
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
READOUT_ROOT = SCRIPT_DIR.parent / "3-organelle-remodeling"
OUTPUT_DIR = SCRIPT_DIR / "comparisons"

sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils import load_stage_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def _per_cell_onset(signal_df: pd.DataFrame, threshold_df: pd.DataFrame) -> pd.DataFrame:
    """First t_rel_minutes where signal exceeds the per-FOV threshold."""
    threshold_lookup = dict(zip(threshold_df["fov_name"].astype(str), threshold_df["threshold"]))
    rows = []
    productive = signal_df[(signal_df["cohort"] == "productive") & signal_df["signal"].notna()]
    for (ds, fov, tid), g in productive.groupby(["dataset_id", "fov_name", "track_id"]):
        threshold = threshold_lookup.get(str(fov))
        if threshold is None:
            continue
        g = g.sort_values("t_rel_minutes")
        above = g["signal"].to_numpy() > threshold
        if not above.any():
            continue
        first_idx = int(np.argmax(above))
        rows.append(
            {
                "dataset_id": ds,
                "fov_name": str(fov),
                "track_id": int(tid),
                "lineage_id": int(g["lineage_id"].iloc[0]),
                "onset_t_rel_minutes": float(g["t_rel_minutes"].iloc[first_idx]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Compute Spearman ρ between phase and matched-fluor onset times."""
    parser = argparse.ArgumentParser(description="Stage 4 claim (a') Spearman ρ.")
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--organelle", required=True, choices=["sec61", "g3bp1"])
    parser.add_argument("--track", required=True, choices=["A-anno", "A-LC", "B"])
    parser.add_argument("--n-permutations", type=int, default=1000)
    args = parser.parse_args()

    config = load_stage_config(args.datasets, args.config)
    cmp_cfg = config["comparisons"][args.comparison]
    candidate_set = cmp_cfg["candidate_set"]

    fluor_signal = pd.read_parquet(READOUT_ROOT / args.track / args.organelle / f"{candidate_set}_signal.parquet")
    fluor_threshold = pd.read_csv(READOUT_ROOT / args.track / args.organelle / f"{candidate_set}_threshold.csv")
    phase_signal = pd.read_parquet(READOUT_ROOT / args.track / "phase" / f"{candidate_set}_signal.parquet")
    phase_threshold = pd.read_csv(READOUT_ROOT / args.track / "phase" / f"{candidate_set}_threshold.csv")

    fluor_onset = _per_cell_onset(fluor_signal, fluor_threshold)
    phase_onset = _per_cell_onset(phase_signal, phase_threshold)

    paired = fluor_onset.merge(
        phase_onset,
        on=["dataset_id", "fov_name", "track_id", "lineage_id"],
        suffixes=("_fluor", "_phase"),
    )
    n_paired = len(paired)
    if n_paired < 5:
        _logger.warning(f"Only {n_paired} paired cells; ρ not informative")
    rho, _p = spearmanr(paired["onset_t_rel_minutes_fluor"], paired["onset_t_rel_minutes_phase"])

    # Permutation null.
    rng = np.random.default_rng(seed=0)
    null_rhos = []
    fluor_arr = paired["onset_t_rel_minutes_fluor"].to_numpy()
    phase_arr = paired["onset_t_rel_minutes_phase"].to_numpy()
    for _ in range(args.n_permutations):
        shuffled = rng.permutation(phase_arr)
        r, _ = spearmanr(fluor_arr, shuffled)
        null_rhos.append(r)
    null_rhos = np.asarray(null_rhos)
    perm_p = float(np.mean(np.abs(null_rhos) >= abs(rho)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.comparison}_phase_vs_{args.organelle}_{args.track}.csv"
    pd.DataFrame(
        [
            {
                "comparison": args.comparison,
                "organelle": args.organelle,
                "track": args.track,
                "n_paired": int(n_paired),
                "spearman_rho": float(rho) if not np.isnan(rho) else np.nan,
                "permutation_p_value": perm_p,
                "n_permutations": int(args.n_permutations),
                "claim_succeeds": bool((not np.isnan(rho)) and abs(rho) >= 0.20 and perm_p < 0.05),
            }
        ]
    ).to_csv(out_path, index=False)
    _logger.info(f"ρ = {rho:.3f}, perm p = {perm_p:.4f} (n={n_paired})")
    _logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
