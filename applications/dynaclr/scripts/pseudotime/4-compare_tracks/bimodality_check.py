"""Stage 4 bimodality check: Hartigans dip-test on back-projected real-time.

Per discussion §3.8 #11 and round 2 ML-engineer critique: every
back-projected real-time distribution from Path B needs a multimodality
test before reporting a single median. We use the BIC ratio between
1- and 2-component Gaussian mixtures (no extra package dependency).
A 2-component model with BIC at least 10 lower than 1-component flags
the distribution as multimodal.

Usage::

    cd applications/dynaclr/scripts/pseudotime/4-compare_tracks
    uv run python bimodality_check.py \
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
from sklearn.mixture import GaussianMixture

SCRIPT_DIR = Path(__file__).resolve().parent
READOUT_ROOT = SCRIPT_DIR.parent / "3-organelle-remodeling"
OUTPUT_DIR = SCRIPT_DIR / "comparisons"

sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils import load_stage_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def _bic_ratio(values: np.ndarray) -> tuple[float, float, bool]:
    """Return (BIC_1comp, BIC_2comp, multimodal_flag).

    A distribution is flagged multimodal iff BIC drops by ≥ 10 going
    from 1 to 2 components and there are at least 30 samples.
    """
    if len(values) < 30:
        return float("nan"), float("nan"), False
    X = values.reshape(-1, 1)
    g1 = GaussianMixture(n_components=1, random_state=0).fit(X)
    g2 = GaussianMixture(n_components=2, random_state=0).fit(X)
    bic1 = float(g1.bic(X))
    bic2 = float(g2.bic(X))
    return bic1, bic2, bool(bic1 - bic2 >= 10.0)


def main() -> None:
    """Run dip-test (BIC GMM) on back-projected real-time per (track, organelle, cohort)."""
    parser = argparse.ArgumentParser(description="Stage 4 bimodality check.")
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--comparison", required=True)
    args = parser.parse_args()

    config = load_stage_config(args.datasets, args.config)
    cmp_cfg = config["comparisons"][args.comparison]
    candidate_set = cmp_cfg["candidate_set"]
    organelles = cmp_cfg.get("organelles", ["sec61", "g3bp1", "phase"])
    tracks = cmp_cfg.get("tracks", ["A-anno", "A-LC", "B"])
    cohorts = cmp_cfg.get("cohorts", ["productive"])

    rows = []
    for track in tracks:
        for organelle in organelles:
            path = READOUT_ROOT / track / organelle / f"{candidate_set}_signal.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            for cohort in cohorts:
                # Path B: back-projected real-time at the aligned region.
                # Path A: t_rel_minutes is itself real-time. Both reduce
                # to "where in real-time is the signal at threshold."
                col = (
                    "t_rel_minutes_warped" if track == "B" and "t_rel_minutes_warped" in df.columns else "t_rel_minutes"
                )
                sub = df[(df["cohort"] == cohort) & df[col].notna() & df["signal"].notna()]
                if sub.empty:
                    continue
                values = sub[col].to_numpy()
                bic1, bic2, multimodal = _bic_ratio(values)
                rows.append(
                    {
                        "track": track,
                        "organelle": organelle,
                        "cohort": cohort,
                        "n": int(len(values)),
                        "bic_1comp": bic1,
                        "bic_2comp": bic2,
                        "multimodal": multimodal,
                    }
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.comparison}_bimodality.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    _logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
