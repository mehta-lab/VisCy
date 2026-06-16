"""Stage 4 cross-track comparison: side-by-side onset population curves.

Reads per-organelle signal parquets from Stage 3 (one per track), bins
by ``t_rel_minutes``, computes binned median + IQR per cohort, and
emits a 9-panel figure (3 organelles × 3 tracks). Computes the
methodological-claim verdict: does Path B's IQR at the headline metric
beat the better of A-anno and A-LC by ≥ 25% per DAG §9.1?

Per discussion §2.2: the methodological claim succeeds if Path B's
population-curve IQR at the headline metric is at least 25% tighter
than the better of the two Path A baselines. Same metric reported in
real-time minutes for all three tracks so the comparison is fair.

Usage::

    cd applications/dynaclr/scripts/pseudotime/4-compare_tracks
    uv run python compare_onsets.py \
        --datasets ../../../configs/pseudotime/datasets.yaml \
        --config ../../../configs/pseudotime/compare_tracks.yaml \
        --comparison zikv_07_24_full
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
READOUT_ROOT = SCRIPT_DIR.parent / "3-organelle-remodeling"
OUTPUT_DIR = SCRIPT_DIR / "comparisons"

sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils import load_stage_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def _load_signal(track: str, organelle: str, candidate_set: str) -> pd.DataFrame:
    """Load a Stage 3 signal parquet."""
    path = READOUT_ROOT / track / organelle / f"{candidate_set}_signal.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Signal parquet not found: {path}")
    return pd.read_parquet(path)


def _binned_summary(
    df: pd.DataFrame,
    bin_edges: np.ndarray,
    cohort: str,
) -> pd.DataFrame:
    """Per-bin median + IQR of ``signal`` for a cohort."""
    sub = df[(df["cohort"] == cohort) & df["signal"].notna() & df["t_rel_minutes"].notna()]
    if sub.empty:
        return pd.DataFrame(columns=["t_rel_bin_center", "median", "q25", "q75", "n"])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bins = pd.cut(sub["t_rel_minutes"], bins=bin_edges, include_lowest=True, labels=False)
    sub = sub.assign(_bin=bins)
    rows = []
    for b, g in sub.groupby("_bin"):
        if g["signal"].size < 3:
            continue
        rows.append(
            {
                "t_rel_bin_center": float(bin_centers[int(b)]),
                "median": float(g["signal"].median()),
                "q25": float(g["signal"].quantile(0.25)),
                "q75": float(g["signal"].quantile(0.75)),
                "n": int(g["signal"].size),
            }
        )
    return pd.DataFrame(rows)


def _iqr_at_zero(summary: pd.DataFrame, target_t_rel: float = 0.0) -> float:
    """IQR width at the bin closest to ``target_t_rel``."""
    if summary.empty:
        return float("nan")
    idx = (summary["t_rel_bin_center"] - target_t_rel).abs().idxmin()
    return float(summary.loc[idx, "q75"] - summary.loc[idx, "q25"])


def _plot_grid(
    summaries: dict[tuple[str, str, str], pd.DataFrame],
    organelles: list[str],
    tracks: list[str],
    cohorts: list[str],
    out_path: Path,
    title: str,
) -> None:
    """3×3 grid: rows = tracks, cols = organelles, cohorts overlaid per panel."""
    fig, axes = plt.subplots(len(tracks), len(organelles), figsize=(4 * len(organelles), 3 * len(tracks)), sharex=True)
    if len(tracks) == 1:
        axes = np.atleast_2d(axes)
    cohort_colors = {"productive": "C3", "mock": "C7", "bystander": "C0"}
    for i, track in enumerate(tracks):
        for j, organelle in enumerate(organelles):
            ax = axes[i, j]
            for cohort in cohorts:
                key = (track, organelle, cohort)
                summary = summaries.get(key)
                if summary is None or summary.empty:
                    continue
                color = cohort_colors.get(cohort, "C2")
                ax.plot(summary["t_rel_bin_center"], summary["median"], color=color, label=cohort)
                ax.fill_between(summary["t_rel_bin_center"], summary["q25"], summary["q75"], color=color, alpha=0.2)
            ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
            if i == 0:
                ax.set_title(organelle)
            if j == 0:
                ax.set_ylabel(f"{track}\nsignal")
            if i == len(tracks) - 1:
                ax.set_xlabel("t_rel_minutes")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run cross-track comparison for one comparison config entry."""
    parser = argparse.ArgumentParser(description="Stage 4 cross-track comparison.")
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--comparison", required=True, help="Name under config['comparisons']")
    args = parser.parse_args()

    config = load_stage_config(args.datasets, args.config)
    comparisons = config.get("comparisons", {})
    if args.comparison not in comparisons:
        raise KeyError(f"Comparison {args.comparison!r} not in {sorted(comparisons)}")
    cmp_cfg = comparisons[args.comparison]
    candidate_set = cmp_cfg["candidate_set"]
    organelles = cmp_cfg.get("organelles", ["sec61", "g3bp1", "phase"])
    tracks = cmp_cfg.get("tracks", ["A-anno", "A-LC", "B"])
    cohorts = cmp_cfg.get("cohorts", ["productive", "mock", "bystander"])
    bin_minutes = float(cmp_cfg.get("bin_minutes", 30.0))
    bin_range = cmp_cfg.get("bin_range_minutes", [-360, 540])
    methodological_fraction = float(cmp_cfg.get("methodological_threshold_fraction", 0.25))
    bin_edges = np.arange(bin_range[0], bin_range[1] + bin_minutes, bin_minutes)

    # Load all signal parquets and compute per-(track, organelle, cohort) summaries.
    summaries: dict[tuple[str, str, str], pd.DataFrame] = {}
    for track in tracks:
        for organelle in organelles:
            try:
                df = _load_signal(track, organelle, candidate_set)
            except FileNotFoundError as exc:
                _logger.warning(f"Skipping {track}/{organelle}: {exc}")
                continue
            for cohort in cohorts:
                summaries[(track, organelle, cohort)] = _binned_summary(df, bin_edges, cohort)

    # Methodological-claim verdict per organelle.
    verdict_rows = []
    for organelle in organelles:
        b_iqr = _iqr_at_zero(summaries.get(("B", organelle, "productive"), pd.DataFrame()))
        a_anno_iqr = _iqr_at_zero(summaries.get(("A-anno", organelle, "productive"), pd.DataFrame()))
        a_lc_iqr = _iqr_at_zero(summaries.get(("A-LC", organelle, "productive"), pd.DataFrame()))
        better_a = (
            min(v for v in (a_anno_iqr, a_lc_iqr) if not np.isnan(v))
            if any(not np.isnan(v) for v in (a_anno_iqr, a_lc_iqr))
            else float("nan")
        )
        ratio = (
            float(b_iqr / better_a) if (better_a and not np.isnan(better_a) and not np.isnan(b_iqr)) else float("nan")
        )
        success = (not np.isnan(ratio)) and (ratio <= 1.0 - methodological_fraction)
        verdict_rows.append(
            {
                "organelle": organelle,
                "B_iqr_at_zero": b_iqr,
                "A_anno_iqr_at_zero": a_anno_iqr,
                "A_LC_iqr_at_zero": a_lc_iqr,
                "better_A_iqr": better_a,
                "B_over_A_ratio": ratio,
                "methodological_success": bool(success),
            }
        )
    verdict_df = pd.DataFrame(verdict_rows)
    _logger.info(f"Methodological-claim verdicts:\n{verdict_df.to_string(index=False)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUTPUT_DIR / f"{args.comparison}.png"
    summary_path = OUTPUT_DIR / f"{args.comparison}_summary.csv"
    verdict_path = OUTPUT_DIR / f"{args.comparison}_verdict.csv"

    long_summary = []
    for (track, organelle, cohort), df in summaries.items():
        if df.empty:
            continue
        df = df.copy()
        df["track"] = track
        df["organelle"] = organelle
        df["cohort"] = cohort
        long_summary.append(df)
    if long_summary:
        pd.concat(long_summary, ignore_index=True).to_csv(summary_path, index=False)
    verdict_df.to_csv(verdict_path, index=False)
    _plot_grid(summaries, organelles, tracks, cohorts, fig_path, title=args.comparison)
    _logger.info(f"Wrote {fig_path}")
    _logger.info(f"Wrote {summary_path}")
    _logger.info(f"Wrote {verdict_path}")


if __name__ == "__main__":
    main()
