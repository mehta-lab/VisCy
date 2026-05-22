r"""Diagnostic: plot per-cell phase + fluor signal traces for paired cells.

Visual companion to ``compare_phase_to_fluor.py``. For every productive
cell that contributes to the Spearman ρ (i.e. crosses both the phase
and the fluorescence threshold), draws a 2-panel row showing:

- Top: phase cosine distance vs ``t_rel_minutes`` with the per-FOV
  threshold and the detected phase onset.
- Bottom: fluor cosine distance vs ``t_rel_minutes`` with the per-FOV
  threshold and the detected fluor onset.

Used to inspect *why* g3bp1 ρ is high (sharp threshold crossings) and
sec61 ρ is unstable (slow / gradual).

Usage::

    cd applications/dynaclr/scripts/pseudotime/4-compare_tracks
    uv run python plot_paired_traces.py \\
        --comparison zikv_07_24_full \\
        --organelle sec61 \\
        --track A-anno
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
READOUT_ROOT = SCRIPT_DIR.parent / "3-organelle-remodeling"
OUTPUT_DIR = SCRIPT_DIR / "comparisons"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)

# Mapping from --comparison to candidate set (matches compare_tracks.yaml).
COMPARISON_TO_CANDIDATE_SET = {
    "zikv_07_24_full": "zikv_productive_07_24",
    "zikv_pooled_full": "zikv_productive_pooled",
}


def _per_cell_onset(signal_df: pd.DataFrame, threshold_df: pd.DataFrame) -> pd.DataFrame:
    """Productive-cell onset = first ``t_rel_minutes`` where signal > FOV threshold."""
    threshold_lookup = dict(zip(threshold_df["fov_name"].astype(str), threshold_df["threshold"]))
    rows = []
    productive = signal_df[
        (signal_df["cohort"] == "productive") & signal_df["signal"].notna() & signal_df["t_rel_minutes"].notna()
    ]
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
                "dataset_id": str(ds),
                "fov_name": str(fov),
                "track_id": int(tid),
                "lineage_id": str(g["lineage_id"].iloc[0]),
                "onset_t_rel_minutes": float(g["t_rel_minutes"].iloc[first_idx]),
                "threshold": float(threshold),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Render per-cell phase + fluor diagnostic traces."""
    parser = argparse.ArgumentParser(description="Plot paired phase/fluor traces.")
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--organelle", required=True, choices=["sec61", "g3bp1"])
    parser.add_argument("--track", required=True, choices=["A-anno", "A-LC", "B"])
    args = parser.parse_args()

    candidate_set = COMPARISON_TO_CANDIDATE_SET[args.comparison]

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
    n = len(paired)
    if n == 0:
        _logger.warning("No paired cells; nothing to plot")
        return
    _logger.info(f"Plotting {n} paired cells")

    paired = paired.sort_values("onset_t_rel_minutes_fluor").reset_index(drop=True)

    fig, axes = plt.subplots(n, 2, figsize=(11, max(2.5 * n, 3.0)), sharex=True, squeeze=False)
    for i, row in paired.iterrows():
        key = (row["dataset_id"], row["fov_name"], int(row["track_id"]))
        phase_track = phase_signal[
            (phase_signal["dataset_id"] == key[0])
            & (phase_signal["fov_name"] == key[1])
            & (phase_signal["track_id"] == key[2])
            & phase_signal["t_rel_minutes"].notna()
        ].sort_values("t_rel_minutes")
        fluor_track = fluor_signal[
            (fluor_signal["dataset_id"] == key[0])
            & (fluor_signal["fov_name"] == key[1])
            & (fluor_signal["track_id"] == key[2])
            & fluor_signal["t_rel_minutes"].notna()
        ].sort_values("t_rel_minutes")

        ax_phase, ax_fluor = axes[i, 0], axes[i, 1]
        ax_phase.plot(phase_track["t_rel_minutes"], phase_track["signal"], color="0.2", lw=1.0)
        ax_phase.axhline(row["threshold_phase"], color="grey", ls=":", lw=0.8)
        ax_phase.axvline(row["onset_t_rel_minutes_phase"], color="C0", ls="--", lw=1.0)
        ax_phase.axvline(0, color="red", ls="-", lw=0.6, alpha=0.6)
        ax_phase.set_ylabel(f"{key[1]}\nt={key[2]}", fontsize=8)
        if i == 0:
            ax_phase.set_title("phase cosine distance", fontsize=9)

        ax_fluor.plot(fluor_track["t_rel_minutes"], fluor_track["signal"], color="0.2", lw=1.0)
        ax_fluor.axhline(row["threshold_fluor"], color="grey", ls=":", lw=0.8)
        ax_fluor.axvline(row["onset_t_rel_minutes_fluor"], color="C1", ls="--", lw=1.0)
        ax_fluor.axvline(0, color="red", ls="-", lw=0.6, alpha=0.6)
        if i == 0:
            ax_fluor.set_title(f"{args.organelle} cosine distance", fontsize=9)

    axes[-1, 0].set_xlabel("t_rel_minutes (anchor=0)")
    axes[-1, 1].set_xlabel("t_rel_minutes (anchor=0)")
    fig.suptitle(
        f"{args.comparison} | {args.organelle} | track={args.track} | n={n} paired",
        fontsize=10,
    )
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.comparison}_traces_phase_vs_{args.organelle}_{args.track}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    _logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
