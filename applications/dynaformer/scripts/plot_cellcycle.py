"""Plot predicted reporter trajectories of cell-cycle runs against the truth.

Usage
-----
python plot_cellcycle.py --models regression_c-radio regression_c-radio_single_frame stain_c-radio --out <dir>

For each model, reads its readout's predictions on the test split (``eval/best_<readout>/predictions_test.csv``;
``--readouts`` picks the readout per model, by default the one its task trains). Writes
``trajectories.png``, the longest test tracks that pass through S phase with every model's Geminin-mKO and PIP-mVenus
prediction over the true trace, the true PIP-FUCCI phase shaded underneath; and ``scatter.png``, predicted against
true levels for every test frame.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynaformer.cellcycle_metrics import phase_strict

RUNS = Path("/hpc/projects/comp.micro/virtual_staining/models/dynaformer/runs/cellcycle")
STATS = Path("/hpc/projects/virtual_staining/datasets/huang-lab/Bill/5-cellcycle/stats.json")
REPORTERS = {"mko": "Geminin-mKO", "mvenus": "PIP-mVenus"}
PHASE_COLORS = {0: "#cfe8cf", 1: "#f6d6b8", 2: "#d9cdea"}  # G1, S, G2/M
PHASE_NAMES = {0: "G1", 1: "S", 2: "G2/M"}


def load(model: str, readout: str, split: str) -> pd.DataFrame:
    """Per-frame truth and one readout's prediction, in counts."""
    table = pd.read_csv(RUNS / model / "eval" / f"best_{readout}" / f"predictions_{split}.csv")
    keep = ["track_id", "t"] + [f"true_{r}" for r in REPORTERS] + [f"{readout}_{r}" for r in REPORTERS]
    return table[keep].rename(columns={f"{readout}_{r}": f"pred_{r}" for r in REPORTERS})


def main() -> None:
    """Draw the trajectory and scatter figures."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--readouts", nargs="+", help="readout per model; default: stain for stain runs, else regression"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--tracks", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    readouts = args.readouts or ["stain" if m.startswith("stain") else "regression" for m in args.models]
    thresholds = json.loads(STATS.read_text())["thresholds"]
    tables = {f"{m} ({r})": load(m, r, args.split) for m, r in zip(args.models, readouts, strict=True)}
    truth = next(iter(tables.values()))
    truth = truth.assign(phase=phase_strict(truth.true_mko.to_numpy(), truth.true_mvenus.to_numpy(), thresholds))

    # Long tracks that pass through S, where the PIP drop and the Geminin rise make the trajectory informative.
    stats = truth.groupby("track_id").agg(n=("t", "size"), s_frames=("phase", lambda p: int((p == 1).sum())))
    chosen = stats[(stats.s_frames >= 5)].sort_values("n", ascending=False).index[: args.tracks]
    args.out.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(chosen), 2, figsize=(14, 2.6 * len(chosen)), sharex=True, squeeze=False)
    for row, track_id in enumerate(chosen):
        true_track = truth[truth.track_id == track_id]
        for col, (reporter, label) in enumerate(REPORTERS.items()):
            ax = axes[row, col]
            for phase, color in PHASE_COLORS.items():
                in_phase = true_track.t[true_track.phase == phase]
                for t in in_phase:
                    ax.axvspan(t - 0.5, t + 0.5, color=color, linewidth=0)
            ax.plot(true_track.t, true_track[f"true_{reporter}"], color="black", linewidth=2, label="truth")
            for k, (name, table) in enumerate(tables.items()):
                track = table[table.track_id == track_id]
                ax.plot(track.t, track[f"pred_{reporter}"], color=colors(k), linewidth=1.3, label=name)
            ax.set_ylabel(f"{label} (counts)")
            ax.set_title(f"track {track_id}: {label}", fontsize=9)
    axes[-1, 0].set_xlabel("frame (10 min)")
    axes[-1, 1].set_xlabel("frame (10 min)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [plt.Rectangle((0, 0), 1, 1, color=c) for c in PHASE_COLORS.values()]
    labels += [f"true {PHASE_NAMES[p]}" for p in PHASE_COLORS]
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out / "trajectories.png", dpi=110)

    fig, axes = plt.subplots(2, len(tables), figsize=(3.6 * len(tables), 7), squeeze=False)
    for k, (name, table) in enumerate(tables.items()):
        for row, (reporter, label) in enumerate(REPORTERS.items()):
            ax = axes[row, k]
            true, pred = table[f"true_{reporter}"], table[f"pred_{reporter}"]
            finite = np.isfinite(pred)
            ax.hexbin(true[finite], pred[finite], gridsize=60, bins="log", mincnt=1, cmap="viridis")
            top = float(np.percentile(true, 99.5))
            ax.plot([0, top], [0, top], color="red", linewidth=0.8)
            ax.set_xlim(-50, top)
            ax.set_ylim(-50, top)
            r = np.corrcoef(true[finite], pred[finite])[0, 1]
            ax.set_title(f"{name}\n{label}: r = {r:.3f}", fontsize=8)
            ax.set_xlabel("true (counts)")
            ax.set_ylabel("predicted (counts)")
    fig.tight_layout()
    fig.savefig(args.out / "scatter.png", dpi=110)
    print(f"Wrote {args.out / 'trajectories.png'} and {args.out / 'scatter.png'}")


if __name__ == "__main__":
    main()
