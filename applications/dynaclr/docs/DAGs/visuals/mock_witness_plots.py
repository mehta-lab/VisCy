"""Generate MOCK example plots for the witness-score LC DAG doc.

Synthetic (illustrative) data only — matches the plot types and Wong palette
produced by the real orchestrated.py so the doc shows what outputs look like.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/home/eduardo.hirata/repos/viscy/applications/dynaclr/docs/DAGs/visuals")
WONG = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442"]
rng = np.random.default_rng(7)

MOCK_TAG = "illustrative — bar/ROC use real run values; hist/F1 synthetic"


def _mock_note(fig):
    fig.text(0.99, 0.01, MOCK_TAG, ha="right", va="bottom", fontsize=7, color="#B00020", style="italic")


def save(fig, name):
    """Watermark ``fig`` and write it to the visuals dir as PNG + PDF."""
    _mock_note(fig)
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# 1) Witness score distribution + gating bands (motivates the labels).
def witness_score_hist():
    """Mock witness-score histogram with control/perturbed humps and dead-zone band."""
    ctrl = rng.normal(1.4, 0.7, 4000)
    pert = rng.normal(-1.4, 0.7, 3600)
    scores = np.concatenate([ctrl, pert])
    t = np.quantile(np.abs(scores), 0.10)  # dead_zone = 0.1

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(-4, 4, 60)
    ax.hist(ctrl, bins=bins, color=WONG[2], alpha=0.7, label="control-well cells (X)")
    ax.hist(pert, bins=bins, color=WONG[4], alpha=0.7, label="perturbed-well cells (Y)")
    ax.axvspan(-t, t, color="gray", alpha=0.25, label=f"dead-zone (|w|≤t, t={t:.2f}) → dropped")
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("witness score  w(z)")
    ax.set_ylabel("cell count")
    ax.set_title("Witness score distribution & gating — witness_state (marker=G3BP1)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "mock_witness_score_hist")


# 2) Per-marker metrics bar chart (mirrors _plot_metrics_bar).
def metrics_bar():
    """Mock per-marker AUROC/accuracy/weighted-F1 bar chart (mirrors _plot_metrics_bar)."""
    # Representative values from a real 2D-MIP-BagOfChannels infectomics run,
    # scored vs ground-truth infection_state (eval_against). Strong where the
    # marker carries infection signal (viral_sensor, SEC61B), near chance where
    # it does not (Phase3D, G3BP1) — the useful discriminating signal.
    markers = ["G3BP1", "SEC61B", "Phase3D", "viral_sensor"]
    auroc = [0.554, 0.838, 0.536, 0.815]
    acc = [0.491, 0.764, 0.580, 0.865]
    wf1 = [0.388, 0.768, 0.466, 0.861]
    metrics = {"AUROC": auroc, "Accuracy": acc, "Weighted F1": wf1}
    colors = ["#0072B2", "#E69F00", "#009E73"]

    x = np.arange(len(markers))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(6, len(markers) * 1.5), 5))
    for i, (name, vals) in enumerate(metrics.items()):
        ax.bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.85)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(markers, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", label="Random (0.5)")
    ax.set_ylabel("Score")
    ax.set_title("witness_state — performance vs infection_state (per marker)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "mock_metrics_bar")


# 3) ROC curves (mirrors _plot_roc_curves, binary control/perturbed).
def roc_curves():
    """Mock per-marker one-vs-rest ROC curves (mirrors _plot_roc_curves)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title("ROC — witness_state vs infection_state (per marker)", fontsize=11)
    aurocs = {"G3BP1": 0.554, "SEC61B": 0.838, "Phase3D": 0.536, "viral_sensor": 0.815}
    for i, (marker, target_auc) in enumerate(aurocs.items()):
        # Build a smooth ROC with roughly the target AUROC.
        fpr = np.linspace(0, 1, 200)
        k = np.interp(target_auc, [0.5, 1.0], [1.0, 12.0])
        tpr = fpr ** (1.0 / k)
        ax.plot(fpr, tpr, color=WONG[i % len(WONG)], linewidth=1.8, label=f"{marker} (AUROC={target_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "mock_roc_curves")


# 4) F1 over time (mirrors _plot_f1_over_time).
def f1_over_time():
    """Mock per-class F1 across hours post-perturbation (mirrors _plot_f1_over_time)."""
    hours = np.arange(0, 49, 6)
    fig, ax = plt.subplots(figsize=(8, 5))
    # control: high, flat; perturbed: rises as phenotype emerges post-infection.
    control_f1 = np.clip(0.9 - 0.02 * rng.standard_normal(len(hours)), 0, 1)
    perturbed_f1 = np.clip(1 / (1 + np.exp(-(hours - 18) / 5)) * 0.9 + 0.05, 0, 1)
    ax.plot(hours, control_f1, marker="o", color=WONG[0], linewidth=2, label="control")
    ax.plot(hours, perturbed_f1, marker="o", color=WONG[1], linewidth=2, label="perturbed")
    ax.set_xlabel("Hours post perturbation")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("F1 over time — witness_state (marker=G3BP1)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "mock_f1_over_time")


if __name__ == "__main__":
    witness_score_hist()
    metrics_bar()
    roc_curves()
    f1_over_time()
