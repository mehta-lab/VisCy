"""
Visualise label-free (phase) infection classifier results.

Produces:
  1. % infected vs time  — annotation (red circles) and prediction (blue squares)
  2. Accuracy + F1 + ROC-AUC vs time  — three lines on one plot
  3. PCA map coloured by classification result (TP/TN/FP/FN)
  4. ROC curve computed from infection_probability vs binary annotation

Input:
  predictions_csv – phase_infection_predictions.csv
  metrics_csv     – phase_infection_metrics_over_time.csv
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.stats_utils import get_significance_stars, add_significance_bar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# %% paths
predictions_csv = Path("phase_infection_predictions.csv")
metrics_csv = Path("phase_infection_metrics_over_time.csv")
output_dir = Path("figures/")

# %% configuration
time_interval_hours = 0.5      # hours per timepoint
start_hpi = 3.0                # hours post infection at first frame

CLASSIFICATION_COLORS = {
    "TP": "red",
    "TN": "blue",
    "FP": "orange",
    "FN": "steelblue",
}

output_dir.mkdir(parents=True, exist_ok=True)


# %% helper: convert timepoint index to hours post infection
def t_to_hpi(t: np.ndarray) -> np.ndarray:
    return start_hpi + t * time_interval_hours


# %% load data
pred_df = pd.read_csv(predictions_csv)
metrics_df = pd.read_csv(metrics_csv)

print(f"Predictions columns: {list(pred_df.columns)}")
print(f"Metrics columns:     {list(metrics_df.columns)}")
print(f"Classification results: {pred_df['classification_result'].unique()}")

# %% convert t to hpi
metrics_df["hpi"] = t_to_hpi(metrics_df["t"].values)
pred_df["hpi"] = t_to_hpi(pred_df["t"].values)

# %% plot 1: % infected vs time
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    metrics_df["hpi"],
    metrics_df["pct_infected_annotation"],
    color="red",
    marker="o",
    markersize=5,
    linewidth=1.8,
    label="Annotation",
)
ax.plot(
    metrics_df["hpi"],
    metrics_df["pct_infected_prediction"],
    color="blue",
    marker="s",
    markersize=5,
    linewidth=1.8,
    linestyle="--",
    label="Prediction",
)

ax.set_xlabel("Hours Post Infection", fontsize=11)
ax.set_ylabel("% Infected", fontsize=11)
ax.set_title("% Infected vs Time — annotation vs prediction", fontsize=12)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
save_path = output_dir / "pct_infected_vs_time.svg"
fig.savefig(save_path, format="svg", bbox_inches="tight")
print(f"Saved: {save_path}")
plt.close(fig)

# %% plot 2: accuracy + F1 + ROC-AUC vs time
fig, ax = plt.subplots(figsize=(7, 5))

metric_specs = [
    ("accuracy", "Accuracy", "steelblue", "-", "o"),
    ("f1",       "F1 Score", "orange",    "--", "s"),
    ("roc_auc",  "ROC-AUC",  "green",     ":", "^"),
]

for col, label, color, ls, marker in metric_specs:
    if col not in metrics_df.columns:
        print(f"Column '{col}' not found in metrics CSV, skipping.")
        continue
    ax.plot(
        metrics_df["hpi"],
        metrics_df[col],
        color=color,
        linestyle=ls,
        marker=marker,
        markersize=5,
        linewidth=1.8,
        label=label,
    )

ax.set_xlabel("Hours Post Infection", fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title("Classifier metrics vs Time", fontsize=12)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
save_path = output_dir / "classifier_metrics_vs_time.svg"
fig.savefig(save_path, format="svg", bbox_inches="tight")
print(f"Saved: {save_path}")
plt.close(fig)

# %% plot 3: PCA map coloured by TP/TN/FP/FN
if "PCA1" not in pred_df.columns or "PCA2" not in pred_df.columns:
    print("PCA columns not found in predictions CSV — skipping PCA plot.")
else:
    fig, ax = plt.subplots(figsize=(7, 6))

    for result, color in CLASSIFICATION_COLORS.items():
        mask = pred_df["classification_result"].astype(str).str.upper() == result
        pts = pred_df[mask]
        ax.scatter(
            pts["PCA1"],
            pts["PCA2"],
            c=color,
            s=5,
            alpha=0.5,
            label=result,
            rasterized=True,
        )

    ax.set_xlabel("PCA1", fontsize=11)
    ax.set_ylabel("PCA2", fontsize=11)
    ax.set_title("PCA map — TP/TN/FP/FN classification", fontsize=12)
    ax.legend(fontsize=9, markerscale=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = output_dir / "PCA_TPTNFPFN_map.svg"
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)

# %% plot 4: ROC curve
required_cols = {"infection_probability", "infection_status"}
if not required_cols.issubset(pred_df.columns):
    print(f"Columns {required_cols} not all found in predictions CSV — skipping ROC curve.")
else:
    # Binarise annotation: infected = 1, uninfected = 0
    y_true = (pred_df["infection_status"].astype(str).str.lower() == "infected").astype(int)
    y_score = pred_df["infection_probability"].values

    if y_true.sum() == 0 or (1 - y_true).sum() == 0:
        print("ROC curve: only one class present — skipping.")
    else:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc_val = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color="steelblue", linewidth=2,
                label=f"ROC AUC = {roc_auc_val:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("ROC Curve — label-free infection classifier", fontsize=12)
        ax.legend(fontsize=10, loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        save_path = output_dir / "roc_curve.svg"
        fig.savefig(save_path, format="svg", bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close(fig)
