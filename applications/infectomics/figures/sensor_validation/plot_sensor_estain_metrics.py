"""
Plot accuracy/F1 curves and % infected vs MOI for sensor validation experiments.

Produces:
  1. Dual y-axis line plot: accuracy (dashed) and F1 (solid) vs MOI for ZIKV and DENV
  2. % infected vs MOI — 4 lines: ZIKV sensor, ZIKV estain, DENV sensor, DENV estain

Input:
  metrics_csv        – infection_metrics_by_MOI.csv
  infection_pct_csv  – infection_percentage_by_MOI.csv
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.stats_utils import get_significance_stars, add_significance_bar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% paths
metrics_csv = Path("infection_metrics_by_MOI.csv")
infection_pct_csv = Path("infection_percentage_by_MOI.csv")
output_dir = Path("figures/")

# %% style configuration
VIRUS_COLORS = {
    "ZIKV": "orange",
    "DENV": "magenta",
}
MODALITY_LINE_STYLE = {
    "sensor": "-",
    "estain": "--",
}
MODALITY_MARKER = {
    "sensor": "o",
    "estain": "s",
}
ESTAIN_COLORS = {
    "ZIKV_sensor": "orange",
    "ZIKV_estain": "red",
    "DENV_sensor": "purple",
    "DENV_estain": "magenta",
}

output_dir.mkdir(parents=True, exist_ok=True)

# %% load data
metrics_df = pd.read_csv(metrics_csv)
infection_pct_df = pd.read_csv(infection_pct_csv)

print(f"Metrics columns: {list(metrics_df.columns)}")
print(f"Infection pct columns: {list(infection_pct_df.columns)}")
print(f"Viruses in metrics: {metrics_df['virus'].unique()}")

# %% plot 1: accuracy and F1 score vs MOI (dual y-axis)
fig, ax1 = plt.subplots(figsize=(7, 5))
ax2 = ax1.twinx()

for virus, color in VIRUS_COLORS.items():
    subset = metrics_df[metrics_df["virus"] == virus].sort_values("MOI")
    if subset.empty:
        continue

    # Accuracy — dashed line on ax1
    ax1.plot(
        subset["MOI"],
        subset["accuracy"],
        color=color,
        linestyle="--",
        marker="^",
        markersize=6,
        linewidth=1.8,
        label=f"{virus} accuracy",
    )

    # F1 — solid line on ax2
    ax2.plot(
        subset["MOI"],
        subset["F1"],
        color=color,
        linestyle="-",
        marker="o",
        markersize=6,
        linewidth=1.8,
        label=f"{virus} F1",
    )

ax1.set_xlabel("MOI", fontsize=11)
ax1.set_ylabel("Accuracy", fontsize=11)
ax2.set_ylabel("F1 Score", fontsize=11)
ax1.set_ylim(0, 1.05)
ax2.set_ylim(0, 1.05)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.set_title("Sensor validation — Accuracy and F1 vs MOI", fontsize=12)

plt.tight_layout()
save_path = output_dir / "sensor_accuracy_f1_vs_MOI.svg"
fig.savefig(save_path, format="svg", bbox_inches="tight")
print(f"Saved: {save_path}")
plt.close(fig)

# %% plot 2: % infected vs MOI — 4 lines
fig, ax = plt.subplots(figsize=(7, 5))

line_specs = [
    ("ZIKV", "sensor"),
    ("ZIKV", "estain"),
    ("DENV", "sensor"),
    ("DENV", "estain"),
]

for virus, modality in line_specs:
    subset = infection_pct_df[
        (infection_pct_df["virus"] == virus) & (infection_pct_df["modality"] == modality)
    ].sort_values("MOI")
    if subset.empty:
        print(f"  No data for {virus} {modality}")
        continue

    color = ESTAIN_COLORS.get(f"{virus}_{modality}", "gray")
    linestyle = MODALITY_LINE_STYLE.get(modality, "-")
    marker = MODALITY_MARKER.get(modality, "o")

    ax.plot(
        subset["MOI"],
        subset["percent_infected"],
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=6,
        linewidth=1.8,
        label=f"{virus} {modality}",
    )

ax.set_xlabel("MOI", fontsize=11)
ax.set_ylabel("% Infected", fontsize=11)
ax.set_title("% Infected vs MOI — sensor vs e-stain", fontsize=12)
ax.legend(fontsize=9, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
save_path = output_dir / "percent_infected_vs_MOI.svg"
fig.savefig(save_path, format="svg", bbox_inches="tight")
print(f"Saved: {save_path}")
plt.close(fig)
