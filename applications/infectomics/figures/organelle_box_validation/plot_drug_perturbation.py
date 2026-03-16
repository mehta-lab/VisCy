"""
Bar / violin plots with significance testing for drug perturbation experiments.

Handles both TOMM20 (edge_density) and LAMP1 (organelle_volume) in one script.
Toggle `experiment_type` at the top to switch between the two.

Input:
  feature_csv   – drug_perturbation_features.csv
  well_map_csv  – well_map.csv
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.stats_utils import get_significance_stars, add_significance_bar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# %% paths
feature_csv = Path("drug_perturbation_features.csv")
well_map_csv = Path("well_map.csv")
output_dir = Path("figures/")

# %% experiment configuration
# Change this variable to switch between TOMM20 and LAMP1 analysis
experiment_type = "TOMM20"   # "TOMM20" or "LAMP1"

EXPERIMENT_CONFIG = {
    "TOMM20": {
        "feature_col": "edge_density",
        "ylabel": "Edge Density",
        "conditions": ["DMSO", "0.5uM Oligomycin", "5uM Oligomycin"],
        "well_slice": slice(3, None),   # wells[3:]
        "title": "TOMM20 — Oligomycin drug perturbation",
        "color": "#DD8452",
    },
    "LAMP1": {
        "feature_col": "organelle_volume",
        "ylabel": "Organelle Volume",
        "conditions": ["DMSO", "0.5uM Bafilomycin", "5uM Bafilomycin"],
        "well_slice": slice(None, 3),  # wells[:3]
        "title": "LAMP1 — Bafilomycin drug perturbation",
        "color": "#4C72B0",
    },
}

cfg = EXPERIMENT_CONFIG[experiment_type]
feature_col = cfg["feature_col"]
conditions = cfg["conditions"]
well_slice = cfg["well_slice"]
ylabel = cfg["ylabel"]
title = cfg["title"]
bar_color = cfg["color"]

output_dir.mkdir(parents=True, exist_ok=True)

# %% load data
features_df = pd.read_csv(feature_csv)
well_map_df = pd.read_csv(well_map_csv)

print(f"Feature columns: {list(features_df.columns)}")
print(f"Well map columns: {list(well_map_df.columns)}")

# Select wells for this experiment type
all_wells = well_map_df["Well ID"].tolist()
selected_wells = all_wells[well_slice]
print(f"Selected wells ({experiment_type}): {selected_wells}")

# Assign conditions to wells (assumes wells are ordered consistently with conditions)
well_condition_map = {}
for well, condition in zip(selected_wells, conditions):
    well_condition_map[well] = condition

# Filter feature data to selected wells and add condition column
features_df = features_df[features_df["well"].isin(selected_wells)].copy()
features_df["condition"] = features_df["well"].map(well_condition_map)

print(f"Rows after filtering: {len(features_df)}")
print(f"Conditions present: {features_df['condition'].unique()}")

# %% group data by condition
condition_data = []
for cond in conditions:
    subset = features_df[features_df["condition"] == cond][feature_col].dropna().values
    condition_data.append(subset)
    print(f"  {cond}: n={len(subset)}, mean={np.mean(subset):.4f}, std={np.std(subset):.4f}")

# %% significance testing (Welch's t-test, condition 0 vs 1 and 0 vs 2)
p_01 = stats.ttest_ind(condition_data[0], condition_data[1], equal_var=False).pvalue
p_02 = stats.ttest_ind(condition_data[0], condition_data[2], equal_var=False).pvalue
print(f"Welch's t-test p-values: DMSO vs cond1 = {p_01:.4e}, DMSO vs cond2 = {p_02:.4e}")

# %% violin plot per condition
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle(f"{title} — violin plot", fontsize=11)

x_positions = np.arange(1, len(conditions) + 1)
parts = ax.violinplot(
    [d for d in condition_data if len(d) > 0],
    positions=x_positions[: len([d for d in condition_data if len(d) > 0])],
    showmeans=True,
    showextrema=True,
)
for pc in parts["bodies"]:
    pc.set_facecolor(bar_color)
    pc.set_alpha(0.7)

# Annotate mean±SD
for i, data in enumerate(condition_data):
    if len(data) == 0:
        continue
    m, s = np.mean(data), np.std(data)
    ax.text(
        x_positions[i],
        m + s + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
        f"{m:.3f}±{s:.3f}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

ax.set_xticks(x_positions)
ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=9)
ax.set_ylabel(ylabel)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
save_path = output_dir / f"{experiment_type}_drug_perturbation_violin.svg"
fig.savefig(save_path, format="svg", bbox_inches="tight")
print(f"Saved: {save_path}")
plt.close(fig)

# %% bar plot with significance brackets
means = [np.mean(d) if len(d) > 0 else 0.0 for d in condition_data]
stds = [np.std(d) if len(d) > 0 else 0.0 for d in condition_data]

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle(f"{title} — bar plot", fontsize=11)

bars = ax.bar(
    x_positions,
    means,
    yerr=stds,
    capsize=5,
    color=bar_color,
    alpha=0.7,
    edgecolor="black",
    linewidth=0.8,
    error_kw={"elinewidth": 1.2, "capthick": 1.2},
)

# n= annotations below each bar
for i, data in enumerate(condition_data):
    ax.text(
        x_positions[i],
        0.01,
        f"n={len(data)}",
        ha="center",
        va="bottom",
        fontsize=7,
        transform=ax.get_xaxis_transform(),
    )

# Significance brackets
y_max = max(m + s for m, s in zip(means, stds)) if means else 1.0
bracket_h = y_max * 0.05
y_bracket1 = y_max + bracket_h
y_bracket2 = y_bracket1 + bracket_h * 2.5

add_significance_bar(ax, x_positions[0], x_positions[1], y_bracket1, bracket_h * 0.5, p_01)
add_significance_bar(ax, x_positions[0], x_positions[2], y_bracket2, bracket_h * 0.5, p_02)

ax.set_xticks(x_positions)
ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=9)
ax.set_ylabel(ylabel)
ax.set_ylim(0, y_bracket2 + bracket_h * 4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
save_path = output_dir / f"{experiment_type}_drug_perturbation_barplot.svg"
fig.savefig(save_path, format="svg", bbox_inches="tight")
print(f"Saved: {save_path}")
plt.close(fig)
