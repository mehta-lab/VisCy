"""
Visualize MMD (Maximum Mean Discrepancy) organelle remodeling scores.

Produces:
  1. Heatmap (organelles × timepoints) for ZIKV and DENV separately.
  2. Line plots: MMD vs timepoint per organelle, ZIKV and DENV as subplots.

Input:
  mmd_csv_dir – directory containing one CSV per dataset,
                each with columns: organelle, timepoint, mmd_zikv, mmd_denv
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.stats_utils import get_significance_stars, add_significance_bar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% paths
mmd_csv_dir = Path("mmd_results/")
output_dir = Path("figures/")

# %% configuration
VIRUS_COLS = {
    "ZIKV": "mmd_zikv",
    "DENV": "mmd_denv",
}

output_dir.mkdir(parents=True, exist_ok=True)

# %% load and concatenate all CSV files in mmd_csv_dir
csv_files = sorted(mmd_csv_dir.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {mmd_csv_dir}")

df_list = []
for csv_file in csv_files:
    tmp = pd.read_csv(csv_file)
    tmp["source_file"] = csv_file.stem
    df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)
print(f"Loaded {len(df)} rows from {len(csv_files)} files.")
print(f"Columns: {list(df.columns)}")
print(f"Organelles: {df['organelle'].unique()}")
print(f"Timepoints: {sorted(df['timepoint'].unique())}")

# %% build pivot tables per virus
organelles = sorted(df["organelle"].unique())
timepoints = sorted(df["timepoint"].unique())

# %% heatmaps
for virus, col in VIRUS_COLS.items():
    if col not in df.columns:
        print(f"Column '{col}' not found, skipping {virus} heatmap.")
        continue

    pivot = df.pivot_table(
        index="organelle",
        columns="timepoint",
        values=col,
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(max(8, len(timepoints)), max(5, len(organelles) * 0.5)))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": f"MMD ({virus})"},
    )
    ax.set_title(f"{virus} — MMD Organelle Remodeling Score", fontsize=12)
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Organelle")
    plt.tight_layout()

    save_path = output_dir / f"{virus}_mmd_heatmap.svg"
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)

# %% line plots: MMD vs timepoint per organelle
n_organelles = len(organelles)
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, max(5, n_organelles * 0.5)),
    sharey=False,
)
fig.suptitle("MMD Remodeling Score vs Timepoint", fontsize=13)

for ax, (virus, col) in zip(axes, VIRUS_COLS.items()):
    if col not in df.columns:
        ax.text(0.5, 0.5, f"No column '{col}'", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(virus)
        continue

    for i, organelle in enumerate(organelles):
        subset = df[df["organelle"] == organelle].sort_values("timepoint")
        if subset.empty:
            continue
        # Aggregate in case of multiple source files
        agg = subset.groupby("timepoint")[col].mean()
        color = color_cycle[i % len(color_cycle)]
        ax.plot(
            agg.index,
            agg.values,
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=color,
            label=organelle,
        )

    ax.set_title(virus, fontsize=11)
    ax.set_xlabel("Timepoint", fontsize=10)
    ax.set_ylabel("MMD Score", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        ncol=1,
    )

plt.tight_layout()
save_path = output_dir / "mmd_line_plots.svg"
fig.savefig(save_path, format="svg", bbox_inches="tight")
print(f"Saved: {save_path}")
plt.close(fig)
