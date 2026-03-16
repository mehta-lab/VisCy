"""
Violin plots of SSIM and PCC between organelle tag and live stain.

Produces one 1×2 figure per organelle (LAMP1, TOMM20) and one combined figure.
Input: live_stain_correlation.csv
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
correlation_csv = Path("live_stain_correlation.csv")
output_dir = Path("figures/")

# %% configuration
METRICS = ["SSIM", "PCC"]
ORGANELLES = ["LAMP1", "TOMM20"]

output_dir.mkdir(parents=True, exist_ok=True)

# %% load data
df = pd.read_csv(correlation_csv)
print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
print(f"Organelles found: {df['organelle_tag'].unique()}")


# %% helper: violin with mean±SD annotation
def violin_with_annotation(ax, data, position, color, width=0.5):
    """Draw a single violin at *position* and annotate with mean±SD."""
    parts = ax.violinplot(
        data,
        positions=[position],
        widths=width,
        showmeans=True,
        showextrema=True,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    mean_val = np.mean(data)
    std_val = np.std(data)
    ax.text(
        position,
        mean_val + std_val + 0.03,
        f"{mean_val:.3f}±{std_val:.3f}",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    return mean_val, std_val


# %% per-organelle figures
colors = {"LAMP1": "#4C72B0", "TOMM20": "#DD8452"}

for organelle in ORGANELLES:
    subset = df[df["organelle_tag"] == organelle].copy()
    if subset.empty:
        print(f"No data for organelle {organelle}, skipping.")
        continue

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    fig.suptitle(f"{organelle} — live stain correlation", fontsize=12)

    for ax, metric in zip(axes, METRICS):
        values = subset[metric].dropna().values
        violin_with_annotation(ax, values, position=1, color=colors.get(organelle, "steelblue"))
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel(metric)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = output_dir / f"{organelle}_live_stain_correlation.svg"
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)

# %% combined figure — all organelles on shared axes
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("All organelles — live stain correlation", fontsize=12)

organelle_list = sorted(df["organelle_tag"].unique())
x_positions = np.arange(1, len(organelle_list) + 1)
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for ax, metric in zip(axes, METRICS):
    for i, organelle in enumerate(organelle_list):
        subset = df[df["organelle_tag"] == organelle][metric].dropna().values
        if len(subset) == 0:
            continue
        color = color_cycle[i % len(color_cycle)]
        violin_with_annotation(ax, subset, position=x_positions[i], color=color)

    ax.set_title(metric)
    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(organelle_list, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
save_path = output_dir / "all_organelles_live_stain_correlation.svg"
fig.savefig(save_path, format="svg", bbox_inches="tight")
print(f"Saved: {save_path}")
plt.close(fig)
