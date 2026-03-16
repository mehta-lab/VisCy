"""
Bar plots of organelle_volume (or any feature) distributions by remodeling_status
and predicted_infection, with significance testing.

Groups:
  - non-remodelled  (remodeling_status == 'control')
  - remodelled      (remodeling_status == 'remodeled')
  - uninfected      (predicted_infection == 'uninfected')
  - infected        (predicted_infection == 'infected')

Significance: Welch's t-test
  - non-remodelled vs remodelled
  - uninfected vs infected

Also provides add_feature_to_df() for merging feature columns onto an embedding
DataFrame by (fov_name, track_id, time_point) keys.

Input:
  features_csv – organelle_features.csv
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.stats_utils import get_significance_stars, add_significance_bar

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# %% paths
features_csv = Path("organelle_features.csv")
output_dir = Path("figures/")

# %% configuration
feature_col = "organelle_volume"   # change to any feature column as needed

output_dir.mkdir(parents=True, exist_ok=True)


# %% utility: merge feature column onto embedding DataFrame
def add_feature_to_df(
    feature_csv: Path,
    embedding_df: pd.DataFrame,
    feature_col: str,
) -> pd.DataFrame:
    """
    Merge a computed feature column onto an embedding DataFrame.

    Matching keys: (fov_name, track_id, time_point).
    Rows in embedding_df without a match receive NaN for the feature column.

    Parameters
    ----------
    feature_csv   : Path to CSV containing at minimum fov_name, track_id,
                    time_point, and the desired feature_col.
    embedding_df  : DataFrame to annotate.
    feature_col   : Name of the feature column to add.

    Returns
    -------
    Annotated copy of embedding_df with feature_col added.
    """
    feat_df = pd.read_csv(feature_csv)[["fov_name", "track_id", "time_point", feature_col]]
    feat_df["track_id"] = pd.to_numeric(feat_df["track_id"], errors="coerce")
    feat_df["time_point"] = pd.to_numeric(feat_df["time_point"], errors="coerce")

    embedding_df = embedding_df.copy()
    embedding_df["track_id"] = pd.to_numeric(embedding_df["track_id"], errors="coerce")
    embedding_df["time_point"] = pd.to_numeric(embedding_df.get("time_point", embedding_df.get("t")), errors="coerce")

    merged = embedding_df.merge(feat_df, on=["fov_name", "track_id", "time_point"], how="left")
    return merged


# %% load features
df = pd.read_csv(features_csv)
print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
print(f"Remodeling labels: {df['remodeling_status'].unique()}")
print(f"Infection labels:  {df['predicted_infection'].unique()}")

# %% extract 4 groups
ctrl_data      = df[df["remodeling_status"].astype(str).str.lower() == "control"][feature_col].dropna().values
remod_data     = df[df["remodeling_status"].astype(str).str.lower() == "remodeled"][feature_col].dropna().values
uninf_data     = df[df["predicted_infection"].astype(str).str.lower() == "uninfected"][feature_col].dropna().values
inf_data       = df[df["predicted_infection"].astype(str).str.lower() == "infected"][feature_col].dropna().values

for label, data in [
    ("non-remodelled", ctrl_data),
    ("remodelled", remod_data),
    ("uninfected", uninf_data),
    ("infected", inf_data),
]:
    print(f"  {label}: n={len(data)}, mean={np.mean(data) if len(data) else float('nan'):.4f}")

# %% significance tests (Welch's t-test)
p_remod = stats.ttest_ind(ctrl_data, remod_data, equal_var=False).pvalue if (len(ctrl_data) and len(remod_data)) else 1.0
p_inf   = stats.ttest_ind(uninf_data, inf_data, equal_var=False).pvalue if (len(uninf_data) and len(inf_data)) else 1.0
print(f"Welch's t-test: control vs remodeled p={p_remod:.4e},  uninfected vs infected p={p_inf:.4e}")

# %% bar plot with significance brackets
x_positions = [0, 0.6, 1.2, 1.8]
bar_colors = ["lightgreen", "orange", "lightblue", "red"]
group_data = [ctrl_data, remod_data, uninf_data, inf_data]
labels = ["non-remodelled", "remodelled", "translocation (−ve)", "translocation (+ve)"]

means = [np.mean(d) if len(d) else 0.0 for d in group_data]
stds  = [np.std(d)  if len(d) else 0.0 for d in group_data]

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(
    x_positions,
    means,
    yerr=stds,
    capsize=5,
    color=[matplotlib.colors.to_rgba(c, alpha=0.5) for c in bar_colors],
    edgecolor="black",
    linewidth=0.8,
    error_kw={"elinewidth": 1.2, "capthick": 1.2},
    width=0.45,
)

# n= annotations
for xi, data in zip(x_positions, group_data):
    ax.text(xi, 0, f"n={len(data)}", ha="center", va="top", fontsize=7,
            transform=ax.get_xaxis_transform())

# Significance brackets
y_max = max(m + s for m, s in zip(means, stds)) if means else 1.0
h = y_max * 0.06

# Pair 1: non-remodelled vs remodelled  (x[0] vs x[1])
y1 = y_max + h
add_significance_bar(ax, x_positions[0], x_positions[1], y1, h * 0.4, p_remod)

# Pair 2: uninfected vs infected  (x[2] vs x[3])
y2 = y_max + h
add_significance_bar(ax, x_positions[2], x_positions[3], y2, h * 0.4, p_inf)

ax.set_ylabel(feature_col.replace("_", " ").title())
ax.set_xticks(x_positions)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_ylim(bottom=0, top=y_max + h * 4)
ax.set_title(f"{feature_col.replace('_', ' ').title()} by remodeling & infection status")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend with coloured rectangles
legend_handles = [
    mpatches.Patch(facecolor=c, alpha=0.5, edgecolor="black", label=lbl)
    for c, lbl in zip(bar_colors, labels)
]
ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

plt.tight_layout()
save_path = output_dir / f"{feature_col}_remodeling_infection_barplot.svg"
fig.savefig(save_path, format="svg", bbox_inches="tight")
print(f"Saved: {save_path}")
plt.close(fig)
