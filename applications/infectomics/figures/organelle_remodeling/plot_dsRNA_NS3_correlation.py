"""
Bar plots of mean PCC (organelle vs dsRNA and organelle vs NS3) across organelles.

Logic:
  1. Compute dsRNA / NS3 pixel thresholds from uninfected rows (mean + 2*std).
  2. Categorize each row by dsRNA_present / ns3_present status.
  3. Filter to '+dsRNA/+NS3' rows.
  4. Remove excluded organelles.
  5. Split by virus (DENV / ZIKV) and plot bar + error bars for each metric.

Saves 4 SVGs: ZIKV_dsRNA_PCC, ZIKV_NS3_PCC, DENV_dsRNA_PCC, DENV_NS3_PCC.

Input: dsRNA_NS3_correlation.csv
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
correlation_csv = Path("dsRNA_NS3_correlation.csv")
output_dir = Path("figures/")

# %% configuration
excluded_organelles = ["pAL10", "pAL27"]

VIRUS_CONDITIONS = {
    "ZIKV": ["zikv", "ZIKV", "Zika", "ZIKA"],
    "DENV": ["denv", "DENV", "Dengue", "DENGUE"],
}

METRICS = {
    "org_dsRNA_PCC": "Organelle vs dsRNA PCC",
    "org_ns3_PCC": "Organelle vs NS3 PCC",
}

output_dir.mkdir(parents=True, exist_ok=True)


# %% helper: categorize infection status
def categorize_status(dsrna_present: bool, ns3_present: bool) -> str:
    """Return a category string based on dsRNA and NS3 presence flags."""
    d = "+dsRNA" if dsrna_present else "-dsRNA"
    n = "+NS3" if ns3_present else "-NS3"
    return f"{d}/{n}"


# %% load data
df = pd.read_csv(correlation_csv)
print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
print(f"Conditions: {df['condition'].unique()}")
print(f"Organelles: {df['organelle'].unique()}")


# %% identify uninfected rows (condition contains 'mock' or 'uninfected')
def is_uninfected(condition: str) -> bool:
    c = str(condition).lower()
    return "mock" in c or "uninfect" in c or "ctrl" in c or "control" in c


uninfected_mask = df["condition"].apply(is_uninfected)
uninfected_df = df[uninfected_mask]

if uninfected_df.empty:
    raise ValueError(
        "Could not identify uninfected rows. "
        "Check that 'condition' column contains 'mock', 'uninfected', or 'control'."
    )

# %% compute pixel thresholds from uninfected rows
dsrna_threshold = (
    uninfected_df["num_pixels_dsRNA"].mean()
    + 2 * uninfected_df["num_pixels_dsRNA"].std()
)
ns3_threshold = (
    uninfected_df["num_pixels_ns3"].mean()
    + 2 * uninfected_df["num_pixels_ns3"].std()
)
print(f"dsRNA threshold: {dsrna_threshold:.2f}")
print(f"NS3 threshold:   {ns3_threshold:.2f}")

# %% add category column
df["dsRNA_present"] = df["num_pixels_dsRNA"] > dsrna_threshold
df["ns3_present"] = df["num_pixels_ns3"] > ns3_threshold
df["category"] = df.apply(
    lambda row: categorize_status(row["dsRNA_present"], row["ns3_present"]), axis=1
)

# %% filter to '+dsRNA/+NS3'
filtered = df[df["category"] == "+dsRNA/+NS3"].copy()
print(f"Rows with +dsRNA/+NS3: {len(filtered)}")

# %% remove excluded organelles
filtered = filtered[~filtered["organelle"].isin(excluded_organelles)].copy()
print(f"Rows after removing excluded organelles: {len(filtered)}")


# %% helper: match condition to virus label
def get_virus_label(condition: str) -> str | None:
    c = str(condition)
    for virus, keywords in VIRUS_CONDITIONS.items():
        if any(kw in c for kw in keywords):
            return virus
    return None


filtered["virus"] = filtered["condition"].apply(get_virus_label)
print(f"Virus assignments: {filtered['virus'].value_counts().to_dict()}")


# %% bar plot function
def plot_pcc_bars(
    data: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    virus_label: str,
    output_path: Path,
) -> None:
    """Bar plot of mean PCC per organelle, sorted by decreasing mean."""
    grouped = data.groupby("organelle")[metric_col].agg(["mean", "std", "count"])
    grouped = grouped.sort_values("mean", ascending=False).reset_index()

    if grouped.empty:
        print(f"  No data for {virus_label} {metric_col}, skipping.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(grouped) * 0.8), 5))
    x = np.arange(len(grouped))
    bars = ax.bar(
        x,
        grouped["mean"],
        yerr=grouped["std"],
        capsize=4,
        color="steelblue" if "dsRNA" in metric_col else "coral",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.7,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )

    # Annotate n= on each bar
    for xi, (_, row) in zip(x, grouped.iterrows()):
        ax.text(
            xi,
            row["mean"] + row["std"] + 0.02,
            f"n={int(row['count'])}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(grouped["organelle"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric_label)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f"{virus_label} — {metric_label}", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# %% generate all 4 plots
for virus in ["ZIKV", "DENV"]:
    virus_df = filtered[filtered["virus"] == virus]
    if virus_df.empty:
        print(f"No data found for {virus}.")
        continue

    for metric_col, metric_label in METRICS.items():
        safe_metric = metric_col.replace("_", "")  # e.g. orgdsRNAPCC
        output_path = output_dir / f"{virus}_{metric_col}.svg"
        plot_pcc_bars(virus_df, metric_col, metric_label, virus, output_path)
