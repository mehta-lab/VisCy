"""
Plot % infected vs time (hours post infection) for ZIKV and DENV at multiple MOIs.

Error bars = SEM across FOVs at each timepoint.
Produces a grid of subplots: n_MOI rows × 2 virus columns.

Input:
  percent_infected_csv – percent_infected.csv
  well_map_csv         – well_map.csv
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
percent_infected_csv = Path("percent_infected.csv")
well_map_csv = Path("well_map.csv")
output_dir = Path("figures/")

# %% configuration
VIRUS_COLORS = {
    "ZIKV": "orange",
    "DENV": "magenta",
}
VIRUS_ORDER = ["ZIKV", "DENV"]

output_dir.mkdir(parents=True, exist_ok=True)

# %% load and merge data
pct_df = pd.read_csv(percent_infected_csv)
well_map_df = pd.read_csv(well_map_csv)

print(f"Percent infected columns: {list(pct_df.columns)}")
print(f"Well map columns: {list(well_map_df.columns)}")

# Merge on well == Well ID
merged = pct_df.merge(
    well_map_df,
    left_on="well",
    right_on="Well ID",
    how="inner",
)

print(f"Rows after merge: {len(merged)}")
print(f"Infection types: {merged['Infection'].unique()}")
print(f"MOI values: {sorted(merged['Multiplicity of infection'].unique())}")

# %% split by virus and collect MOIs
# Assume 'Infection' column contains virus name (ZIKV / DENV / mock etc.)
moi_values = sorted(merged["Multiplicity of infection"].dropna().unique())
virus_types = [v for v in VIRUS_ORDER if v in merged["Infection"].str.upper().values
               or any(v in str(x).upper() for x in merged["Infection"].unique())]

# Fallback: use unique infection values that are not mock
all_infections = merged["Infection"].unique()
if not virus_types:
    virus_types = [v for v in all_infections if "mock" not in str(v).lower()]
    print(f"Falling back to infection types: {virus_types}")

n_moi = len(moi_values)
n_virus = len(virus_types)

if n_moi == 0 or n_virus == 0:
    raise ValueError("No MOI or virus data found. Check CSV contents and column names.")

# %% helper: match infection label to virus name
def matches_virus(infection_label: str, virus: str) -> bool:
    """Case-insensitive check if infection_label contains the virus name."""
    return virus.upper() in str(infection_label).upper()

# %% build grid of subplots
fig, axes = plt.subplots(
    n_moi,
    n_virus,
    figsize=(5 * n_virus, 4 * n_moi),
    sharex=False,
    sharey=True,
    squeeze=False,
)
fig.suptitle("% Infected vs Hours Post Infection", fontsize=13, y=1.01)

for row_idx, moi in enumerate(moi_values):
    for col_idx, virus in enumerate(virus_types):
        ax = axes[row_idx][col_idx]

        # Filter rows for this MOI × virus
        mask = (
            merged["Multiplicity of infection"] == moi
        ) & merged["Infection"].apply(lambda x: matches_virus(x, virus))
        subset = merged[mask].copy()

        if subset.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{virus} MOI={moi}")
            continue

        # Use 'time' column (hours post infection) from percent_infected_csv
        # If a separate HPI column exists in well_map use that
        time_col = "time" if "time" in subset.columns else "Hours post infection"
        if time_col not in subset.columns:
            # Try to find any time-like column
            time_col = [c for c in subset.columns if "time" in c.lower() or "hour" in c.lower()][0]

        # Compute mean and SEM across FOVs at each timepoint
        grouped = subset.groupby(time_col)["percent_infected"]
        mean_vals = grouped.mean()
        sem_vals = grouped.sem().fillna(0)
        time_points = mean_vals.index.values

        color = VIRUS_COLORS.get(virus, "steelblue")
        ax.errorbar(
            time_points,
            mean_vals.values,
            yerr=sem_vals.values,
            fmt="-o",
            color=color,
            markersize=5,
            linewidth=1.8,
            capsize=4,
            elinewidth=1.2,
            label=f"{virus} MOI={moi}",
        )

        ax.set_xlabel("Hours Post Infection", fontsize=9)
        ax.set_ylabel("% Infected", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title(f"{virus}  MOI = {moi}", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

plt.tight_layout()
save_path = output_dir / "infection_dynamics_grid.svg"
fig.savefig(save_path, format="svg", bbox_inches="tight")
print(f"Saved: {save_path}")
plt.close(fig)
