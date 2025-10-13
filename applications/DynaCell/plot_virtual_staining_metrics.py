# %%
from pathlib import Path


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


csv_path = Path("~/Documents/dynacell/metrics/virtual_staining/master_metrics.csv").expanduser()
df = pd.read_csv(csv_path)

hek_time_step = 15  # in minutes
a549_time_step = 10  # in minutes

# Add a new column to df for time in hours, depending on cell_type
def compute_time_hours(row):
    if row["cell_type"].lower() == "hek293t":
        return row["time"] * hek_time_step / 60
    elif row["cell_type"].lower() == "a549":
        return row["time"] * a549_time_step / 60
    else:
        return np.nan

df["time_hours"] = df.apply(compute_time_hours, axis=1)

# %% Group by model - VSCyto3D and CellDiff, and plot pearson over time
# for each cell line, organelle, and infection condition

# Make sure the plots directory exists
plots_dir = csv_path.parent / "plots"
plots_dir.mkdir(exist_ok=True)

# Group by cell line and organelle, plot both infection conditions in one plot
# Define a consistent color mapping for models
model_colors = {
    "VSCyto3D": "#1f77b4",    # blue
    "CellDiff": "#ff7f0e",    # orange
    # Add more models here if needed
}

# %%
# Plot each group individually in a 2x3 grid of subplots

# Get all unique combinations of cell_type, organelle, infection_condition
# Keep nuclei on the top row and membrane on the bottom row
groups_nuclei = [g for g in df.groupby(["cell_type", "organelle", "infection_condition"]).groups.keys() if g[1].lower() == "nuclei"]
groups_membrane = [g for g in df.groupby(["cell_type", "organelle", "infection_condition"]).groups.keys() if g[1].lower() == "membrane"]
groups = groups_nuclei + groups_membrane

n_groups = len(groups)
n_rows, n_cols = 2, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()

for idx, ((cell_type, organelle, infection_condition), ax) in enumerate(zip(groups, axes)):
    group = df[
        (df["cell_type"] == cell_type) &
        (df["organelle"] == organelle) &
        (df["infection_condition"] == infection_condition)
    ]
    if group.empty:
        ax.set_visible(False)
        continue
    for model, sub_group in group.groupby("model"):
        color = model_colors.get(model, None)
        ax.plot(
            sub_group["time_hours"],
            sub_group["pearson"],
            label=model,
            color=color,
            marker="o"
        )
    ax.set_title(f"{cell_type}, {organelle}, {infection_condition}")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Pearson")
    ax.set_ylim(-0.1, 0.85)
    ax.legend()

# Hide any unused subplots
for j in range(idx + 1, n_rows * n_cols):
    axes[j].set_visible(False)

plt.tight_layout()
# plt.show()
plt.savefig(plots_dir / "virtual_staining_metrics_grid.png")
plt.close()

# %% Compare mock and infected conditions
for (cell_line, organelle), group in df.groupby(["cell_type", "organelle"]):
    if not group.empty:
        plt.figure()
        for (infection_condition, model), sub_group in group.groupby(["infection_condition", "model"]):
            if infection_condition.lower() == "mock":
                linestyle = "-"
            else:
                linestyle = "--"
            color = model_colors.get(model, None)
            plt.plot(
                sub_group["time_hours"],
                sub_group["pearson"],
                label=f"{model} ({infection_condition})",
                linestyle=linestyle,
                color=color,
            )
        plt.ylim(-0.1, 0.85)
        plt.legend()
        plt.title(f"{cell_line} {organelle}")
        plt.xlabel("Time [hours]")
        plt.ylabel("Pearson Cross-Correlation Coefficient")
        plt.savefig(plots_dir / f"{cell_line}_{organelle}_all_conditions.png")
        plt.close()

# %% Compare cell lines
# Plot for nuclei and membrane, comparing A549 (mock only) and HEK293T (all conditions), for both models

for organelle in ["nuclei", "membrane"]:
    plt.figure(figsize=(8, 6))
    for model in model_colors.keys():
        # A549, mock only (solid line)
        group_a549 = df[
            (df["cell_type"] == "A549") &
            (df["organelle"].str.lower() == organelle) &
            (df["infection_condition"].str.lower() == "mock") &
            (df["model"] == model)
        ]
        if not group_a549.empty:
            plt.plot(
                group_a549["time_hours"],
                group_a549["pearson"],
                label=f"A549 ({model})",
                color=model_colors[model],
                linestyle="-"
            )
        # HEK293T, mock only (dashed line)
        group_hek_mock = df[
            (df["cell_type"] == "HEK293T") &
            (df["organelle"].str.lower() == organelle) &
            (df["infection_condition"].str.lower() == "mock") &
            (df["model"] == model)
        ]
        if not group_hek_mock.empty:
            plt.plot(
                group_hek_mock["time_hours"],
                group_hek_mock["pearson"],
                label=f"HEK293T ({model})",
                color=model_colors[model],
                linestyle="--"
            )
    plt.ylim(-0.1, 0.85)
    plt.xlabel("Time [hours]")
    plt.ylabel("Pearson Cross-Correlation Coefficient")
    plt.title(f"Virtual Staining: {organelle.capitalize()} (A549 mock vs HEK293T mock)")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(plots_dir / f"compare_A549_HEK293T_{organelle}.png")
    plt.close()

# %%
