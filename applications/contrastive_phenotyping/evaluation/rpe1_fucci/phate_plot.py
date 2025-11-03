# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.dimensionality_reduction import compute_phate

# %%
test_data_features_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_rpe_fucci_leger_weigert/0-phenotyping/rpe_fucci_test_data_ckpt264.zarr"
)
test_drugs_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_rpe_fucci_leger_weigert/0-phenotyping/rpe_fucci_test_drugs_ckpt264.zarr"
)
cell_cycle_labels_path = "/hpc/projects/organelle_phenotyping/models/rpe_fucci/pseudolabels/cell_cycle_labels.csv"

# %% Load embeddings and annotations.

test_features = read_embedding_dataset(test_data_features_path)
# test_drugs = read_embedding_dataset(test_drugs_path)

# Load cell cycle labels
cell_cycle_labels_df = pd.read_csv(cell_cycle_labels_path, dtype={"dataset_name": str})

# Create a combined identifier for matching
sample_coords = test_features.coords["sample"].values
fov_names = [coord[0] for coord in sample_coords]
ids = [coord[1] for coord in sample_coords]

# Create DataFrame with embeddings and identifiers
embedding_df = pd.DataFrame(
    {
        "dataset_name": fov_names,
        "timepoint": ids,
    }
)

# Merge with cell cycle labels
merged_data = embedding_df.merge(
    cell_cycle_labels_df, on=["dataset_name", "timepoint"], how="inner"
)

print(f"Original embeddings: {len(embedding_df)}")
print(f"Cell cycle labels: {len(cell_cycle_labels_df)}")
print(f"Merged data: {len(merged_data)}")
print(f"Cell cycle distribution:\n{merged_data['cell_cycle_state'].value_counts()}")

# Get corresponding features for merged samples
merged_indices = merged_data.index.values
cell_cycle_states = merged_data["cell_cycle_state"].values

# %%
# compute phate
phate_kwargs = {
    "knn": 10,
    "decay": 20,
    "n_components": 2,
    "gamma": 1,
    "t": "auto",
    "n_jobs": -1,
}

phate_model, phate_embedding = compute_phate(test_features, **phate_kwargs)
# %%

# Define colorblind-friendly palette for cell cycle states (blue/orange as requested)
cycle_colors = {"G1": "#1f77b4", "G2": "#ff7f0e", "S": "#9467bd"}

plt.figure(figsize=(10, 10))
sns.scatterplot(
    x=phate_embedding[merged_indices, 0],
    y=phate_embedding[merged_indices, 1],
    hue=cell_cycle_states,
    palette=cycle_colors,
    alpha=0.6,
)
plt.title("PHATE Embedding Colored by Cell Cycle State")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")


# %%
# Plot the PHATE embedding from the xarray

plt.figure(figsize=(10, 10))
sns.scatterplot(
    x=test_features["PHATE1"][merged_indices],
    y=test_features["PHATE2"][merged_indices],
    hue=cell_cycle_states,
    palette=cycle_colors,
    alpha=0.6,
)
plt.title("PHATE1 vs PHATE2 Colored by Cell Cycle State")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
# %%
# plot the 3D PHATE embedding (Note: seaborn scatterplot doesn't support 3D, using matplotlib)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

for state in ["G1", "G2", "S"]:
    mask = cell_cycle_states == state
    ax.scatter(
        test_features["PHATE1"][merged_indices][mask],
        test_features["PHATE2"][merged_indices][mask],
        test_features["PHATE3"][merged_indices][mask],
        c=cycle_colors[state],
        alpha=0.6,
        label=state,
    )

ax.set_xlabel("PHATE1")
ax.set_ylabel("PHATE2")
ax.set_zlabel("PHATE3")
ax.set_title("3D PHATE Embedding Colored by Cell Cycle State")
ax.legend()

# %%
# Plot the PHATE embedding from test_drugs (commented out since not loaded)
# plt.figure(figsize=(10, 10))
# sns.scatterplot(
#     x=test_drugs["PHATE1"],
#     y=test_drugs["PHATE2"],
#     # hue=test_drugs["t"],
#     alpha=0.5,
# )
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# PHATE1 vs PHATE2
sns.scatterplot(
    x=test_features["PHATE1"][merged_indices],
    y=test_features["PHATE2"][merged_indices],
    hue=cell_cycle_states,
    palette=cycle_colors,
    alpha=0.6,
    ax=axes[0],
)
axes[0].set_title("PHATE1 vs PHATE2")
axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# PHATE1 vs PHATE3
sns.scatterplot(
    x=test_features["PHATE1"][merged_indices],
    y=test_features["PHATE3"][merged_indices],
    hue=cell_cycle_states,
    palette=cycle_colors,
    alpha=0.6,
    ax=axes[1],
)
axes[1].set_title("PHATE1 vs PHATE3")
axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# PHATE2 vs PHATE3
sns.scatterplot(
    x=test_features["PHATE2"][merged_indices],
    y=test_features["PHATE3"][merged_indices],
    hue=cell_cycle_states,
    palette=cycle_colors,
    alpha=0.6,
    ax=axes[2],
)
axes[2].set_title("PHATE2 vs PHATE3")
axes[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.show()
# %%
