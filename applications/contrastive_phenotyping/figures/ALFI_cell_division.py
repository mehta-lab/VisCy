
# %% Figure on ALFI cell division model showing 
# (a) Euclidean distance over a cell division event and 
# (b) difference between trajectory of cell in time-aware and classical method over division event

from pathlib import Path
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from viscy.representation.embedding_writer import read_embedding_dataset

# %% Task A: plot the Eucledian distance for a dividing cell

# Paths to datasets
feature_paths = {
    "7 min interval": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_7mins.zarr",
    "14 min interval": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_14mins.zarr",
    "28 min interval": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_28mins.zarr",
    "56 min interval": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_56mins.zarr",
    "91 min interval": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_91mins.zarr",
    "Classical": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_classical.zarr",
}

track_well = '/0/2/0'
parent_id = 3   # 11
daughter1_track = 4  # 12
daughter2_track = 5  # 13

# %% plot the eucledian distance over time lag for a parent cell for different time intervals

def compute_displacement_track(fov_name, track_id, current_time, distance_metric="euclidean_squared", max_delta_t=10):

    fov_names = embedding_dataset["fov_name"].values
    track_ids = embedding_dataset["track_id"].values
    timepoints = embedding_dataset["t"].values
    embeddings = embedding_dataset["features"].values

    # find index where fov_name, track_id and current_time match
    i = np.where(
        (fov_names == fov_name)
        & (track_ids == track_id)
        & (timepoints == current_time)
    )[0][0]
    current_embedding = embeddings[i].reshape(1, -1)

    # Check if max_delta_t is provided, otherwise use the maximum timepoint
    if max_delta_t is None:
        max_delta_t = timepoints.max()

    displacement_per_delta_t = defaultdict(list)

    # Compute displacements for each delta t
    for delta_t in range(1, max_delta_t + 1):
        future_time = current_time + delta_t
        matching_indices = np.where(
            (fov_names == fov_name)
            & (track_ids == track_id)
            & (timepoints == future_time)
        )[0]

        if len(matching_indices) == 1:
            if distance_metric == "euclidean_squared":
                future_embedding = embeddings[matching_indices[0]].reshape(1, -1)
                displacement = np.sum((current_embedding - future_embedding) ** 2)
            elif distance_metric == "cosine":
                future_embedding = embeddings[matching_indices[0]].reshape(1, -1)
                displacement = cosine_similarity(
                    current_embedding, future_embedding
                )
            displacement_per_delta_t[delta_t].append(displacement)
    
    return displacement_per_delta_t

# %% plot the eucledian distance for a parent cell

plt.figure(figsize=(10, 6))
for label, path in feature_paths.items():
    embedding_dataset = read_embedding_dataset(path)
    displacement_per_delta_t = compute_displacement_track(track_well, parent_id, 1)
    delta_ts = sorted(displacement_per_delta_t.keys())
    displacements = [np.mean(displacement_per_delta_t[delta_t]) for delta_t in delta_ts]
    plt.plot(delta_ts, displacements, label=label)

plt.xlabel("Time Interval (delta t)")
plt.ylabel("Displacement (Euclidean Distance)")
plt.title("Displacement vs Time Interval for Parent Cell")
plt.legend()
plt.show()

# %% Task B: plot the phate map and overlay the dividing cell trajectory

# for time-aware model uncomment the next three lines
# embedding_dataset = read_embedding_dataset(
#     "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_28mins.zarr"
# )

# for classical model uncomment the next three line
embedding_dataset = read_embedding_dataset(
    "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_classical.zarr"
)

PHATE1 = embedding_dataset["PHATE1"].values
PHATE2 = embedding_dataset["PHATE2"].values

# %% plot PHATE map based on the embedding dataset time points

sns.scatterplot(
    x=embedding_dataset["PHATE1"], y=embedding_dataset["PHATE2"], hue=embedding_dataset["t"], s=7, alpha=0.8
)

# %% color using human annotation for cell cycle state

def load_annotation(da, path, name, categories: dict | None = None):
    annotation = pd.read_csv(path)
    # annotation_columns = annotation.columns.tolist()
    # print(annotation_columns)
    annotation["fov_name"] = "/" + annotation["fov ID"]
    annotation = annotation.set_index(["fov_name", "id"])
    mi = pd.MultiIndex.from_arrays(
        [da["fov_name"].values, da["id"].values], names=["fov_name", "id"]
    )
    selected = annotation.reindex(mi)[name]
    if categories:
        selected = selected.astype("category").cat.rename_categories(categories)
    return selected


# %% load the cell cycle state annotation

ann_root = Path(
    "/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets"
)

division = load_annotation(
    embedding_dataset,
    ann_root / "test_annotations.csv",
    "division",
    {0: "interphase", 1: "mitosis"},
)

# %% find a parent that divides to two daughter cells for ploting trajectory

cell_parent = embedding_dataset.where(embedding_dataset["fov_name"] == track_well, drop=True).where(
    embedding_dataset["track_id"] == parent_id, drop=True
)
cell_parent = cell_parent["PHATE1"].values, cell_parent["PHATE2"].values
cell_parent = pd.DataFrame(np.column_stack(cell_parent), columns=["PHATE1", "PHATE2"])

cell_daughter1 = embedding_dataset.where(embedding_dataset["fov_name"] == track_well, drop=True).where(
    embedding_dataset["track_id"] == daughter1_track, drop=True
)
cell_daughter1 = cell_daughter1["PHATE1"].values, cell_daughter1["PHATE2"].values
cell_daughter1 = pd.DataFrame(np.column_stack(cell_daughter1), columns=["PHATE1", "PHATE2"])

cell_daughter2 = embedding_dataset.where(embedding_dataset["fov_name"] == track_well, drop=True).where(
    embedding_dataset["track_id"] == daughter2_track, drop=True
)
cell_daughter2 = cell_daughter2["PHATE1"].values, cell_daughter2["PHATE2"].values
cell_daughter2 = pd.DataFrame(np.column_stack(cell_daughter2), columns=["PHATE1", "PHATE2"])

# %% Plot: display one arrow at end of trajectory of cell overlayed on PHATE

sns.scatterplot(
    x=embedding_dataset["PHATE1"],
    y=embedding_dataset["PHATE2"],
    hue=division,
    palette={"interphase": "steelblue", "mitosis": "orangered", -1: "green"},
    s=7,
    alpha=0.5,
)

# sns.lineplot(x=cell_parent["PHATE1"], y=cell_parent["PHATE2"], color="black", linewidth=2)
# sns.lineplot(
#     x=cell_daughter1["PHATE1"], y=cell_daughter1["PHATE2"], color="blue", linewidth=2
# )
# sns.lineplot(
#     x=cell_daughter2["PHATE1"], y=cell_daughter2["PHATE2"], color="red", linewidth=2
# )

parent_arrow = FancyArrowPatch(
    (cell_parent["PHATE1"].values[28], cell_parent["PHATE2"].values[28]),
    (cell_parent["PHATE1"].values[35], cell_parent["PHATE2"].values[35]),
    color="black",
    arrowstyle="->",
    mutation_scale=20,  # reduce the size of arrowhead by half
    lw=2,
    shrinkA=0,
    shrinkB=0,
)
plt.gca().add_patch(parent_arrow)
parent_arrow = FancyArrowPatch(
    (cell_parent["PHATE1"].values[35], cell_parent["PHATE2"].values[35]),
    (cell_parent["PHATE1"].values[38], cell_parent["PHATE2"].values[38]),
    color="black",
    arrowstyle="->",
    mutation_scale=20,  # reduce the size of arrowhead by half
    lw=2,
    shrinkA=0,
    shrinkB=0,
)
plt.gca().add_patch(parent_arrow)
daughter1_arrow = FancyArrowPatch(
    (cell_daughter1["PHATE1"].values[0], cell_daughter1["PHATE2"].values[0]),
    (cell_daughter1["PHATE1"].values[1], cell_daughter1["PHATE2"].values[1]),
    color="blue",
    arrowstyle="->",
    mutation_scale=20,  # reduce the size of arrowhead by half
    lw=2,
    shrinkA=0,
    shrinkB=0,
)
plt.gca().add_patch(daughter1_arrow)
daughter1_arrow = FancyArrowPatch(
    (cell_daughter1["PHATE1"].values[1], cell_daughter1["PHATE2"].values[1]),
    (cell_daughter1["PHATE1"].values[10], cell_daughter1["PHATE2"].values[10]),
    color="blue",
    arrowstyle="->",
    mutation_scale=20,  # reduce the size of arrowhead by half
    lw=2,
    shrinkA=0,
    shrinkB=0,
)
plt.gca().add_patch(daughter1_arrow)
daughter2_arrow = FancyArrowPatch(
    (cell_daughter2["PHATE1"].values[0], cell_daughter2["PHATE2"].values[0]),
    (cell_daughter2["PHATE1"].values[1], cell_daughter2["PHATE2"].values[1]),
    color="red",
    arrowstyle="->",
    mutation_scale=20,  # reduce the size of arrowhead by half
    lw=2,
    shrinkA=0,
    shrinkB=0,
)
plt.gca().add_patch(daughter2_arrow)
daughter2_arrow = FancyArrowPatch(
    (cell_daughter2["PHATE1"].values[1], cell_daughter2["PHATE2"].values[1]),
    (cell_daughter2["PHATE1"].values[10], cell_daughter2["PHATE2"].values[10]),
    color="red",
    arrowstyle="->",
    mutation_scale=20,  # reduce the size of arrowhead by half
    lw=2,
    shrinkA=0,
    shrinkB=0,
)
plt.gca().add_patch(daughter2_arrow)

# %%
