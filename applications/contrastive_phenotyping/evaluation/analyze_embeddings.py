# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import load_annotation
from viscy.representation.evaluation.dimensionality_reduction import (
    compute_pca,
    compute_umap,
)

# %% Paths and parameters

path_embedding = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_51.zarr"
)
path_annotations_infection = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred/extracted_inf_state.csv"
)
path_annotations_division = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/"
)

path_tracks = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/4.1-tracking/test_tracking_4.zarr"
)

path_images = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/2-register/registered_chunked.zarr"
)

# %% Load embeddings and annotations.

dataset = read_embedding_dataset(path_embedding)
# load all unprojected features:
features = dataset["features"]
# or select a well:
# features - features[features["fov_name"].str.contains("B/4")]
features

feb_infection = load_annotation(
    dataset,
    path_annotations_infection,
    "infection_state",
    {0.0: "background", 1.0: "uninfected", 2.0: "infected"},
)

# %% interactive quality control: principal components
# Compute principal components and ranks of embeddings and projections.

# compute rank
rank_features = np.linalg.matrix_rank(dataset["features"].values)
rank_projections = np.linalg.matrix_rank(dataset["projections"].values)

pca_features, pca_projections, pca_df = compute_pca(dataset)

# Plot explained variance and rank
plt.plot(
    pca_features.explained_variance_ratio_, label=f"features, rank={rank_features}"
)
plt.plot(
    pca_projections.explained_variance_ratio_,
    label=f"projections, rank={rank_projections}",
)
plt.legend()
plt.xlabel("n_components")
plt.ylabel("explained variance ratio")
plt.xlim([0, 50])
plt.show()

# Density plot of first two principal components of features and projections.
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.kdeplot(data=pca_df, x="PCA1", y="PCA2", ax=ax[0], fill=True, cmap="Blues")
sns.kdeplot(data=pca_df, x="PCA1_proj", y="PCA2_proj", ax=ax[1], fill=True, cmap="Reds")
ax[0].set_title("Density plot of PCA1 vs PCA2 (features)")
ax[1].set_title("Density plot of PCA1 vs PCA2 (projections)")
plt.show()

# %% interactive quality control: UMAP
# Compute UMAP embeddings
umap_features, umap_projections, umap_df = compute_umap(dataset)

# %%
# Plot UMAP embeddings as density plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.kdeplot(data=umap_df, x="UMAP1", y="UMAP2", ax=ax[0], fill=True, cmap="Blues")
sns.kdeplot(
    data=umap_df, x="UMAP1_proj", y="UMAP2_proj", ax=ax[1], fill=True, cmap="Reds"
)
ax[0].set_title("Density plot of UMAP1 vs UMAP2 (features)")
ax[1].set_title("Density plot of UMAP1 vs UMAP2 (projections)")
plt.show()

# %% interactive quality control: pairwise distances


# %% Evaluation: infection score

## Overlay UMAP and infection state
## Linear classification accuracy
## Clustering accuracy

# %% Evaluation: cell division

# %% Evaluation: correlation between principal components and computed features
