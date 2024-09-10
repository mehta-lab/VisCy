# %% Imports
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP


from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import load_annotation


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

rank_features = np.linalg.matrix_rank(dataset["features"].values)
rank_projections = np.linalg.matrix_rank(dataset["projections"].values)

PCA_features = PCA().fit(dataset["features"].values)
PCA_projection = PCA().fit(dataset["projections"].values)

plt.plot(
    PCA_features.explained_variance_ratio_, label=f"features, rank={rank_features}"
)
plt.plot(
    PCA_projection.explained_variance_ratio_,
    label=f"projections, rank={rank_projections}",
)
plt.legend()
plt.xlabel("n_components")
plt.show()

# %% interactive quality control: UMAP

# %% interactive quality control: pairwise distances


# %% Evaluation: infection score

## Overlay UMAP and infection state
## Linear classification accuracy
## Clustering accuracy

# %% Evaluation: cell division

# %% Evaluation: correlation between principal components and computed features
