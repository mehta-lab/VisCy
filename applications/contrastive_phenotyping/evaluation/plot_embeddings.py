# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import (
    dataset_of_tracks,
    load_annotation,
)

# %% Paths and parameters.

features_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/contrastive_tune_augmentations/predict/2024_06_13/l2_projection_batchnorm-128p.zarr"
)
data_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/2-register/registered_chunked.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/4.1-tracking/test_tracking_4.zarr"
)

# %%
embedding_dataset = read_embedding_dataset(features_path)
embedding_dataset

# %%
# Compute PCA of the features and projections to estimate the number of components to keep.
PCA_features = PCA(n_components=100).fit(embedding_dataset["features"].values)
PCA_projection = PCA(n_components=100).fit(embedding_dataset["projections"].values)

plt.plot(PCA_features.explained_variance_ratio_, label="features")
plt.plot(PCA_projection.explained_variance_ratio_, label="projections")
plt.legend()
plt.xlabel("n_components")
plt.show()

# TODO: Include the followiing in the standard report.
# * Explained variance of the features and projections.
# * The UMAPs of the features and projections.
# * 2D image of the embeddings of features and projections of test tracks (e.g., infected, uninfected, dividing, non-dividing).
# * Heatmaps of annotations over UMAPs.


# %%
print(np.linalg.matrix_rank(embedding_dataset["features"].values))
print(np.linalg.matrix_rank(embedding_dataset["projections"].values))

# %%
# Extract a track from the dataset and visualize its features.

fov_name = "/0/1/000000"  # "/B/4/4" FOV names can change between datasets.
track_id = 21
all_tracks_FOV = embedding_dataset.sel(fov_name=fov_name)
a_track_in_FOV = all_tracks_FOV.sel(track_id=track_id)
# Why is sample dimension ~22000 long after the dataset is sliced by FOV and by track_id?
indices = np.arange(a_track_in_FOV.sizes["sample"])
features_track = a_track_in_FOV["features"]
time_stamp = features_track["t"][indices].astype(str)

px.imshow(
    features_track.values[indices],
    labels={
        "x": "feature",
        "y": "t",
        "color": "value",
    },  # change labels to match our metadata
    y=time_stamp,
    # show fov_name as y-axis
)
# normalize individual features.

scaled_features_track = StandardScaler().fit_transform(features_track.values)
px.imshow(
    scaled_features_track,
    labels={
        "x": "feature",
        "y": "t",
        "color": "value",
    },  # change labels to match our metadata
    y=time_stamp,
    # show fov_name as y-axis
)
# Scaled features are centered around 0 with a standard deviation of 1.
# Each feature is individually normalized along the time dimension.

plt.plot(np.mean(scaled_features_track, axis=1), label="scaled_mean")
plt.plot(np.std(scaled_features_track, axis=1), label="scaled_std")
plt.plot(np.mean(features_track.values, axis=1), label="mean")
plt.plot(np.std(features_track.values, axis=1), label="std")
plt.legend()
plt.xlabel("t")
plt.show()

# %%
# Create the montage of the images of the cells in the track.

source_channel = ["Phase3D", "RFP"]
z_range = (28, 43)
predict_dataset = dataset_of_tracks(
    data_path,
    tracks_path,
    [fov_name],
    [track_id],
    z_range=z_range,
    source_channel=source_channel,
)

phase = np.stack([p["anchor"][0, 7].numpy() for p in predict_dataset])
fluor = np.stack([np.max(p["anchor"][1].numpy(), axis=0) for p in predict_dataset])

# %% Naive loop to iterate over the images and display

for t in range(len(predict_dataset)):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(phase[t].squeeze(), cmap="gray")
    axes[0].set_title("Phase")
    axes[0].axis("off")
    axes[1].imshow(fluor[t].squeeze(), cmap="gray")
    axes[1].set_title("Fluor")
    axes[1].axis("off")
    plt.title(f"t={t}")
    plt.show()

# %% display the track in napari
# import os

# import napari

# os.environ["DISPLAY"] = ":1"
# viewer = napari.Viewer()
# viewer.add_image(phase, name="Phase", colormap="gray")
# viewer.add_image(fluor, name="Fluor", colormap="magenta")

# %%
# Compute UMAP over all features
features = embedding_dataset["features"]
# or select a well:
# features = features[features["fov_name"].str.contains("B/4")]

scaled_features = StandardScaler().fit_transform(features.values)
umap = UMAP()
# Fit UMAP on all features
embedding = umap.fit_transform(scaled_features)


# %%
# Add UMAP coordinates to the dataset

features = (
    features.assign_coords(UMAP1=("sample", embedding[:, 0]))
    .assign_coords(UMAP2=("sample", embedding[:, 1]))
    .set_index(sample=["UMAP1", "UMAP2"], append=True)
)
features


sns.scatterplot(
    x=features["UMAP1"], y=features["UMAP2"], hue=features["t"], s=7, alpha=0.8
)

# %%
# Transform the track features
scaled_features_track_umap = umap.transform(scaled_features_track)
plt.plot(scaled_features_track_umap[:, 0], scaled_features_track_umap[:, 1])
plt.plot(scaled_features_track_umap[0, 0], scaled_features_track_umap[0, 1], marker="o")
plt.plot(
    scaled_features_track_umap[-1, 0], scaled_features_track_umap[-1, 1], marker="x"
)
for i in range(1, len(scaled_features_track_umap) - 1):
    plt.plot(
        scaled_features_track_umap[i, 0],
        scaled_features_track_umap[i, 1],
        marker=".",
        color="blue",
    )
plt.show()

# %%
# examine random features
random_samples = np.random.randint(0, embedding_dataset.sizes["sample"], 700)
# concatenate fov_name, track_id, and t to create a unique sample identifier
sample_id = (
    features["fov_name"][random_samples]
    + "-"
    + features["track_id"][random_samples].astype(str)
    + "-"
    + features["t"][random_samples].astype(str)
)
px.imshow(
    scaled_features[random_samples],
    labels={
        "x": "feature",
        "y": "sample",
        "color": "value",
    },  # change labels to match our metadata
    y=sample_id,
    # show fov_name as y-axis
)
# %%
ann_root = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/4.1-tracking"
)

infection = load_annotation(
    features,
    ann_root / "tracking_v1_infection.csv",
    "infection class",
    {0.0: "background", 1.0: "uninfected", 2.0: "infected"},
)
division = load_annotation(
    features,
    ann_root / "cell_division_state.csv",
    "division",
    {0: "non-dividing", 2: "dividing"},
)


# %%
sns.scatterplot(x=features["UMAP1"], y=features["UMAP2"], hue=division, s=7, alpha=0.8)

# %%
sns.scatterplot(x=features["UMAP1"], y=features["UMAP2"], hue=infection, s=7, alpha=0.8)

# %%
ax = sns.histplot(x=features["UMAP1"], y=features["UMAP2"], hue=infection, bins=64)
sns.move_legend(ax, loc="lower left")

# %%
sns.displot(
    x=features["UMAP1"],
    y=features["UMAP2"],
    kind="hist",
    col=infection,
    bins=64,
    cmap="inferno",
)

# %%
# interactive scatter plot to associate clusters with specific cells
df = pd.DataFrame({k: v for k, v in features.coords.items() if k != "features"})
df["infection"] = infection.values
df["division"] = division.values
df["well"] = df["fov_name"].str.rsplit("/", n=1).str[0]
df["fov_track_id"] = df["fov_name"] + "-" + df["track_id"].astype(str)
# select row B (DENV)
df = df[df["fov_name"].str.contains("B")]
df.sort_values("t", inplace=True)

g = px.scatter(
    data_frame=df,
    x="UMAP1",
    y="UMAP2",
    symbol="infection",
    color="well",
    hover_name="fov_name",
    hover_data=["id", "t", "track_id"],
    animation_frame="t",
    animation_group="fov_track_id",
)
g.update_layout(width=800, height=600)


# %%
# cluster features in heatmap directly
# this is very slow for large datasets even with fastcluster installed
inf_codes = pd.Series(infection.values.codes, name="infection")
lut = dict(zip(inf_codes.unique(), "brw"))
row_colors = inf_codes.map(lut)

g = sns.clustermap(
    scaled_features, row_colors=row_colors.to_numpy(), col_cluster=False, cbar_pos=None
)
g.yaxis.set_ticks([])
# %%
