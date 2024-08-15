# %%
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import matplotlib.pyplot as plt
from viscy.light.embedding_writer import read_embedding_dataset
from viscy.data.triplet import TripletDataset

# %% Paths

features_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/contrastive_tune_augmentations/predict/2024_02_04/tokenized-drop_path_0_0.zarr"
)
data_path = Path(
    "/hpc/projects/virtual_staining/2024_02_04_A549_DENV_ZIKV_timelapse/registered_chunked.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/7.1-seg_track/tracking_v1.zarr"
)

# %%
embedding_dataset = read_embedding_dataset(features_path)
embedding_dataset

# %%
# Extract a track from the dataset and visualize its features.

fov_name = "/B/4/4"
track_id = 71
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

# %%
# load all unprojected features:
features = embedding_dataset["features"]
# or select a well:
# features = features[features["fov_name"].str.contains("B/4")]
features

# %%
# examine raw features
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
    features.values[random_samples],
    labels={
        "x": "feature",
        "y": "sample",
        "color": "value",
    },  # change labels to match our metadata
    y=sample_id,
    # show fov_name as y-axis
)

# %%
scaled_features = StandardScaler().fit_transform(features.values)

umap = UMAP()

embedding = umap.fit_transform(scaled_features)
features = (
    features.assign_coords(UMAP1=("sample", embedding[:, 0]))
    .assign_coords(UMAP2=("sample", embedding[:, 1]))
    .set_index(sample=["UMAP1", "UMAP2"], append=True)
)
features

# %%
sns.scatterplot(
    x=features["UMAP1"], y=features["UMAP2"], hue=features["t"], s=7, alpha=0.8
)


# %%
def load_annotation(da, path, name, categories: dict | None = None):
    annotation = pd.read_csv(path)
    annotation["fov_name"] = "/" + annotation["fov ID"]
    annotation = annotation.set_index(["fov_name", "id"])
    mi = pd.MultiIndex.from_arrays(
        [da["fov_name"].values, da["id"].values], names=["fov_name", "id"]
    )
    selected = annotation.loc[mi][name]
    if categories:
        selected = selected.astype("category").cat.rename_categories(categories)
    return selected


# %%
ann_root = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/7.1-seg_track"
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

px.scatter(
    data_frame=pd.DataFrame(
        {k: v for k, v in features.coords.items() if k != "features"}
    ),
    x="UMAP1",
    y="UMAP2",
    color=(infection.astype(str) + " " + division.astype(str)).rename("annotation"),
    hover_name="fov_name",
    hover_data=["id", "t"],
)

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
