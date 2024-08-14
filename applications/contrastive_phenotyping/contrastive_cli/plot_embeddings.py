# %%
from pathlib import Path

import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import numpy as np
from viscy.light.embedding_writer import read_embedding_dataset

# %%
dataset = read_embedding_dataset(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/contrastive_tune_augmentations/predict/2024_02_04-tokenized-drop_path_0_0.zarr"
)
dataset

# %%
# load all unprojected features:
features = dataset["features"]
# or select a well:
# features = features[features["fov_name"].str.contains("B/4")]
features

# %%
# examine raw features
random_samples = np.random.randint(0, dataset.sizes["sample"], 700)
# concatenate fov_name, track_id, and t to create a unique sample identifier
sample_id = [
    str(dataset["fov_name"][idx].values)
    + "/"
    + str(dataset["track_id"][idx].values)
    + "_"
    + str(dataset["t"][idx].values)
    for idx in random_samples
]
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
