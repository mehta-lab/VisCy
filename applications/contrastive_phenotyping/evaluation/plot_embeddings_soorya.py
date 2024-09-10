# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from viscy.representation.embedding_writer import read_embedding_dataset

# %%
dataset = read_embedding_dataset(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/code_testing_soorya/Ver2_updateTracking_refineModel/predictions/Feb_test_2chan_128patch_128projDim/2chan_128patch_20ckpt_Feb_test.zarr"
)
dataset

# %%
# load all unprojected features:
features = dataset["features"]
# or select a well:
# features - features[features["fov_name"].str.contains("B/4")]
features


# %% perform principal componenet analysis of features

from sklearn.decomposition import PCA

pca = PCA(n_components=4)
# scaled_features = StandardScaler().fit_transform(features.values)
# pca_features = pca.fit_transform(scaled_features)
pca_features = pca.fit_transform(features.values)

features = (
    features.assign_coords(PCA1=("sample", pca_features[:, 0]))
    .assign_coords(PCA2=("sample", pca_features[:, 1]))
    .assign_coords(PCA3=("sample", pca_features[:, 2]))
    .assign_coords(PCA4=("sample", pca_features[:, 3]))
    .set_index(sample=["PCA1", "PCA2", "PCA3", "PCA4"], append=True)
)

# %% plot PCA components

plt.figure(figsize=(10, 10))
sns.scatterplot(x=features["PCA1"], y=features["PCA2"], hue=features["t"], s=7, alpha=0.8)

# %% umap with 2 components
scaled_features = StandardScaler().fit_transform(features.values)

umap = UMAP()

embedding = umap.fit_transform(features.values)
features = (
    features.assign_coords(UMAP1=("sample", embedding[:, 0]))
    .assign_coords(UMAP2=("sample", embedding[:, 1]))
    .set_index(sample=["UMAP1", "UMAP2"], append=True)
)
features

# %%
# scaled_features = StandardScaler().fit_transform(features.values)

# umap = UMAP(n_components=4)

# embedding = umap.fit_transform(scaled_features)
# features = (
#     features.assign_coords(UMAP1=("sample", embedding[:, 0]))
#     .assign_coords(UMAP2=("sample", embedding[:, 1]))
#     .assign_coords(UMAP3=("sample", embedding[:, 2]))
#     .assign_coords(UMAP4=("sample", embedding[:, 3]))
#     .set_index(sample=["UMAP1", "UMAP2", "UMAP3", "UMAP4"], append=True)
# )
# features

# %%
sns.scatterplot(
    x=features["UMAP1"], y=features["UMAP2"], hue=features["t"], s=7, alpha=0.8
)

# %%
def load_annotation(da, path, name, categories: dict | None = None):
    annotation = pd.read_csv(path)
    # annotation_columns = annotation.columns.tolist()
    # print(annotation_columns)
    annotation["fov_name"] = "/" + annotation["fov_name"]
    annotation = annotation.set_index(["fov_name", "id"])
    mi = pd.MultiIndex.from_arrays(
        [da["fov_name"].values, da["id"].values], names=["fov_name", "id"]
    )
    selected = annotation.loc[mi][name]
    if categories:
        selected = selected.astype("category").cat.rename_categories(categories)
    return selected


# %%
# ann_root = Path(
#     "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/7.1-seg_track"
# )

# infection = load_annotation(
#     features,
#     ann_root / "tracking_v1_infection.csv",
#     "infection class",
#     {0.0: "background", 1.0: "uninfected", 2.0: "infected"},
# )
# division = load_annotation(
#     features,
#     ann_root / "cell_division_state.csv",
#     "division",
#     {0: "non-dividing", 2: "dividing"},
# )


# %% new annotation

ann_root = Path("/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred")

infection = load_annotation(
    features,
    ann_root / "extracted_inf_state.csv",
    "infection_state",
    {0.0: "background", 1.0: "uninfected", 2.0: "infected"},
)

# %%
sns.scatterplot(x=features["UMAP1"], y=features["UMAP2"], hue=division, s=7, alpha=0.8)

# %%
sns.scatterplot(x=features["UMAP1"], y=features["UMAP2"], hue=infection, s=7, alpha=0.8)

# %% plot PCA components with infection hue
sns.scatterplot(x=features["PCA1"], y=features["PCA2"], hue=infection, s=7, alpha=0.8)

# %%
ax = sns.histplot(x=features["UMAP1"], y=features["UMAP2"], hue=infection, bins=64)
sns.move_legend(ax, loc="lower left")

# %% see the histogram distribution of UMAP1 and UMAP2 for each infection state
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

fig = px.scatter(
    data_frame=pd.DataFrame(
        {k: v for k, v in features.coords.items() if k != "features"}
    ),
    x="UMAP1",
    y="UMAP2",
    color=(infection.astype(str) + " " + division.astype(str)).rename("annotation"),
    hover_name="fov_name",
    hover_data=["track_id", "t"],
)
fig.update_traces(marker=dict(size=3))

# %% interactive PCA plot

fig = px.scatter(
    data_frame=pd.DataFrame(
        {k: v for k, v in features.coords.items() if k != "features"}
    ),
    x="PCA1",
    y="PCA2",
    color=(infection.astype(str) + " " + division.astype(str)).rename("annotation"),
    hover_name="fov_name",
    hover_data=["track_id", "t"],
)
fig.update_traces(marker=dict(size=3))

# %% cluster cells in PCA1 vs PCA2 space using Gaussian Mixture Model

import numpy as np
import seaborn as sns
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2)
PCA1_array = features["PCA1"].values.reshape(-1, 1)
PCA2_array = features["PCA2"].values.reshape(-1, 1)
gmm.fit(np.concatenate((PCA1_array, PCA2_array), axis=1))

GMM_predict = gmm.predict(np.concatenate((PCA1_array, PCA2_array), axis=1))
features = features.assign_coords(gmm=("sample", GMM_predict))
# display the clustering results
fig = px.scatter(
    data_frame=pd.DataFrame(
        {k: v for k, v in features.coords.items() if k != "features"}
    ),
    x="PCA1",
    y="PCA2",
    color=features["gmm"].astype(str),
    hover_name="fov_name",
    hover_data=["track_id", "t"],
)
fig.update_traces(marker=dict(size=3))

# %%
# cluster features in heatmap directly
inf_codes = pd.Series(infection.values.codes, name="infection")
lut = dict(zip(inf_codes.unique(), "brw"))
row_colors = inf_codes.map(lut)

g = sns.clustermap(
    scaled_features, row_colors=row_colors.to_numpy(), col_cluster=False, cbar_pos=None
)
g.yaxis.set_ticks([])

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
    data_frame=df[df["infection"].isin(["uninfected", "infected"])],
    x="UMAP1",
    y="UMAP2",
    symbol="well",
    color="infection",
    hover_name="fov_name",
    hover_data=["id", "t", "track_id"],
    animation_frame="t",
    animation_group="fov_track_id",
)
g.update_layout(width=800, height=600)

# %% video frame for scatter across supervised infection annotation

df = pd.DataFrame({k: v for k, v in features.coords.items() if k != "features"})
df["infection"] = infection.values
df["division"] = division.values
df["well"] = df["fov_name"].str.rsplit("/", n=1).str[0]
df["fov_track_id"] = df["fov_name"] + "-" + df["track_id"].astype(str)
df.sort_values("t", inplace=True)

for time in range(48):
    plt.clf()  # Clear the previous plot
    sns.scatterplot(
        data=df[(df["infection"].isin(["uninfected", "infected"])) & (df["t"] == time)],
        x="UMAP1",
        y="UMAP2",
        hue="infection",
        palette={"uninfected": "blue", "infected": "red", "background": "black"},
        s=12,
    )
    plt.legend().remove()
    plt.xlim(-7, 15)
    plt.ylim(2, 15)

    plt.savefig(f"/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/obsolete/videos/Supervised/scatter_infection_" + str(time).zfill(3) + ".png")

# %% video frame for scatter across virus type or wells

# for time in range(48):
#     sns.scatterplot(
#         data=df[(df["t"] == time)],
#         x="UMAP1",
#         y="UMAP2",
#         hue="well",
#         palette={"/B/3": "blue", "/A/3": "blue", "/B/4": "red", "/A/4": "green"},
#         s=12,
#     )

df_well_B4 = df[df['well'] == '/B/4']  # DENV, MOI 5
df_well_A4 = df[df['well'] == '/A/4']  # ZIka, MOI 5
df_well_Mock = df[(df['well'] == '/B/3') | (df['well'] == '/A/3')]  # Mock

for time in range(48):
    plt.clf()
    sns.scatterplot(
        data=df_well_B4[(df_well_B4["t"] == time)],
        x="UMAP1",
        y="UMAP2",
        hue="infection",
        palette={"uninfected": "black", "infected": "black", "background": "black"},
        s=12,
    )
    plt.legend().remove()
    plt.xlim(-7, 15)
    plt.ylim(2, 15)
    plt.title(f"Time: {(time*0.5)+3} hours post infection")
    plt.savefig(f"/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/obsolete/videos/Dengue/scatter_Dengue_infection_" + str(time).zfill(3) + ".png")

for time in range(48):
    plt.clf()
    sns.scatterplot(
        data=df_well_A4[(df_well_A4["t"] == time)],
        x="UMAP1",
        y="UMAP2",
        hue="infection",
        palette={"uninfected": "black", "infected": "black", "background": "black"},
        s=12,
    )
    plt.legend().remove()
    plt.title(f"Time: {(time*0.5)+3} hours post infection")
    plt.xlim(-7, 15)
    plt.ylim(2, 15)

    plt.savefig(f"/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/obsolete/videos/Zika/scatter_Zika_infection_" + str(time).zfill(3) + ".png")

for time in range(48):
    plt.clf()
    sns.scatterplot(
        data=df_well_Mock[(df_well_Mock["t"] == time)],
        x="UMAP1",
        y="UMAP2",
        hue="infection",
        palette={"uninfected": "black", "infected": "black", "background": "black"},
        s=12,
    )
    plt.legend().remove()
    plt.title(f"Time: {(time*0.5)+3} hours post infection")
    plt.xlim(-7, 15)
    plt.ylim(2, 15)

    plt.savefig(f"/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/obsolete/videos/Mock/scatter_Mock_infection_" + str(time).zfill(3) + ".png")

# do the plot next for the baove three conditions with palette: "Mock": "black", "Zika": "blue", "Dengue": "red"
for time in range(48):
    plt.clf()
    sns.scatterplot(
        data=df[(df["t"] == time)],
        x="UMAP1",
        y="UMAP2",
        hue="well",
        palette={"/B/3": "black", "/A/3": "black", "/B/4": "red", "/A/4": "blue"},
        s=12,
    )
    plt.xlim(-7, 15)
    plt.ylim(2, 15)
    plt.title(f"Time: {(time*0.5)+3} hours post infection")
    plt.legend().remove()

    plt.savefig(f"/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/obsolete/videos/Well/scatter_well_" + str(time).zfill(3) + ".png")

# %% video frame for scatter across division state for 30 cells

# div_csv_path = '/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/7.1-seg_track/track_Feb.csv'
# df_div = pd.read_csv(div_csv_path)

# plot for well A3, FOVs 0, 1, 10, 11, 12,and 13
selected_fovs = df[df['fov_name'].isin(['/A/3/0', '/A/3/1', '/A/3/10', '/A/3/11', '/A/3/12', '/A/3/13'])]

for time in range(48):
    plt.clf()
    sns.scatterplot(
        data=selected_fovs[(selected_fovs["t"] == time)],
        x="UMAP1",
        y="UMAP2",
        hue="division",
        palette={"non-dividing": "blue", "dividing": "red"},
        s=12,
    )
    plt.legend().remove()
    plt.xlim(-7, 15)
    plt.ylim(2, 15)
    plt.title(f"Time: {(time*0.5)+3} hours post infection")

    plt.savefig(f"/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/obsolete/videos/Division/scatter_division_" + str(time).zfill(3) + ".png")

# making videos
# ffmpeg -r 2 -f image2 -pattern_type glob -i "*?png" -vcodec libx264 -crf 20 -pix_fmt yuv420p output.mp4

# %% display flow field plot for df over time for one dengue infected cell 

import matplotlib.pyplot as plt

# Group the features by track_id and fov_name
grouped_features = df_well_B4.groupby(["track_id", "fov_name"])

# Create a new column for the UMAP1 and UMAP2 coordinates
df_well_B4["UMAP1_track"] = np.nan
df_well_B4["UMAP2_track"] = np.nan

# Iterate over the groups and assign UMAP coordinates to each track
for group_name, group_data in grouped_features:
    track_id, fov_name = group_name
    umap1 = group_data["UMAP1"].values
    umap2 = group_data["UMAP2"].values
    df_well_B4.loc[(df_well_B4["track_id"] == track_id) & (df_well_B4["fov_name"] == fov_name), "UMAP1_track"] = umap1
    df_well_B4.loc[(df_well_B4["track_id"] == track_id) & (df_well_B4["fov_name"] == fov_name), "UMAP2_track"] = umap2

# Compute the flow field for each cell
flow_field = np.gradient(df_well_B4[["UMAP1_track", "UMAP2_track"]].values, axis=0)

# Plot the flow field with reduced density
plt.figure(figsize=(10, 10))
plt.quiver(df_well_B4["UMAP1_track"], df_well_B4["UMAP2_track"], flow_field[:, 0], flow_field[:, 1], scale=10)
plt.xlim(-7, 15)
plt.ylim(2, 15)
plt.show()


# %% show the umap flow field of cell 30 in well B4, fov 4 with time as velocity

df_well_B4_4_30 = df[(df['fov_name'] == '/B/4/4') & (df['track_id'] == 30)]
df_well_B4_4_30.sort_values('t', inplace=True)

flow_field = np.gradient(df_well_B4_4_30[["UMAP1", "UMAP2"]].values, axis=0)

plt.figure(figsize=(10, 10))
plt.quiver(df_well_B4_4_30["UMAP1"], df_well_B4_4_30["UMAP2"], flow_field[:, 0], flow_field[:, 1], scale=10, color='r')
plt.xlim(-7, 15)
plt.ylim(2, 15)
plt.show()

df_well_A4_9_5 = df[(df['fov_name'] == '/A/4/9') & (df['track_id'] == 21)]
df_well_A4_9_5.sort_values('t', inplace=True)

flow_field = np.gradient(df_well_A4_9_5[["UMAP1", "UMAP2"]].values, axis=0)

plt.figure(figsize=(10, 10))
plt.quiver(df_well_A4_9_5["UMAP1"], df_well_A4_9_5["UMAP2"], flow_field[:, 0], flow_field[:, 1], scale=10, color='r')
plt.xlim(-7, 15)
plt.ylim(2, 15)
plt.show()

# %% use linear classifier to predict infection state from UMAP coordinates

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X = features[["UMAP1", "UMAP2"]].values.astype(int)
y = infection.values.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# %% use linear classifier to predict infection state from PCA coordinates

X = features[["PCA1", "PCA2"]].values
y = infection.values.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# %% use gaussian mixture model to cluster cells in PCA space
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2)
PCA1_array = features["PCA1"].values.reshape(-1, 1)
PCA2_array = features["PCA2"].values.reshape(-1, 1)

gmm.fit(np.concatenate((PCA1_array, PCA2_array), axis=1))

GMM_predict = gmm.predict(np.concatenate((PCA1_array, PCA2_array), axis=1))
features = features.assign_coords(gmm=("sample", GMM_predict))

# display the clustering results
fig = px.scatter(
    data_frame=pd.DataFrame(
        {k: v for k, v in features.coords.items() if k != "features"}
    ),
    x="PCA1",
    y="PCA2",
    color=features["gmm"].astype(str),
    hover_name="fov_name",
    hover_data=["track_id", "t"],
)

fig.update_traces(marker=dict(size=3))

# %%
