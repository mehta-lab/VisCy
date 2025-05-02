"""Use pre-trained ImageNet models to extract features from images."""

# %%

import pandas as pd
import seaborn as sns
import timm
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from viscy.data.triplet import TripletDataModule
from viscy.transforms import ScaleIntensityRangePercentilesd, NormalizeSampled
import phate
from xarray import Dataset
import xarray as xr
from typing import Literal
from viscy.representation.evaluation.clustering import (
    pairwise_distance_matrix,
    rank_nearest_neighbors,
)

# from viscy.representation.evaluation.distance import compute_piece_wise_dissimilarity, analyze_and_plot_distances

# %% function to compute phate from embedding values


def compute_phate(embeddings, n_components=2, knn=15, decay=0.5, **phate_kwargs):

    # Compute PHATE embeddings
    phate_model = phate.PHATE(
        n_components=n_components, knn=knn, decay=decay, **phate_kwargs
    )
    phate_embedding = phate_model.fit_transform(embeddings)

    return phate_embedding


# %%
model = timm.create_model("convnext_tiny", pretrained=True).eval().to("cuda")

# %%
#   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.
#  / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \
# '-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'

# for ALFI division dataset

dm = TripletDataModule(
    data_path="/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets/float_phase_ome_zarr_output_test.zarr",
    tracks_path="/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets/track_phase_ome_zarr_output_test.zarr",
    source_channel=["DIC"],
    z_range=(0, 1),
    batch_size=128,
    num_workers=8,
    initial_yx_patch_size=(192, 192),
    final_yx_patch_size=(192, 192),
    normalizations=[
        ScaleIntensityRangePercentilesd(
            keys=["DIC"], lower=50, upper=99, b_min=0.0, b_max=1.0
        )
    ],
)
dm.prepare_data()
dm.setup("predict")

# %%
features = []
indices = []

with torch.inference_mode():
    for batch in tqdm(dm.predict_dataloader()):
        image = batch["anchor"][:, :, 0]
        rgb_image = image.repeat(1, 3, 1, 1).to("cuda")
        features.append(model.forward_features(rgb_image))
        indices.append(batch["index"])

# %%
pooled = torch.cat(features).mean(dim=(2, 3)).cpu().numpy()
tracks = pd.concat([pd.DataFrame(idx) for idx in indices])

# %% PCA and phate computation

scaled_features = StandardScaler().fit_transform(pooled)
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

phate_embedding = compute_phate(
    embeddings=pooled,
    n_components=2,
    knn=5,
    decay=40,
    n_jobs=15,
)

# %% add pooled to dataframe naming each column with feature_i
for i, feature in enumerate(pooled.T):
    tracks[f"feature_{i}"] = feature
# add pca features to dataframe naming each column with pca_i
for i, feature in enumerate(pca_features.T):
    tracks[f"pca_{i}"] = feature
# add phate features to dataframe naming each column with phate_i
for i, feature in enumerate(phate_embedding.T):
    tracks[f"phate_{i}"] = feature

# save the dataframe as csv
tracks.to_csv(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/code/ALFI/imagenet_pretrained_features.csv",
    index=False,
)

# %% load the dataframe
tracks = pd.read_csv(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/code/ALFI/imagenet_pretrained_features.csv"
)
SECONDS_PER_FRAME = 7 * 60  # seconds

# %% load annotations


ann_root = Path(
    "/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets"
)
ann_path = ann_root / "test_annotations.csv"
annotation = pd.read_csv(ann_path)
annotation["fov_name"] = "/" + annotation["fov ID"]

# Initialize the division column with NaN values
tracks["division"] = float("nan")

# Populate division values by matching fov_name and track_id and t
for index, row in annotation.iterrows():
    mask = (
        (tracks["fov_name"] == row["fov_name"])
        & (tracks["track_id"] == row["track_id"])
        & (tracks["t"] == row["t"])
    )
    tracks.loc[mask, "division"] = row["division"]

# Print number of NaN values to verify the matching
print(f"Number of NaNs in division column: {tracks['division'].isna().sum()}")

# remove rows with division = -1
tracks = tracks[tracks["division"] != -1]

# %% plot PCA and phatemaps with annotations
plt.figure(figsize=(10, 10))
ax = sns.scatterplot(
    x=tracks["pca_0"],
    y=tracks["pca_1"],
    hue=tracks["division"],
    palette={0: "steelblue", 1: "orange"},
    legend=False,
)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/ALFI/imagenet_pretrained_PCmap.png",
    dpi=300,
)

# add interactive plotly plot
# import plotly.express as px
# fig = px.scatter(tracks, x="pca_0", y="pca_1", color="division", hover_data=["fov_name", "track_id", "t"])
# fig.show()

# phatemap with annotations
plt.figure(figsize=(10, 10))
ax = sns.scatterplot(
    x=tracks["phate_0"],
    y=tracks["phate_1"],
    hue=tracks["division"],
    palette={0: "steelblue", 1: "orange"},
    legend=False,
)
ax.set_xlabel("PHATE1")
ax.set_ylabel("PHATE2")
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/ALFI/imagenet_pretrained_phatemap.png",
    dpi=300,
)

# %% compute the accuracy of the model using a linear classifier

# remove rows with division = -1
tracks = tracks[tracks["division"] != -1]

# dataframe for training set, fov names starts with "/B/4/6" or "/B/4/7" or "/A/3/"
data_train_val = tracks[
    tracks["fov_name"].str.contains("/0/0/0")
    | tracks["fov_name"].str.contains("/0/1/0")
    | tracks["fov_name"].str.contains("/0/2/0")
]

data_test = tracks[
    tracks["fov_name"].str.contains("/0/3/0")
    | tracks["fov_name"].str.contains("/0/4/0")
]

x_train = data_train_val.drop(
    columns=[
        "division",
        "fov_name",
        "t",
        "track_id",
        "id",
        "parent_id",
        "parent_track_id",
        "pca_0",
        "pca_1",
    ]
)
y_train = data_train_val["division"]

# train a logistic regression model
clf = LogisticRegression(random_state=0).fit(x_train, y_train)

# test the trained classifer on the other half of the data

x_test = data_test.drop(
    columns=[
        "division",
        "fov_name",
        "t",
        "track_id",
        "id",
        "parent_id",
        "parent_track_id",
        "pca_0",
        "pca_1",
    ]
)
y_test = data_test["division"]

# predict the infection state for the testing set
y_pred = clf.predict(x_test)

# compute the accuracy of the classifier

accuracy = np.mean(y_pred == y_test)
# save the accuracy for final ploting
print(f"Accuracy of model: {accuracy}")


# %% find a parent that divides to two daughter cells for ploting trajectory

track_well = "/0/2/0"
parent_id = 3  # 11
daughter1_track = 4  # 12
daughter2_track = 5  # 13

# %%
cell_parent = (
    tracks[(tracks["fov_name"] == track_well) & (tracks["track_id"] == parent_id)][
        ["phate_0", "phate_1"]
    ]
    .reset_index(drop=True)
    .iloc[::2]
)

cell_daughter1 = (
    tracks[
        (tracks["fov_name"] == track_well) & (tracks["track_id"] == daughter1_track)
    ][["phate_0", "phate_1"]]
    .reset_index(drop=True)
    .iloc[::2]
)

cell_daughter2 = (
    tracks[
        (tracks["fov_name"] == track_well) & (tracks["track_id"] == daughter2_track)
    ][["phate_0", "phate_1"]]
    .reset_index(drop=True)
    .iloc[::2]
)


# %% Plot: display one arrow at end of trajectory of cell overlayed on PHATE

sns.scatterplot(
    x=tracks["phate_0"],
    y=tracks["phate_1"],
    hue=tracks["division"],
    palette={0: "steelblue", 1: "orange"},
    s=7,
    alpha=0.5,
)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("PHATE1", fontsize=14)
plt.ylabel("PHATE2", fontsize=14)
plt.legend([])

# sns.lineplot(x=cell_parent["PHATE1"], y=cell_parent["PHATE2"], color="black", linewidth=2)
# sns.lineplot(
#     x=cell_daughter1["PHATE1"], y=cell_daughter1["PHATE2"], color="blue", linewidth=2
# )
# sns.lineplot(
#     x=cell_daughter2["PHATE1"], y=cell_daughter2["PHATE2"], color="red", linewidth=2
# )

from matplotlib.patches import FancyArrowPatch

parent_arrow = FancyArrowPatch(
    (cell_parent["phate_0"].values[28], cell_parent["phate_1"].values[28]),
    (cell_parent["phate_0"].values[35], cell_parent["phate_1"].values[35]),
    color="black",
    arrowstyle="->",
    mutation_scale=20,  # reduce the size of arrowhead by half
    lw=2,
    shrinkA=0,
    shrinkB=0,
)
plt.gca().add_patch(parent_arrow)
parent_arrow = FancyArrowPatch(
    (cell_parent["phate_0"].values[35], cell_parent["phate_1"].values[35]),
    (cell_parent["phate_0"].values[38], cell_parent["phate_1"].values[38]),
    color="black",
    arrowstyle="->",
    mutation_scale=20,  # reduce the size of arrowhead by half
    lw=2,
    shrinkA=0,
    shrinkB=0,
)
plt.gca().add_patch(parent_arrow)
daughter1_arrow = FancyArrowPatch(
    (cell_daughter1["phate_0"].values[0], cell_daughter1["phate_1"].values[0]),
    (cell_daughter1["phate_0"].values[1], cell_daughter1["phate_1"].values[1]),
    color="blue",
    arrowstyle="->",
    mutation_scale=20,  # reduce the size of arrowhead by half
    lw=2,
    shrinkA=0,
    shrinkB=0,
)
plt.gca().add_patch(daughter1_arrow)
daughter1_arrow = FancyArrowPatch(
    (cell_daughter1["phate_0"].values[1], cell_daughter1["phate_1"].values[1]),
    (cell_daughter1["phate_0"].values[10], cell_daughter1["phate_1"].values[10]),
    color="blue",
    arrowstyle="->",
    mutation_scale=20,  # reduce the size of arrowhead by half
    lw=2,
    shrinkA=0,
    shrinkB=0,
)
plt.gca().add_patch(daughter1_arrow)
daughter2_arrow = FancyArrowPatch(
    (cell_daughter2["phate_0"].values[0], cell_daughter2["phate_1"].values[0]),
    (cell_daughter2["phate_0"].values[1], cell_daughter2["phate_1"].values[1]),
    color="red",
    arrowstyle="->",
    mutation_scale=20,  # reduce the size of arrowhead by half
    lw=2,
    shrinkA=0,
    shrinkB=0,
)
plt.gca().add_patch(daughter2_arrow)
daughter2_arrow = FancyArrowPatch(
    (cell_daughter2["phate_0"].values[1], cell_daughter2["phate_1"].values[1]),
    (cell_daughter2["phate_0"].values[10], cell_daughter2["phate_1"].values[10]),
    color="red",
    arrowstyle="->",
    mutation_scale=20,  # reduce the size of arrowhead by half
    lw=2,
    shrinkA=0,
    shrinkB=0,
)
plt.gca().add_patch(daughter2_arrow)

plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/ALFI/appendix_ALFI_div_track_imageNet.png",
    dpi=300,
)


#   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.
#  / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \
# '-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'


# %% for sensor infection dataset with phase and rfp

dm = TripletDataModule(
    data_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr",
    tracks_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr",
    source_channel=["Phase3D", "RFP"],
    batch_size=128,
    num_workers=8,
    z_range=(24, 29),
    initial_yx_patch_size=(128, 128),
    final_yx_patch_size=(128, 128),
    normalizations=[
        ScaleIntensityRangePercentilesd(
            keys=["RFP"], lower=50, upper=99, b_min=0.0, b_max=1.0
        ),
        NormalizeSampled(
            keys=["Phase3D"], level="fov_statistics", subtrahend="mean", divisor="std"
        ),
    ],
)
dm.prepare_data()
dm.setup("predict")

# %%
features = []
indices = []

with torch.inference_mode():
    for batch in tqdm(dm.predict_dataloader()):
        # Get both channels and handle dimensions properly
        phase = batch["anchor"][:, 0]  # Shape: [batch, z, h, w]
        phase = phase[:, 2]  # Get middle slice: [batch, h, w]
        rfp = batch["anchor"][:, 1]  # Shape: [batch, z, h, w]
        rfp = rfp.max(dim=1)[0]  # Max project z: [batch, h, w]

        # Create RGB image using phase for all channels and adding RFP to red channel
        rgb_image = torch.stack(
            [
                phase + rfp,  # Red channel: phase + RFP
                phase,  # Green channel: phase only
                phase,  # Blue channel: phase only
            ],
            dim=1,
        ).to(
            "cuda"
        )  # Final shape: [batch, 3, h, w]

        features.append(model.forward_features(rgb_image))
        indices.append(batch["index"])

# %%
pooled = torch.cat(features).mean(dim=(2, 3)).cpu().numpy()
tracks = pd.concat([pd.DataFrame(idx) for idx in indices])

# %% PCA and phate computation

scaled_features = StandardScaler().fit_transform(pooled)
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

phate_embedding = compute_phate(
    embeddings=pooled,
    n_components=2,
    knn=5,
    decay=40,
    n_jobs=15,
)

# %% add pooled to dataframe naming each column with feature_i
for i, feature in enumerate(pooled.T):
    tracks[f"feature_{i}"] = feature
# add pca features to dataframe naming each column with pca_i
for i, feature in enumerate(pca_features.T):
    tracks[f"pca_{i}"] = feature
# add phate features to dataframe naming each column with phate_i
for i, feature in enumerate(phate_embedding.T):
    tracks[f"phate_{i}"] = feature

# # save the dataframe as csv
# tracks.to_csv("/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/code/ALFI/imagenet_pretrained_features.csv", index=False)

# %% load the dataframe
# tracks = pd.read_csv("/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/code/ALFI/imagenet_pretrained_features.csv")

# %% load annotations

ann_root = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred"
)
ann_path = ann_root / "extracted_inf_state.csv"
annotation = pd.read_csv(ann_path)
annotation["fov_name"] = "/" + annotation["fov_name"]

# Initialize the infection column with NaN values
tracks["infection"] = float("nan")

# Populate infection values by matching fov_name and track_id
for index, row in tracks.iterrows():
    matching_annotations = annotation.loc[
        (annotation["fov_name"] == row["fov_name"])
        & (annotation["track_id"] == row["track_id"]),
        "infection_state",
    ]
    if len(matching_annotations) > 0:
        tracks.loc[index, "infection"] = matching_annotations.iloc[0]

# find number of NaNs in infection column
print(f"Number of NaNs in infection column: {tracks['infection'].isna().sum()}")

# remove rows with infection = 0
tracks = tracks[tracks["infection"] != 0]

# %% plot PCA and phatemaps with annotations
plt.figure(figsize=(10, 10))
ax = sns.scatterplot(
    x=tracks["pca_0"],
    y=tracks["pca_1"],
    hue=tracks["infection"],
    palette={1: "steelblue", 2: "orange"},
    legend=False,
)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/imagenet_pretrained_PCmap.png",
    dpi=300,
)

# phatemap with annotations
plt.figure(figsize=(10, 10))
ax = sns.scatterplot(
    x=tracks["phate_0"],
    y=tracks["phate_1"],
    hue=tracks["infection"],
    palette={1: "steelblue", 2: "orange"},
    legend=False,
)
ax.set_xlabel("PHATE1")
ax.set_ylabel("PHATE2")
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/imagenet_pretrained_phatemap.png",
    dpi=300,
)

# %% compute the accuracy of the model using a linear classifier

# dataframe for training set, fov names starts with "/B/4/6" or "/B/4/7" or "/A/3/"
data_train_val = tracks[
    tracks["fov_name"].str.contains("/B/4/6")
    | tracks["fov_name"].str.contains("/B/4/7")
    | tracks["fov_name"].str.contains("/A/3/")
]

data_test = tracks[
    tracks["fov_name"].str.contains("/B/4/8")
    | tracks["fov_name"].str.contains("/B/4/9")
    | tracks["fov_name"].str.contains("/B/3/")
]

x_train = data_train_val.drop(
    columns=[
        "infection",
        "fov_name",
        "t",
        "track_id",
        "id",
        "parent_id",
        "parent_track_id",
        "pca_0",
        "pca_1",
    ]
)
y_train = data_train_val["infection"]

# train a logistic regression model
clf = LogisticRegression(random_state=0).fit(x_train, y_train)

# test the trained classifer on the other half of the data

x_test = data_test.drop(
    columns=[
        "infection",
        "fov_name",
        "t",
        "track_id",
        "id",
        "parent_id",
        "parent_track_id",
        "pca_0",
        "pca_1",
    ]
)
y_test = data_test["infection"]

# predict the infection state for the testing set
y_pred = clf.predict(x_test)

# compute the accuracy of the classifier

accuracy = np.mean(y_pred == y_test)
# save the accuracy for final ploting
print(f"Accuracy of model: {accuracy}")

# %% plot the infection state over time
# plot the predicted infection state over time for /B/3 well and /B/4 well
time_points_test = np.unique(data_test["t"])

infected_true_cntrl = []
infected_true_infected = []

for time in time_points_test:
    infected_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/3"))
        & (data_test["t"] == time)
        & (data_test["infection"] == 2)
    ].shape[0]
    total_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/3")) & (data_test["t"] == time)
    ].shape[0]
    infected_true_cntrl.append(infected_cell * 100 / total_cell)
    infected_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/4"))
        & (data_test["t"] == time)
        & (data_test["infection"] == 2)
    ].shape[0]
    total_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/4")) & (data_test["t"] == time)
    ].shape[0]
    infected_true_infected.append(infected_cell * 100 / total_cell)

plt.plot(
    time_points_test * 0.5 + 3,
    infected_true_cntrl,
    label="mock true",
    color="steelblue",
    linestyle="--",
)
plt.plot(
    time_points_test * 0.5 + 3,
    infected_true_infected,
    label="MOI true",
    color="orange",
    linestyle="--",
)
plt.ylim(0, 100)
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/imagenet_pretrained_infection_over_time.svg",
    dpi=300,
)


# %% for sensor infection dataset with rfp only

dm = TripletDataModule(
    data_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr",
    tracks_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr",
    source_channel=["RFP"],
    batch_size=128,
    num_workers=8,
    z_range=(24, 29),
    initial_yx_patch_size=(128, 128),
    final_yx_patch_size=(128, 128),
    normalizations=[
        ScaleIntensityRangePercentilesd(
            keys=["RFP"], lower=50, upper=99, b_min=0.0, b_max=1.0
        ),
    ],
)
dm.prepare_data()
dm.setup("predict")

# %%
features = []
indices = []

with torch.inference_mode():
    for batch in tqdm(dm.predict_dataloader()):
        # Get both channels and handle dimensions properly
        rfp = batch["anchor"][:, 0]  # Shape: [batch, z, h, w]
        rfp = rfp.max(dim=1)[0]  # Max project z: [batch, h, w]

        # Create RGB image using phase for all channels and adding RFP to red channel
        rgb_image = torch.stack(
            [
                rfp,  # Red channel: RFP
                rfp,  # Green channel: RFP
                rfp,  # Blue channel: RFP
            ],
            dim=1,
        ).to(
            "cuda"
        )  # Final shape: [batch, 3, h, w]

        features.append(model.forward_features(rgb_image))
        indices.append(batch["index"])

# %%
pooled = torch.cat(features).mean(dim=(2, 3)).cpu().numpy()
tracks = pd.concat([pd.DataFrame(idx) for idx in indices])

# %% PCA and phate computation

scaled_features = StandardScaler().fit_transform(pooled)
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

phate_embedding = compute_phate(
    embeddings=pooled,
    n_components=2,
    knn=5,
    decay=40,
    n_jobs=15,
)

# %% add pooled to dataframe naming each column with feature_i
for i, feature in enumerate(pooled.T):
    tracks[f"feature_{i}"] = feature
# add pca features to dataframe naming each column with pca_i
for i, feature in enumerate(pca_features.T):
    tracks[f"pca_{i}"] = feature
# add phate features to dataframe naming each column with phate_i
for i, feature in enumerate(phate_embedding.T):
    tracks[f"phate_{i}"] = feature

# # save the dataframe as csv
# tracks.to_csv("/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/code/ALFI/imagenet_pretrained_features.csv", index=False)

# %% load the dataframe
# tracks = pd.read_csv("/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/code/ALFI/imagenet_pretrained_features.csv")

# %% load annotations

ann_root = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred"
)
ann_path = ann_root / "extracted_inf_state.csv"
annotation = pd.read_csv(ann_path)
annotation["fov_name"] = "/" + annotation["fov_name"]

# Initialize the infection column with NaN values
tracks["infection"] = float("nan")

# Populate infection values by matching fov_name and track_id
for index, row in annotation.iterrows():
    mask = (
        (tracks["fov_name"] == row["fov_name"])
        & (tracks["track_id"] == row["track_id"])
        & (tracks["t"] == row["t"])
    )
    tracks.loc[mask, "infection"] = row["infection_state"]

# find number of NaNs in infection column
print(f"Number of NaNs in infection column: {tracks['infection'].isna().sum()}")

# remove rows with infection = 0
tracks = tracks[tracks["infection"] != 0]

# %% save the tracks as csv
tracks.to_csv(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/imagenet_pretrained_infected_features.csv",
    index=False,
)

# %% load the tracks
tracks = pd.read_csv(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/imagenet_pretrained_infected_features.csv"
)
SECONDS_PER_FRAME = 30 * 60  # seconds

# %% plot PCA and phatemaps with annotations
plt.figure(figsize=(10, 10))
ax = sns.scatterplot(
    x=tracks["pca_0"],
    y=tracks["pca_1"],
    hue=tracks["infection"],
    palette={1: "steelblue", 2: "orange"},
    legend=False,
)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/imagenet_pretrained_PCmap.png",
    dpi=300,
)

# phatemap with annotations
plt.figure(figsize=(10, 10))
ax = sns.scatterplot(
    x=tracks["phate_0"],
    y=tracks["phate_1"],
    hue=tracks["infection"],
    palette={1: "steelblue", 2: "orange"},
    legend=False,
)
ax.set_xlabel("PHATE1")
ax.set_ylabel("PHATE2")
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/imagenet_pretrained_phatemap.png",
    dpi=300,
)

# %% compute the accuracy of the model using a linear classifier

# dataframe for training set, fov names starts with "/B/4/6" or "/B/4/7" or "/A/3/"
data_train_val = tracks[
    tracks["fov_name"].str.contains("/B/4/6")
    | tracks["fov_name"].str.contains("/B/4/7")
    | tracks["fov_name"].str.contains("/A/3/")
]

data_test = tracks[
    tracks["fov_name"].str.contains("/B/4/8")
    | tracks["fov_name"].str.contains("/B/4/9")
    | tracks["fov_name"].str.contains("/B/3/")
]

x_train = data_train_val.drop(
    columns=[
        "infection",
        "fov_name",
        "t",
        "track_id",
        "id",
        "parent_id",
        "parent_track_id",
        "pca_0",
        "pca_1",
    ]
)
y_train = data_train_val["infection"]

# train a logistic regression model
clf = LogisticRegression(random_state=0).fit(x_train, y_train)

# test the trained classifer on the other half of the data

x_test = data_test.drop(
    columns=[
        "infection",
        "fov_name",
        "t",
        "track_id",
        "id",
        "parent_id",
        "parent_track_id",
        "pca_0",
        "pca_1",
    ]
)
y_test = data_test["infection"]

# predict the infection state for the testing set
y_pred = clf.predict(x_test)

# compute the accuracy of the classifier

accuracy = np.mean(y_pred == y_test)
# save the accuracy for final ploting
print(f"Accuracy of model: {accuracy}")

# %% plot the infection state over time
# plot the predicted infection state over time for /B/3 well and /B/4 well
time_points_test = np.unique(data_test["t"])

infected_true_cntrl = []
infected_true_infected = []

for time in time_points_test:
    infected_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/3"))
        & (data_test["t"] == time)
        & (data_test["infection"] == 2)
    ].shape[0]
    total_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/3")) & (data_test["t"] == time)
    ].shape[0]
    infected_true_cntrl.append(infected_cell * 100 / total_cell)
    infected_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/4"))
        & (data_test["t"] == time)
        & (data_test["infection"] == 2)
    ].shape[0]
    total_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/4")) & (data_test["t"] == time)
    ].shape[0]
    infected_true_infected.append(infected_cell * 100 / total_cell)

plt.plot(
    time_points_test * 0.5 + 3,
    infected_true_cntrl,
    label="mock true",
    color="steelblue",
    linestyle="--",
)
plt.plot(
    time_points_test * 0.5 + 3,
    infected_true_infected,
    label="MOI true",
    color="orange",
    linestyle="--",
)
plt.ylim(-10, 110)
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/imagenet_pretrained_infection_over_time.svg",
    dpi=300,
)


# %% find an uninfected and infected track and overlay on scatterplot

infected_fov = "/B/4/9"
infected_track = 42
uninfected_fov = "/A/3/9"
uninfected_track = 19  # or 23

# %%
cell_uninfected = tracks[
    (tracks["fov_name"] == uninfected_fov) & (tracks["track_id"] == uninfected_track)
][["phate_0", "phate_1"]].reset_index(drop=True)

cell_infected = tracks[
    (tracks["fov_name"] == infected_fov) & (tracks["track_id"] == infected_track)
][["phate_0", "phate_1"]].reset_index(drop=True)

# %% Plot: display one arrow at end of trajectory of cell overlayed on PHATE

sns.scatterplot(
    x=tracks["phate_0"],
    y=tracks["phate_1"],
    hue=tracks["infection"],
    palette={1: "steelblue", 2: "orange"},
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

from matplotlib.patches import FancyArrowPatch


def add_arrows(df, color):
    for i in range(df.shape[0] - 1):
        start = df.iloc[i]
        end = df.iloc[i + 1]
        arrow = FancyArrowPatch(
            (start["phate_0"], start["phate_1"]),
            (end["phate_0"], end["phate_1"]),
            color=color,
            arrowstyle="-",
            mutation_scale=10,  # reduce the size of arrowhead by half
            lw=1,
            shrinkA=0,
            shrinkB=0,
        )
        plt.gca().add_patch(arrow)


# Apply arrows to the trajectories
add_arrows(cell_uninfected, color="blue")
add_arrows(cell_infected, color="red")

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("PHATE1", fontsize=14)
plt.ylabel("PHATE2", fontsize=14)
plt.legend([])

plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/imagenet_pretrained_infection_track.png",
    dpi=300,
)


#   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.
#  / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \
# '-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'


# %% compute metrics (pairwise distance, dynamic range) on embeddings

from viscy.representation.evaluation.distance import (
    compute_piece_wise_dissimilarity,
    analyze_and_plot_distances,
)


def compute_embedding_distances(
    embeddings,
    distance_metric: Literal["cosine", "euclidean", "normalized_euclidean"] = "cosine",
) -> pd.DataFrame:
    """
    Compute and save pairwise distances between embeddings.
    """
    feature_columns = [f"feature_{i}" for i in range(768)]
    features = embeddings[feature_columns].to_numpy()

    if distance_metric != "euclidean":
        features = StandardScaler().fit_transform(features)

    # Compute the distance matrix
    cross_dist = pairwise_distance_matrix(features, metric=distance_metric)

    # Normalize by sqrt of embedding dimension if using euclidean distance
    if distance_metric == "euclidean":
        cross_dist /= np.sqrt(features.shape[1])

    rank_fractions = rank_nearest_neighbors(cross_dist, normalize=True)

    # Create a DataFrame with track information
    features_df = pd.DataFrame(
        {
            "track_id": embeddings["track_id"],
            "t": embeddings["t"],
            "fov_name": embeddings["fov_name"],
        }
    )

    # Compute piece-wise dissimilarity and rank difference
    piece_wise_dissimilarity_per_track, _ = compute_piece_wise_dissimilarity(
        features_df, cross_dist, rank_fractions
    )

    all_dissimilarity = np.concatenate(piece_wise_dissimilarity_per_track)

    # Random sampling values in the dissimilarity matrix
    n_samples = len(all_dissimilarity)
    random_indices = np.random.randint(0, len(cross_dist), size=(n_samples, 2))
    sampled_values = cross_dist[random_indices[:, 0], random_indices[:, 1]]

    # Create and save DataFrame
    distributions_df = pd.DataFrame(
        {
            "adjacent_frame": pd.Series(all_dissimilarity),
            "random_sampling": pd.Series(sampled_values),
        }
    )

    return distributions_df


# Compute distances and metrics
distance_df = compute_embedding_distances(tracks, distance_metric="cosine")
metrics = analyze_and_plot_distances(
    distance_df,
    output_file_path=Path(
        "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/ALFI/imagenet_pretrained_distance_plot.pdf"
    ),
    overwrite=True,
)
print(f"Pairwise distance adjacent frame: {metrics['dissimilarity_mean']}")
print(f"Pairwise distance random sampling: {metrics['random_mean']}")
print(
    f"Ratio of pairwise distance dynamic range: {metrics['random_mean'] / metrics['dissimilarity_mean']}"
)
print(f"Dynamic range: {metrics['dynamic_range']}")


# %% MSD slope
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List
from viscy.representation.evaluation.distance import compute_displacement_statistics


def compute_displacement(
    embedding_dataset,
    distance_metric: Literal["euclidean_squared", "cosine"] = "euclidean_squared",
    max_delta_t: int = None,
) -> Dict[int, List[float]]:
    """Compute displacements between embeddings at different time differences.

    For each time difference τ, computes distances between embeddings of the same cell
    separated by τ timepoints. Supports multiple distance metrics.

    Parameters
    ----------
    embedding_dataset : xarray.Dataset
        Dataset containing embeddings and metadata with the following variables:
        - features: (N, D) array of embeddings
        - fov_name: (N,) array of field of view names
        - track_id: (N,) array of cell track IDs
        - t: (N,) array of timepoints
    distance_metric : str, optional
        The metric to use for computing distances between embeddings.
        Valid options are:
        - "euclidean_squared": Squared Euclidean distance (default)
        - "cosine": Cosine similarity
    max_delta_t : int, optional
        Maximum time difference τ to compute displacements for.
        If None, uses the maximum possible time difference in the dataset.

    Returns
    -------
    Dict[int, List[float]]
        Dictionary mapping time difference τ to list of displacements.
        Each displacement value represents the distance between a pair of
        embeddings from the same cell separated by τ timepoints.
    """

    # Get data from dataset
    fov_names = embedding_dataset["fov_name"].values
    track_ids = embedding_dataset["track_id"].values
    timepoints = embedding_dataset["t"].values
    feature_columns = [f"feature_{i}" for i in range(768)]
    embeddings = embedding_dataset[feature_columns].to_numpy()

    # Check if max_delta_t is provided, otherwise use the maximum timepoint
    if max_delta_t is None:
        max_delta_t = timepoints.max()

    displacement_per_delta_t = defaultdict(list)
    # Process each sample
    for i in tqdm(range(len(fov_names)), desc="Processing FOVs"):
        fov_name = fov_names[i]
        track_id = track_ids[i]
        current_time = timepoints[i]
        current_embedding = embeddings[i].reshape(1, -1)

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
    return dict(displacement_per_delta_t)


embedding_dimension = tracks.shape[1]
# Compute displacements
displacements = compute_displacement(
    embedding_dataset=tracks,
    distance_metric="euclidean_squared",
)
means, stds = compute_displacement_statistics(displacements)
# Sort by delta_t for plotting
delta_t = sorted(means.keys())
mean_values = [means[delta_t] for delta_t in delta_t]
std_values = [stds[delta_t] for delta_t in delta_t]
delta_t_seconds = [i * SECONDS_PER_FRAME for i in delta_t]
# Filter out non-positive values for log scale
valid_mask = np.array(mean_values) > 0
valid_delta_t = np.array(delta_t_seconds)[valid_mask]
valid_means = np.array(mean_values)[valid_mask]

# Calculate slopes for different regions
log_delta_t = np.log(valid_delta_t)
log_means = np.log(valid_means)
n_points = len(log_delta_t)
early_end = n_points // 3
early_slope, _ = np.polyfit(log_delta_t[:early_end], log_means[:early_end], 1)
early_slope = early_slope / (2 * embedding_dimension)
print(f"Early slope: {early_slope}")

# %%
