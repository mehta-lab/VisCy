"""Use pre-trained ImageNet models to extract features from ALFI images."""

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
from viscy.transforms import ScaleIntensityRangePercentilesd
import phate

from viscy.representation.evaluation.distance import (
    compute_embedding_distances,
    analyze_and_plot_distances,
)
from viscy.representation.evaluation.distance import (
    compute_displacement_statistics,
    compute_displacement,
)

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

# %% for ALFI division dataset

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

# %% compute metrics

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

plt.figure(figsize=(10, 6))
plt.plot(
    delta_t_seconds,
    mean_values,
    "-",
    color="orange",
    alpha=0.5,
    zorder=1,
)
plt.scatter(
    delta_t_seconds,
    mean_values,
    color="orange",
    s=20,
    label="imagenet",
    zorder=2,
)
plt.xlabel("Time Shift (seconds)", fontsize=18)
plt.ylabel("Mean Square Displacement", fontsize=18)
plt.ylim(0, 900)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=16, frameon=False)
plt.tight_layout()
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/ALFI/imagenet_pretrained_msd_linear.svg",
    dpi=300,
)

# Calculate slopes for different regions
log_delta_t = np.log(valid_delta_t)
log_means = np.log(valid_means)
n_points = len(log_delta_t)
early_end = n_points // 3
early_slope, _ = np.polyfit(log_delta_t[:early_end], log_means[:early_end], 1)
early_slope = early_slope / (2 * embedding_dimension)
print(f"Early slope: {early_slope}")

#  Plot MSD vs time (log-log scale with slopes)
plt.figure(figsize=(10, 6))
plt.plot(
    log_delta_t,
    log_means,
    "-",
    color="orange",
    alpha=0.5,
    zorder=1,
)
plt.scatter(
    log_delta_t,
    log_means,
    color="orange",
    s=20,
    zorder=2,
    label="imagenet",
    # label=f"{interval_label} (α_early={early_slope:.2e}, α_mid={mid_slope:.2e})",
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("log of Time Shift (seconds)", fontsize=18)
plt.ylabel("log of Mean Square Displacement", fontsize=18)
plt.ylim(2, 7)
plt.grid(True, alpha=0.3, which="both")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16, frameon=False)
plt.tight_layout()
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/ALFI/imagenet_pretrained_msd_log.svg",
    dpi=300,
)

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
cell_parent = tracks[
    (tracks["fov_name"] == track_well) & (tracks["track_id"] == parent_id)
][["phate_0", "phate_1"]].reset_index(drop=True)

cell_daughter1 = tracks[
    (tracks["fov_name"] == track_well) & (tracks["track_id"] == daughter1_track)
][["phate_0", "phate_1"]].reset_index(drop=True)

cell_daughter2 = tracks[
    (tracks["fov_name"] == track_well) & (tracks["track_id"] == daughter2_track)
][["phate_0", "phate_1"]].reset_index(drop=True)


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

# %%
