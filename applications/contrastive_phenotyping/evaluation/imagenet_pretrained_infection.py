"""Use pre-trained ImageNet models to extract features from images."""

# %%
from pathlib import Path
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
from viscy.representation.evaluation.distance import (
    compute_displacement_statistics,
    compute_displacement,
)
from viscy.representation.evaluation.distance import (
    compute_embedding_distances,
    analyze_and_plot_distances,
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
                rfp,  # Red channel: RFP, can be phase as well
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

# %% Compute distances and metrics

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

# %%
