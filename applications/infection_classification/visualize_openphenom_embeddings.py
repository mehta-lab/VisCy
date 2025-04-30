# %%
"""Load saved OpenPhenom embeddings and visualize with PCA and PHATE."""

import sys
from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import seaborn as sns
import torch
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Load model directly
from transformers import AutoModel

from viscy.data.triplet import TripletDataModule
from viscy.transforms import NormalizeSampled, ScaleIntensityRangePercentilesd

plt.style.use(
    "/home/eduardo.hirata/repos/viscy/applications/contrastive_phenotyping/figures/figure.mplstyle"
)


# Function to compute PHATE
def compute_phate(embeddings, n_components=2, knn=15, decay=1, **phate_kwargs):
    phate_model = phate.PHATE(
        n_components=n_components, knn=knn, decay=decay, **phate_kwargs
    )
    phate_embedding = phate_model.fit_transform(embeddings)
    return phate_embedding


def save_legend_as_pdf(ax, filename):
    """Extract and save the legend as a separate PDF file."""
    # Get the legend
    leg = ax.get_legend()

    # Create a new figure for the legend
    fig_leg = plt.figure(figsize=(2, 1))

    # Add the legend to the new figure
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis("off")

    # Recreate the legend in the new figure
    legend = ax_leg.legend(
        *ax.get_legend_handles_labels(),
        frameon=True,
        loc="center",
        title=leg.get_title().get_text() if leg.get_title() else None,
    )

    # Save the legend figure
    fig_leg.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig_leg)

    # Remove the legend from the original plot
    ax.get_legend().remove()


class OpenPhenomWrapper:
    def __init__(self):
        try:
            self.model = AutoModel.from_pretrained(
                "recursionpharma/OpenPhenom", trust_remote_code=True
            )
            self.model.eval()
            self.model.to("cuda")
        except ImportError:
            raise ImportError(
                "Please install the OpenPhenom dependencies: "
                "pip install git+https://github.com/recursionpharma/maes_microscopy.git"
            )

    def extract_features(self, x):
        """Extract features from the input images.

        Args:
            x: Input tensor of shape [B, C, D, H, W] or [B, C, H, W]

        Returns:
            Features of shape [B, 384]
        """
        # OpenPhenom expects [B, C, H, W] but our data is [B, C, D, H, W]
        # If 5D input, take middle slice or average across D
        if x.dim() == 5:
            # Take middle slice
            d = x.shape[2]
            x = x[:, :, d // 2, :, :]

        # Convert to uint8 as OpenPhenom expects uint8 inputs
        if x.dtype != torch.uint8:
            model_device = self.model.device
            x = (
                ((x - x.min()) / (x.max() - x.min()) * 255)
                .clamp(0, 255)
                .to(torch.uint8)
            ).to(model_device)

        # Get embeddings
        self.model.return_channelwise_embeddings = False
        with torch.no_grad():
            embeddings = self.model.predict(x)

        return embeddings


#   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.
#  / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \
# '-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'
#
# Define paths for the outputs
output_dir = Path(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/openphenom"
)
output_dir.mkdir(parents=True, exist_ok=True)
# %%
# INFECTION DATASET

# Setup OpenPhenom wrapper model
openphenom = OpenPhenomWrapper()

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

print("Extracting features...")
features = []
indices = []

with torch.inference_mode():
    for batch in tqdm(dm.predict_dataloader()):
        # Get both channels and handle dimensions properly
        img = batch["anchor"]
        features.append(openphenom.extract_features(img))
        indices.append(batch["index"])

pooled = torch.cat(features).cpu().numpy()
tracks = pd.concat([pd.DataFrame(idx) for idx in indices])

output_path = Path(output_dir / "openphenom_pretrained_features_infection_rfp_only.csv")
print(f"Saving features to {output_path}")
# Save tracks and pooled features
tracks.to_csv(output_path, index=False)
np.save(output_dir / "pooled_features_infection_rfp_only.npy", pooled)

# %%
# Load the features and pool them
tracks_path = output_dir / "openphenom_pretrained_features_infection_rfp_only.csv"
tracks = pd.read_csv(tracks_path)
pooled = np.load(output_dir / "pooled_features_infection_rfp_only.npy")

# Compute PCA for original features
print("Computing PCA on original features...")
scaled_features = StandardScaler().fit_transform(pooled)
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Compute PHATE for original features
print("Computing PHATE on original features...")
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

# %%
# Compute PCA of the original features
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
    output_dir / "openphenom_infection_pca.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
# Visualize PHATE of original features
plt.figure(figsize=(10, 10))
ax = sns.scatterplot(
    x=tracks["phate_0"],
    y=tracks["phate_1"],
    hue=tracks["infection"],
    palette={1: "steelblue", 2: "orange"},
    legend="brief",
    rasterized=True,
    s=20,  # Point size
)
ax.set_xlabel("PHATE1")
ax.set_ylabel("PHATE2")

# Save legend separately
save_legend_as_pdf(ax, output_dir / "openphenom_infection_rfp_only_phate_legend.pdf")

# Save the main figure without legend
plt.savefig(
    output_dir / "openphenom_infection_rfp_only_phate.pdf", dpi=300, bbox_inches="tight"
)
plt.savefig(
    output_dir / "openphenom_infection_rfp_only_phate.png", dpi=300, bbox_inches="tight"
)
# plt.close()
# %%
# Accuracy plot
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
    label="mock OpenPhenom",
    color="magenta",
    linestyle="-",
)
plt.plot(
    time_points_test * 0.5 + 3,
    infected_true_infected,
    label="MOI OpenPhenom",
    color="green",
    linestyle="-",
)
plt.ylim(0, 100)
plt.savefig(output_dir / "openphenom_infection_rfp_only_over_time.pdf", dpi=300)
plt.legend()

# %%
#   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.
#  / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \
# '-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'

# ALFI DATASET
print("Loading OpenPhenom model...")
openphenom = OpenPhenomWrapper()

print("Setting up data module...")
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

print("Extracting features...")
features = []
indices = []

with torch.inference_mode():
    for batch in tqdm(dm.predict_dataloader()):
        image = batch["anchor"][:, :, 0]
        rgb_image = image.repeat(1, 3, 1, 1).to("cuda")
        features.append(openphenom.extract_features(rgb_image))
        indices.append(batch["index"])
# %%
print("Processing features...")
pooled = torch.cat(features).cpu().numpy()
tracks = pd.concat([pd.DataFrame(idx) for idx in indices])
output_path = Path(output_dir / "openphenom_pretrained_features_ALFI.csv")
tracks.to_csv(output_path, index=False)
np.save(output_dir / "pooled_features_ALFI.npy", pooled)
print(f"Saving features to {output_path}")

# %%
print("Computing PCA and PHATE...")
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
# %%
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
    output_dir / "openphenom_ALFI_pca.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
# Visualize PHATE of original features
plt.figure(figsize=(10, 10))
ax = sns.scatterplot(
    x=tracks["phate_0"],
    y=tracks["phate_1"],
    hue=tracks["division"],
    palette={0: "steelblue", 1: "orange"},
    legend="brief",
    rasterized=True,
    s=20,  # Point size
)
ax.set_xlabel("PHATE1")
ax.set_ylabel("PHATE2")

# Save legend separately
save_legend_as_pdf(ax, output_dir / "openphenom_infection_phate_legend.pdf")

# Save the main figure without legend
plt.savefig(output_dir / "openphenom_ALFI_phate.pdf", dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "openphenom_ALFI_phate.png", dpi=300, bbox_inches="tight")

# %%
