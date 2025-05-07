"""Use pre-trained OpenPhenom models to extract features from images."""

# %%
import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from viscy.data.triplet import TripletDataModule
from viscy.transforms import ScaleIntensityRangePercentilesd, NormalizeSampled
import phate
import xarray as xr
from typing import Literal

# Load model directly
from transformers import AutoModel


# %% function to compute phate from embedding values
def compute_phate(embeddings, n_components=2, knn=15, decay=0.5, **phate_kwargs):
    # Compute PHATE embeddings
    phate_model = phate.PHATE(
        n_components=n_components, knn=knn, decay=decay, **phate_kwargs
    )
    phate_embedding = phate_model.fit_transform(embeddings)
    return phate_embedding


# %% OpenPhenom Wrapper
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
        # OpenPhenom expects [B, C, H, W] but our data might be [B, C, D, H, W]
        # If 5D input, take middle slice or average across D
        if x.dim() == 5:
            # Take middle slice or average across D dimension
            d = x.shape[2]
            x = x[:, :, d // 2, :, :]

        # Convert to uint8 as OpenPhenom expects uint8 inputs
        if x.dtype != torch.uint8:
            x = (x * 255).clamp(0, 255).to(torch.uint8)

        # Get embeddings
        self.model.return_channelwise_embeddings = False
        with torch.no_grad():
            embeddings = self.model.predict(x)

        return embeddings


# %% Initialize OpenPhenom model
print("Loading OpenPhenom model...")
openphenom = OpenPhenomWrapper()

# %%
# For infection dataset with phase and RFP
print("Setting up data module...")
dm = TripletDataModule(
    data_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr",
    tracks_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr",
    source_channel=["Phase3D", "RFP"],
    batch_size=32,  # Lower batch size for OpenPhenom which is larger
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
print("Extracting features...")
features = []
indices = []

with torch.inference_mode():
    for batch in tqdm(dm.predict_dataloader()):
        # Get both channels and handle dimensions properly
        phase = batch["anchor"][:, 0]  # Phase channel
        rfp = batch["anchor"][:, 1]  # RFP channel

        # Create image for OpenPhenom (OpenPhenom handles multi-channel inputs natively)
        img = batch["anchor"].to("cuda")

        # Extract features using OpenPhenom
        batch_features = openphenom.extract_features(img)
        features.append(batch_features.cpu())
        indices.append(batch["index"])

# %%
print("Processing features...")
pooled = torch.cat(features).numpy()
tracks = pd.concat([pd.DataFrame(idx) for idx in indices])

# %% PCA and phate computation
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

# %% Add features to dataframe
for i, feature in enumerate(pooled.T):
    tracks[f"feature_{i}"] = feature
# Add PCA features to dataframe
for i, feature in enumerate(pca_features.T):
    tracks[f"pca_{i}"] = feature
# Add PHATE features to dataframe
for i, feature in enumerate(phate_embedding.T):
    tracks[f"phate_{i}"] = feature

# %% Save the extracted features
output_path = Path(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/openphenom_pretrained_features.csv"
)
print(f"Saving features to {output_path}")
tracks.to_csv(output_path, index=False)

# %% Load annotations
print("Loading infection annotations...")
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

# Find number of NaNs in infection column
print(f"Number of NaNs in infection column: {tracks['infection'].isna().sum()}")

# Remove rows with infection = 0 (background class)
tracks = tracks[tracks["infection"] != 0]

# %% Apply recommended batch correction for OpenPhenom features
print("Applying batch correction to OpenPhenom features...")


def batch_correction(embeddings, batch_ids, control_mask=None):
    """
    Apply batch correction to embeddings as recommended in OpenPhenom documentation.

    Args:
        embeddings: numpy array of shape [N, D] with embeddings
        batch_ids: numpy array of shape [N] with batch identifiers
        control_mask: boolean mask of shape [N] indicating control samples

    Returns:
        Batch-corrected embeddings
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # If control mask not provided, assume all samples are for fitting
    if control_mask is None:
        control_mask = np.ones(len(embeddings), dtype=bool)

    # Step 1: Fit PCA on control samples
    pca = PCA(n_components=min(embeddings.shape[1], 100))
    pca.fit(embeddings[control_mask])

    # Step 2: Transform all embeddings with PCA
    transformed = pca.transform(embeddings)

    # Step 3: For each batch, fit StandardScaler on controls and transform all
    corrected = np.zeros_like(transformed)
    unique_batches = np.unique(batch_ids)

    for batch in unique_batches:
        batch_mask = batch_ids == batch
        batch_controls = batch_mask & control_mask

        # If no controls in this batch, use all samples
        if not np.any(batch_controls):
            batch_controls = batch_mask

        # Fit scaler on controls from this batch
        scaler = StandardScaler()
        scaler.fit(transformed[batch_controls])

        # Transform all samples from this batch
        corrected[batch_mask] = scaler.transform(transformed[batch_mask])

    return corrected


# Extract fov as batch ID
fov_batch_ids = np.array([name.split("/")[1] for name in tracks["fov_name"]])
# Create control mask (uninfected cells)
control_mask = tracks["infection"] == 1

# Get all feature columns
feature_cols = [col for col in tracks.columns if col.startswith("feature_")]
feature_matrix = tracks[feature_cols].values

# Apply batch correction
corrected_features = batch_correction(feature_matrix, fov_batch_ids, control_mask)

# Replace original features with corrected ones
for i in range(corrected_features.shape[1]):
    tracks[f"corrected_feature_{i}"] = corrected_features[:, i]

# %% Save embeddings in multiple formats

# 1. Save as NumPy arrays
output_dir = Path(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection"
)
output_dir.mkdir(exist_ok=True, parents=True)

# Save original features
np.save(output_dir / "openphenom_original_features.npy", feature_matrix)
# Save batch-corrected features
np.save(output_dir / "openphenom_corrected_features.npy", corrected_features)

# %%
output_dir = Path(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection"
)
output_dir.mkdir(exist_ok=True, parents=True)
# Load the embeddings
feature_matrix = np.load(output_dir / "openphenom_original_features.npy")
corrected_features = np.load(output_dir / "openphenom_corrected_features.npy")
# Replace original features with corrected ones
for i in range(corrected_features.shape[1]):
    tracks[f"corrected_feature_{i}"] = corrected_features[:, i]
# %% Plot PCA and PHATE maps with annotations
print("Plotting visualizations...")
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
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/openphenom_pretrained_PCmap.png",
    dpi=300,
)

# PHATE map with annotations
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
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/openphenom_pretrained_phatemap.png",
    dpi=300,
)

# %% Compute accuracy using logistic regression
print("Computing classification accuracy...")
# Split data into training and testing sets
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

# Train on original features
print("Training on original features...")
x_train = data_train_val.filter(regex="^feature_")
y_train = data_train_val["infection"]

# Train logistic regression model
clf_original = LogisticRegression(random_state=0, max_iter=1000).fit(x_train, y_train)

# Test on original features
x_test = data_test.filter(regex="^feature_")
y_test = data_test["infection"]

# Predict infection state
y_pred_original = clf_original.predict(x_test)

# Compute accuracy
accuracy_original = np.mean(y_pred_original == y_test)
print(f"Accuracy with original OpenPhenom features: {accuracy_original:.4f}")

# Train on batch-corrected features
print("Training on batch-corrected features...")
x_train_corrected = data_train_val.filter(regex="^corrected_feature_")
y_train = data_train_val["infection"]

# Train logistic regression model
clf_corrected = LogisticRegression(random_state=0, max_iter=1000).fit(
    x_train_corrected, y_train
)

# Test on batch-corrected features
x_test_corrected = data_test.filter(regex="^corrected_feature_")
y_test = data_test["infection"]

# Predict infection state
y_pred_corrected = clf_corrected.predict(x_test_corrected)

# Compute accuracy
accuracy_corrected = np.mean(y_pred_corrected == y_test)
print(f"Accuracy with batch-corrected OpenPhenom features: {accuracy_corrected:.4f}")

# %% Plot infection state over time
print("Plotting infection state over time...")
# Plot the predicted infection state over time for /B/3 well and /B/4 well
time_points_test = np.unique(data_test["t"])

infected_true_cntrl = []
infected_true_infected = []
infected_pred_cntrl = []
infected_pred_infected = []

for time in time_points_test:
    # Ground truth for control wells
    control_at_time = data_test[
        (data_test["fov_name"].str.startswith("/B/3")) & (data_test["t"] == time)
    ]
    if len(control_at_time) > 0:
        infected_count = sum(control_at_time["infection"] == 2)
        total_count = len(control_at_time)
        infected_true_cntrl.append(infected_count * 100 / total_count)
    else:
        infected_true_cntrl.append(np.nan)

    # Ground truth for infected wells
    infected_at_time = data_test[
        (data_test["fov_name"].str.startswith("/B/4")) & (data_test["t"] == time)
    ]
    if len(infected_at_time) > 0:
        infected_count = sum(infected_at_time["infection"] == 2)
        total_count = len(infected_at_time)
        infected_true_infected.append(infected_count * 100 / total_count)
    else:
        infected_true_infected.append(np.nan)

    # Add model predictions to the test data
    data_test.loc[:, "predicted_infection"] = y_pred_corrected

    # Model predictions for control wells
    control_pred_count = sum(control_at_time["predicted_infection"] == 2)
    if len(control_at_time) > 0:
        infected_pred_cntrl.append(control_pred_count * 100 / len(control_at_time))
    else:
        infected_pred_cntrl.append(np.nan)

    # Model predictions for infected wells
    infected_pred_count = sum(infected_at_time["predicted_infection"] == 2)
    if len(infected_at_time) > 0:
        infected_pred_infected.append(infected_pred_count * 100 / len(infected_at_time))
    else:
        infected_pred_infected.append(np.nan)

# Create the time course plot
plt.figure(figsize=(10, 6))
plt.plot(
    time_points_test * 0.5 + 3,
    infected_true_cntrl,
    label="Mock (ground truth)",
    color="steelblue",
    linestyle="--",
    marker="o",
)
plt.plot(
    time_points_test * 0.5 + 3,
    infected_true_infected,
    label="Infected (ground truth)",
    color="orange",
    linestyle="--",
    marker="o",
)
plt.plot(
    time_points_test * 0.5 + 3,
    infected_pred_cntrl,
    label="Mock (predicted)",
    color="steelblue",
    marker="x",
)
plt.plot(
    time_points_test * 0.5 + 3,
    infected_pred_infected,
    label="Infected (predicted)",
    color="orange",
    marker="x",
)

plt.xlabel("Time (hours)")
plt.ylabel("% Infected cells")
plt.title("Infection Time Course - OpenPhenom Features")
plt.legend()
plt.grid(True)
plt.ylim(0, 100)
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/openphenom_pretrained_infection_over_time.png",
    dpi=300,
)

print("Analysis complete!")

# %%
