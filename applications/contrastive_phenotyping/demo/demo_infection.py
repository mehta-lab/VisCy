# %%
import os
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import seaborn as sns
import timm
import torch
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Slider
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xarray import Dataset

from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import dataset_of_tracks
from viscy.representation.evaluation.clustering import (
    pairwise_distance_matrix,
    rank_nearest_neighbors,
)
from viscy.representation.evaluation.dimensionality_reduction import (
    compute_pca,
    compute_phate,
)
from viscy.transforms import NormalizeSampled, ScaleIntensityRangePercentilesd

# %%
# Set the style for the plots
DARK_MODE = False
plt.style.use("dark_background" if DARK_MODE else "default")
plt.style.use(
    "/home/eduardo.hirata/repos/viscy/applications/contrastive_phenotyping/figures/figure.mplstyle"
)
# Define different colors based on dark mode
if DARK_MODE:
    annotation_colors = {"uinfected": "blue", "infected": "red"}
    text_color = "white"
else:
    annotation_colors = {"uinfected": "blue", "infected": "red"}
    text_color = "black"
# %%

input_data_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr"
tracks_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr"
dynaclr_features_path = "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/rev6_NTXent_sensorPhase_infection/2chan_160patch_94ckpt_rev6_2.zarr"

imagenet_features_path = ""

output_dir = Path("./imagenet_vs_dynaclr/infection")
output_dir.mkdir(parents=True, exist_ok=True)


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


def add_phate_arrows(df, color):
    from matplotlib.patches import FancyArrowPatch

    for i in range(df.shape[0] - 1):
        start = df.iloc[i]
        end = df.iloc[i + 1]
        arrow = FancyArrowPatch(
            (start["PHATE1"], start["PHATE2"]),
            (end["PHATE1"], end["PHATE2"]),
            color=color,
            arrowstyle="-",
            mutation_scale=10,  # reduce the size of arrowhead by half
            lw=1,
            shrinkA=0,
            shrinkB=0,
        )
        plt.gca().add_patch(arrow)


# %%
def imagenet_prediction(
    data_path,
    tracks_path,
    source_channel,
    z_range=(0, 1),
    initial_yx_patch_size=(192, 192),
    final_yx_patch_size=(192, 192),
    path_to_save=None,
):
    # Load the ImageNet pretrained model
    model = timm.create_model("convnext_tiny", pretrained=True).eval().to("cuda")
    model.eval()

    # Setup Data Loader
    dm = TripletDataModule(
        data_path=input_data_path,
        tracks_path=tracks_path,
        source_channel=source_channel,
        batch_size=32,
        num_workers=10,
        z_range=z_range,
        initial_yx_patch_size=initial_yx_patch_size,
        final_yx_patch_size=final_yx_patch_size,
        normalizations=[
            ScaleIntensityRangePercentilesd(
                keys=["RFP"], lower=50, upper=99, b_min=0.0, b_max=1.0
            ),
        ],
    )
    dm.prepare_data()
    dm.setup("predict")
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
    pooled = torch.cat(features).mean(dim=(2, 3)).cpu().numpy()
    tracks_df = pd.concat([pd.DataFrame(idx) for idx in indices])

    _, phate_embedding = compute_phate(
        embedding_dataset=pooled,
        n_components=2,
        knn=5,
        decay=40,
        n_jobs=15,
    )

    for i, feature in enumerate(pooled.T):
        tracks_df[f"feature_{i}"] = feature
    for i, feature in enumerate(phate_embedding.T):
        tracks_df[f"PHATE{i+1}"] = feature

    if path_to_save:
        tracks_df.to_csv(path_to_save, index=False)
    return pooled, tracks_df


source_channel = ["RFP"]
z_range = (16, 21)
initial_yx_patch_size = (192, 192)
final_yx_patch_size = (192, 192)
# pooled, imagenet_pred_df = imagenet_prediction(
#     data_path=input_data_path,
#     tracks_path=tracks_path,
#     source_channel=source_channel,
#     z_range=z_range,
#     initial_yx_patch_size=initial_yx_patch_size,
#     final_yx_patch_size=final_yx_patch_size,
#     path_to_save=output_dir / "imagenet_tracks_phate.csv",
# )

# %%
imagenet_pred_df = pd.read_csv(output_dir / "imagenet_tracks_phate.csv")
print("Loading infection annotations...")
ann_root = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred"
)
ann_path = ann_root / "extracted_inf_state.csv"
annotation = pd.read_csv(ann_path)
annotation["fov_name"] = "/" + annotation["fov_name"]

# Initialize the infection column with NaN values
imagenet_pred_df["infection"] = float("nan")

# Populate infection values by matching fov_name and track_id
for index, row in annotation.iterrows():
    mask = (
        (imagenet_pred_df["fov_name"] == row["fov_name"])
        & (imagenet_pred_df["track_id"] == row["track_id"])
        & (imagenet_pred_df["t"] == row["t"])
    )
    imagenet_pred_df.loc[mask, "infection"] = row["infection_state"]


# find number of NaNs in infection column
print(
    f"Number of NaNs in infection column: {imagenet_pred_df['infection'].isna().sum()}"
)

# remove rows with infection = 0
imagenet_pred_df = imagenet_pred_df[imagenet_pred_df["infection"] != 0]
# %%
# Import utility functions
from utils import create_plotly_visualization

# Selected tracks to highlight
infected_fov = "/B/4/9"
infected_track = 42
uninfected_fov = "/A/3/9"
uninfected_track = 19


# Create an interactive Plotly visualization with time slider
fig = create_plotly_visualization(
    imagenet_pred_df,
    highlight_tracks={
        1: [("/A/3/9", 19)],  # Uninfected tracks
        2: [("/B/4/9", 42)],  # Infected tracks
    },
    df_coordinates=["PHATE1", "PHATE2"],
    time_column="t",
    category_column="infection",
    category_labels={1: "Uninfected", 2: "Infected"},
    category_colors={1: "cornflowerblue", 2: "salmon"},
    highlight_colors={1: "blue", 2: "red"},
    title_prefix="ImageNet PHATE Embedding",
    plot_size_xy=(500, 500),
)
fig.show()
# %%
from skimage.exposure import rescale_intensity

# Cache the images
fov_name_mock = "/B/3/9"
track_id_mock = [100]
fov_name_inf = "/B/4/9"
track_id_inf = [44]
z_range = (24, 29)
yx_patch_size = (128, 128)
channels_to_display = ["Phase3D", "RFP"]

fov_name_mock_list = [fov_name_mock] * len(track_id_mock)
fov_name_inf_list = [fov_name_inf] * len(track_id_inf)

conditions_to_compare = {
    "uinfected": {
        "fov_name_list": fov_name_mock_list,
        "track_id_list": track_id_mock,
    },
    "infected": {
        "fov_name_list": fov_name_inf_list,
        "track_id_list": track_id_inf,
    },
}
image_cache = {}
for condition, condition_data in conditions_to_compare.items():
    dm = TripletDataModule(
        data_path=input_data_path,
        tracks_path=tracks_path,
        source_channel=channels_to_display,
        z_range=z_range,
        initial_yx_patch_size=yx_patch_size,
        final_yx_patch_size=yx_patch_size,
        include_fov_names=condition_data["fov_name_list"]
        * len(condition_data["track_id_list"]),
        include_track_ids=condition_data["track_id_list"],
        predict_cells=True,
        batch_size=1,
    )
    dm.setup("predict")

    # Cache the condition
    condition_key = f"{condition}_cache"
    image_cache[condition_key] = {
        "fov_name": None,
        "track_id": None,
        "images_by_timepoint": {},
    }

    for i, patch in enumerate(dm.predict_dataloader()):
        fov_name = patch["index"]["fov_name"][0]
        track_id = patch["index"]["track_id"][0]
        images = patch["anchor"].numpy()[0]
        t = int(patch["index"]["t"][0])

        # Store metadata if first time
        if image_cache[condition_key]["fov_name"] is None:
            image_cache[condition_key]["fov_name"] = fov_name
            image_cache[condition_key]["track_id"] = track_id

        z_idx = images.shape[1] // 2
        C, Z, Y, X = images.shape
        # CYX
        image_out = np.zeros((C, 1, Y, X), dtype=np.float32)
        for c_idx, channel in enumerate(channels_to_display):
            if channel in ["Phase3D", "DIC", "BF"]:
                image_out[c_idx] = images[c_idx, z_idx]
                # Normalize 0-mean, unit variance
                image_out[c_idx] = (
                    image_out[c_idx] - image_out[c_idx].mean()
                ) / image_out[c_idx].std()
                # rescale intensity
                image_out[c_idx] = rescale_intensity(image_out[c_idx], out_range=(0, 1))
            else:
                # For fluoresnce max projection and then normalize
                image_out[c_idx] = np.max(images[c_idx], axis=0)
                # Percentile normalization
                lower, upper = np.percentile(image_out[c_idx], (50, 99))
                image_out[c_idx] = (image_out[c_idx] - lower) / (upper - lower)
                # Rescale intensity
                image_out[c_idx] = rescale_intensity(image_out[c_idx], out_range=(0, 1))

        # Store by timepoint
        image_cache[condition_key]["images_by_timepoint"][t] = image_out

    print(
        f"Cached {condition_key} with {len(image_cache[condition_key]['images_by_timepoint'])} timepoints"
    )

# %%
# Plotly visualization of the Phase and RFP images
from utils import create_image_visualization

# Create the visualization using our utility function
fig = create_image_visualization(
    image_cache=image_cache,
    subplot_titles=[
        "Uinfected Phase",
        "Uinfected Viral Sensor",
        "Infected Phase",
        "Infected Viral Sensor",
    ],
    condition_keys=["uinfected_cache", "infected_cache"],
    channel_colormaps=["gray", "magma"],
    plot_size_xy=(800, 800),
    # horizontal_spacing=0.05,
    # vertical_spacing=0.1,
)

# Show the figure
fig.show()

# %%
# Load the DyaCLR embeddings
dynaclr_embeddings = read_embedding_dataset(dynaclr_features_path)
dynaclr_features = dynaclr_embeddings["features"]
dynaclr_features_df = dynaclr_features["sample"].to_dataframe().reset_index(drop=True)
# Compute PHATE embedding
_, phate_embedding = compute_phate(
    embedding_dataset=dynaclr_features,
    n_components=2,
    knn=5,
    decay=40,
    n_jobs=15,
)
for i, feature in enumerate(phate_embedding.T):
    dynaclr_features_df[f"PHATE{i+1}"] = feature
# %%
# Initialize the infection column with NaN values
dynaclr_features_df["infection"] = float("nan")

# Populate infection values by matching fov_name and track_id
for index, row in annotation.iterrows():
    mask = (
        (dynaclr_features_df["fov_name"] == row["fov_name"])
        & (dynaclr_features_df["track_id"] == row["track_id"])
        & (dynaclr_features_df["t"] == row["t"])
    )
    dynaclr_features_df.loc[mask, "infection"] = row["infection_state"]


# find number of NaNs in infection column
print(
    f"Number of NaNs in infection column: {dynaclr_features_df['infection'].isna().sum()}"
)

# remove rows with infection = 0
dynaclr_features_df = dynaclr_features_df[dynaclr_features_df["infection"] != 0]
# %%
# Create an interactive Plotly visualization with time slider
fig = create_plotly_visualization(
    dynaclr_features_df,
    highlight_tracks={
        1: [("/A/3/9", 19)],  # Uninfected tracks
        2: [("/B/4/9", 42)],  # Infected tracks
    },
    df_coordinates=["PHATE1", "PHATE2"],
    time_column="t",
    category_column="infection",
    category_labels={1: "Uninfected", 2: "Infected"},
    category_colors={1: "cornflowerblue", 2: "salmon"},
    highlight_colors={1: "blue", 2: "red"},
    title_prefix="DynaCLR PHATE Embedding",
    plot_size_xy=(500, 500),
)
fig.show()


# %%
create_combined_visualization(
    image_cache,
    imagenet_pred_df,
    dynaclr_features_df,
    highlight_tracks={
        1: [("/A/3/9", 19)],  # Uninfected tracks
        2: [("/B/4/9", 42)],  # Infected tracks
    },
    subplot_titles=[
        "Uinfected Phase",
        "Uinfected Viral Sensor",
        "Infected Phase",
        "Infected Viral Sensor",
    ],
    condition_keys=["uinfected_cache", "infected_cache"],
    channel_colormaps=["gray", "magma"],
    category_colors={1: "cornflowerblue", 2: "salmon"},
    highlight_colors={1: "blue", 2: "red"},
    category_labels={1: "Uninfected", 2: "Infected"},
    plot_size_xy=(1800, 600),
)

# %%
