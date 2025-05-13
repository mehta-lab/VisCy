#!/usr/bin/env python3
# Demo: Comparing ImageNet vs DynaCLR for Cell Infection Analysis

# %% [markdown]
# # Cell Infection Analysis: ImageNet vs DynaCLR
#
# This tutorial demonstrates how to:
# 1. Use ImageNet pre-trained features for analyzing cell infection
# 2. Compare with DynaCLR learned features
# 3. Visualize the differences between approaches

# %% [markdown]
# ## Setup and Imports

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from skimage.exposure import rescale_intensity
from tqdm import tqdm

from utils import create_combined_visualization, create_plotly_visualization
from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.dimensionality_reduction import compute_phate
from viscy.transforms import ScaleIntensityRangePercentilesd

# %% [markdown]
# ## Set Data Paths

# %%
input_data_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr"
tracks_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr"
dynaclr_features_path = "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/rev6_NTXent_sensorPhase_infection/2chan_160patch_94ckpt_rev6_2.zarr"

output_dir = Path("./imagenet_vs_dynaclr/infection")
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Step 1: Compute ImageNet Features
#
# We'll use a pre-trained ConvNext model to extract features from our cell images


def imagenet_prediction(
    data_path,
    tracks_path,
    source_channel,
    z_range=(0, 1),
    initial_yx_patch_size=(192, 192),
    final_yx_patch_size=(192, 192),
    path_to_save=None,
):
    model = timm.create_model("convnext_tiny", pretrained=True).eval().to("cuda")
    model.eval()

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
            rfp = batch["anchor"][:, 0]
            rfp = rfp.max(dim=1)[0]

            rgb_image = torch.stack(
                [
                    rfp,
                    rfp,
                    rfp,
                ],
                dim=1,
            ).to("cuda")

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


# %% [markdown]
# ### Run ImageNet Feature Extraction

# %%
source_channel = ["RFP"]
z_range = (24, 29)
patch_size = (128, 128)

print("Computing ImageNet features...")
pooled, imagenet_pred_df = imagenet_prediction(
    data_path=input_data_path,
    tracks_path=tracks_path,
    source_channel=source_channel,
    z_range=z_range,
    initial_yx_patch_size=patch_size,
    final_yx_patch_size=patch_size,
    path_to_save=output_dir / "imagenet_tracks_phate.csv",
)
print("ImageNet features saved to:", output_dir / "imagenet_tracks_phate.csv")

# %% [markdown]
# ## Step 2: Load Infection Annotations
#
# Now we'll load the infection state annotations and merge them with our features

# %%
imagenet_pred_df = pd.read_csv(output_dir / "imagenet_tracks_phate.csv")
print("Loading infection annotations...")
ann_root = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred"
)
ann_path = ann_root / "extracted_inf_state.csv"
annotation = pd.read_csv(ann_path)
annotation["fov_name"] = "/" + annotation["fov_name"]

imagenet_pred_df["infection"] = float("nan")

for index, row in annotation.iterrows():
    mask = (
        (imagenet_pred_df["fov_name"] == row["fov_name"])
        & (imagenet_pred_df["track_id"] == row["track_id"])
        & (imagenet_pred_df["t"] == row["t"])
    )
    imagenet_pred_df.loc[mask, "infection"] = row["infection_state"]

print(
    f"Number of NaNs in infection column: {imagenet_pred_df['infection'].isna().sum()}"
)
imagenet_pred_df = imagenet_pred_df[imagenet_pred_df["infection"] != 0]
print(f"Number of rows after filtering: {len(imagenet_pred_df)}")

# %% [markdown]
# ## Step 3: Visualize ImageNet Features
#
# Create an interactive visualization of the ImageNet embeddings with time slider

# %%
infected_fov = "/B/4/9"
infected_track = 42
uninfected_fov = "/A/3/9"
uninfected_track = 19

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

# %% [markdown]
# ## Step 4: Cache Sample Images for Visualization
#
# We'll cache representative images of infected and uninfected cells

# %%
print("Caching sample images...")
fov_name_mock = "/B/3/9"
track_id_mock = [100]
fov_name_inf = "/B/4/9"
track_id_inf = [44]
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

        if image_cache[condition_key]["fov_name"] is None:
            image_cache[condition_key]["fov_name"] = fov_name
            image_cache[condition_key]["track_id"] = track_id

        z_idx = images.shape[1] // 2
        C, Z, Y, X = images.shape
        image_out = np.zeros((C, 1, Y, X), dtype=np.float32)
        for c_idx, channel in enumerate(channels_to_display):
            if channel in ["Phase3D", "DIC", "BF"]:
                image_out[c_idx] = images[c_idx, z_idx]
                image_out[c_idx] = (
                    image_out[c_idx] - image_out[c_idx].mean()
                ) / image_out[c_idx].std()
                image_out[c_idx] = rescale_intensity(image_out[c_idx], out_range=(0, 1))
            else:
                image_out[c_idx] = np.max(images[c_idx], axis=0)
                lower, upper = np.percentile(image_out[c_idx], (50, 99))
                image_out[c_idx] = (image_out[c_idx] - lower) / (upper - lower)
                image_out[c_idx] = rescale_intensity(image_out[c_idx], out_range=(0, 1))

        image_cache[condition_key]["images_by_timepoint"][t] = image_out

    print(
        f"Cached {condition_key} with {len(image_cache[condition_key]['images_by_timepoint'])} timepoints"
    )

# %% [markdown]
# ## Step 5: Load and Process DynaCLR Features
#
# Now we'll load the features from our specialized DynaCLR model

# %%
print("Loading DynaCLR features...")
dynaclr_embeddings = read_embedding_dataset(dynaclr_features_path)
dynaclr_features = dynaclr_embeddings["features"]
dynaclr_features_df = dynaclr_features["sample"].to_dataframe().reset_index(drop=True)

print("Computing PHATE embedding for DynaCLR features...")
_, phate_embedding = compute_phate(
    embedding_dataset=dynaclr_features,
    n_components=2,
    knn=5,
    decay=40,
    n_jobs=15,
)
for i, feature in enumerate(phate_embedding.T):
    dynaclr_features_df[f"PHATE{i+1}"] = feature

# %% [markdown]
# ### Add Infection Annotations to DynaCLR Features

# %%
dynaclr_features_df["infection"] = float("nan")

for index, row in annotation.iterrows():
    mask = (
        (dynaclr_features_df["fov_name"] == row["fov_name"])
        & (dynaclr_features_df["track_id"] == row["track_id"])
        & (dynaclr_features_df["t"] == row["t"])
    )
    dynaclr_features_df.loc[mask, "infection"] = row["infection_state"]

print(
    f"Number of NaNs in infection column: {dynaclr_features_df['infection'].isna().sum()}"
)
dynaclr_features_df = dynaclr_features_df[dynaclr_features_df["infection"] != 0]
print(f"Number of rows after filtering: {len(dynaclr_features_df)}")

# %% [markdown]
# ## Step 6: Create Combined Visualization
#
# Finally, we'll create a combined visualization comparing ImageNet and DynaCLR approaches

# %%
print("Creating combined visualization...")
create_combined_visualization(
    image_cache,
    imagenet_pred_df,
    dynaclr_features_df,
    highlight_tracks={
        1: [("/A/3/9", 19)],  # Uninfected tracks
        2: [("/B/4/9", 42)],  # Infected tracks
    },
    subplot_titles=[
        "Uninfected Phase",
        "Uninfected Viral Sensor",
        "Infected Phase",
        "Infected Viral Sensor",
    ],
    condition_keys=["uinfected_cache", "infected_cache"],
    channel_colormaps=["gray", "magma"],
    category_colors={1: "cornflowerblue", 2: "salmon"},
    highlight_colors={1: "blue", 2: "red"},
    category_labels={1: "Uninfected", 2: "Infected"},
    plot_size_xy=(1200, 600),
    title_location="top",
)

# %% [markdown]
# ## Conclusion
#
# This tutorial demonstrated the comparison between general purpose ImageNet features and
# specialized DynaCLR features for analyzing cell infection states. The visualizations
# show how each approach groups infected vs. uninfected cells in feature space.

# %%
