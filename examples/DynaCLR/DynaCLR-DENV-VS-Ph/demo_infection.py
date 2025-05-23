# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python (dynaclr)
#     language: python
#     name: dynaclr
# ---

# %% [markdown]
# # Demo: Comparing DynaCLR vs ImageNet Embeddings for Cell Infection Analysis
#
# This tutorial demonstrates how to:
# 1. Use ImageNet pre-trained features for analyzing cell infection
# 2. Compare with DynaCLR learned features
# 3. Visualize the differences between approaches

# %% [markdown]
# ## Setup and Imports

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.exposure import rescale_intensity

from utils import (
    create_combined_visualization,
)
from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import read_embedding_dataset

# %% [markdown]
# ## Set Data Paths
#
# The data, tracks and annotations can be downloaded from [here](https://drive.google.com/drive/folders/1-_0000000000000000000000000000000000000000)
# You can load the precomputed features from [here](https://drive.google.com/drive/folders/1-_0000000000000000000000000000000000000000)
#
# ## Note:
#
# Alternatively, you can run the CLI to compute the features yourself following the instructions in the [README.md](https://github.com/viscy-ai/viscy/tree/main/examples/DynaCLR/smooth_embeddings)

# %%
# TODO: Update the paths to the downloaded data
# Point to the *.zarr files
input_data_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr"  # Replace with path to registered_test.zarr
tracks_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr"  # Replace with path to  track_test.zarr
ann_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred/extracted_inf_state.csv"  # Replace with path to extracted_inf_state.csv
)

# TODO: Update the path to the DynaCL and ImageNet features
dynaclr_features_path = "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/rev6_NTXent_sensorPhase_infection/2chan_160patch_94ckpt_rev6_2_phate.zarr"
imagenet_features_path = "/home/eduardo.hirata/repos/viscy/applications/benchmarking/DynaCLR/ImageNet/20240204_A549_DENV_ZIKV_sensor_only_imagenet.zarr"

# %% [markdown]
# ## Load the embeddings and annotations
# Load the embeddings you downloaded and append the human annotations to the dataframe

# %%
# Load the embeddings
dynaclr_embeddings = read_embedding_dataset(dynaclr_features_path)
imagenet_embeddings = read_embedding_dataset(imagenet_features_path)

dynaclr_features_df = dynaclr_embeddings["sample"].to_dataframe().reset_index(drop=True)
imagenet_features_df = (
    imagenet_embeddings["sample"].to_dataframe().reset_index(drop=True)
)

# Load the annotations and create a dataframe with the infection state
annotation = pd.read_csv(ann_path)
annotation["fov_name"] = "/" + annotation["fov_name"]

imagenet_features_df["infection"] = float("nan")

for index, row in annotation.iterrows():
    mask = (
        (imagenet_features_df["fov_name"] == row["fov_name"])
        & (imagenet_features_df["track_id"] == row["track_id"])
        & (imagenet_features_df["t"] == row["t"])
    )
    imagenet_features_df.loc[mask, "infection"] = row["infection_state"]
    mask = (
        (dynaclr_features_df["fov_name"] == row["fov_name"])
        & (dynaclr_features_df["track_id"] == row["track_id"])
        & (dynaclr_features_df["t"] == row["t"])
    )
    dynaclr_features_df.loc[mask, "infection"] = row["infection_state"]

# Filter out rows with infection state 0
imagenet_features_df = imagenet_features_df[imagenet_features_df["infection"] != 0]
dynaclr_features_df = dynaclr_features_df[dynaclr_features_df["infection"] != 0]

# %% [markdown]
# ## Choosing a representative track for visualization

# %%
# NOTE: We have chosen these tracks to be representative of the data. Feel free to open the dataset and select other tracks
fov_name_mock = "/A/3/9"
track_id_mock = [19]
fov_name_inf = "/B/4/9"
track_id_inf = [42]

# Default parameters for the test dataset
z_range = (0, 1)
z_range = (24, 29)
yx_patch_size = (160, 160)

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

print("Caching sample images...")
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

# %%
print("Creating Cell Images and PHATE Embeddings Visualization...")
create_combined_visualization(
    image_cache,
    imagenet_features_df,
    dynaclr_features_df,
    highlight_tracks={
        1: [(fov_name_mock, track_id_mock[0])],  # Uninfected tracks
        2: [(fov_name_inf, track_id_inf[0])],  # Infected tracks
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
# Time-aware sampling improved temporal continutiy and dynamic range of embeddings.
# These improvements can be seen in the PHATE projections of DynaCLR.
# The embeddings show smoother and higher dynamic range.
#
