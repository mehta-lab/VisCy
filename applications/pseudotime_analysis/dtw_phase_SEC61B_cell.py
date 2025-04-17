# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from iohub import open_ome_zarr
from plotting_utils import (
    align_image_stacks,
    create_consensus_embedding,
    find_pattern_matches,
    identify_lineages,
    plot_reference_vs_full_lineages,
)

from viscy.representation.embedding_writer import read_embedding_dataset

# Create a custom logger for just this script
logger = logging.getLogger("viscy")
logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add a console handler specifically for this logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")  # Simplified format
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Make sure the logger isn't affected by parent loggers
logger.propagate = False

# Test the logger
logger.info("DTW analysis logger is configured and working!")

NAPARI = True
ANNOTATE_FOV = False

if NAPARI:
    import os

    import napari

    os.environ["DISPLAY"] = ":1"
    viewer = napari.Viewer()


# %%
input_data_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_DENV.zarr"
)


feature_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/predictions/timeAware_2chan__ntxent_192patch_70ckpt_rev7_GT.zarr"
)
# feature_path = Path(
#     "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/3-phenotyping/predictions_infection/2chan_192patch_100ckpt_timeAware_ntxent_GT.zarr"
# )
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/4-track-gt/2024_11_07_A549_SEC61_ZIKV_DENV_2_cropped.zarr"
)
embeddings_dataset = read_embedding_dataset(feature_path)
feature_df = embeddings_dataset["sample"].to_dataframe().reset_index(drop=True)

lineages = identify_lineages(feature_df)

logger.info(f"Found {len(lineages)} distinct lineages")

filtered_lineages = []
min_timepoints = 20
for fov_id, track_ids in lineages:
    # Get all rows for this lineage
    lineage_rows = feature_df[
        (feature_df["fov_name"] == fov_id) & (feature_df["track_id"].isin(track_ids))
    ]

    # Count the total number of timepoints
    total_timepoints = len(lineage_rows)

    # Only keep lineages with at least min_timepoints
    if total_timepoints >= min_timepoints:
        filtered_lineages.append((fov_id, track_ids))
logger.info(
    f"Found {len(filtered_lineages)} lineages with at least {min_timepoints} timepoints"
)
# %%
if ANNOTATE_FOV:
    # Display fov for QC
    fov = "C/2/001001"
    labels_dataset = open_ome_zarr(tracks_path / fov)
    fluor_dataset = open_ome_zarr(input_data_path / fov)
    channels_to_display = [
        "raw GFP EX488 EM525-45",
        "raw mCherry EX561 EM600-37",
    ]
    channels_indices = [
        fluor_dataset.get_channel_index(channel) for channel in channels_to_display
    ]

    viewer.add_image(
        fluor_dataset[0][:, channels_indices[0]], colormap="green", blending="additive"
    )
    viewer.add_image(
        fluor_dataset[0][:, channels_indices[1]],
        colormap="magenta",
        blending="additive",
    )
    labels_layer = viewer.add_labels(labels_dataset[0][:, 0], blending="translucent")
    labels_layer.opacity = 0.5
    labels_layer.rendering = "translucent"

    viewer.dims.ndisplay = 3

# %%
CONDITION = (
    "remodelling_w_sensor"  # remodelling_no_sensor, remodelling_w_sensor, cell_division
)

if CONDITION == "remodelling_no_sensor":
    # Get the reference lineage
    # From the B/2 SEC61B-DV
    reference_lineage_fov = "/B/2/001000"
    reference_lineage_track_id = 104
    reference_timepoints = [55, 70]  # organelle remodeling

# From the C/2 SEC61B-DV-pl40
elif CONDITION == "remodelling_w_sensor":
    # reference_lineage_fov = "/C/2/000001"
    # reference_lineage_track_id = 115
    # reference_timepoints = [47, 70] #sensor rellocalization and partial remodelling

    # reference_lineage_fov = "/C/2/000001"
    # reference_lineage_track_id = 158
    # reference_timepoints = [44, 74]  # sensor rellocalization and partial remodelling

    reference_lineage_fov = "/C/2/001000"
    reference_lineage_track_id = [129]
    reference_timepoints = [8, 70]  # sensor rellocalization and partial remodelling

    # From nuclear rellocalization
    # reference_lineage_fov = "/C/2/001001"
    # reference_lineage_track_id = [130, 131, 132]
    # reference_timepoints = [10, 80]  # sensor rellocalization and partial remodelling

    # reference_lineage_fov = "/C/2/001001"
    # reference_lineage_track_id = [160, 161]
    # reference_timepoints = [10, 80]  # sensor rellocalization and partial remodelling

    # reference_lineage_fov = "/C/2/001001"
    # reference_lineage_track_id = [126, 127]
    # reference_timepoints = [20, 70]  # sensor rellocalization and partial remodelling

elif CONDITION == "infection":
    reference_lineage_fov = "/C/2/001000"
    reference_lineage_track_id = [129]
    reference_timepoints = [8, 70]  # sensor rellocalization and partial remodelling

# Cell division
elif CONDITION == "cell_division":
    reference_lineage_fov = "/C/2/000001"
    reference_lineage_track_id = [107, 108, 109]
    reference_timepoints = [25, 70]

# Get the reference pattern
reference_pattern = None
reference_lineage = []
for fov_id, track_ids in filtered_lineages:
    if fov_id == reference_lineage_fov and all(
        track_id in track_ids for track_id in reference_lineage_track_id
    ):
        logger.info(
            f"Found reference pattern for {fov_id} {reference_lineage_track_id}"
        )
        reference_pattern = embeddings_dataset.sel(
            sample=(fov_id, reference_lineage_track_id)
        ).features.values
        reference_lineage.append(reference_pattern)
        break
if reference_pattern is None:
    raise ValueError(
        f"Reference pattern not found for {reference_lineage_fov} {reference_lineage_track_id}"
    )
reference_pattern = np.concatenate(reference_lineage)
reference_pattern = reference_pattern[reference_timepoints[0] : reference_timepoints[1]]

# %%
# Find all matches to the reference pattern
all_match_positions = find_pattern_matches(
    reference_pattern,
    filtered_lineages,
    embeddings_dataset,
    window_step_fraction=0.1,
    num_candidates=3,
    method="bernd_clifford",
    save_path=f"./SEC61B/20241107_SEC61B_{CONDITION}_matching_lineages.csv",
)

# %%
# Get the top N aligned cells
n_cells = 5
top_n_aligned_cells = all_match_positions.head(n_cells)
# %%
# Align the matches to reference
all_lineage_images, all_aligned_stacks = align_image_stacks(
    reference_pattern,
    top_n_aligned_cells,
    input_data_path,
    tracks_path,
    source_channels=[
        "Phase3D",
        "raw GFP EX488 EM525-45",
        "raw mCherry EX561 EM600-37",
    ],
    yx_patch_size=(192, 192),
    z_range=(10, 30),
    view_ref_sector_only=True,
    napari_viewer=viewer if NAPARI else None,
)

# %%
# Get an average of the aligned embeddings sections to the reference pattern

consensus_embedding = create_consensus_embedding(
    reference_pattern,
    top_n_aligned_cells,
    embeddings_dataset,
)
# saving the consensus embedding as xarray dataset
consensus_embedding_ds = xr.Dataset({"sample": (["t", "channel"], consensus_embedding)})
consensus_embedding_ds.to_zarr(
    f"./SEC61B/20241107_SEC61B_{CONDITION}_consensus_embedding.zarr", mode="w"
)
plot_reference_vs_full_lineages(
    reference_pattern,
    top_n_aligned_cells,
    embeddings_dataset,
    save_path=f"./SEC61B/20241107_SEC61B_{CONDITION}_reference_vs_full_lineages.png",
)


# %%
# Check how this compares to the average aligned embeddings
all_match_positions_wrt_consensus = find_pattern_matches(
    consensus_embedding,
    filtered_lineages,
    embeddings_dataset,
    window_step_fraction=0.25,  # Step size as fraction of reference pattern length
    num_candidates=3,
    save_path=f"./SEC61B/20241107_SEC61B_{CONDITION}_consensus_matching_lineages.csv",
)
top_n_aligned_cells_wrt_consensus = all_match_positions_wrt_consensus.head(n_cells)

plot_reference_vs_full_lineages(
    consensus_embedding,
    top_n_aligned_cells_wrt_consensus,
    embeddings_dataset,
    save_path=f"./SEC61B/20241107_SEC61B_{CONDITION}_consensus_vs_full_lineages.png",
)
# %%
all_lineage_images, all_aligned_stacks = align_image_stacks(
    reference_pattern=consensus_embedding,
    top_aligned_cells=top_n_aligned_cells_wrt_consensus,
    input_data_path=input_data_path,
    tracks_path=tracks_path,
    source_channels=["Phase3D", "raw GFP EX488 EM525-45", "raw mCherry EX561 EM600-37"],
    yx_patch_size=(192, 192),
    z_range=(10, 30),
    view_ref_sector_only=True,
    napari_viewer=viewer if NAPARI else None,
)
# %%
