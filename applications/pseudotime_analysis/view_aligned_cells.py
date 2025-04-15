# %%
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import seaborn as sns
from iohub import open_ome_zarr

from utils import filter_lineages_by_timepoints, identify_lineages, load_annotation
from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import read_embedding_dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# %%
os.environ["DISPLAY"] = ":1"
viewer = napari.Viewer()

# %%
dataset_path = Path(
    "/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets/float_phase_ome_zarr_output_test.zarr"
)
tracks_path = Path(
    "/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets/track_phase_ome_zarr_output_test.zarr"
)
test_data_embedding_path = Path(
    "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_7mins.zarr"
)
annotation_path = "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/fixed_test_annotations.csv"
annotation_df = pd.read_csv(annotation_path)
annotation_df["fov_name"] = "/" + annotation_df["fov ID"]

alignment_results_path = Path("./ALFI_test_dataset_alignment_results.csv")

alignment_results_df = pd.read_csv(alignment_results_path).reset_index(drop=True)
test_data_timeaware_embeddings = read_embedding_dataset(test_data_embedding_path)

features_df = (
    test_data_timeaware_embeddings["sample"].to_dataframe().reset_index(drop=True)
)

lineages = identify_lineages(annotation_path)
logger.info(f"Found {len(lineages)} distinct lineages")

# Filter lineages with fewer than 10 timepoints
filtered_lineages = filter_lineages_by_timepoints(
    lineages, annotation_path, min_timepoints=25
)
# %%
# We have the alignments results, get the top 10 cells with the highest alignment score
alignment_results_df = alignment_results_df.sort_values(
    by="iou_match_rate", ascending=False
)
top_n_cells = alignment_results_df.head(10)

# Print the ioU match  for these top 10 per lineage_label
for lineage_label, lineage_df in top_n_cells.groupby("lineage_label"):
    print(f"Lineage {lineage_label}: {lineage_df['iou_match_rate'].mean()}")

# %%

# NOTE Hardcoded chosen visually to get the reference iamges
reference_embeddings = None
reference_tracks = [14, 15]
reference_fov = "/0/2/0"

# Find first ocurrence of division=1 for FOV '0/2/0' track_id=14
division_event = annotation_df[
    (annotation_df["fov_name"] == "/0/2/0") & (annotation_df["track_id"] == 14)
]
division_event = division_event[division_event["division"] == 1]
division_event = division_event.iloc[0]
division_tidx = division_event["t"]

data_module = TripletDataModule(
    data_path=dataset_path,
    tracks_path=tracks_path,
    include_fov_names=[reference_fov] * len(reference_tracks),
    include_track_ids=reference_tracks,
    source_channel=["DIC"],
    z_range=(0, 1),
    initial_yx_patch_size=(256, 256),
    final_yx_patch_size=(256, 256),
    batch_size=1,
    num_workers=12,
    normalizations=None,
    predict_cells=True,
)
data_module.setup("predict")

reference_stack = []
reference_timepoints = []
for batch in data_module.predict_dataloader():
    image = batch["anchor"].numpy()
    indices = batch["index"]
    track_id = indices["track_id"].tolist()
    t = indices["t"].tolist()
    reference_stack.append(image[0])
    reference_timepoints.append(t[0])

reference_stack = np.stack(reference_stack, axis=0)
reference_timepoints = np.array(reference_timepoints)
viewer.add_image(reference_stack, name="Reference lineage")

# %%
all_aligned_stacks = []
all_aligned_stacks.append(reference_stack)

# Process each lineage and align it to the reference
for lineage_label, lineage_df in top_n_cells.groupby("lineage_label"):
    fov_id = lineage_df["fov_id"].iloc[0]
    track_ids_list = lineage_df["track_ids"].iloc[0]
    track_ids_list = eval(lineage_df["track_ids"].to_list()[0])

    logger.info(f"Aligning lineage {lineage_label} to reference")
    data_module = TripletDataModule(
        data_path=dataset_path,
        tracks_path=tracks_path,
        include_fov_names=[fov_id] * len(track_ids_list),
        include_track_ids=track_ids_list,
        source_channel=["DIC"],
        z_range=(0, 1),
        initial_yx_patch_size=(256, 256),
        final_yx_patch_size=(256, 256),
        batch_size=1,
        num_workers=12,
        normalizations=None,
        predict_cells=True,
    )
    data_module.setup("predict")

    # Get the alignment results (warping path) for this lineage
    alignment_results = alignment_results_df[
        alignment_results_df["lineage_label"] == lineage_label
    ]["warping_path"].tolist()

    if not alignment_results:
        logger.warning(f"No alignment results found for lineage {lineage_label}")
        continue

    warping_path = eval(alignment_results[0])
    warping_path = np.array(warping_path)

    # # Find the division timepoint in the warping path
    # division_indices = np.where(warping_path[:, 0] == division_tidx)[0]
    # if len(division_indices) == 0:
    #     logger.warning(
    #         f"Division timepoint not found in warping path for lineage {lineage_label}"
    #     )
    #     continue

    # warp_tidx_start = division_indices[0]

    # Collect images from the lineage
    lineage_images = []
    lineage_timepoints = []

    for batch in data_module.predict_dataloader():
        image = batch["anchor"].numpy()[0]  # Extract single image
        t = batch["index"]["t"].item()  # Get timepoint
        lineage_images.append(image)
        lineage_timepoints.append(t)

    lineage_images = np.array(lineage_images)
    lineage_timepoints = np.array(lineage_timepoints)

    # Create an aligned stack based on the warping path
    aligned_stack = np.zeros_like(reference_stack)

    # Map each reference timepoint to the corresponding lineage timepoint using the warping path
    for i, ref_t in enumerate(reference_timepoints):
        # Find this reference timepoint in the warping path
        matches = np.where(warping_path[:, 0] == ref_t)[0]
        if len(matches) > 0:
            # Get the corresponding lineage timepoint
            match_idx = matches[0]  # Use the first match if multiple exist
            lineage_t = warping_path[match_idx, 1]

            # Find this timepoint in our lineage data
            lineage_idx = np.where(lineage_timepoints == lineage_t)[0]
            if len(lineage_idx) > 0:
                # Copy the image to the aligned stack
                aligned_stack[i] = lineage_images[lineage_idx[0]]

    all_aligned_stacks.append(aligned_stack)
    viewer.add_image(aligned_stack, name=f"Lineage {lineage_label}")

    # %%
    #
    alignment_results = alignment_results_df[
        alignment_results_df["lineage_label"] == lineage_label
    ]["warping_path"].tolist()

    if not alignment_results:
        logger.warning(f"No alignment results found for lineage {lineage_label}")
        continue

    warping_path = eval(alignment_results[0])
    warping_path = np.array(warping_path)
    # Find the division timepoint in the warping path
    division_indices = np.where(warping_path[:, 0] == division_tidx)[0]
    if len(division_indices) == 0:
        logger.warning(
            f"Division timepoint not found in warping path for lineage {lineage_label}"
        )
        continue

    warp_tidx_start = division_indices[0]
    print(warp_tidx_start)

# %%
# Let's compare with the more robust alignment method from plotting_utils.py
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from plotting_utils import align_image_stacks

# Prepare the reference pattern as the embeddings for the reference tracks
reference_embeddings = []
for track_id in reference_tracks:
    track_emb = test_data_timeaware_embeddings.sel(
        sample=(reference_fov, track_id)
    ).features.values
    reference_embeddings.append(track_emb)
reference_pattern = np.concatenate(reference_embeddings, axis=0)

# Create a DataFrame in the format expected by align_image_stacks
aligned_cells_df = []
for _, row in top_n_cells.iterrows():
    fov_id = row["fov_id"]
    track_ids_list = (
        eval(row["track_ids"])
        if isinstance(row["track_ids"], str)
        else row["track_ids"]
    )

    # Get warping path in the correct format (list of tuples)
    warping_path = (
        eval(row["warping_path"])
        if isinstance(row["warping_path"], str)
        else row["warping_path"]
    )
    warping_path = [
        (int(ref_idx), int(query_idx)) for ref_idx, query_idx in warping_path
    ]

    aligned_cells_df.append(
        {
            "fov_name": fov_id,
            "track_ids": track_ids_list,
            "warp_path": warping_path,
            "start_timepoint": 0,  # We'll align from the beginning
            "end_timepoint": len(reference_pattern),
            "distance": row.get("distance", 0.0),
        }
    )

aligned_cells_df = pd.DataFrame(aligned_cells_df)

# Use the align_image_stacks function
lineage_images, robust_aligned_stacks = align_image_stacks(
    reference_pattern=reference_pattern,
    top_aligned_cells=aligned_cells_df,
    input_data_path=dataset_path,
    tracks_path=tracks_path,
    source_channels=["DIC"],
    yx_patch_size=(256, 256),
    z_range=(0, 1),
    view_ref_sector_only=True,
    napari_viewer=None,  # We'll add to the viewer manually
)

# Add the robustly aligned stacks to the viewer
for i, aligned_stack in enumerate(robust_aligned_stacks):
    lineage_info = aligned_cells_df.iloc[i]
    fov_name = lineage_info["fov_name"]
    track_id = lineage_info["track_ids"][0] if lineage_info["track_ids"] else "unknown"
    viewer.add_image(aligned_stack, name=f"Robust_{fov_name}_track_{track_id}")
