# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtaidistance.dtw_ndim import warping_path
from plotting_utils import (
    align_and_average_embeddings,
    align_image_stacks,
    find_pattern_matches,
    plot_reference_aligned_average,
    plot_reference_vs_full_lineages,
    create_consensus_embedding,
)

from viscy.representation.embedding_writer import read_embedding_dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

NAPARI = True

if NAPARI:
    import os

    import napari

    os.environ["DISPLAY"] = ":1"
    viewer = napari.Viewer()


# %%
#
def identify_lineages(tracking_df: pd.DataFrame) -> list[tuple[str, list[int]]]:
    """
    Identifies all distinct lineages in the cell tracking data, following only
    one branch after each division event.

    Args:
        annotations_path: Path to the annotations CSV file

    Returns:
        A list of tuples, where each tuple contains (fov_id, [track_ids])
        representing a single branch lineage within a single FOV
    """
    # Read the CSV file

    # Process each FOV separately to handle repeated track_ids
    all_lineages = []

    # Group by FOV
    for fov_id, fov_df in tracking_df.groupby("fov_name"):
        # Create a dictionary to map tracks to their parents within this FOV
        child_to_parent = {}

        # Group by track_id and get the first row for each track to find its parent
        for track_id, track_group in fov_df.groupby("track_id"):
            first_row = track_group.iloc[0]
            parent_track_id = first_row["parent_track_id"]

            if parent_track_id != -1:
                child_to_parent[track_id] = parent_track_id

        # Find root tracks (those without parents or with parent_track_id = -1)
        all_tracks = set(fov_df["track_id"].unique())
        child_tracks = set(child_to_parent.keys())
        root_tracks = all_tracks - child_tracks

        # Build a parent-to-children mapping
        parent_to_children = {}
        for child, parent in child_to_parent.items():
            if parent not in parent_to_children:
                parent_to_children[parent] = []
            parent_to_children[parent].append(child)

        # Function to get a single branch from each parent
        # We'll choose the first child in the list (arbitrary choice)
        def get_single_branch(track_id):
            branch = [track_id]
            if track_id in parent_to_children:
                # Choose only the first child (you could implement other selection criteria)
                first_child = parent_to_children[track_id][0]
                branch.extend(get_single_branch(first_child))
            return branch

        # Build lineages starting from root tracks within this FOV
        for root_track in root_tracks:
            # Get a single branch from this root
            lineage_tracks = get_single_branch(root_track)
            all_lineages.append((fov_id, lineage_tracks))

    return all_lineages


# %%
input_data_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_ZIKV_DENV.zarr"
)
feature_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/predictions/timeAware_2chan__ntxent_192patch_70ckpt_rev7_GT.zarr"
)
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
# Get the reference lineage
# From the B/2 SEC61B-DV
# reference_lineage_fov = "/B/2/001000"
# reference_lineage_track_id = 104
# reference_timepoints = [55, 70]  # organelle remodeling

# From the C/2 SEC61B-DV-pl40

# reference_lineage_fov = "/C/2/000001"
# reference_lineage_track_id = 115
# reference_timepoints = [47, 70] #sensor rellocalization and partial remodelling

# reference_lineage_fov = "/C/2/000001"
# reference_lineage_track_id = 158
# reference_timepoints = [44, 74]  # sensor rellocalization and partial remodelling

# Cell division
reference_lineage_fov = "/C/2/000001"
reference_lineage_track_id = 108
reference_timepoints = [40, 67]


ref_total_timepoints = reference_timepoints[1] - reference_timepoints[0]

# Get the reference pattern
for fov_id, track_ids in filtered_lineages:
    if fov_id == reference_lineage_fov and reference_lineage_track_id in track_ids:
        logger.info(
            f"Found reference pattern for {fov_id} {reference_lineage_track_id}"
        )
        reference_pattern = embeddings_dataset.sel(
            sample=(fov_id, reference_lineage_track_id)
        ).features.values
        break
if reference_pattern is None:
    raise ValueError(
        f"Reference pattern not found for {reference_lineage_fov} {reference_lineage_track_id}"
    )
reference_pattern = reference_pattern[reference_timepoints[0] : reference_timepoints[1]]

# %%
# Find all matches to the reference pattern
all_match_positions = find_pattern_matches(
    reference_pattern,
    filtered_lineages,
    embeddings_dataset,
    window_step_fraction=0.25,
    num_candidates=3,
    save_path="./SEC61B/20241107_SEC61B_organelle_remodeling_matching_lineages.csv",
)

# %%
# Get the top N aligned cells
n_cells = 3
top_n_aligned_cells = all_match_positions.head(n_cells)

# %%

# Align the matches to reference
all_lineage_images, all_aligned_stacks = align_image_stacks(
    reference_pattern,
    top_n_aligned_cells,
    input_data_path,
    tracks_path,
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

# %%
# Call the plotting functions
plot_reference_aligned_average(
    reference_pattern,
    top_n_aligned_cells,
    embeddings_dataset,
    save_path="./SEC61B/reference_aligned_average_embeddings.png",
)

plot_reference_vs_full_lineages(
    reference_pattern,
    top_n_aligned_cells,
    embeddings_dataset,
    save_path="./SEC61B/reference_vs_full_lineages.png",
)


# %%
# Check how this compares to the average aligned embeddings
all_match_positions_wrt_average = find_pattern_matches(
    consensus_embedding,
    filtered_lineages,
    embeddings_dataset,
    window_step_fraction=0.25,  # Step size as fraction of reference pattern length
    num_candidates=3,
    save_path="./SEC61B/20241107_SEC61B_average_aligned_embeddings_matching_lineages.csv",
)
top_n_aligned_cells_wrt_average = all_match_positions_wrt_average.head(n_cells)
plot_reference_aligned_average(
    consensus_embedding,
    top_n_aligned_cells_wrt_average,
    embeddings_dataset,
    save_path="./SEC61B/reference_aligned_average_embeddings_wrt_average.png",
)

plot_reference_vs_full_lineages(
    consensus_embedding,
    top_n_aligned_cells_wrt_average,
    embeddings_dataset,
    save_path="./SEC61B/reference_vs_full_lineages_wrt_average.png",
)
# %%
all_lineage_images, all_aligned_stacks = align_image_stacks(
    consensus_embedding,
    top_n_aligned_cells_wrt_average,
    input_data_path,
    tracks_path,
    view_ref_sector_only=True,
    napari_viewer=viewer if NAPARI else None,
)

# %%
