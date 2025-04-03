# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtaidistance.dtw_ndim import warping_path
from iohub.ngff import open_ome_zarr
from scipy.spatial.distance import cdist
from tqdm import tqdm

from viscy.data.triplet import TripletDataModule
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


# DTW matching
def find_best_match_dtw(
    lineage: np.ndarray,
    reference_pattern: np.ndarray,
    num_candidates: int = 5,
    window_step: int = 5,
    max_distance: float = float("inf"),
):
    """Find the best matches in a lineage using DTW.

    Args:
        lineage: The lineage to search (t,embeddings).
        reference_pattern: The pattern to search for (t,embeddings).
        num_candidates: The number of candidates to return. Defaults to 5.
        window_step: The step size for the window. Defaults to 5.
        max_distance: Maximum distance threshold to consider a match. Defaults to infinity.
    """
    dtw_results = []
    n_windows = len(lineage) - len(reference_pattern) + 1

    if n_windows <= 0:
        return []

    for i in range(0, n_windows, window_step):
        window = lineage[i : i + len(reference_pattern)]
        path, dist = warping_path(
            window,
            reference_pattern,
            include_distance=True,
        )
        if dist <= max_distance:
            dtw_results.append((i, path, dist))

    sorted_dtw_results = sorted(dtw_results, key=lambda x: x[2])[:num_candidates]

    return sorted_dtw_results


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

reference_lineage_fov = "/C/2/000001"
reference_lineage_track_id = 158
reference_timepoints = [44, 74]  # sensor rellocalization and partial remodelling
# Cell division
# reference_lineage_fov = "/C/2/000001"
# reference_lineage_track_id = 108
# reference_timepoints = [47, 67]

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
all_match_positions = {
    "fov_name": [],
    "track_ids": [],
    "distance": [],
    "warp_path": [],
    "start_timepoint": [],
    "end_timepoint": [],
}
for i, (fov_name, track_ids) in tqdm(
    enumerate(filtered_lineages), total=len(filtered_lineages)
):
    lineages = []
    for track_id in track_ids:
        lineage = embeddings_dataset.sel(sample=(fov_name, track_id)).features.values
        lineages.append(lineage)
    lineages = np.concatenate(lineages, axis=0)
    best_matches = find_best_match_dtw(
        lineages,
        reference_pattern=reference_pattern,
        num_candidates=3,
        window_step=ref_total_timepoints // 4,
    )
    if len(best_matches) > 0:
        best_pos, best_path, best_dist = best_matches[0]
        all_match_positions["fov_name"].append(fov_name)
        all_match_positions["track_ids"].append(track_ids)
        all_match_positions["distance"].append(best_dist)
        all_match_positions["warp_path"].append(best_path)
        all_match_positions["start_timepoint"].append(best_pos)
        all_match_positions["end_timepoint"].append(best_pos + len(reference_pattern))
    else:
        logger.debug(f"No best matches found for {fov_name} {track_ids}")
        all_match_positions["fov_name"].append(fov_name)
        all_match_positions["track_ids"].append(track_ids)
        all_match_positions["distance"].append(None)
        all_match_positions["warp_path"].append(None)
        all_match_positions["start_timepoint"].append(None)
        all_match_positions["end_timepoint"].append(None)

all_match_positions = pd.DataFrame(all_match_positions)
all_match_positions = all_match_positions.dropna()
all_match_positions.to_csv(
    "./SEC61B/20241107_SEC61B_organelle_remodeling_matching_lineages.csv", index=False
)
# %%
# Inspect top 5 aligned cells in napari
# Get the sorted top 5 aligned cells
all_match_positions = all_match_positions.sort_values(by="distance", ascending=True)
# TODO: remove the hardcoded number 3
top_5_aligned_cells = all_match_positions.head(3)

# Get the embeddings for the top 5 aligned cells
top_5_aligned_cells_embeddings = []
for _, row in top_5_aligned_cells.iterrows():
    fov_name = row["fov_name"]
    track_ids = row["track_ids"]
    embeddings = embeddings_dataset.sel(sample=(fov_name, track_ids)).features.values
    top_5_aligned_cells_embeddings.append(embeddings)

# %%
# Align the matches
# Average the matches
# Align cells to reference
# Plot the results
VIEW_REF_SECTOR_ONLY = True
all_lineage_images = []
all_aligned_stacks = []
for idx, row in top_5_aligned_cells.iterrows():
    fov_name = row["fov_name"]
    track_ids = row["track_ids"]
    warp_path = row["warp_path"]
    start_time = row["start_timepoint"]
    end_time = row["end_timepoint"]

    # Initialize the data module
    data_module = TripletDataModule(
        data_path=input_data_path,
        tracks_path=tracks_path,
        source_channel=["Phase3D", "raw GFP EX488 EM525-45"],
        z_range=[10, 30],
        initial_yx_patch_size=(192, 192),
        final_yx_patch_size=(192, 192),
        batch_size=1,
        num_workers=12,
        predict_cells=True,
        include_fov_names=[fov_name],
        include_track_ids=track_ids,
    )
    data_module.setup("predict")

    # Get the images and timepoints for the lineage
    lineage_images = []
    for batch in data_module.predict_dataloader():
        image = batch["anchor"].numpy()
        t = batch["index"]["t"].item()
        lineage_images.append(image)

    lineage_images = np.array(lineage_images)
    all_lineage_images.append(lineage_images)

    # Create an aligned stack based on the warping path
    # Here we want to show the points where the reference pattern matches the lineage
    if VIEW_REF_SECTOR_ONLY:
        aligned_stack = np.zeros(
            (len(reference_pattern),) + lineage_images.shape[1:],
            dtype=lineage_images.dtype,
        )

        # Create a mapping from reference indices to lineage indices
        ref_to_lineage = {}
        for ref_idx, query_idx in warp_path:
            lineage_idx = int(start_time + query_idx)
            if 0 <= lineage_idx < len(lineage_images):
                ref_to_lineage[ref_idx] = lineage_idx

        # Fill the aligned stack, ensuring every reference index is covered
        for ref_idx in range(len(reference_pattern)):
            if ref_idx in ref_to_lineage:
                aligned_stack[ref_idx] = lineage_images[ref_to_lineage[ref_idx]]
            else:
                # Handle missing indices - could interpolate or leave as zeros
                # For now, find the closest available reference index
                if ref_to_lineage:
                    closest_ref_idx = min(
                        ref_to_lineage.keys(), key=lambda x: abs(x - ref_idx)
                    )
                    aligned_stack[ref_idx] = lineage_images[
                        ref_to_lineage[closest_ref_idx]
                    ]

        all_aligned_stacks.append(aligned_stack)
        if NAPARI:
            viewer.add_image(aligned_stack, name=f"Aligned {fov_name}")
    else:
        # View the whole lineage shifted by the start time
        aligned_stack = lineage_images[start_time:]
        all_aligned_stacks.append(aligned_stack)
        if NAPARI:
            viewer.add_image(
                aligned_stack, name=f"Aligned_{fov_name}_track_{track_ids[0]}"
            )

# %%
# Get an average of the aligned embeddings sections to the reference pattern
all_aligned_embeddings = []
for idx, row in top_5_aligned_cells.iloc[1:].iterrows():
    fov_name = row["fov_name"]
    track_ids = row["track_ids"]
    warp_path = row["warp_path"]
    start_time = int(row["start_timepoint"])
    end_time = int(row["end_timepoint"])

    # Ensuring the references are the same shape
    embeddings = embeddings_dataset.sel(sample=(fov_name, track_ids)).features.values
    embedding_segment = embeddings[start_time:end_time]

    # Align the embedding segment to the reference pattern
    aligned_segment = np.zeros_like(reference_pattern)
    for ref_idx, query_idx in warp_path:
        aligned_segment[ref_idx] = embedding_segment[query_idx]

    all_aligned_embeddings.append(aligned_segment)

all_aligned_embeddings = np.array(all_aligned_embeddings)
# Average the aligned embeddings 768D x T embeddings
# NOTE: we can also do the median
average_aligned_embeddings = np.mean(all_aligned_embeddings, axis=0)

# %%
# Plot the reference embedding, the aligned embeddings, and the average aligned embedding
plt.figure(figsize=(15, 10))
# Get the reference pattern embeddings
reference_embeddings = reference_pattern

# Calculate average aligned embeddings
all_aligned_embeddings = []
for idx, row in top_5_aligned_cells.iterrows():
    fov_name = row["fov_name"]
    track_ids = row["track_ids"]
    warp_path = row["warp_path"]
    start_time = int(row["start_timepoint"])

    # Get the lineage embeddings
    lineage_embeddings = embeddings_dataset.sel(
        sample=(fov_name, track_ids)
    ).features.values

    # Create aligned embeddings using the warping path
    aligned_embeddings = np.zeros(
        (len(reference_pattern), lineage_embeddings.shape[1]),
        dtype=lineage_embeddings.dtype,
    )

    # Create mapping from reference to lineage
    ref_to_lineage = {}
    for ref_idx, query_idx in warp_path:
        lineage_idx = int(start_time + query_idx)
        if 0 <= lineage_idx < len(lineage_embeddings):
            ref_to_lineage[ref_idx] = lineage_idx

    # Fill aligned embeddings
    for ref_idx in range(len(reference_pattern)):
        if ref_idx in ref_to_lineage:
            aligned_embeddings[ref_idx] = lineage_embeddings[ref_to_lineage[ref_idx]]
        elif ref_to_lineage:
            closest_ref_idx = min(ref_to_lineage.keys(), key=lambda x: abs(x - ref_idx))
            aligned_embeddings[ref_idx] = lineage_embeddings[
                ref_to_lineage[closest_ref_idx]
            ]

    all_aligned_embeddings.append(aligned_embeddings)

# Calculate average aligned embeddings
average_aligned_embeddings = np.mean(all_aligned_embeddings, axis=0)

# Plot dimension 0
plt.subplot(2, 1, 1)
# Plot reference pattern
plt.plot(
    range(len(reference_embeddings)),
    reference_embeddings[:, 0],
    label="Reference",
    color="black",
    linewidth=3,
)

# Plot each aligned embedding
for i, aligned_embeddings in enumerate(all_aligned_embeddings):
    plt.plot(
        range(len(aligned_embeddings)),
        aligned_embeddings[:, 0],
        label=f"Aligned {i}",
        alpha=0.4,
        linestyle="--",
    )

# Plot average aligned embedding
plt.plot(
    range(len(average_aligned_embeddings)),
    average_aligned_embeddings[:, 0],
    label="Average Aligned",
    color="red",
    linewidth=2,
)

plt.title("Dimension 0: Reference, Aligned, and Average Embeddings")
plt.xlabel("Reference Time Index")
plt.ylabel("Embedding Value")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot dimension 1
plt.subplot(2, 1, 2)
# Plot reference pattern
plt.plot(
    range(len(reference_embeddings)),
    reference_embeddings[:, 1],
    label="Reference",
    color="black",
    linewidth=3,
)

# Plot each aligned embedding
for i, aligned_embeddings in enumerate(all_aligned_embeddings):
    plt.plot(
        range(len(aligned_embeddings)),
        aligned_embeddings[:, 1],
        label=f"Aligned {i}",
        alpha=0.4,
        linestyle="--",
    )

# Plot average aligned embedding
plt.plot(
    range(len(average_aligned_embeddings)),
    average_aligned_embeddings[:, 1],
    label="Average Aligned",
    color="red",
    linewidth=2,
)

plt.title("Dimension 1: Reference, Aligned, and Average Embeddings")
plt.xlabel("Reference Time Index")
plt.ylabel("Embedding Value")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./SEC61B/reference_aligned_average_embeddings.png", dpi=300)
plt.show()

# %%
# Visualize where the reference pattern matches in each full lineage
plt.figure(figsize=(15, 15))

# First, plot the reference pattern for comparison
plt.subplot(len(top_5_aligned_cells) + 1, 2, 1)
plt.plot(
    range(len(reference_pattern)),
    reference_pattern[:, 0],
    label="Reference Dim 0",
    color="black",
    linewidth=2,
)
plt.title("Reference Pattern - Dimension 0")
plt.xlabel("Time Index")
plt.ylabel("Embedding Value")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(len(top_5_aligned_cells) + 1, 2, 2)
plt.plot(
    range(len(reference_pattern)),
    reference_pattern[:, 1],
    label="Reference Dim 1",
    color="black",
    linewidth=2,
)
plt.title("Reference Pattern - Dimension 1")
plt.xlabel("Time Index")
plt.ylabel("Embedding Value")
plt.grid(True, alpha=0.3)
plt.legend()

# Then plot each lineage with the matched section highlighted
for i, (_, row) in enumerate(top_5_aligned_cells.iloc[1:].iterrows()):
    fov_name = row["fov_name"]
    track_ids = row["track_ids"]
    warp_path = row["warp_path"]
    start_time = row["start_timepoint"]
    distance = row["distance"]

    # Get the full lineage embeddings
    lineage_embeddings = embeddings_dataset.sel(
        sample=(fov_name, track_ids)
    ).features.values

    # Create a subplot for dimension 0
    plt.subplot(len(top_5_aligned_cells) + 1, 2, 2 * i + 3)

    # Plot the full lineage
    plt.plot(
        range(len(lineage_embeddings)),
        lineage_embeddings[:, 0],
        label="Full Lineage",
        color="blue",
        alpha=0.7,
    )

    # Highlight the matched section
    matched_indices = set()
    for _, query_idx in warp_path:
        lineage_idx = int(start_time + query_idx)
        if 0 <= lineage_idx < len(lineage_embeddings):
            matched_indices.add(lineage_idx)

    matched_indices = sorted(list(matched_indices))
    if matched_indices:
        plt.plot(
            matched_indices,
            [lineage_embeddings[idx, 0] for idx in matched_indices],
            "ro-",
            label=f"Matched Section (DTW dist={distance:.2f})",
            linewidth=2,
        )

        # Add vertical lines to mark the start and end of the matched section
        plt.axvline(x=min(matched_indices), color="red", linestyle="--", alpha=0.5)
        plt.axvline(x=max(matched_indices), color="red", linestyle="--", alpha=0.5)

        # Add text labels
        plt.text(
            min(matched_indices),
            min(lineage_embeddings[:, 0]),
            f"Start: {min(matched_indices)}",
            color="red",
            fontsize=10,
        )
        plt.text(
            max(matched_indices),
            min(lineage_embeddings[:, 0]),
            f"End: {max(matched_indices)}",
            color="red",
            fontsize=10,
        )

    plt.title(f"Lineage {i} ({fov_name}) - Dimension 0")
    plt.xlabel("Lineage Time")
    plt.ylabel("Embedding Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Create a subplot for dimension 1
    plt.subplot(len(top_5_aligned_cells) + 1, 2, 2 * i + 4)

    # Plot the full lineage
    plt.plot(
        range(len(lineage_embeddings)),
        lineage_embeddings[:, 1],
        label="Full Lineage",
        color="green",
        alpha=0.7,
    )

    # Highlight the matched section
    if matched_indices:
        plt.plot(
            matched_indices,
            [lineage_embeddings[idx, 1] for idx in matched_indices],
            "ro-",
            label=f"Matched Section (DTW dist={distance:.2f})",
            linewidth=2,
        )

        # Add vertical lines to mark the start and end of the matched section
        plt.axvline(x=min(matched_indices), color="red", linestyle="--", alpha=0.5)
        plt.axvline(x=max(matched_indices), color="red", linestyle="--", alpha=0.5)

        # Add text labels
        plt.text(
            min(matched_indices),
            min(lineage_embeddings[:, 1]),
            f"Start: {min(matched_indices)}",
            color="red",
            fontsize=10,
        )
        plt.text(
            max(matched_indices),
            min(lineage_embeddings[:, 1]),
            f"End: {max(matched_indices)}",
            color="red",
            fontsize=10,
        )

    plt.title(f"Lineage {i} ({fov_name}) - Dimension 1")
    plt.xlabel("Lineage Time")
    plt.ylabel("Embedding Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./SEC61B/reference_vs_full_lineages.png", dpi=300)
plt.show()


# %%
