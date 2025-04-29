# %%
import ast
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
from tqdm import tqdm

from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.dimensionality_reduction import compute_pca

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

if NAPARI:
    import os

    import napari

    os.environ["DISPLAY"] = ":1"
    viewer = napari.Viewer()


# %%
CONDITION_EMBEDDINGS = "phase_n_organelle"
CONDITION_TO_ALIGN = "infection"

input_data_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_DENV.zarr"
)

if CONDITION_EMBEDDINGS == "phase_n_organelle":
    feature_path = Path(
        "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/predictions/timeAware_2chan__ntxent_192patch_70ckpt_rev7_GT.zarr"
    )
elif CONDITION_EMBEDDINGS == "organelle_only":
    feature_path = Path(
        "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/predictions/timeAware_organelle_only_ntxent_192patch_ckpt52_rev8_GT.zarr"
    )
elif CONDITION_EMBEDDINGS == "phase_n_sensor":
    feature_path = Path(
        "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/3-phenotyping/predictions_infection/2chan_192patch_100ckpt_timeAware_ntxent_GT.zarr"
    )
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/4-track-gt/2024_11_07_A549_SEC61_ZIKV_DENV_2_cropped.zarr"
)
embeddings_dataset = read_embedding_dataset(feature_path)
feature_df = embeddings_dataset["sample"].to_dataframe().reset_index(drop=True)

# Logic to find lineages
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
# Find the reference pattern
if CONDITION_TO_ALIGN == "phase_n_organelle":
    # Get the reference lineage
    # From the B/2 SEC61B-DV
    reference_lineage_fov = "/B/2/001000"
    reference_lineage_track_id = 104
    reference_timepoints = [55, 70]  # organelle remodeling

# From the C/2 SEC61B-DV-pl40
elif (
    CONDITION_TO_ALIGN == "remodelling_w_sensor"
    or CONDITION_TO_ALIGN == "organelle_only"
):
    reference_lineage_fov = "/C/2/001000"
    reference_lineage_track_id = [129]
    reference_timepoints = [8, 70]  # sensor rellocalization and partial remodelling

elif CONDITION_TO_ALIGN == "infection":
    reference_lineage_fov = "/C/2/001000"
    reference_lineage_track_id = [129]
    reference_timepoints = [8, 70]  # sensor rellocalization and partial remodelling

# Cell division
elif CONDITION_TO_ALIGN == "cell_division":
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

output_root = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/figure"
)
output_save_dir = (
    output_root / f"SEC61B/features_{CONDITION_EMBEDDINGS}_by_{CONDITION_TO_ALIGN}"
)
output_save_dir.mkdir(parents=True, exist_ok=True)

# Find the matching lineages
matches_df_path = Path(
    output_root / f"SEC61B/20241107_SEC61B_{CONDITION_TO_ALIGN}_matching_lineages.csv"
)
# Find all matches to the reference pattern
all_match_positions = pd.read_csv(matches_df_path)
# Get the top N aligned cells
n_cells = 5
top_n_aligned_cells = all_match_positions.head(n_cells)
# %%
source_channels = [
    "Phase3D",
    "raw GFP EX488 EM525-45",
    "raw mCherry EX561 EM600-37",
]
yx_patch_size = (192, 192)
z_range = (10, 30)
view_ref_sector_only = (True,)

all_lineage_images = []
all_aligned_stacks = []
all_unaligned_stacks = []

# Get aligned and unaligned stacks
top_aligned_cells = top_n_aligned_cells
napari_viewer = viewer if NAPARI else None

for idx, row in tqdm(
    top_aligned_cells.iterrows(),
    total=len(top_aligned_cells),
    desc="Aligning images",
):
    fov_name = row["fov_name"]
    track_ids = ast.literal_eval(row["track_ids"])
    warp_path = ast.literal_eval(row["warp_path"])
    start_time = int(row["start_timepoint"])

    print(f"Aligning images for {fov_name} with track ids: {track_ids}")
    data_module = TripletDataModule(
        data_path=input_data_path,
        tracks_path=tracks_path,
        source_channel=source_channels,
        z_range=z_range,
        initial_yx_patch_size=yx_patch_size,
        final_yx_patch_size=yx_patch_size,
        batch_size=1,
        num_workers=12,
        predict_cells=True,
        include_fov_names=[fov_name] * len(track_ids),
        include_track_ids=track_ids,
    )
    data_module.setup("predict")

    # Get the images for the lineage
    lineage_images = []
    for batch in data_module.predict_dataloader():
        image = batch["anchor"].numpy()[0]
        lineage_images.append(image)

    lineage_images = np.array(lineage_images)
    all_lineage_images.append(lineage_images)
    print(f"Lineage images shape: {np.array(lineage_images).shape}")

    # Create an aligned stack based on the warping path
    if view_ref_sector_only:
        aligned_stack = np.zeros(
            (len(reference_pattern),) + lineage_images.shape[-4:],
            dtype=lineage_images.dtype,
        )
        unaligned_stack = np.zeros(
            (len(reference_pattern),) + lineage_images.shape[-4:],
            dtype=lineage_images.dtype,
        )

        # Map each reference timepoint to the corresponding lineage timepoint
        for ref_idx in range(len(reference_pattern)):
            # Find matches in warping path for this reference index
            matches = [(i, q) for i, q in warp_path if i == ref_idx]
            unaligned_stack[ref_idx] = lineage_images[ref_idx]
            if matches:
                # Get the corresponding lineage timepoint (first match if multiple)
                print(f"Found match for ref idx: {ref_idx}")
                match = matches[0]
                query_idx = match[1]
                lineage_idx = int(start_time + query_idx)
                print(
                    f"Lineage index: {lineage_idx}, start time: {start_time}, query idx: {query_idx}, ref idx: {ref_idx}"
                )
                # Copy the image if it's within bounds
                if 0 <= lineage_idx < len(lineage_images):
                    aligned_stack[ref_idx] = lineage_images[lineage_idx]
                else:
                    # Find nearest valid timepoint if out of bounds
                    nearest_idx = min(max(0, lineage_idx), len(lineage_images) - 1)
                    aligned_stack[ref_idx] = lineage_images[nearest_idx]
            else:
                # If no direct match, find closest reference timepoint in warping path
                print(f"No match found for ref idx: {ref_idx}")
                all_ref_indices = [i for i, _ in warp_path]
                if all_ref_indices:
                    closest_ref_idx = min(
                        all_ref_indices, key=lambda x: abs(x - ref_idx)
                    )
                    closest_matches = [
                        (i, q) for i, q in warp_path if i == closest_ref_idx
                    ]

                    if closest_matches:
                        closest_query_idx = closest_matches[0][1]
                        lineage_idx = int(start_time + closest_query_idx)

                        if 0 <= lineage_idx < len(lineage_images):
                            aligned_stack[ref_idx] = lineage_images[lineage_idx]
                        else:
                            # Bound to valid range
                            nearest_idx = min(
                                max(0, lineage_idx), len(lineage_images) - 1
                            )
                            aligned_stack[ref_idx] = lineage_images[nearest_idx]

        all_aligned_stacks.append(aligned_stack)
        all_unaligned_stacks.append(unaligned_stack)

all_aligned_stacks = np.array(all_aligned_stacks)
all_unaligned_stacks = np.array(all_unaligned_stacks)

# %%
# Plot the aligned and unaligned stacks
if NAPARI:
    for idx, row in top_aligned_cells.reset_index().iterrows():
        fov_name = row["fov_name"]
        track_ids = ast.literal_eval(row["track_ids"])

        aligned_stack = all_aligned_stacks[idx]
        unaligned_stack = all_unaligned_stacks[idx]

        unaligned_gfp_mip = np.max(unaligned_stack[:, 1, :, :], axis=1)
        aligned_gfp_mip = np.max(aligned_stack[:, 1, :, :], axis=1)
        unaligned_mcherry_mip = np.max(unaligned_stack[:, 2, :, :], axis=1)
        aligned_mcherry_mip = np.max(aligned_stack[:, 2, :, :], axis=1)

        z_slice = 15
        unaligned_phase = unaligned_stack[:, 0, z_slice, :]
        aligned_phase = aligned_stack[:, 0, z_slice, :]

        # unaligned
        viewer.add_image(
            unaligned_gfp_mip,
            name=f"unaligned_gfp_{fov_name}_{track_ids[0]}",
            colormap="green",
            contrast_limits=(106, 215),
        )
        viewer.add_image(
            unaligned_mcherry_mip,
            name=f"unaligned_mcherry_{fov_name}_{track_ids[0]}",
            colormap="magenta",
            contrast_limits=(106, 190),
        )
        viewer.add_image(
            unaligned_phase,
            name=f"unaligned_phase_{fov_name}_{track_ids[0]}",
            colormap="gray",
            contrast_limits=(-0.74, 0.4),
        )
        # aligned
        viewer.add_image(
            aligned_gfp_mip,
            name=f"aligned_gfp_{fov_name}_{track_ids[0]}",
            colormap="green",
            contrast_limits=(106, 215),
        )
        viewer.add_image(
            aligned_mcherry_mip,
            name=f"aligned_mcherry_{fov_name}_{track_ids[0]}",
            colormap="magenta",
            contrast_limits=(106, 190),
        )
        viewer.add_image(
            aligned_phase,
            name=f"aligned_phase_{fov_name}_{track_ids[0]}",
            colormap="gray",
            contrast_limits=(-0.74, 0.4),
        )
    viewer.grid.enabled = True
    viewer.grid.shape = (-1, 6)


# %%
def plot_unaligned_vs_aligned_embeddings(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_dataset: xr.Dataset,
    output_save_dir: Path,
):
    """
    Plot the unaligned and aligned embeddings using PCA dimensions (PC1 and PC2) compared to the reference track.
    Shows the entire timeline of each lineage with the alignment section highlighted.

    Args:
        reference_pattern: The reference pattern embeddings
        top_aligned_cells: DataFrame with alignment information
        embeddings_dataset: Dataset containing embeddings
        output_save_dir: Path to save the output figures
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Collect all embedding data for PCA
    all_embeddings = []

    # Add reference pattern to PCA calculation
    all_embeddings.append(reference_pattern)

    # Get all lineage embeddings for PCA calculation
    all_lineage_data = {}  # Store full lineage data for each track

    # Process each lineage once to collect all data
    for _, row in top_aligned_cells.iterrows():
        fov_name = row["fov_name"]
        track_ids = ast.literal_eval(row["track_ids"])

        # Process each track in this lineage
        for track_id in track_ids:
            key = (fov_name, track_id)
            if key not in all_lineage_data:
                # Get full embedding data for this track
                track_data = embeddings_dataset.sel(
                    sample=(fov_name, track_id)
                ).features.values
                all_lineage_data[key] = track_data
                all_embeddings.append(track_data)

    # Combine all embeddings for PCA
    all_embeddings_flat = np.vstack(all_embeddings)

    # Standardize data
    scaler = StandardScaler()
    all_embeddings_scaled = scaler.fit_transform(all_embeddings_flat)

    # Apply PCA
    pca = PCA(n_components=2)
    pca.fit(all_embeddings_scaled)

    # Get the explained variance
    explained_variance = pca.explained_variance_ratio_ * 100
    logger.info(
        f"PCA explained variance: PC1 {explained_variance[0]:.2f}%, PC2 {explained_variance[1]:.2f}%"
    )

    # Transform reference pattern
    reference_scaled = scaler.transform(reference_pattern)
    reference_pca = pca.transform(reference_scaled)

    # Process each aligned cell for visualization
    for i, (_, row) in enumerate(top_aligned_cells.iterrows()):
        fov_name = row["fov_name"]
        track_ids = ast.literal_eval(row["track_ids"])
        warp_path = ast.literal_eval(row["warp_path"])
        start_time = int(row["start_timepoint"])
        distance = row["distance"]

        logger.info(f"Processing lineage {i+1}: {fov_name}, tracks {track_ids}")

        # Get the complete lineage embeddings for all timepoints
        full_lineage_embeddings = []

        for track_id in track_ids:
            key = (fov_name, track_id)
            track_data = all_lineage_data[key]
            full_lineage_embeddings.append(track_data)

        # Concatenate all tracks in this lineage
        full_lineage_embeddings = np.concatenate(full_lineage_embeddings)

        # Apply PCA to the full lineage
        full_lineage_scaled = scaler.transform(full_lineage_embeddings)
        full_lineage_pca = pca.transform(full_lineage_scaled)

        # Create aligned embeddings using the warping path (using original embeddings)
        aligned_embeddings = np.zeros_like(reference_pattern)

        # Map each reference timepoint to the corresponding lineage timepoint using warp path
        for ref_idx, query_idx in warp_path:
            lineage_idx = int(start_time + query_idx)
            if 0 <= lineage_idx < len(full_lineage_embeddings):
                aligned_embeddings[ref_idx] = full_lineage_embeddings[lineage_idx]

        # Fill in any missing values for the aligned embeddings
        ref_indices_in_path = set(i for i, _ in warp_path)
        for ref_idx in range(len(reference_pattern)):
            if ref_idx not in ref_indices_in_path and ref_indices_in_path:
                closest_ref_idx = min(
                    ref_indices_in_path, key=lambda x: abs(x - ref_idx)
                )
                closest_matches = [(i, q) for i, q in warp_path if i == closest_ref_idx]
                if closest_matches:
                    closest_query_idx = closest_matches[0][1]
                    lineage_idx = int(start_time + closest_query_idx)
                    if 0 <= lineage_idx < len(full_lineage_embeddings):
                        aligned_embeddings[ref_idx] = full_lineage_embeddings[
                            lineage_idx
                        ]

        # Apply PCA to aligned embeddings
        aligned_scaled = scaler.transform(aligned_embeddings)
        aligned_pca = pca.transform(aligned_scaled)

        # Create figure with 2 rows and 2 columns
        plt.figure(figsize=(20, 16))

        # ---- Row 1: Full lineage with alignment highlighted (PCA) ----

        # PC1 - Full lineage with all timepoints
        plt.subplot(2, 2, 1)

        # Plot the entire lineage with one color
        plt.plot(
            range(len(full_lineage_pca)),
            full_lineage_pca[:, 0],
            label="Full Lineage",
            color="blue",
            linewidth=1.5,
        )

        # Highlight alignment section
        alignment_indices = []
        for _, query_idx in warp_path:
            lineage_idx = int(start_time + query_idx)
            if 0 <= lineage_idx < len(full_lineage_embeddings):
                alignment_indices.append(lineage_idx)

        if alignment_indices:
            alignment_indices = sorted(alignment_indices)

            # Plot the aligned section in red
            for j in range(1, len(alignment_indices)):
                if (
                    alignment_indices[j] - alignment_indices[j - 1] == 1
                ):  # Adjacent indices
                    plt.plot(
                        [alignment_indices[j - 1], alignment_indices[j]],
                        [
                            full_lineage_pca[alignment_indices[j - 1], 0],
                            full_lineage_pca[alignment_indices[j], 0],
                        ],
                        color="red",
                        linewidth=2.5,
                    )

            # Plot individual aligned points (in case they're not contiguous)
            plt.scatter(
                alignment_indices,
                full_lineage_pca[alignment_indices, 0],
                color="red",
                s=30,
                zorder=5,
                label="Aligned Section",
            )

            # Add a vertical line to mark alignment start
            plt.axvline(
                x=start_time, color="green", linestyle="--", label="Alignment Start"
            )

        plt.title(f"Lineage {i+1} Full Timeline - PC1 ({fov_name}, tracks {track_ids})")
        plt.xlabel("Timepoint")
        plt.ylabel(f"PC1 ({explained_variance[0]:.2f}%)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # PC2 - Full lineage
        plt.subplot(2, 2, 2)

        # Plot the entire lineage with one color
        plt.plot(
            range(len(full_lineage_pca)),
            full_lineage_pca[:, 1],
            label="Full Lineage",
            color="blue",
            linewidth=1.5,
        )

        # Highlight alignment section
        if alignment_indices:
            # Plot the aligned section in red
            for j in range(1, len(alignment_indices)):
                if (
                    alignment_indices[j] - alignment_indices[j - 1] == 1
                ):  # Adjacent indices
                    plt.plot(
                        [alignment_indices[j - 1], alignment_indices[j]],
                        [
                            full_lineage_pca[alignment_indices[j - 1], 1],
                            full_lineage_pca[alignment_indices[j], 1],
                        ],
                        color="red",
                        linewidth=2.5,
                    )

            # Plot individual aligned points
            plt.scatter(
                alignment_indices,
                full_lineage_pca[alignment_indices, 1],
                color="red",
                s=30,
                zorder=5,
                label="Aligned Section",
            )

            # Add a vertical line to mark alignment start
            plt.axvline(
                x=start_time, color="green", linestyle="--", label="Alignment Start"
            )

        plt.title(f"Lineage {i+1} Full Timeline - PC2 ({fov_name}, tracks {track_ids})")
        plt.xlabel("Timepoint")
        plt.ylabel(f"PC2 ({explained_variance[1]:.2f}%)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # ---- Row 2: Alignment comparison with reference pattern ----

        # Create unaligned embeddings window starting from the alignment point
        unaligned_window = full_lineage_embeddings[
            start_time : start_time + len(reference_pattern)
        ]
        # Pad if needed
        if len(unaligned_window) < len(reference_pattern):
            pad_length = len(reference_pattern) - len(unaligned_window)
            padding = np.zeros((pad_length, unaligned_window.shape[1]))
            unaligned_window = np.vstack((unaligned_window, padding))
        # Trim if needed
        elif len(unaligned_window) > len(reference_pattern):
            unaligned_window = unaligned_window[: len(reference_pattern)]

        # Transform unaligned to PCA
        unaligned_scaled = scaler.transform(unaligned_window)
        unaligned_pca = pca.transform(unaligned_scaled)

        # PC1 - Aligned vs Reference
        plt.subplot(2, 2, 3)
        plt.plot(
            range(len(reference_pca)),
            reference_pca[:, 0],
            label="Reference",
            color="black",
            linewidth=2,
        )
        plt.plot(
            range(len(unaligned_pca)),
            unaligned_pca[:, 0],
            label="Unaligned",
            color="blue",
            linestyle="--",
            alpha=0.7,
        )
        plt.plot(
            range(len(aligned_pca)),
            aligned_pca[:, 0],
            label="Aligned",
            color="red",
            alpha=0.7,
        )
        plt.title(f"Alignment Comparison - PC1")
        plt.xlabel("Reference Timepoint")
        plt.ylabel(f"PC1 ({explained_variance[0]:.2f}%)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # PC2 - Aligned vs Reference
        plt.subplot(2, 2, 4)
        plt.plot(
            range(len(reference_pca)),
            reference_pca[:, 1],
            label="Reference",
            color="black",
            linewidth=2,
        )
        plt.plot(
            range(len(unaligned_pca)),
            unaligned_pca[:, 1],
            label="Unaligned",
            color="blue",
            linestyle="--",
            alpha=0.7,
        )
        plt.plot(
            range(len(aligned_pca)),
            aligned_pca[:, 1],
            label="Aligned",
            color="red",
            alpha=0.7,
        )
        plt.title(f"Alignment Comparison - PC2")
        plt.xlabel("Reference Timepoint")
        plt.ylabel(f"PC2 ({explained_variance[1]:.2f}%)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        save_path = output_save_dir / f"lineage_{i+1}_pca_full_timeline.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")


# Plot the unaligned and aligned embeddings
plot_unaligned_vs_aligned_embeddings(
    reference_pattern,
    top_n_aligned_cells,
    embeddings_dataset,
    output_save_dir,
)


# %%
def plot_reference_vs_full_lineages(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_dataset: xr.Dataset,
    save_path: str | None = None,
) -> np.ndarray:
    """
    Visualize where the reference pattern matches in each full lineage.

    Args:
        reference_pattern: The reference pattern embeddings
        top_aligned_cells: DataFrame with alignment information
        embeddings_dataset: Dataset containing embeddings
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(15, 15))

    # Get all lineage embeddings to determine global x-limits
    all_lineage_lengths = []
    for _, row in top_aligned_cells.iterrows():
        fov_name = row["fov_name"]
        track_ids = ast.literal_eval(row["track_ids"])
        lineage_embeddings = embeddings_dataset.sel(
            sample=(fov_name, track_ids)
        ).features.values
        all_lineage_lengths.append(len(lineage_embeddings))

    max_lineage_length = (
        max(all_lineage_lengths) if all_lineage_lengths else len(reference_pattern)
    )

    # First, plot the reference pattern for comparison
    plt.subplot(len(top_aligned_cells) + 1, 2, 1)
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
    plt.xlim(0, max_lineage_length)

    plt.subplot(len(top_aligned_cells) + 1, 2, 2)
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
    plt.xlim(0, max_lineage_length)

    # Then plot each lineage with the matched section highlighted
    for i, (_, row) in enumerate(top_aligned_cells.iterrows()):
        fov_name = row["fov_name"]
        track_ids = ast.literal_eval(row["track_ids"])
        warp_path = ast.literal_eval(row["warp_path"])
        start_time = row["start_timepoint"]
        distance = row["distance"]

        # Get the full lineage embeddings
        lineage_embeddings = embeddings_dataset.sel(
            sample=(fov_name, track_ids)
        ).features.values

        # Create a subplot for dimension 0
        plt.subplot(len(top_aligned_cells) + 1, 2, 2 * i + 3)

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

        plt.title(f"Lineage {i} ({fov_name}) Track {track_ids[0]} - Dimension 0")
        plt.xlabel("Lineage Time")
        plt.ylabel("Embedding Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max_lineage_length)

        # Create a subplot for dimension 1
        plt.subplot(len(top_aligned_cells) + 1, 2, 2 * i + 4)

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
        plt.xlim(0, max_lineage_length)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


# %%
plot_reference_vs_full_lineages(
    reference_pattern,
    top_n_aligned_cells,
    embeddings_dataset,
    save_path=output_root
    / f"SEC61B/20241107_SEC61B_features_{CONDITION_EMBEDDINGS}_reference_vs_full_lineages.png",
)


# %%
def plot_direct_overlay_comparison(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_dataset: xr.Dataset,
    output_save_dir: Path,
    condition: str,
):
    """
    Create a direct overlay comparison between reference, unaligned, and aligned patterns.

    Args:
        reference_pattern: The reference pattern embeddings
        top_aligned_cells: DataFrame with alignment information
        embeddings_dataset: Dataset containing embeddings
        output_save_dir: Path to save the output figures
        condition: String describing the condition
    """
    n_cells = len(top_aligned_cells)
    fig, axes = plt.subplots(n_cells, 2, figsize=(16, 4 * n_cells))

    # If only one row, reshape axes for easier indexing
    if n_cells == 1:
        axes = axes.reshape(1, -1)

    # Process each aligned cell
    for i, (_, row) in enumerate(top_aligned_cells.iterrows()):
        fov_name = row["fov_name"]
        track_ids = ast.literal_eval(row["track_ids"])
        warp_path = ast.literal_eval(row["warp_path"])
        start_time = int(row["start_timepoint"])
        distance = row["distance"]

        # Get the full lineage embeddings
        lineage_embeddings = embeddings_dataset.sel(
            sample=(fov_name, track_ids)
        ).features.values

        # Extract the unaligned window matching the reference pattern length
        unaligned_window = lineage_embeddings[
            start_time : start_time + len(reference_pattern)
        ]

        # Pad or trim to match reference length
        if len(unaligned_window) < len(reference_pattern):
            pad_length = len(reference_pattern) - len(unaligned_window)
            padding = np.zeros((pad_length, unaligned_window.shape[1]))
            unaligned_window = np.vstack((unaligned_window, padding))
        elif len(unaligned_window) > len(reference_pattern):
            unaligned_window = unaligned_window[: len(reference_pattern)]

        # Create aligned embeddings using the warping path
        aligned_embeddings = np.zeros_like(reference_pattern)

        # Map each reference timepoint to the corresponding lineage timepoint
        for ref_idx, query_idx in warp_path:
            lineage_idx = int(start_time + query_idx)
            if 0 <= lineage_idx < len(lineage_embeddings):
                aligned_embeddings[ref_idx] = lineage_embeddings[lineage_idx]

        # Fill in any missing values for the aligned embeddings
        ref_indices_in_path = set(i for i, _ in warp_path)
        for ref_idx in range(len(reference_pattern)):
            if ref_idx not in ref_indices_in_path and ref_indices_in_path:
                closest_ref_idx = min(
                    ref_indices_in_path, key=lambda x: abs(x - ref_idx)
                )
                closest_matches = [(i, q) for i, q in warp_path if i == closest_ref_idx]
                if closest_matches:
                    closest_query_idx = closest_matches[0][1]
                    lineage_idx = int(start_time + closest_query_idx)
                    if 0 <= lineage_idx < len(lineage_embeddings):
                        aligned_embeddings[ref_idx] = lineage_embeddings[lineage_idx]

        time_points = np.arange(len(reference_pattern))

        # Plot dimension 0 for all patterns
        ax = axes[i, 0]
        ax.plot(
            time_points, reference_pattern[:, 0], "k-", linewidth=2.5, label="Reference"
        )
        ax.plot(
            time_points,
            unaligned_window[:, 0],
            "b--",
            linewidth=1.5,
            label="Unaligned",
        )
        ax.plot(
            time_points, aligned_embeddings[:, 0], "r-", linewidth=1.5, label="Aligned"
        )

        ax.set_title(f"Dimension 0 - Lineage {i+1} ({fov_name})")
        ax.set_xlabel("Reference Time Point")
        ax.set_ylabel("Embedding Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot dimension 1 for all patterns
        ax = axes[i, 1]
        ax.plot(
            time_points, reference_pattern[:, 1], "k-", linewidth=2.5, label="Reference"
        )
        ax.plot(
            time_points,
            unaligned_window[:, 1],
            "b--",
            linewidth=1.5,
            label="Unaligned",
        )
        ax.plot(
            time_points, aligned_embeddings[:, 1], "r-", linewidth=1.5, label="Aligned"
        )

        ax.set_title(f"Dimension 1 - Lineage {i+1} (DTW dist={distance:.2f})")
        ax.set_xlabel("Reference Time Point")
        ax.set_ylabel("Embedding Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Direct Comparison: Reference vs Unaligned vs Aligned - {condition.upper()}",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure
    save_path = (
        output_save_dir
        / f"20241107_SEC61B_{CONDITION_EMBEDDINGS}_direct_overlay_comparison.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Direct overlay comparison saved to {save_path}")

    return save_path


# Plot direct overlay comparison
overlay_comparison_path = plot_direct_overlay_comparison(
    reference_pattern,
    top_n_aligned_cells,
    embeddings_dataset,
    output_save_dir,
    CONDITION_EMBEDDINGS,
)


# %%
def create_trajectory_comparison_video(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_dataset: xr.Dataset,
    all_unaligned_stacks: np.ndarray,
    all_aligned_stacks: np.ndarray,
    output_path: Path,
    lineage_idx: int = 0,
    fps: int = 5,
):
    """
    Create a video comparing unaligned vs aligned trajectory with a moving point over time.

    Args:
        reference_pattern: The reference pattern embeddings
        top_aligned_cells: DataFrame with alignment information
        embeddings_dataset: Dataset containing embeddings
        all_unaligned_stacks: Array of unaligned image stacks
        all_aligned_stacks: Array of aligned image stacks
        output_path: Path to save the video .mp4
        lineage_idx: Index of the lineage to visualize (default: 0)
        fps: Frames per second for the video (default: 5)
    """
    import matplotlib.animation as animation
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    # Set dark theme
    plt.style.use("dark_background")

    # Get the row for the selected lineage
    row = top_aligned_cells.iloc[lineage_idx]
    fov_name = row["fov_name"]
    track_ids = ast.literal_eval(row["track_ids"])
    warp_path = ast.literal_eval(row["warp_path"])
    start_time = int(row["start_timepoint"])

    # Get the selected stacks
    unaligned_stack = all_unaligned_stacks[lineage_idx]
    aligned_stack = all_aligned_stacks[lineage_idx]

    # Get the lineage embeddings
    lineage_embeddings = embeddings_dataset.sel(
        sample=(fov_name, track_ids)
    ).features.values

    # Extract the unaligned window matching the reference pattern length
    unaligned_window = lineage_embeddings[
        start_time : start_time + len(reference_pattern)
    ]

    # Pad or trim to match reference length
    if len(unaligned_window) < len(reference_pattern):
        pad_length = len(reference_pattern) - len(unaligned_window)
        padding = np.zeros((pad_length, unaligned_window.shape[1]))
        unaligned_window = np.vstack((unaligned_window, padding))
    elif len(unaligned_window) > len(reference_pattern):
        unaligned_window = unaligned_window[: len(reference_pattern)]

    # Create aligned embeddings using the warping path
    aligned_embeddings = np.zeros_like(reference_pattern)

    # Map each reference timepoint to the corresponding lineage timepoint
    for ref_idx, query_idx in warp_path:
        lineage_idx = int(start_time + query_idx)
        if 0 <= lineage_idx < len(lineage_embeddings):
            aligned_embeddings[ref_idx] = lineage_embeddings[lineage_idx]

    # Fill in any missing values for the aligned embeddings
    ref_indices_in_path = set(i for i, _ in warp_path)
    for ref_idx in range(len(reference_pattern)):
        if ref_idx not in ref_indices_in_path and ref_indices_in_path:
            closest_ref_idx = min(ref_indices_in_path, key=lambda x: abs(x - ref_idx))
            closest_matches = [(i, q) for i, q in warp_path if i == closest_ref_idx]
            if closest_matches:
                closest_query_idx = closest_matches[0][1]
                lineage_idx = int(start_time + closest_query_idx)
                if 0 <= lineage_idx < len(lineage_embeddings):
                    aligned_embeddings[ref_idx] = lineage_embeddings[lineage_idx]

    # Create the figure with a 2x5 grid layout
    fig = plt.figure(figsize=(22, 8), facecolor="black")

    # Define the layout of subplots with reduced spacing and width ratios for rectangular dimension plots
    gs = fig.add_gridspec(2, 4, wspace=0.05, hspace=0.1, width_ratios=[1, 1, 2, 2])

    # Create the subplots for unaligned row (top row)
    ax_unaligned_phase = fig.add_subplot(gs[0, 0])  # Phase
    ax_unaligned_gfp = fig.add_subplot(gs[0, 1])  # GFP
    ax_unaligned_mcherry = fig.add_subplot(gs[0, 2])  # mCherry
    ax_dim0 = fig.add_subplot(gs[0, 3])  # Dimension 0 plot
    ax_dim1 = fig.add_subplot(gs[1, 3])  # Dimension 1 plot

    # Create the subplots for aligned row (bottom row)
    ax_aligned_phase = fig.add_subplot(gs[1, 0])  # Phase
    ax_aligned_gfp = fig.add_subplot(gs[1, 1])  # GFP
    ax_aligned_mcherry = fig.add_subplot(gs[1, 2])  # mCherry

    # Set background color for all axes
    for ax in [
        ax_unaligned_phase,
        ax_unaligned_gfp,
        ax_unaligned_mcherry,
        ax_aligned_phase,
        ax_aligned_gfp,
        ax_aligned_mcherry,
        ax_dim0,
        ax_dim1,
    ]:
        ax.set_facecolor("black")

    # Select z-slice for images
    z_slice = 15

    # Time points for the plots
    time_points = np.arange(len(reference_pattern))

    # Define contrast limits for each channel
    phase_limits = (-0.74, 0.4)
    gfp_limits = (106, 215)
    mcherry_limits = (106, 170)

    # Get the initial images
    unaligned_phase_init = unaligned_stack[0, 0, z_slice, :]
    unaligned_gfp_init = unaligned_stack[0, 1, z_slice, :]
    unaligned_mcherry_init = unaligned_stack[0, 2, z_slice, :]

    aligned_phase_init = aligned_stack[0, 0, z_slice, :]
    aligned_gfp_init = aligned_stack[0, 1, z_slice, :]
    aligned_mcherry_init = aligned_stack[0, 2, z_slice, :]

    # Create normalized data for proper visualization
    from matplotlib.colors import Normalize

    phase_norm = Normalize(vmin=phase_limits[0], vmax=phase_limits[1])
    gfp_norm = Normalize(vmin=gfp_limits[0], vmax=gfp_limits[1])
    mcherry_norm = Normalize(vmin=mcherry_limits[0], vmax=mcherry_limits[1])

    # Function to safely normalize values to [0,1] range
    def safe_norm(norm_func, data):
        normalized = norm_func(data)
        return np.clip(normalized, 0, 1)

    # Plot the initial images for phase (grayscale)
    unaligned_phase_img = ax_unaligned_phase.imshow(
        unaligned_phase_init, cmap="gray", vmin=phase_limits[0], vmax=phase_limits[1]
    )
    ax_unaligned_phase.set_title(f"Unaligned Phase", color="white")
    ax_unaligned_phase.axis("off")

    aligned_phase_img = ax_aligned_phase.imshow(
        aligned_phase_init, cmap="gray", vmin=phase_limits[0], vmax=phase_limits[1]
    )
    ax_aligned_phase.set_title(f"Aligned Phase", color="white")
    ax_aligned_phase.axis("off")

    # Create RGB images for GFP (green channel)
    unaligned_gfp_rgb = np.zeros((*unaligned_gfp_init.shape, 3))
    unaligned_gfp_rgb[:, :, 1] = safe_norm(
        gfp_norm, unaligned_gfp_init
    )  # Green channel

    aligned_gfp_rgb = np.zeros((*aligned_gfp_init.shape, 3))
    aligned_gfp_rgb[:, :, 1] = safe_norm(gfp_norm, aligned_gfp_init)  # Green channel

    # Create RGB images for mCherry (magenta channel = red + blue)
    unaligned_mcherry_rgb = np.zeros((*unaligned_mcherry_init.shape, 3))
    unaligned_mcherry_rgb[:, :, 0] = safe_norm(
        mcherry_norm, unaligned_mcherry_init
    )  # Red component
    unaligned_mcherry_rgb[:, :, 2] = safe_norm(
        mcherry_norm, unaligned_mcherry_init
    )  # Blue component

    aligned_mcherry_rgb = np.zeros((*aligned_mcherry_init.shape, 3))
    aligned_mcherry_rgb[:, :, 0] = safe_norm(
        mcherry_norm, aligned_mcherry_init
    )  # Red component
    aligned_mcherry_rgb[:, :, 2] = safe_norm(
        mcherry_norm, aligned_mcherry_init
    )  # Blue component

    # Plot the RGB fluorescence images
    unaligned_gfp_img = ax_unaligned_gfp.imshow(unaligned_gfp_rgb)
    ax_unaligned_gfp.set_title(f"Unaligned GFP", color="white")
    ax_unaligned_gfp.axis("off")

    aligned_gfp_img = ax_aligned_gfp.imshow(aligned_gfp_rgb)
    ax_aligned_gfp.set_title(f"Aligned GFP", color="white")
    ax_aligned_gfp.axis("off")

    unaligned_mcherry_img = ax_unaligned_mcherry.imshow(unaligned_mcherry_rgb)
    ax_unaligned_mcherry.set_title(f"Unaligned mCherry", color="white")
    ax_unaligned_mcherry.axis("off")

    aligned_mcherry_img = ax_aligned_mcherry.imshow(aligned_mcherry_rgb)
    ax_aligned_mcherry.set_title(f"Aligned mCherry", color="white")
    ax_aligned_mcherry.axis("off")

    # Plot all trajectories
    ax_dim0.plot(
        time_points, reference_pattern[:, 0], "w-", linewidth=2.5, label="Reference"
    )
    ax_dim0.plot(
        time_points, unaligned_window[:, 0], "c--", linewidth=1.5, label="Unaligned"
    )
    ax_dim0.plot(
        time_points, aligned_embeddings[:, 0], "r-", linewidth=1.5, label="Aligned"
    )
    ax_dim0.set_title(f"Dimension 0 - Lineage ({fov_name})", color="white")
    ax_dim0.set_xlabel("Reference Time Point", color="white")
    ax_dim0.set_ylabel("Embedding Value", color="white")
    ax_dim0.tick_params(colors="white")
    ax_dim0.legend(facecolor="black", edgecolor="white")
    ax_dim0.grid(True, alpha=0.3, color="gray")

    ax_dim1.plot(
        time_points, reference_pattern[:, 1], "w-", linewidth=2.5, label="Reference"
    )
    ax_dim1.plot(
        time_points, unaligned_window[:, 1], "c--", linewidth=1.5, label="Unaligned"
    )
    ax_dim1.plot(
        time_points, aligned_embeddings[:, 1], "r-", linewidth=1.5, label="Aligned"
    )
    ax_dim1.set_title(f"Dimension 1 - Lineage ({fov_name})", color="white")
    ax_dim1.set_xlabel("Reference Time Point", color="white")
    ax_dim1.set_ylabel("Embedding Value", color="white")
    ax_dim1.tick_params(colors="white")
    ax_dim1.legend(facecolor="black", edgecolor="white")
    ax_dim1.grid(True, alpha=0.3, color="gray")

    # Add moving points to the trajectory plots
    (point_ref_dim0,) = ax_dim0.plot(0, reference_pattern[0, 0], "wo", markersize=10)
    (point_unaligned_dim0,) = ax_dim0.plot(
        0, unaligned_window[0, 0], "co", markersize=10
    )
    (point_aligned_dim0,) = ax_dim0.plot(
        0, aligned_embeddings[0, 0], "ro", markersize=10
    )

    (point_ref_dim1,) = ax_dim1.plot(0, reference_pattern[0, 1], "wo", markersize=10)
    (point_unaligned_dim1,) = ax_dim1.plot(
        0, unaligned_window[0, 1], "co", markersize=10
    )
    (point_aligned_dim1,) = ax_dim1.plot(
        0, aligned_embeddings[0, 1], "ro", markersize=10
    )

    # Instead of tight_layout, manually adjust the figure margins
    fig.subplots_adjust(
        left=0.02, right=0.98, bottom=0.05, top=0.90, wspace=0.05, hspace=0.3
    )

    # Create the animation function
    def animate(frame):
        # Get current frame's images
        unaligned_phase = unaligned_stack[frame, 0, z_slice, :]
        unaligned_gfp = unaligned_stack[frame, 1, z_slice, :]
        unaligned_mcherry = unaligned_stack[frame, 2, z_slice, :]

        aligned_phase = aligned_stack[frame, 0, z_slice, :]
        aligned_gfp = aligned_stack[frame, 1, z_slice, :]
        aligned_mcherry = aligned_stack[frame, 2, z_slice, :]

        # Create RGB images for the current frame
        # GFP (green channel)
        unaligned_gfp_rgb = np.zeros((*unaligned_gfp.shape, 3))
        unaligned_gfp_rgb[:, :, 1] = safe_norm(gfp_norm, unaligned_gfp)  # Green channel

        aligned_gfp_rgb = np.zeros((*aligned_gfp.shape, 3))
        aligned_gfp_rgb[:, :, 1] = safe_norm(gfp_norm, aligned_gfp)  # Green channel

        # mCherry (magenta channel = red + blue)
        unaligned_mcherry_rgb = np.zeros((*unaligned_mcherry.shape, 3))
        unaligned_mcherry_rgb[:, :, 0] = safe_norm(
            mcherry_norm, unaligned_mcherry
        )  # Red component
        unaligned_mcherry_rgb[:, :, 2] = safe_norm(
            mcherry_norm, unaligned_mcherry
        )  # Blue component

        aligned_mcherry_rgb = np.zeros((*aligned_mcherry.shape, 3))
        aligned_mcherry_rgb[:, :, 0] = safe_norm(
            mcherry_norm, aligned_mcherry
        )  # Red component
        aligned_mcherry_rgb[:, :, 2] = safe_norm(
            mcherry_norm, aligned_mcherry
        )  # Blue component

        # Update the images
        unaligned_phase_img.set_array(unaligned_phase)
        unaligned_gfp_img.set_array(unaligned_gfp_rgb)
        unaligned_mcherry_img.set_array(unaligned_mcherry_rgb)

        aligned_phase_img.set_array(aligned_phase)
        aligned_gfp_img.set_array(aligned_gfp_rgb)
        aligned_mcherry_img.set_array(aligned_mcherry_rgb)

        # Update the trajectory points - must use sequences not single values
        point_ref_dim0.set_data([frame], [reference_pattern[frame, 0]])
        point_unaligned_dim0.set_data([frame], [unaligned_window[frame, 0]])
        point_aligned_dim0.set_data([frame], [aligned_embeddings[frame, 0]])

        point_ref_dim1.set_data([frame], [reference_pattern[frame, 1]])
        point_unaligned_dim1.set_data([frame], [unaligned_window[frame, 1]])
        point_aligned_dim1.set_data([frame], [aligned_embeddings[frame, 1]])

        # Return all artists that were updated
        return [
            unaligned_phase_img,
            unaligned_gfp_img,
            unaligned_mcherry_img,
            aligned_phase_img,
            aligned_gfp_img,
            aligned_mcherry_img,
            point_ref_dim0,
            point_unaligned_dim0,
            point_aligned_dim0,
            point_ref_dim1,
            point_unaligned_dim1,
            point_aligned_dim1,
        ]

    # Create the animation
    ani = animation.FuncAnimation(
        fig, animate, frames=len(reference_pattern), interval=1000 / fps, blit=True
    )

    # Save the animation
    ani.save(
        output_path,
        writer="ffmpeg",
        fps=fps,
        dpi=300,
        extra_args=["-vcodec", "libx264"],
    )

    logger.info(f"Trajectory comparison video saved to {save_path}")
    return save_path


# Call the function to create the video for the first lineage
if len(all_aligned_stacks) > 0 and len(all_unaligned_stacks) > 0:
    for idx, row in top_n_aligned_cells.reset_index().iterrows():
        fov_name = row["fov_name"][1:].replace("/", "_")
        track_ids = row["track_ids"]
        save_path = output_save_dir / "heterogeneity_vs_alignment"
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = (
            save_path / f"20241107_SEC61B_fov_{fov_name}_track_{track_ids[0]}.mp4"
        )

        video_path = create_trajectory_comparison_video(
            reference_pattern,
            top_n_aligned_cells,
            embeddings_dataset,
            all_unaligned_stacks,
            all_aligned_stacks,
            output_path=save_path,
            lineage_idx=idx,
            fps=5,
        )


# %%
def plot_multiple_lineages_unaligned(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_dataset: xr.Dataset,
    all_unaligned_stacks: np.ndarray,
    output_path: Path,
    num_lineages: int = 3,
    z_slice: int = 15,
    time_point: int = 20,  # Example time point to visualize
):
    """
    Create a plot showing multiple lineages in rows with phase/GFP/mCherry side by side,
    followed by dimension plots for unaligned trajectories only.

    Each row shows:
    - Phase, GFP, mCherry images for unaligned data
    - Dimension 0 and Dimension 1 plots showing unaligned trajectory vs reference

    Args:
        reference_pattern: The reference pattern embeddings
        top_aligned_cells: DataFrame with alignment information
        embeddings_dataset: Dataset containing embeddings
        all_unaligned_stacks: Array of unaligned image stacks
        output_path: Path to save the output figure
        num_lineages: Number of lineages to display (default: 3)
        z_slice: Z-slice to use for images (default: 15)
        time_point: Time point to display in the visualization (default: 20)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # Set dark theme
    plt.style.use("dark_background")

    # Limit to requested number of lineages
    num_lineages = min(num_lineages, len(top_aligned_cells), len(all_unaligned_stacks))

    # Create figure
    fig = plt.figure(figsize=(20, 4 * num_lineages), facecolor="black")

    # Define column ratios - standard single spacing value
    gs = fig.add_gridspec(
        num_lineages, 5, wspace=0.2, hspace=0.3, width_ratios=[1.5, 1.5, 1.5, 2, 2]
    )

    # Define contrast limits for each channel
    phase_limits = (-0.74, 0.4)
    gfp_limits = (106, 215)
    mcherry_limits = (106, 190)

    # Function to safely normalize values to [0,1] range
    def safe_norm(norm_func, data):
        normalized = norm_func(data)
        return np.clip(normalized, 0, 1)

    # Create normalizers for each channel
    phase_norm = Normalize(vmin=phase_limits[0], vmax=phase_limits[1])
    gfp_norm = Normalize(vmin=gfp_limits[0], vmax=gfp_limits[1])
    mcherry_norm = Normalize(vmin=mcherry_limits[0], vmax=mcherry_limits[1])

    # Process each lineage
    for i in range(num_lineages):
        # Get data for this lineage
        row = top_aligned_cells.iloc[i]
        fov_name = row["fov_name"]
        track_ids = ast.literal_eval(row["track_ids"])
        start_time = int(row["start_timepoint"])

        # Get unaligned stack for this lineage
        unaligned_stack = all_unaligned_stacks[i]

        # Make sure time_point is valid
        valid_time_point = min(time_point, unaligned_stack.shape[0] - 1)

        # Get the lineage embeddings
        lineage_embeddings = embeddings_dataset.sel(
            sample=(fov_name, track_ids)
        ).features.values

        # Extract the unaligned window
        unaligned_window = lineage_embeddings[
            start_time : start_time + len(reference_pattern)
        ]

        # Pad or trim to match reference length
        if len(unaligned_window) < len(reference_pattern):
            pad_length = len(reference_pattern) - len(unaligned_window)
            padding = np.zeros((pad_length, unaligned_window.shape[1]))
            unaligned_window = np.vstack((unaligned_window, padding))
        elif len(unaligned_window) > len(reference_pattern):
            unaligned_window = unaligned_window[: len(reference_pattern)]

        # Create axes for this lineage's row
        ax_phase = fig.add_subplot(gs[i, 0])
        ax_gfp = fig.add_subplot(gs[i, 1])
        ax_mcherry = fig.add_subplot(gs[i, 2])
        ax_dim0 = fig.add_subplot(gs[i, 3])
        ax_dim1 = fig.add_subplot(gs[i, 4])

        # Set background color for all axes
        for ax in [ax_phase, ax_gfp, ax_mcherry, ax_dim0, ax_dim1]:
            ax.set_facecolor("black")

        # Get the images for the selected time point
        phase_img = unaligned_stack[valid_time_point, 0, z_slice, :]
        gfp_img = unaligned_stack[valid_time_point, 1, z_slice, :]
        mcherry_img = unaligned_stack[valid_time_point, 2, z_slice, :]

        # Display phase image
        ax_phase.imshow(
            phase_img, cmap="gray", vmin=phase_limits[0], vmax=phase_limits[1]
        )
        ax_phase.set_title(
            f"Phase - {fov_name}\nTrack {track_ids[0]}", color="white", fontsize=10
        )
        ax_phase.axis("off")

        # Create RGB image for GFP (green channel)
        gfp_rgb = np.zeros((*gfp_img.shape, 3))
        gfp_rgb[:, :, 1] = safe_norm(gfp_norm, gfp_img)  # Green channel
        ax_gfp.imshow(gfp_rgb)
        ax_gfp.set_title("GFP", color="white")
        ax_gfp.axis("off")

        # Create RGB image for mCherry (magenta = red + blue)
        mcherry_rgb = np.zeros((*mcherry_img.shape, 3))
        mcherry_rgb[:, :, 0] = safe_norm(mcherry_norm, mcherry_img)  # Red component
        mcherry_rgb[:, :, 2] = safe_norm(mcherry_norm, mcherry_img)  # Blue component
        ax_mcherry.imshow(mcherry_rgb)
        ax_mcherry.set_title("mCherry", color="white")
        ax_mcherry.axis("off")

        # Plot dimension 0 trajectory
        time_points = np.arange(len(reference_pattern))
        ax_dim0.plot(
            time_points, reference_pattern[:, 0], "w-", linewidth=2.5, label="Reference"
        )
        ax_dim0.plot(
            time_points, unaligned_window[:, 0], "c--", linewidth=1.5, label="Unaligned"
        )

        # Highlight current time point
        if valid_time_point < len(time_points):
            ax_dim0.plot(
                valid_time_point,
                unaligned_window[valid_time_point, 0],
                "co",
                markersize=10,
            )

        ax_dim0.set_title(f"Dimension 0", color="white")
        ax_dim0.set_xlabel("Time Point", color="white")
        ax_dim0.set_ylabel("Embedding Value", color="white")
        ax_dim0.tick_params(colors="white")
        ax_dim0.legend(facecolor="black", edgecolor="white")
        ax_dim0.grid(True, alpha=0.3, color="gray")

        # Plot dimension 1 trajectory
        ax_dim1.plot(
            time_points, reference_pattern[:, 1], "w-", linewidth=2.5, label="Reference"
        )
        ax_dim1.plot(
            time_points, unaligned_window[:, 1], "c--", linewidth=1.5, label="Unaligned"
        )

        # Highlight current time point
        if valid_time_point < len(time_points):
            ax_dim1.plot(
                valid_time_point,
                unaligned_window[valid_time_point, 1],
                "co",
                markersize=10,
            )

        ax_dim1.set_title(f"Dimension 1", color="white")
        ax_dim1.set_xlabel("Time Point", color="white")
        ax_dim1.set_ylabel("Embedding Value", color="white")
        ax_dim1.tick_params(colors="white")
        ax_dim1.legend(facecolor="black", edgecolor="white")
        ax_dim1.grid(True, alpha=0.3, color="gray")

    # Add main title
    plt.suptitle(
        f"Unaligned Trajectories - Multiple Lineages (TP:{time_point})",
        fontsize=16,
        color="white",
        y=0.98,
    )

    # Adjust figure margins
    fig.subplots_adjust(
        left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.1, hspace=0.3
    )

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # plt.close(fig)

    return output_path


# %%
# Create the comparison plot for unaligned trajectories
# NOTE for testing used lineage 3
if len(all_unaligned_stacks) > 0:
    output_file = output_save_dir / "20241107_SEC61B_unaligned_comparison.png"
    unaligned_plot = plot_multiple_lineages_unaligned(
        reference_pattern,
        top_n_aligned_cells,
        embeddings_dataset,
        output_path=output_file,
        num_lineages=3,
        time_point=20,
    )


# %%
def create_multiple_lineages_video(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_dataset: xr.Dataset,
    image_stacks: np.ndarray,  # Either all_unaligned_stacks or all_aligned_stacks
    output_path: Path,
    mode: str = "unaligned",  # "unaligned" or "aligned"
    num_lineages: int = 3,
    z_slice: int = 15,
    fps: int = 5,
    show_title: bool = False,  # New parameter to control title visibility
):
    """
    Create a video showing multiple lineages with phase/GFP/mCherry side by side,
    followed by dimension plots for either unaligned or aligned trajectories.

    Args:
        reference_pattern: The reference pattern embeddings
        top_aligned_cells: DataFrame with alignment information
        embeddings_dataset: Dataset containing embeddings
        image_stacks: Array of image stacks (unaligned or aligned)
        output_path: Path to save the video file
        mode: "unaligned" or "aligned" to determine which trajectories to display
        num_lineages: Number of lineages to display (default: 3)
        z_slice: Z-slice to use for images (default: 15)
        fps: Frames per second for the video (default: 5)
        show_title: Whether to show the main title with frame counter (default: False)
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # Set dark theme
    plt.style.use("dark_background")

    # Limit to requested number of lineages
    num_lineages = min(num_lineages, len(top_aligned_cells), len(image_stacks))

    # Create figure with rows for each lineage
    fig = plt.figure(figsize=(20, 4 * num_lineages), facecolor="black")

    # Define column ratios - more space for images, increased spacing between images and plots
    gs = fig.add_gridspec(
        num_lineages, 5, wspace=0.2, hspace=0.3, width_ratios=[1.5, 1.5, 1.5, 2, 2]
    )

    # Define contrast limits for each channel
    phase_limits = (-0.74, 0.4)
    gfp_limits = (106, 215)
    mcherry_limits = (106, 190)

    # Function to safely normalize values to [0,1] range
    def safe_norm(norm_func, data):
        normalized = norm_func(data)
        return np.clip(normalized, 0, 1)

    # Create normalizers for each channel
    phase_norm = Normalize(vmin=phase_limits[0], vmax=phase_limits[1])
    gfp_norm = Normalize(vmin=gfp_limits[0], vmax=gfp_limits[1])
    mcherry_norm = Normalize(vmin=mcherry_limits[0], vmax=mcherry_limits[1])

    # Store all axes and image objects for later updating
    all_axes = []
    image_objs = []
    trajectory_points = []

    # Maximum number of frames to simulate
    max_frames = len(reference_pattern)

    # Process each lineage
    for i in range(num_lineages):
        # Get data for this lineage
        row = top_aligned_cells.iloc[i]
        fov_name = row["fov_name"]
        track_ids = ast.literal_eval(row["track_ids"])
        start_time = int(row["start_timepoint"])
        warp_path = ast.literal_eval(row["warp_path"])

        # Get image stack for this lineage
        stack = image_stacks[i]

        # Get the lineage embeddings
        lineage_embeddings = embeddings_dataset.sel(
            sample=(fov_name, track_ids)
        ).features.values

        # Create properly formatted trajectories based on mode
        if mode == "unaligned":
            # Extract the unaligned window
            traj_window = lineage_embeddings[
                start_time : start_time + len(reference_pattern)
            ]

            # Pad or trim to match reference length
            if len(traj_window) < len(reference_pattern):
                pad_length = len(reference_pattern) - len(traj_window)
                padding = np.zeros((pad_length, traj_window.shape[1]))
                traj_window = np.vstack((traj_window, padding))
            elif len(traj_window) > len(reference_pattern):
                traj_window = traj_window[: len(reference_pattern)]

        elif mode == "aligned":
            # Create aligned embeddings using the warping path
            traj_window = np.zeros_like(reference_pattern)

            # Map each reference timepoint to the corresponding lineage timepoint
            for ref_idx, query_idx in warp_path:
                lineage_idx = int(start_time + query_idx)
                if 0 <= lineage_idx < len(lineage_embeddings):
                    traj_window[ref_idx] = lineage_embeddings[lineage_idx]

            # Fill in any missing values for the aligned embeddings
            ref_indices_in_path = set(i for i, _ in warp_path)
            for ref_idx in range(len(reference_pattern)):
                if ref_idx not in ref_indices_in_path and ref_indices_in_path:
                    closest_ref_idx = min(
                        ref_indices_in_path, key=lambda x: abs(x - ref_idx)
                    )
                    closest_matches = [
                        (i, q) for i, q in warp_path if i == closest_ref_idx
                    ]
                    if closest_matches:
                        closest_query_idx = closest_matches[0][1]
                        lineage_idx = int(start_time + closest_query_idx)
                        if 0 <= lineage_idx < len(lineage_embeddings):
                            traj_window[ref_idx] = lineage_embeddings[lineage_idx]

        # Create axes for this lineage's row
        ax_phase = fig.add_subplot(gs[i, 0])
        ax_gfp = fig.add_subplot(gs[i, 1])
        ax_mcherry = fig.add_subplot(gs[i, 2])
        ax_dim0 = fig.add_subplot(gs[i, 3])
        ax_dim1 = fig.add_subplot(gs[i, 4])

        all_axes.append([ax_phase, ax_gfp, ax_mcherry, ax_dim0, ax_dim1])

        # Set background color for all axes
        for ax in [ax_phase, ax_gfp, ax_mcherry, ax_dim0, ax_dim1]:
            ax.set_facecolor("black")

        # Get the initial images (frame 0)
        init_phase = stack[0, 0, z_slice, :]
        init_gfp = stack[0, 1, z_slice, :]
        init_mcherry = stack[0, 2, z_slice, :]

        # Display phase image (placeholder for animation)
        phase_img = ax_phase.imshow(
            init_phase, cmap="gray", vmin=phase_limits[0], vmax=phase_limits[1]
        )
        # ax_phase.set_title(
        #     f"Phase - {fov_name}\nTrack {track_ids[0]}", color="white", fontsize=10
        # )
        ax_phase.axis("off")

        # Create RGB image for GFP (green channel)
        init_gfp_rgb = np.zeros((*init_gfp.shape, 3))
        init_gfp_rgb[:, :, 1] = safe_norm(gfp_norm, init_gfp)  # Green channel
        gfp_img = ax_gfp.imshow(init_gfp_rgb)
        # ax_gfp.set_title("GFP", color="white")
        ax_gfp.axis("off")

        # Create RGB image for mCherry (magenta = red + blue)
        init_mcherry_rgb = np.zeros((*init_mcherry.shape, 3))
        init_mcherry_rgb[:, :, 0] = safe_norm(
            mcherry_norm, init_mcherry
        )  # Red component
        init_mcherry_rgb[:, :, 2] = safe_norm(
            mcherry_norm, init_mcherry
        )  # Blue component
        mcherry_img = ax_mcherry.imshow(init_mcherry_rgb)
        # ax_mcherry.set_title("mCherry", color="white")
        ax_mcherry.axis("off")

        # Store image objects for updating in animation
        image_objs.append([phase_img, gfp_img, mcherry_img])

        # Plot dimension 0 trajectory
        time_points = np.arange(len(reference_pattern))
        ax_dim0.plot(
            time_points, reference_pattern[:, 0], "w-", linewidth=2.5, label="Reference"
        )
        ax_dim0.plot(
            time_points,
            traj_window[:, 0],
            "c--" if mode == "unaligned" else "r-",
            linewidth=1.5,
            label=mode.capitalize(),
        )

        # Add point marker for current position (will be updated in animation)
        (point_dim0,) = ax_dim0.plot(
            [0],
            [traj_window[0, 0]],
            "co" if mode == "unaligned" else "ro",
            markersize=10,
        )

        ax_dim0.set_title(f"Dimension 0", color="white")
        ax_dim0.set_xlabel("Time Point", color="white")
        ax_dim0.set_ylabel("Embedding Value", color="white")
        ax_dim0.tick_params(colors="white")
        ax_dim0.legend(facecolor="black", edgecolor="white")
        ax_dim0.grid(True, alpha=0.3, color="gray")

        # Plot dimension 1 trajectory
        ax_dim1.plot(
            time_points, reference_pattern[:, 1], "w-", linewidth=2.5, label="Reference"
        )
        ax_dim1.plot(
            time_points,
            traj_window[:, 1],
            "c--" if mode == "unaligned" else "r-",
            linewidth=1.5,
            label=mode.capitalize(),
        )

        # Add point marker for current position (will be updated in animation)
        (point_dim1,) = ax_dim1.plot(
            [0],
            [traj_window[0, 1]],
            "co" if mode == "unaligned" else "ro",
            markersize=10,
        )

        ax_dim1.set_title(f"Dimension 1", color="white")
        ax_dim1.set_xlabel("Time Point", color="white")
        ax_dim1.set_ylabel("Embedding Value", color="white")
        ax_dim1.tick_params(colors="white")
        ax_dim1.legend(facecolor="black", edgecolor="white")
        ax_dim1.grid(True, alpha=0.3, color="gray")

        # Store trajectory point objects for animation updates
        trajectory_points.append([point_dim0, point_dim1, traj_window])

    # Add main title with frame counter only if show_title is True
    title = None
    if show_title:
        title = plt.suptitle(
            f"{mode.capitalize()} Trajectories - Multiple Lineages (Frame: 0/{max_frames-1})",
            fontsize=16,
            color="white",
            y=0.98,
        )

    # Adjust figure margins - slightly different depending on whether title is shown
    if show_title:
        fig.subplots_adjust(
            left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.05, hspace=0.3
        )
    else:
        fig.subplots_adjust(
            left=0.02, right=0.98, bottom=0.05, top=0.98, wspace=0.05, hspace=0.3
        )

    # Create the animation function
    def animate(frame):
        # Update title with frame number if title is shown
        updated_artists = []
        if show_title:
            title.set_text(
                f"{mode.capitalize()} Trajectories - Multiple Lineages (Frame: {frame}/{max_frames-1})"
            )
            updated_artists = [title]

        # Update each lineage's data
        for i in range(num_lineages):
            # Get the images for this frame
            stack = image_stacks[i]

            # Make sure we don't go out of bounds
            valid_frame = min(frame, stack.shape[0] - 1)

            # Get the current frame's images
            phase_img = stack[valid_frame, 0, z_slice, :]
            gfp_img = stack[valid_frame, 1, z_slice, :]
            mcherry_img = stack[valid_frame, 2, z_slice, :]

            # Create RGB image for GFP
            gfp_rgb = np.zeros((*gfp_img.shape, 3))
            gfp_rgb[:, :, 1] = safe_norm(gfp_norm, gfp_img)  # Green channel

            # Create RGB image for mCherry
            mcherry_rgb = np.zeros((*mcherry_img.shape, 3))
            mcherry_rgb[:, :, 0] = safe_norm(mcherry_norm, mcherry_img)  # Red component
            mcherry_rgb[:, :, 2] = safe_norm(
                mcherry_norm, mcherry_img
            )  # Blue component

            # Update the images
            phase_obj, gfp_obj, mcherry_obj = image_objs[i]
            phase_obj.set_array(phase_img)
            gfp_obj.set_array(gfp_rgb)
            mcherry_obj.set_array(mcherry_rgb)

            # Update trajectory points
            point_dim0, point_dim1, traj_data = trajectory_points[i]
            point_dim0.set_data([frame], [traj_data[frame, 0]])
            point_dim1.set_data([frame], [traj_data[frame, 1]])

            # Add all updated artists
            updated_artists.extend(
                [phase_obj, gfp_obj, mcherry_obj, point_dim0, point_dim1]
            )

        return updated_artists

    # Create animation
    ani = animation.FuncAnimation(
        fig, animate, frames=max_frames, interval=1000 / fps, blit=True
    )

    # Save the animation
    ani.save(
        output_path,
        writer="ffmpeg",
        fps=fps,
        dpi=150,
        extra_args=["-vcodec", "libx264"],
    )

    plt.close(fig)

    return output_path


# %%
# Create video files for both unaligned and aligned trajectories
if len(all_unaligned_stacks) > 0 and len(all_aligned_stacks) > 0:
    # Create directory for output
    save_path = output_save_dir / f"multiple_lineages_videos"
    save_path.mkdir(parents=True, exist_ok=True)

    # Create unaligned video
    unaligned_video_path = save_path / f"20241107_SEC61B_unaligned_trajectories.mp4"
    create_multiple_lineages_video(
        reference_pattern,
        top_n_aligned_cells,
        embeddings_dataset,
        output_path=unaligned_video_path,
        mode="unaligned",
        num_lineages=3,
        fps=5,
        show_title=False,  # Set to False to remove the title
    )
    print(f"Created video files at {unaligned_video_path}")

    # Create aligned video
    aligned_video_path = save_path / f"20241107_SEC61B_aligned_trajectories.mp4"
    create_multiple_lineages_video(
        reference_pattern,
        top_n_aligned_cells,
        embeddings_dataset,
        output_path=aligned_video_path,
        mode="aligned",
        num_lineages=3,
        fps=5,
        show_title=False,  # Set to False to remove the title
    )
    print(f"Created video files at {aligned_video_path}")

# %%
# Compute the PCA
n_components = 8
_, _, embeddings_PCA_df = compute_pca(embeddings_dataset, n_components=n_components)


# %%
def create_pca_trajectory_video(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_PCA_df: pd.DataFrame,
    all_unaligned_stacks: np.ndarray,
    all_aligned_stacks: np.ndarray,
    output_path: Path,
    lineage_idx: int = 0,
    fps: int = 5,
):
    """
    Create a video comparing unaligned vs aligned trajectories with a moving point over time,
    using pre-computed PCA components (PC1 and PC2).

    Args:
        reference_pattern: The reference pattern embeddings
        top_aligned_cells: DataFrame with alignment information
        embeddings_PCA_df: DataFrame with pre-computed PCA components
        all_unaligned_stacks: Array of unaligned image stacks
        all_aligned_stacks: Array of aligned image stacks
        output_path: Path to save the video .mp4
        lineage_idx: Index of the lineage to visualize (default: 0)
        fps: Frames per second for the video (default: 5)
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # Set dark theme
    plt.style.use("dark_background")

    # Get the row for the selected lineage
    row = top_aligned_cells.iloc[lineage_idx]
    fov_name = row["fov_name"]
    track_ids = ast.literal_eval(row["track_ids"])
    warp_path = ast.literal_eval(row["warp_path"])
    start_time = int(row["start_timepoint"])

    # Get the selected stacks
    unaligned_stack = all_unaligned_stacks[lineage_idx]
    aligned_stack = all_aligned_stacks[lineage_idx]

    # Filter PCA data for the reference lineage
    reference_fov = "/C/2/001000"  # Based on variables in script
    reference_track_id = 129  # Based on variables in script

    reference_pca = embeddings_PCA_df[
        (embeddings_PCA_df["fov_name"] == reference_fov)
        & (embeddings_PCA_df["track_id"] == reference_track_id)
    ].sort_values(by="t")

    # Get reference timepoints [8, 70] as defined in the script
    reference_start = 8
    reference_end = 70
    reference_pca = reference_pca[
        (reference_pca["t"] >= reference_start) & (reference_pca["t"] < reference_end)
    ].reset_index(drop=True)

    # Get target lineage PCA data
    lineage_pca = (
        embeddings_PCA_df[
            (embeddings_PCA_df["fov_name"] == fov_name)
            & (embeddings_PCA_df["track_id"].isin(track_ids))
        ]
        .sort_values(by="t")
        .reset_index(drop=True)
    )

    # Extract the unaligned window
    unaligned_pca = lineage_pca.iloc[
        start_time : start_time + len(reference_pattern)
    ].reset_index(drop=True)

    # If unaligned_pca is shorter than reference_pattern, pad with last value
    if len(unaligned_pca) < len(reference_pattern):
        last_row = unaligned_pca.iloc[-1].copy()
        for i in range(len(reference_pattern) - len(unaligned_pca)):
            unaligned_pca = pd.concat(
                [unaligned_pca, pd.DataFrame([last_row])], ignore_index=True
            )

    # Create aligned PCA data using the warping path
    aligned_pca = pd.DataFrame(columns=reference_pca.columns)

    # Map each reference timepoint to the corresponding lineage timepoint
    for ref_idx, query_idx in warp_path:
        lineage_idx = int(start_time + query_idx)
        if 0 <= lineage_idx < len(lineage_pca):
            # Add a row to aligned_pca at position ref_idx
            if ref_idx >= len(aligned_pca):
                # If ref_idx is beyond current length, extend aligned_pca
                missing_rows = ref_idx - len(aligned_pca)
                if missing_rows > 0:
                    # Add empty rows
                    empty_rows = pd.DataFrame(
                        [pd.Series(dtype="float64")] * missing_rows,
                        columns=aligned_pca.columns,
                    )
                    aligned_pca = pd.concat(
                        [aligned_pca, empty_rows], ignore_index=True
                    )
                # Add the mapped row
                aligned_pca = pd.concat(
                    [aligned_pca, pd.DataFrame([lineage_pca.iloc[lineage_idx]])],
                    ignore_index=True,
                )
            else:
                # Replace existing row
                aligned_pca.iloc[ref_idx] = lineage_pca.iloc[lineage_idx]

    # Fill any remaining gaps in aligned_pca with nearest values
    aligned_pca = aligned_pca.fillna(method="ffill").fillna(method="bfill")

    # Ensure all dataframes have same length as reference pattern
    reference_pca = reference_pca.iloc[: len(reference_pattern)].reset_index(drop=True)
    unaligned_pca = unaligned_pca.iloc[: len(reference_pattern)].reset_index(drop=True)
    aligned_pca = aligned_pca.iloc[: len(reference_pattern)].reset_index(drop=True)

    # Create the figure with a 2x5 grid layout
    fig = plt.figure(figsize=(22, 8), facecolor="black")

    # Define the layout of subplots with reduced spacing and width ratios for rectangular dimension plots
    gs = fig.add_gridspec(2, 4, wspace=0.05, hspace=0.1, width_ratios=[1, 1, 2, 2])

    # Create the subplots for unaligned row (top row)
    ax_unaligned_phase = fig.add_subplot(gs[0, 0])  # Phase
    ax_unaligned_gfp = fig.add_subplot(gs[0, 1])  # GFP
    ax_unaligned_mcherry = fig.add_subplot(gs[0, 2])  # mCherry
    ax_pc1 = fig.add_subplot(gs[0, 3])  # PC1 plot
    ax_pc2 = fig.add_subplot(gs[1, 3])  # PC2 plot

    # Create the subplots for aligned row (bottom row)
    ax_aligned_phase = fig.add_subplot(gs[1, 0])  # Phase
    ax_aligned_gfp = fig.add_subplot(gs[1, 1])  # GFP
    ax_aligned_mcherry = fig.add_subplot(gs[1, 2])  # mCherry

    # Set background color for all axes
    for ax in [
        ax_unaligned_phase,
        ax_unaligned_gfp,
        ax_unaligned_mcherry,
        ax_aligned_phase,
        ax_aligned_gfp,
        ax_aligned_mcherry,
        ax_pc1,
        ax_pc2,
    ]:
        ax.set_facecolor("black")

    # Select z-slice for images
    z_slice = 15

    # Time points for the plots
    time_points = np.arange(len(reference_pattern))

    # Define contrast limits for each channel
    phase_limits = (-0.74, 0.4)
    gfp_limits = (106, 215)
    mcherry_limits = (106, 190)

    # Get the initial images
    unaligned_phase_init = unaligned_stack[0, 0, z_slice, :]
    unaligned_gfp_init = unaligned_stack[0, 1, z_slice, :]
    unaligned_mcherry_init = unaligned_stack[0, 2, z_slice, :]

    aligned_phase_init = aligned_stack[0, 0, z_slice, :]
    aligned_gfp_init = aligned_stack[0, 1, z_slice, :]
    aligned_mcherry_init = aligned_stack[0, 2, z_slice, :]

    # Create normalized data for proper visualization
    phase_norm = Normalize(vmin=phase_limits[0], vmax=phase_limits[1])
    gfp_norm = Normalize(vmin=gfp_limits[0], vmax=gfp_limits[1])
    mcherry_norm = Normalize(vmin=mcherry_limits[0], vmax=mcherry_limits[1])

    # Function to safely normalize values to [0,1] range
    def safe_norm(norm_func, data):
        normalized = norm_func(data)
        return np.clip(normalized, 0, 1)

    # Plot the initial images for phase (grayscale)
    unaligned_phase_img = ax_unaligned_phase.imshow(
        unaligned_phase_init, cmap="gray", vmin=phase_limits[0], vmax=phase_limits[1]
    )
    ax_unaligned_phase.set_title(f"Unaligned Phase", color="white")
    ax_unaligned_phase.axis("off")

    aligned_phase_img = ax_aligned_phase.imshow(
        aligned_phase_init, cmap="gray", vmin=phase_limits[0], vmax=phase_limits[1]
    )
    ax_aligned_phase.set_title(f"Aligned Phase", color="white")
    ax_aligned_phase.axis("off")

    # Create RGB images for GFP (green channel)
    unaligned_gfp_rgb = np.zeros((*unaligned_gfp_init.shape, 3))
    unaligned_gfp_rgb[:, :, 1] = safe_norm(
        gfp_norm, unaligned_gfp_init
    )  # Green channel

    aligned_gfp_rgb = np.zeros((*aligned_gfp_init.shape, 3))
    aligned_gfp_rgb[:, :, 1] = safe_norm(gfp_norm, aligned_gfp_init)  # Green channel

    # Create RGB images for mCherry (magenta channel = red + blue)
    unaligned_mcherry_rgb = np.zeros((*unaligned_mcherry_init.shape, 3))
    unaligned_mcherry_rgb[:, :, 0] = safe_norm(
        mcherry_norm, unaligned_mcherry_init
    )  # Red component
    unaligned_mcherry_rgb[:, :, 2] = safe_norm(
        mcherry_norm, unaligned_mcherry_init
    )  # Blue component

    aligned_mcherry_rgb = np.zeros((*aligned_mcherry_init.shape, 3))
    aligned_mcherry_rgb[:, :, 0] = safe_norm(
        mcherry_norm, aligned_mcherry_init
    )  # Red component
    aligned_mcherry_rgb[:, :, 2] = safe_norm(
        mcherry_norm, aligned_mcherry_init
    )  # Blue component

    # Plot the RGB fluorescence images
    unaligned_gfp_img = ax_unaligned_gfp.imshow(unaligned_gfp_rgb)
    ax_unaligned_gfp.set_title(f"Unaligned GFP", color="white")
    ax_unaligned_gfp.axis("off")

    aligned_gfp_img = ax_aligned_gfp.imshow(aligned_gfp_rgb)
    ax_aligned_gfp.set_title(f"Aligned GFP", color="white")
    ax_aligned_gfp.axis("off")

    unaligned_mcherry_img = ax_unaligned_mcherry.imshow(unaligned_mcherry_rgb)
    ax_unaligned_mcherry.set_title(f"Unaligned mCherry", color="white")
    ax_unaligned_mcherry.axis("off")

    aligned_mcherry_img = ax_aligned_mcherry.imshow(aligned_mcherry_rgb)
    ax_aligned_mcherry.set_title(f"Aligned mCherry", color="white")
    ax_aligned_mcherry.axis("off")

    # Plot all trajectories using PCA
    ax_pc1.plot(
        time_points,
        reference_pca["PCA1"].values,
        "w-",
        linewidth=2.5,
        label="Reference",
    )
    ax_pc1.plot(
        time_points,
        unaligned_pca["PCA1"].values,
        "c--",
        linewidth=1.5,
        label="Unaligned",
    )
    ax_pc1.plot(
        time_points, aligned_pca["PCA1"].values, "r-", linewidth=1.5, label="Aligned"
    )
    ax_pc1.set_title(f"PC1 - Lineage ({fov_name})", color="white")
    ax_pc1.set_xlabel("Reference Time Point", color="white")
    ax_pc1.set_ylabel("PC1", color="white")
    ax_pc1.tick_params(colors="white")
    ax_pc1.legend(facecolor="black", edgecolor="white")
    ax_pc1.grid(True, alpha=0.3, color="gray")

    ax_pc2.plot(
        time_points,
        reference_pca["PCA2"].values,
        "w-",
        linewidth=2.5,
        label="Reference",
    )
    ax_pc2.plot(
        time_points,
        unaligned_pca["PCA2"].values,
        "c--",
        linewidth=1.5,
        label="Unaligned",
    )
    ax_pc2.plot(
        time_points, aligned_pca["PCA2"].values, "r-", linewidth=1.5, label="Aligned"
    )
    ax_pc2.set_title(f"PC2 - Lineage ({fov_name})", color="white")
    ax_pc2.set_xlabel("Reference Time Point", color="white")
    ax_pc2.set_ylabel("PC2", color="white")
    ax_pc2.tick_params(colors="white")
    ax_pc2.legend(facecolor="black", edgecolor="white")
    ax_pc2.grid(True, alpha=0.3, color="gray")

    # Add moving points to the trajectory plots
    (point_ref_pc1,) = ax_pc1.plot(
        0, reference_pca["PCA1"].iloc[0], "wo", markersize=10
    )
    (point_unaligned_pc1,) = ax_pc1.plot(
        0, unaligned_pca["PCA1"].iloc[0], "co", markersize=10
    )
    (point_aligned_pc1,) = ax_pc1.plot(
        0, aligned_pca["PCA1"].iloc[0], "ro", markersize=10
    )

    (point_ref_pc2,) = ax_pc2.plot(
        0, reference_pca["PCA2"].iloc[0], "wo", markersize=10
    )
    (point_unaligned_pc2,) = ax_pc2.plot(
        0, unaligned_pca["PCA2"].iloc[0], "co", markersize=10
    )
    (point_aligned_pc2,) = ax_pc2.plot(
        0, aligned_pca["PCA2"].iloc[0], "ro", markersize=10
    )

    # Instead of tight_layout, manually adjust the figure margins
    fig.subplots_adjust(
        left=0.02, right=0.98, bottom=0.05, top=0.90, wspace=0.05, hspace=0.3
    )

    # Create the animation function
    def animate(frame):
        # Get current frame's images
        unaligned_phase = unaligned_stack[frame, 0, z_slice, :]
        unaligned_gfp = unaligned_stack[frame, 1, z_slice, :]
        unaligned_mcherry = unaligned_stack[frame, 2, z_slice, :]

        aligned_phase = aligned_stack[frame, 0, z_slice, :]
        aligned_gfp = aligned_stack[frame, 1, z_slice, :]
        aligned_mcherry = aligned_stack[frame, 2, z_slice, :]

        # Create RGB images for the current frame
        # GFP (green channel)
        unaligned_gfp_rgb = np.zeros((*unaligned_gfp.shape, 3))
        unaligned_gfp_rgb[:, :, 1] = safe_norm(gfp_norm, unaligned_gfp)  # Green channel

        aligned_gfp_rgb = np.zeros((*aligned_gfp.shape, 3))
        aligned_gfp_rgb[:, :, 1] = safe_norm(gfp_norm, aligned_gfp)  # Green channel

        # mCherry (magenta channel = red + blue)
        unaligned_mcherry_rgb = np.zeros((*unaligned_mcherry.shape, 3))
        unaligned_mcherry_rgb[:, :, 0] = safe_norm(
            mcherry_norm, unaligned_mcherry
        )  # Red component
        unaligned_mcherry_rgb[:, :, 2] = safe_norm(
            mcherry_norm, unaligned_mcherry
        )  # Blue component

        aligned_mcherry_rgb = np.zeros((*aligned_mcherry.shape, 3))
        aligned_mcherry_rgb[:, :, 0] = safe_norm(
            mcherry_norm, aligned_mcherry
        )  # Red component
        aligned_mcherry_rgb[:, :, 2] = safe_norm(
            mcherry_norm, aligned_mcherry
        )  # Blue component

        # Update the images
        unaligned_phase_img.set_array(unaligned_phase)
        unaligned_gfp_img.set_array(unaligned_gfp_rgb)
        unaligned_mcherry_img.set_array(unaligned_mcherry_rgb)

        aligned_phase_img.set_array(aligned_phase)
        aligned_gfp_img.set_array(aligned_gfp_rgb)
        aligned_mcherry_img.set_array(aligned_mcherry_rgb)

        # Update the trajectory points - must use sequences not single values
        point_ref_pc1.set_data([frame], [reference_pca["PCA1"].iloc[frame]])
        point_unaligned_pc1.set_data([frame], [unaligned_pca["PCA1"].iloc[frame]])
        point_aligned_pc1.set_data([frame], [aligned_pca["PCA1"].iloc[frame]])

        point_ref_pc2.set_data([frame], [reference_pca["PCA2"].iloc[frame]])
        point_unaligned_pc2.set_data([frame], [unaligned_pca["PCA2"].iloc[frame]])
        point_aligned_pc2.set_data([frame], [aligned_pca["PCA2"].iloc[frame]])

        # Return all artists that were updated
        return [
            unaligned_phase_img,
            unaligned_gfp_img,
            unaligned_mcherry_img,
            aligned_phase_img,
            aligned_gfp_img,
            aligned_mcherry_img,
            point_ref_pc1,
            point_unaligned_pc1,
            point_aligned_pc1,
            point_ref_pc2,
            point_unaligned_pc2,
            point_aligned_pc2,
        ]

    # Create the animation
    ani = animation.FuncAnimation(
        fig, animate, frames=len(reference_pattern), interval=1000 / fps, blit=True
    )

    # Save the animation
    ani.save(
        output_path,
        writer="ffmpeg",
        fps=fps,
        dpi=300,
        extra_args=["-vcodec", "libx264"],
    )

    logger.info(f"PCA trajectory comparison video saved to {output_path}")
    return output_path


# %%
# Create PCA trajectory comparison videos
if len(all_aligned_stacks) > 0 and len(all_unaligned_stacks) > 0:
    for idx, row in top_n_aligned_cells.reset_index().iterrows():
        fov_name = row["fov_name"][1:].replace("/", "_")
        track_ids = row["track_ids"]
        save_path = output_save_dir / "heterogeneity_vs_alignment_pca"
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = (
            save_path
            / f"20241107_SEC61B_pca_df_fov_{fov_name}_track_{track_ids[0]}.mp4"
        )

        video_path = create_pca_trajectory_video(
            reference_pattern,
            top_n_aligned_cells,
            embeddings_PCA_df,
            all_unaligned_stacks,
            all_aligned_stacks,
            output_path=save_path,
            lineage_idx=idx,
            fps=5,
        )


# %%
def create_multiple_lineages_pca_video(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_PCA_df: pd.DataFrame,
    image_stacks: np.ndarray,  # Either all_unaligned_stacks or all_aligned_stacks
    output_path: Path,
    mode: str = "unaligned",  # "unaligned" or "aligned"
    num_lineages: int = 3,
    z_slice: int = 15,
    fps: int = 5,
    show_title: bool = False,  # Parameter to control title visibility
):
    """
    Create a video showing multiple lineages with phase/GFP/mCherry side by side,
    followed by PC1 and PC2 plots for either unaligned or aligned trajectories.
    Uses pre-computed PCA data directly.

    Args:
        reference_pattern: The reference pattern embeddings
        top_aligned_cells: DataFrame with alignment information
        embeddings_PCA_df: DataFrame with pre-computed PCA components
        image_stacks: Array of image stacks (unaligned or aligned)
        output_path: Path to save the video file
        mode: "unaligned" or "aligned" to determine which trajectories to display
        num_lineages: Number of lineages to display (default: 3)
        z_slice: Z-slice to use for images (default: 15)
        fps: Frames per second for the video (default: 5)
        show_title: Whether to show the main title with frame counter (default: False)
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # Set dark theme
    plt.style.use("dark_background")

    # Limit to requested number of lineages
    num_lineages = min(num_lineages, len(top_aligned_cells), len(image_stacks))

    # Create figure with rows for each lineage
    fig = plt.figure(figsize=(20, 4 * num_lineages), facecolor="black")

    # Define column ratios - more space for images, increased spacing between images and plots
    gs = fig.add_gridspec(
        num_lineages, 5, wspace=0.2, hspace=0.3, width_ratios=[1.5, 1.5, 1.5, 2, 2]
    )

    # Define contrast limits for each channel
    phase_limits = (-0.74, 0.4)
    gfp_limits = (106, 215)
    mcherry_limits = (106, 190)

    # Function to safely normalize values to [0,1] range
    def safe_norm(norm_func, data):
        normalized = norm_func(data)
        return np.clip(normalized, 0, 1)

    # Create normalizers for each channel
    phase_norm = Normalize(vmin=phase_limits[0], vmax=phase_limits[1])
    gfp_norm = Normalize(vmin=gfp_limits[0], vmax=gfp_limits[1])
    mcherry_norm = Normalize(vmin=mcherry_limits[0], vmax=mcherry_limits[1])

    # Get reference lineage PCA data
    reference_fov = "/C/2/001000"  # Based on variables in script
    reference_track_id = 129  # Based on variables in script

    reference_pca = embeddings_PCA_df[
        (embeddings_PCA_df["fov_name"] == reference_fov)
        & (embeddings_PCA_df["track_id"] == reference_track_id)
    ].sort_values(by="t")

    # Get reference timepoints [8, 70] as defined in the script
    reference_start = 8
    reference_end = 70
    reference_pca = reference_pca[
        (reference_pca["t"] >= reference_start) & (reference_pca["t"] < reference_end)
    ].reset_index(drop=True)

    # Collect all trajectory data for visualization
    all_lineage_data = []
    max_frames = len(reference_pattern)

    # Process each lineage to get PCA trajectories
    for i in range(num_lineages):
        # Get data for this lineage
        row = top_aligned_cells.iloc[i]
        fov_name = row["fov_name"]
        track_ids = ast.literal_eval(row["track_ids"])
        start_time = int(row["start_timepoint"])
        warp_path = ast.literal_eval(row["warp_path"])

        # Get lineage PCA data
        lineage_pca = (
            embeddings_PCA_df[
                (embeddings_PCA_df["fov_name"] == fov_name)
                & (embeddings_PCA_df["track_id"].isin(track_ids))
            ]
            .sort_values(by="t")
            .reset_index(drop=True)
        )

        # Create trajectory based on mode
        if mode == "unaligned":
            # Extract unaligned window
            traj_pca = lineage_pca.iloc[
                start_time : start_time + len(reference_pattern)
            ].reset_index(drop=True)

            # Pad if needed
            if len(traj_pca) < len(reference_pattern):
                last_row = traj_pca.iloc[-1].copy()
                for _ in range(len(reference_pattern) - len(traj_pca)):
                    traj_pca = pd.concat(
                        [traj_pca, pd.DataFrame([last_row])], ignore_index=True
                    )

            # Trim if needed
            traj_pca = traj_pca.iloc[: len(reference_pattern)].reset_index(drop=True)

        else:  # aligned mode
            # Create aligned trajectory using warping path
            traj_pca = pd.DataFrame(columns=lineage_pca.columns)

            # Map each reference timepoint to the corresponding lineage timepoint
            for ref_idx, query_idx in warp_path:
                lineage_idx = int(start_time + query_idx)
                if 0 <= lineage_idx < len(lineage_pca):
                    # Add a row to traj_pca at position ref_idx
                    if ref_idx >= len(traj_pca):
                        # If ref_idx is beyond current length, extend traj_pca
                        missing_rows = ref_idx - len(traj_pca)
                        if missing_rows > 0:
                            # Add empty rows
                            empty_rows = pd.DataFrame(
                                [pd.Series(dtype="float64")] * missing_rows,
                                columns=traj_pca.columns,
                            )
                            traj_pca = pd.concat(
                                [traj_pca, empty_rows], ignore_index=True
                            )
                        # Add the mapped row
                        traj_pca = pd.concat(
                            [traj_pca, pd.DataFrame([lineage_pca.iloc[lineage_idx]])],
                            ignore_index=True,
                        )
                    else:
                        # Replace existing row
                        traj_pca.iloc[ref_idx] = lineage_pca.iloc[lineage_idx]

            # Fill any remaining gaps with nearest values
            traj_pca = traj_pca.fillna(method="ffill").fillna(method="bfill")

            # Ensure correct length
            traj_pca = traj_pca.iloc[: len(reference_pattern)].reset_index(drop=True)

            # If still too short, pad with last value
            if len(traj_pca) < len(reference_pattern):
                last_row = (
                    traj_pca.iloc[-1].copy()
                    if not traj_pca.empty
                    else pd.Series(dtype="float64")
                )
                for _ in range(len(reference_pattern) - len(traj_pca)):
                    traj_pca = pd.concat(
                        [traj_pca, pd.DataFrame([last_row])], ignore_index=True
                    )

        all_lineage_data.append(
            {"fov_name": fov_name, "track_ids": track_ids, "traj_pca": traj_pca}
        )

    # Ensure reference has correct length
    reference_pca = reference_pca.iloc[: len(reference_pattern)].reset_index(drop=True)

    # Store all axes and image objects for later updating
    all_axes = []
    image_objs = []
    trajectory_points = []

    # Process each lineage for visualization
    for i in range(num_lineages):
        lineage_data = all_lineage_data[i]
        fov_name = lineage_data["fov_name"]
        track_ids = lineage_data["track_ids"]
        traj_pca = lineage_data["traj_pca"]

        # Get image stack for this lineage
        stack = image_stacks[i]

        # Create axes for this lineage's row
        ax_phase = fig.add_subplot(gs[i, 0])
        ax_gfp = fig.add_subplot(gs[i, 1])
        ax_mcherry = fig.add_subplot(gs[i, 2])
        ax_pc1 = fig.add_subplot(gs[i, 3])
        ax_pc2 = fig.add_subplot(gs[i, 4])

        all_axes.append([ax_phase, ax_gfp, ax_mcherry, ax_pc1, ax_pc2])

        # Set background color for all axes
        for ax in [ax_phase, ax_gfp, ax_mcherry, ax_pc1, ax_pc2]:
            ax.set_facecolor("black")

        # Get the initial images (frame 0)
        init_phase = stack[0, 0, z_slice, :]
        init_gfp = stack[0, 1, z_slice, :]
        init_mcherry = stack[0, 2, z_slice, :]

        # Display phase image (placeholder for animation)
        phase_img = ax_phase.imshow(
            init_phase, cmap="gray", vmin=phase_limits[0], vmax=phase_limits[1]
        )
        ax_phase.set_title(
            f"Phase - {fov_name}\nTrack {track_ids[0]}", color="white", fontsize=10
        )
        ax_phase.axis("off")

        # Create RGB image for GFP (green channel)
        init_gfp_rgb = np.zeros((*init_gfp.shape, 3))
        init_gfp_rgb[:, :, 1] = safe_norm(gfp_norm, init_gfp)  # Green channel
        gfp_img = ax_gfp.imshow(init_gfp_rgb)
        ax_gfp.set_title("GFP", color="white")
        ax_gfp.axis("off")

        # Create RGB image for mCherry (magenta = red + blue)
        init_mcherry_rgb = np.zeros((*init_mcherry.shape, 3))
        init_mcherry_rgb[:, :, 0] = safe_norm(
            mcherry_norm, init_mcherry
        )  # Red component
        init_mcherry_rgb[:, :, 2] = safe_norm(
            mcherry_norm, init_mcherry
        )  # Blue component
        mcherry_img = ax_mcherry.imshow(init_mcherry_rgb)
        ax_mcherry.set_title("mCherry", color="white")
        ax_mcherry.axis("off")

        # Store image objects for updating in animation
        image_objs.append([phase_img, gfp_img, mcherry_img])

        # Plot dimension PC1 trajectory
        time_points = np.arange(len(reference_pattern))
        ax_pc1.plot(
            time_points,
            reference_pca["PCA1"].values,
            "w-",
            linewidth=2.5,
            label="Reference",
        )
        ax_pc1.plot(
            time_points,
            traj_pca["PCA1"].values,
            "c--" if mode == "unaligned" else "r-",
            linewidth=1.5,
            label=mode.capitalize(),
        )

        # Add point marker for current position (will be updated in animation)
        (point_pc1,) = ax_pc1.plot(
            [0],
            [traj_pca["PCA1"].iloc[0]],
            "co" if mode == "unaligned" else "ro",
            markersize=10,
        )

        ax_pc1.set_title(f"PC1", color="white")
        ax_pc1.set_xlabel("Time Point", color="white")
        ax_pc1.set_ylabel("PC1", color="white")
        ax_pc1.tick_params(colors="white")
        ax_pc1.legend(facecolor="black", edgecolor="white")
        ax_pc1.grid(True, alpha=0.3, color="gray")

        # Plot dimension PC2 trajectory
        ax_pc2.plot(
            time_points,
            reference_pca["PCA2"].values,
            "w-",
            linewidth=2.5,
            label="Reference",
        )
        ax_pc2.plot(
            time_points,
            traj_pca["PCA2"].values,
            "c--" if mode == "unaligned" else "r-",
            linewidth=1.5,
            label=mode.capitalize(),
        )

        # Add point marker for current position (will be updated in animation)
        (point_pc2,) = ax_pc2.plot(
            [0],
            [traj_pca["PCA2"].iloc[0]],
            "co" if mode == "unaligned" else "ro",
            markersize=10,
        )

        ax_pc2.set_title(f"PC2", color="white")
        ax_pc2.set_xlabel("Time Point", color="white")
        ax_pc2.set_ylabel("PC2", color="white")
        ax_pc2.tick_params(colors="white")
        ax_pc2.legend(facecolor="black", edgecolor="white")
        ax_pc2.grid(True, alpha=0.3, color="gray")

        # Store trajectory point objects for animation updates
        trajectory_points.append([point_pc1, point_pc2, traj_pca])

    # Add main title with frame counter only if show_title is True
    title = None
    if show_title:
        title = plt.suptitle(
            f"{mode.capitalize()} Trajectories (PCA) - Multiple Lineages (Frame: 0/{max_frames-1})",
            fontsize=16,
            color="white",
            y=0.98,
        )

    # Adjust figure margins
    if show_title:
        fig.subplots_adjust(
            left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.05, hspace=0.3
        )
    else:
        fig.subplots_adjust(
            left=0.02, right=0.98, bottom=0.05, top=0.98, wspace=0.05, hspace=0.3
        )

    # Create the animation function
    def animate(frame):
        # Update title with frame number if title is shown
        updated_artists = []
        if show_title:
            title.set_text(
                f"{mode.capitalize()} Trajectories (PCA) - Multiple Lineages (Frame: {frame}/{max_frames-1})"
            )
            updated_artists = [title]

        # Update each lineage's data
        for i in range(num_lineages):
            # Get the images for this frame
            stack = image_stacks[i]

            # Make sure we don't go out of bounds
            valid_frame = min(frame, stack.shape[0] - 1)

            # Get the current frame's images
            phase_img = stack[valid_frame, 0, z_slice, :]
            gfp_img = stack[valid_frame, 1, z_slice, :]
            mcherry_img = stack[valid_frame, 2, z_slice, :]

            # Create RGB image for GFP
            gfp_rgb = np.zeros((*gfp_img.shape, 3))
            gfp_rgb[:, :, 1] = safe_norm(gfp_norm, gfp_img)  # Green channel

            # Create RGB image for mCherry
            mcherry_rgb = np.zeros((*mcherry_img.shape, 3))
            mcherry_rgb[:, :, 0] = safe_norm(mcherry_norm, mcherry_img)  # Red component
            mcherry_rgb[:, :, 2] = safe_norm(
                mcherry_norm, mcherry_img
            )  # Blue component

            # Update the images
            phase_obj, gfp_obj, mcherry_obj = image_objs[i]
            phase_obj.set_array(phase_img)
            gfp_obj.set_array(gfp_rgb)
            mcherry_obj.set_array(mcherry_rgb)

            # Update trajectory points
            point_pc1, point_pc2, traj_pca = trajectory_points[i]
            point_pc1.set_data([frame], [traj_pca["PCA1"].iloc[frame]])
            point_pc2.set_data([frame], [traj_pca["PCA2"].iloc[frame]])

            # Add all updated artists
            updated_artists.extend(
                [phase_obj, gfp_obj, mcherry_obj, point_pc1, point_pc2]
            )

        return updated_artists

    # Create animation
    ani = animation.FuncAnimation(
        fig, animate, frames=max_frames, interval=1000 / fps, blit=True
    )

    # Save the animation
    ani.save(
        output_path,
        writer="ffmpeg",
        fps=fps,
        dpi=150,
        extra_args=["-vcodec", "libx264"],
    )

    plt.close(fig)
    logger.info(f"Multiple lineages PCA video saved to {output_path}")
    return output_path


# %%
# Create PCA multiple lineages videos using pre-computed PCA data
if len(all_unaligned_stacks) > 0 and len(all_aligned_stacks) > 0:
    # Create directory for output
    save_path = output_save_dir / f"multiple_lineages_pca_df_videos"
    save_path.mkdir(parents=True, exist_ok=True)

    # Create unaligned PCA video
    unaligned_pca_video_path = (
        save_path / f"20241107_SEC61B_unaligned_pca_df_trajectories.mp4"
    )

    _, _, pca_df = compute_pca(embeddings_dataset, n_components=8)
    create_multiple_lineages_pca_video(
        reference_pattern,
        top_n_aligned_cells,
        pca_df,
        output_path=unaligned_pca_video_path,
        mode="unaligned",
        num_lineages=3,
        fps=5,
        show_title=False,
    )
    print(f"Created PCA video file at {unaligned_pca_video_path}")

    # Create aligned PCA video
    aligned_pca_video_path = (
        save_path / f"20241107_SEC61B_aligned_pca_df_trajectories.mp4"
    )
    create_multiple_lineages_pca_video(
        reference_pattern,
        top_n_aligned_cells,
        pca_df,
        output_path=aligned_pca_video_path,
        mode="aligned",
        num_lineages=3,
        fps=5,
        show_title=False,
    )
    print(f"Created PCA video file at {aligned_pca_video_path}")


# %%
def plot_pc1_vs_pc2_trajectories(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_PCA_df: pd.DataFrame,
    output_path: Path,
    num_lineages: int = 3,
    plot_aligned: bool = True,
    plot_unaligned: bool = True,
):
    """
    Create a 2D visualization of PC1 vs PC2 for both aligned and unaligned trajectories.

    Args:
        reference_pattern: The reference pattern embeddings
        top_aligned_cells: DataFrame with alignment information
        embeddings_PCA_df: DataFrame with pre-computed PCA components
        output_path: Path to save the output figure
        num_lineages: Number of lineages to display (default: 3)
        plot_aligned: Whether to plot aligned trajectories (default: True)
        plot_unaligned: Whether to plot unaligned trajectories (default: True)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    import matplotlib.cm as cm

    # Set up figure with dark background
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="black")
    ax.set_facecolor("black")

    # Limit to requested number of lineages
    num_lineages = min(num_lineages, len(top_aligned_cells))

    # Get reference lineage PCA data
    reference_fov = "/C/2/001000"
    reference_track_id = 129

    reference_pca = embeddings_PCA_df[
        (embeddings_PCA_df["fov_name"] == reference_fov)
        & (embeddings_PCA_df["track_id"] == reference_track_id)
    ].sort_values(by="t")

    # Get reference timepoints [8, 70]
    reference_start = 8
    reference_end = 70
    reference_pca = reference_pca[
        (reference_pca["t"] >= reference_start) & (reference_pca["t"] < reference_end)
    ].reset_index(drop=True)

    # Ensure reference has correct length
    reference_pca = reference_pca.iloc[: len(reference_pattern)].reset_index(drop=True)

    # Plot reference trajectory
    ref_line = ax.plot(
        reference_pca["PCA1"],
        reference_pca["PCA2"],
        "w-",
        linewidth=3,
        label="Reference",
    )
    ref_color = to_rgba(ref_line[0].get_color())

    # Create a colormap for trajectory progression
    cmap = cm.plasma

    # Plot start and end points for reference
    ax.scatter(
        reference_pca["PCA1"].iloc[0],
        reference_pca["PCA2"].iloc[0],
        s=150,
        color="white",
        edgecolor="black",
        linewidth=1.5,
        marker="o",
        label="Start",
    )
    ax.scatter(
        reference_pca["PCA1"].iloc[-1],
        reference_pca["PCA2"].iloc[-1],
        s=150,
        color="white",
        edgecolor="black",
        linewidth=1.5,
        marker="s",
        label="End",
    )

    # Add color gradient to reference trajectory
    for i in range(len(reference_pca) - 1):
        progress = i / (len(reference_pca) - 1)
        color = cmap(progress)
        ax.plot(
            [reference_pca["PCA1"].iloc[i], reference_pca["PCA1"].iloc[i + 1]],
            [reference_pca["PCA2"].iloc[i], reference_pca["PCA2"].iloc[i + 1]],
            color=color,
            linewidth=5,
            alpha=0.8,
        )

    # Define colors for unaligned and aligned
    unaligned_color = "cyan"
    aligned_colors = ["red", "orange", "magenta"]

    # Process each lineage
    for i in range(num_lineages):
        # Get data for this lineage
        row = top_aligned_cells.iloc[i]
        fov_name = row["fov_name"]
        track_ids = ast.literal_eval(row["track_ids"])
        start_time = int(row["start_timepoint"])
        warp_path = ast.literal_eval(row["warp_path"])

        # Get lineage PCA data
        lineage_pca = (
            embeddings_PCA_df[
                (embeddings_PCA_df["fov_name"] == fov_name)
                & (embeddings_PCA_df["track_id"].isin(track_ids))
            ]
            .sort_values(by="t")
            .reset_index(drop=True)
        )

        # Plot unaligned trajectory
        if plot_unaligned:
            # Extract unaligned window
            unaligned_pca = lineage_pca.iloc[
                start_time : start_time + len(reference_pattern)
            ].reset_index(drop=True)

            # Pad if needed
            if len(unaligned_pca) < len(reference_pattern):
                last_row = unaligned_pca.iloc[-1].copy()
                for _ in range(len(reference_pattern) - len(unaligned_pca)):
                    unaligned_pca = pd.concat(
                        [unaligned_pca, pd.DataFrame([last_row])], ignore_index=True
                    )

            # Trim if needed
            unaligned_pca = unaligned_pca.iloc[: len(reference_pattern)].reset_index(
                drop=True
            )

            # Plot unaligned trajectory
            un_line = ax.plot(
                unaligned_pca["PCA1"],
                unaligned_pca["PCA2"],
                "--",
                color=unaligned_color,
                linewidth=1.5,
                alpha=0.7,
                label="Unaligned" if i == 0 else "",
            )

            # Plot unaligned start and end
            ax.scatter(
                unaligned_pca["PCA1"].iloc[0],
                unaligned_pca["PCA2"].iloc[0],
                s=80,
                color=unaligned_color,
                marker="o",
                alpha=0.7,
            )
            ax.scatter(
                unaligned_pca["PCA1"].iloc[-1],
                unaligned_pca["PCA2"].iloc[-1],
                s=80,
                color=unaligned_color,
                marker="s",
                alpha=0.7,
            )

            # Add progression points every few steps
            step = max(1, len(unaligned_pca) // 10)
            for j in range(0, len(unaligned_pca), step):
                progress = j / (len(unaligned_pca) - 1) if len(unaligned_pca) > 1 else 0
                ax.scatter(
                    unaligned_pca["PCA1"].iloc[j],
                    unaligned_pca["PCA2"].iloc[j],
                    s=30,
                    color=cmap(progress),
                    alpha=0.7,
                )

        # Plot aligned trajectory
        if plot_aligned:
            # Create aligned trajectory using warping path
            aligned_pca = pd.DataFrame(columns=lineage_pca.columns)

            # Map each reference timepoint to the corresponding lineage timepoint
            for ref_idx, query_idx in warp_path:
                lineage_idx = int(start_time + query_idx)
                if 0 <= lineage_idx < len(lineage_pca):
                    # Add a row to aligned_pca at position ref_idx
                    if ref_idx >= len(aligned_pca):
                        # If ref_idx is beyond current length, extend aligned_pca
                        missing_rows = ref_idx - len(aligned_pca)
                        if missing_rows > 0:
                            # Add empty rows
                            empty_rows = pd.DataFrame(
                                [pd.Series(dtype="float64")] * missing_rows,
                                columns=aligned_pca.columns,
                            )
                            aligned_pca = pd.concat(
                                [aligned_pca, empty_rows], ignore_index=True
                            )
                        # Add the mapped row
                        aligned_pca = pd.concat(
                            [
                                aligned_pca,
                                pd.DataFrame([lineage_pca.iloc[lineage_idx]]),
                            ],
                            ignore_index=True,
                        )
                    else:
                        # Replace existing row
                        aligned_pca.iloc[ref_idx] = lineage_pca.iloc[lineage_idx]

            # Fill any remaining gaps with nearest values
            aligned_pca = aligned_pca.fillna(method="ffill").fillna(method="bfill")

            # Ensure correct length
            aligned_pca = aligned_pca.iloc[: len(reference_pattern)].reset_index(
                drop=True
            )

            # If still too short, pad with last value
            if len(aligned_pca) < len(reference_pattern):
                last_row = (
                    aligned_pca.iloc[-1].copy()
                    if not aligned_pca.empty
                    else pd.Series(dtype="float64")
                )
                for _ in range(len(reference_pattern) - len(aligned_pca)):
                    aligned_pca = pd.concat(
                        [aligned_pca, pd.DataFrame([last_row])], ignore_index=True
                    )

            # Plot aligned trajectory
            aligned_color = aligned_colors[i % len(aligned_colors)]
            al_line = ax.plot(
                aligned_pca["PCA1"],
                aligned_pca["PCA2"],
                "-",
                color=aligned_color,
                linewidth=2.5,
                label=f"Aligned {i+1}",
            )

            # Plot aligned start and end
            ax.scatter(
                aligned_pca["PCA1"].iloc[0],
                aligned_pca["PCA2"].iloc[0],
                s=100,
                color=aligned_color,
                marker="o",
            )
            ax.scatter(
                aligned_pca["PCA1"].iloc[-1],
                aligned_pca["PCA2"].iloc[-1],
                s=100,
                color=aligned_color,
                marker="s",
            )

            # Add progression points every few steps
            step = max(1, len(aligned_pca) // 10)
            for j in range(0, len(aligned_pca), step):
                progress = j / (len(aligned_pca) - 1) if len(aligned_pca) > 1 else 0
                ax.scatter(
                    aligned_pca["PCA1"].iloc[j],
                    aligned_pca["PCA2"].iloc[j],
                    s=40,
                    color=cmap(progress),
                    alpha=0.8,
                )

    # Add colorbar to show progression
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Temporal Progression", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.outline.set_edgecolor("white")
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color="white")

    # Set labels and title
    ax.set_xlabel("PC1", color="white")
    ax.set_ylabel("PC2", color="white")
    ax.set_title("PC1 vs PC2 Trajectory Alignment", color="white", fontsize=16)

    # Style the plot
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2, color="gray")

    # Set legend
    ax.legend(loc="upper right", facecolor="black", edgecolor="white", framealpha=0.7)

    # Save figure
    plt.tight_layout()
    # plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # plt.close()

    logger.info(f"PC1 vs PC2 trajectory plot saved to {output_path}")
    return output_path


# %%
# Create PC1 vs PC2 trajectory visualizations
save_path = output_save_dir / "pca_trajectories"
save_path.mkdir(parents=True, exist_ok=True)

# Plot both aligned and unaligned trajectories together
plot_pc1_vs_pc2_trajectories(
    reference_pattern,
    top_n_aligned_cells,
    embeddings_PCA_df,
    output_path=save_path / "20241107_SEC61B_pc1_vs_pc2_trajectories.png",
    num_lineages=3,
    plot_aligned=True,
    plot_unaligned=True,
)

# Plot only aligned trajectories
plot_pc1_vs_pc2_trajectories(
    reference_pattern,
    top_n_aligned_cells,
    embeddings_PCA_df,
    output_path=save_path / "20241107_SEC61B_pc1_vs_pc2_aligned_only.png",
    num_lineages=3,
    plot_aligned=True,
    plot_unaligned=False,
)


# Create an animation of PC1 vs PC2 movement
def create_pc1_vs_pc2_animation(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_PCA_df: pd.DataFrame,
    output_path: Path,
    num_lineages: int = 3,
    fps: int = 10,
):
    """
    Create an animation of PC1 vs PC2 trajectories with markers moving along the paths.

    Args:
        reference_pattern: The reference pattern embeddings
        top_aligned_cells: DataFrame with alignment information
        embeddings_PCA_df: DataFrame with pre-computed PCA components
        output_path: Path to save the animation
        num_lineages: Number of lineages to visualize
        fps: Frames per second for the animation
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import to_rgba
    import matplotlib.cm as cm

    # Set up figure with dark background
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="black")
    ax.set_facecolor("black")

    # Limit to requested number of lineages
    num_lineages = min(num_lineages, len(top_aligned_cells))

    # Get reference lineage PCA data
    reference_fov = "/C/2/001000"
    reference_track_id = 129

    reference_pca = embeddings_PCA_df[
        (embeddings_PCA_df["fov_name"] == reference_fov)
        & (embeddings_PCA_df["track_id"] == reference_track_id)
    ].sort_values(by="t")

    # Get reference timepoints [8, 70]
    reference_start = 8
    reference_end = 70
    reference_pca = reference_pca[
        (reference_pca["t"] >= reference_start) & (reference_pca["t"] < reference_end)
    ].reset_index(drop=True)

    # Ensure reference has correct length
    reference_pca = reference_pca.iloc[: len(reference_pattern)].reset_index(drop=True)

    # Get unaligned and aligned trajectories for each lineage
    all_unaligned = []
    all_aligned = []

    # Process each lineage
    for i in range(num_lineages):
        # Get data for this lineage
        row = top_aligned_cells.iloc[i]
        fov_name = row["fov_name"]
        track_ids = ast.literal_eval(row["track_ids"])
        start_time = int(row["start_timepoint"])
        warp_path = ast.literal_eval(row["warp_path"])

        # Get lineage PCA data
        lineage_pca = (
            embeddings_PCA_df[
                (embeddings_PCA_df["fov_name"] == fov_name)
                & (embeddings_PCA_df["track_id"].isin(track_ids))
            ]
            .sort_values(by="t")
            .reset_index(drop=True)
        )

        # Extract unaligned window
        unaligned_pca = lineage_pca.iloc[
            start_time : start_time + len(reference_pattern)
        ].reset_index(drop=True)

        # Pad if needed
        if len(unaligned_pca) < len(reference_pattern):
            last_row = unaligned_pca.iloc[-1].copy()
            for _ in range(len(reference_pattern) - len(unaligned_pca)):
                unaligned_pca = pd.concat(
                    [unaligned_pca, pd.DataFrame([last_row])], ignore_index=True
                )

        # Trim if needed
        unaligned_pca = unaligned_pca.iloc[: len(reference_pattern)].reset_index(
            drop=True
        )
        all_unaligned.append(unaligned_pca)

        # Create aligned trajectory using warping path
        aligned_pca = pd.DataFrame(columns=lineage_pca.columns)

        # Map each reference timepoint to the corresponding lineage timepoint
        for ref_idx, query_idx in warp_path:
            lineage_idx = int(start_time + query_idx)
            if 0 <= lineage_idx < len(lineage_pca):
                # Add a row to aligned_pca at position ref_idx
                if ref_idx >= len(aligned_pca):
                    # If ref_idx is beyond current length, extend aligned_pca
                    missing_rows = ref_idx - len(aligned_pca)
                    if missing_rows > 0:
                        # Add empty rows
                        empty_rows = pd.DataFrame(
                            [pd.Series(dtype="float64")] * missing_rows,
                            columns=aligned_pca.columns,
                        )
                        aligned_pca = pd.concat(
                            [aligned_pca, empty_rows], ignore_index=True
                        )
                    # Add the mapped row
                    aligned_pca = pd.concat(
                        [aligned_pca, pd.DataFrame([lineage_pca.iloc[lineage_idx]])],
                        ignore_index=True,
                    )
                else:
                    # Replace existing row
                    aligned_pca.iloc[ref_idx] = lineage_pca.iloc[lineage_idx]

        # Fill any remaining gaps with nearest values
        aligned_pca = aligned_pca.fillna(method="ffill").fillna(method="bfill")

        # Ensure correct length
        aligned_pca = aligned_pca.iloc[: len(reference_pattern)].reset_index(drop=True)

        # If still too short, pad with last value
        if len(aligned_pca) < len(reference_pattern):
            last_row = (
                aligned_pca.iloc[-1].copy()
                if not aligned_pca.empty
                else pd.Series(dtype="float64")
            )
            for _ in range(len(reference_pattern) - len(aligned_pca)):
                aligned_pca = pd.concat(
                    [aligned_pca, pd.DataFrame([last_row])], ignore_index=True
                )

        all_aligned.append(aligned_pca)

    # Define colors for unaligned and aligned
    unaligned_color = "cyan"
    aligned_colors = ["red", "orange", "magenta"]

    # Plot all trajectories first (static)
    # Reference
    ax.plot(
        reference_pca["PCA1"],
        reference_pca["PCA2"],
        "w-",
        linewidth=2.5,
        alpha=0.8,
        label="Reference",
    )

    # Unaligned
    for i in range(num_lineages):
        ax.plot(
            all_unaligned[i]["PCA1"],
            all_unaligned[i]["PCA2"],
            "--",
            color=unaligned_color,
            linewidth=1.5,
            alpha=0.5,
            label="Unaligned" if i == 0 else "",
        )

    # Aligned
    for i in range(num_lineages):
        aligned_color = aligned_colors[i % len(aligned_colors)]
        ax.plot(
            all_aligned[i]["PCA1"],
            all_aligned[i]["PCA2"],
            "-",
            color=aligned_color,
            linewidth=2,
            alpha=0.7,
            label=f"Aligned {i+1}",
        )

    # Create moving points
    (ref_point,) = ax.plot([], [], "wo", markersize=12)
    unaligned_points = [
        ax.plot([], [], "co", markersize=10)[0] for _ in range(num_lineages)
    ]
    aligned_points = [
        ax.plot(
            [], [], "o", color=aligned_colors[i % len(aligned_colors)], markersize=10
        )[0]
        for i in range(num_lineages)
    ]

    # Add trails that show recent positions
    trail_length = 10
    ref_trail = [ax.plot([], [], "w-", alpha=0.5)[0] for _ in range(trail_length)]

    unaligned_trails = []
    for _ in range(num_lineages):
        unaligned_trails.append(
            [
                ax.plot([], [], "-", color=unaligned_color, alpha=0.3)[0]
                for _ in range(trail_length)
            ]
        )

    aligned_trails = []
    for i in range(num_lineages):
        aligned_trails.append(
            [
                ax.plot(
                    [],
                    [],
                    "-",
                    color=aligned_colors[i % len(aligned_colors)],
                    alpha=0.3,
                )[0]
                for _ in range(trail_length)
            ]
        )

    # Set labels and title
    ax.set_xlabel("PC1", color="white")
    ax.set_ylabel("PC2", color="white")
    ax.set_title("PC1 vs PC2 Trajectory Animation", color="white", fontsize=16)

    # Style the plot
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2, color="gray")

    # Add frame counter
    frame_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        color="white",
        fontsize=12,
        verticalalignment="top",
    )

    # Set legend
    ax.legend(loc="upper right", facecolor="black", edgecolor="white", framealpha=0.7)

    def update_trail(trail, xs, ys, current_idx):
        for i, trail_segment in enumerate(trail):
            if current_idx - i >= 0 and current_idx - i + 1 < len(xs):
                trail_segment.set_data(
                    xs[current_idx - i : current_idx - i + 2],
                    ys[current_idx - i : current_idx - i + 2],
                )
            else:
                trail_segment.set_data([], [])

    def animate(frame):
        # Update frame counter
        frame_text.set_text(f"Frame: {frame}/{len(reference_pattern)-1}")

        # Update reference point
        ref_point.set_data(
            reference_pca["PCA1"].iloc[frame], reference_pca["PCA2"].iloc[frame]
        )

        # Update reference trail
        update_trail(
            ref_trail, reference_pca["PCA1"].values, reference_pca["PCA2"].values, frame
        )

        # Update each lineage's points
        for i in range(num_lineages):
            # Unaligned
            unaligned_points[i].set_data(
                all_unaligned[i]["PCA1"].iloc[frame],
                all_unaligned[i]["PCA2"].iloc[frame],
            )

            # Aligned
            aligned_points[i].set_data(
                all_aligned[i]["PCA1"].iloc[frame], all_aligned[i]["PCA2"].iloc[frame]
            )

            # Update trails
            update_trail(
                unaligned_trails[i],
                all_unaligned[i]["PCA1"].values,
                all_unaligned[i]["PCA2"].values,
                frame,
            )

            update_trail(
                aligned_trails[i],
                all_aligned[i]["PCA1"].values,
                all_aligned[i]["PCA2"].values,
                frame,
            )

        # Return all artists that were updated
        artists = [ref_point, frame_text]
        artists.extend(unaligned_points)
        artists.extend(aligned_points)
        artists.extend(ref_trail)
        for trail in unaligned_trails:
            artists.extend(trail)
        for trail in aligned_trails:
            artists.extend(trail)

        return artists

    # Determine axis limits to include all trajectories
    all_pc1 = []
    all_pc2 = []

    all_pc1.extend(reference_pca["PCA1"].values)
    all_pc2.extend(reference_pca["PCA2"].values)

    for i in range(num_lineages):
        all_pc1.extend(all_unaligned[i]["PCA1"].values)
        all_pc2.extend(all_unaligned[i]["PCA2"].values)
        all_pc1.extend(all_aligned[i]["PCA1"].values)
        all_pc2.extend(all_aligned[i]["PCA2"].values)

    # Set axis limits with some padding
    pc1_min, pc1_max = min(all_pc1), max(all_pc1)
    pc2_min, pc2_max = min(all_pc2), max(all_pc2)

    # Add 10% padding
    pc1_range = pc1_max - pc1_min
    pc2_range = pc2_max - pc2_min

    ax.set_xlim(pc1_min - 0.1 * pc1_range, pc1_max + 0.1 * pc1_range)
    ax.set_ylim(pc2_min - 0.1 * pc2_range, pc2_max + 0.1 * pc2_range)

    # Create animation
    ani = animation.FuncAnimation(
        fig, animate, frames=len(reference_pattern), interval=1000 / fps, blit=True
    )

    # Save animation
    ani.save(
        output_path,
        writer="ffmpeg",
        fps=fps,
        dpi=150,
        extra_args=["-vcodec", "libx264"],
    )

    plt.close()
    logger.info(f"PC1 vs PC2 animation saved to {output_path}")
    return output_path


# Create PC1 vs PC2 animation
create_pc1_vs_pc2_animation(
    reference_pattern,
    top_n_aligned_cells,
    embeddings_PCA_df,
    output_path=save_path / "20241107_SEC61B_pc1_vs_pc2_animation.mp4",
    num_lineages=3,
    fps=10,
)


# %%
def plot_multiple_lineages_pca(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_PCA_df: pd.DataFrame,
    image_stacks: np.ndarray,  # Either all_unaligned_stacks or all_aligned_stacks
    output_path: Path,
    mode: str = "unaligned",  # "unaligned" or "aligned"
    pc_columns: list = ["PCA1", "PCA2"],  # PCA components to plot
    num_lineages: int = 3,
    z_slice: int = 15,
    time_point: int = 20,  # Example time point to visualize
):
    """
    Create a plot showing multiple lineages in rows with phase/GFP/mCherry side by side,
    followed by specified PC components for unaligned or aligned trajectories.

    Args:
        reference_pattern: The reference pattern embeddings
        top_aligned_cells: DataFrame with alignment information
        embeddings_PCA_df: DataFrame with pre-computed PCA components
        image_stacks: Array of image stacks (unaligned or aligned)
        output_path: Path to save the output figure
        mode: "unaligned" or "aligned" to determine which trajectories to display
        pc_columns: List of PC columns to plot (default: ["PCA1", "PCA2"])
        num_lineages: Number of lineages to display (default: 3)
        z_slice: Z-slice to use for images (default: 15)
        time_point: Time point to display in the visualization (default: 20)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # Set dark theme
    plt.style.use("dark_background")

    # Limit to requested number of lineages
    num_lineages = min(num_lineages, len(top_aligned_cells), len(image_stacks))
    num_pcs = len(pc_columns)

    # Create figure with rows for each lineage, columns for images and PCs
    fig = plt.figure(figsize=(20, 4 * num_lineages), facecolor="black")

    # Define column ratios - equal space for images and PCs
    column_widths = [1.5, 1.5, 1.5]  # For phase, GFP, mCherry
    column_widths.extend([2] * num_pcs)  # For each PC plot

    gs = fig.add_gridspec(
        num_lineages, 3 + num_pcs, wspace=0.2, hspace=0.3, width_ratios=column_widths
    )

    # Define contrast limits for each channel
    phase_limits = (-0.74, 0.4)
    gfp_limits = (106, 215)
    mcherry_limits = (106, 190)

    # Function to safely normalize values to [0,1] range
    def safe_norm(norm_func, data):
        normalized = norm_func(data)
        return np.clip(normalized, 0, 1)

    # Create normalizers for each channel
    phase_norm = Normalize(vmin=phase_limits[0], vmax=phase_limits[1])
    gfp_norm = Normalize(vmin=gfp_limits[0], vmax=gfp_limits[1])
    mcherry_norm = Normalize(vmin=mcherry_limits[0], vmax=mcherry_limits[1])

    # Get reference lineage PCA data
    reference_fov = "/C/2/001000"
    reference_track_id = 129

    reference_pca = embeddings_PCA_df[
        (embeddings_PCA_df["fov_name"] == reference_fov)
        & (embeddings_PCA_df["track_id"] == reference_track_id)
    ].sort_values(by="t")

    # Get reference timepoints [8, 70]
    reference_start = 8
    reference_end = 70
    reference_pca = reference_pca[
        (reference_pca["t"] >= reference_start) & (reference_pca["t"] < reference_end)
    ].reset_index(drop=True)

    # Ensure reference has correct length
    reference_pca = reference_pca.iloc[: len(reference_pattern)].reset_index(drop=True)

    # Collect all trajectory data for each lineage
    lineage_data = []
    for i in range(num_lineages):
        # Get data for this lineage
        row = top_aligned_cells.iloc[i]
        fov_name = row["fov_name"]
        track_ids = ast.literal_eval(row["track_ids"])
        start_time = int(row["start_timepoint"])
        warp_path = ast.literal_eval(row["warp_path"])

        # Get lineage PCA data
        full_pca = (
            embeddings_PCA_df[
                (embeddings_PCA_df["fov_name"] == fov_name)
                & (embeddings_PCA_df["track_id"].isin(track_ids))
            ]
            .sort_values(by="t")
            .reset_index(drop=True)
        )

        # Extract trajectory based on mode
        if mode == "unaligned":
            # Extract unaligned window
            traj_pca = full_pca.iloc[
                start_time : start_time + len(reference_pattern)
            ].reset_index(drop=True)

            # Pad if needed
            if len(traj_pca) < len(reference_pattern):
                last_row = (
                    traj_pca.iloc[-1].copy()
                    if not traj_pca.empty
                    else pd.Series(dtype="float64")
                )
                for _ in range(len(reference_pattern) - len(traj_pca)):
                    traj_pca = pd.concat(
                        [traj_pca, pd.DataFrame([last_row])], ignore_index=True
                    )

            # Trim if needed
            traj_pca = traj_pca.iloc[: len(reference_pattern)].reset_index(drop=True)

        else:  # aligned mode
            # Create aligned trajectory using warping path
            traj_pca = pd.DataFrame(columns=full_pca.columns)

            # Map each reference timepoint to the corresponding lineage timepoint
            for ref_idx, query_idx in warp_path:
                lineage_idx = int(start_time + query_idx)
                if 0 <= lineage_idx < len(full_pca):
                    # Add a row to traj_pca at position ref_idx
                    if ref_idx >= len(traj_pca):
                        # If ref_idx is beyond current length, extend traj_pca
                        missing_rows = ref_idx - len(traj_pca)
                        if missing_rows > 0:
                            # Add empty rows
                            empty_rows = pd.DataFrame(
                                [pd.Series(dtype="float64")] * missing_rows,
                                columns=traj_pca.columns,
                            )
                            traj_pca = pd.concat(
                                [traj_pca, empty_rows], ignore_index=True
                            )
                        # Add the mapped row
                        traj_pca = pd.concat(
                            [traj_pca, pd.DataFrame([full_pca.iloc[lineage_idx]])],
                            ignore_index=True,
                        )
                    else:
                        # Replace existing row
                        traj_pca.iloc[ref_idx] = full_pca.iloc[lineage_idx]

            # Fill any remaining gaps with nearest values
            traj_pca = traj_pca.fillna(method="ffill").fillna(method="bfill")

            # Ensure correct length
            traj_pca = traj_pca.iloc[: len(reference_pattern)].reset_index(drop=True)

            # If still too short, pad with last value
            if len(traj_pca) < len(reference_pattern):
                last_row = (
                    traj_pca.iloc[-1].copy()
                    if not traj_pca.empty
                    else pd.Series(dtype="float64")
                )
                for _ in range(len(reference_pattern) - len(traj_pca)):
                    traj_pca = pd.concat(
                        [traj_pca, pd.DataFrame([last_row])], ignore_index=True
                    )

        lineage_data.append(
            {"fov_name": fov_name, "track_ids": track_ids, "pca_df": traj_pca}
        )

    # Process each lineage
    for i in range(num_lineages):
        data = lineage_data[i]
        fov_name = data["fov_name"]
        track_ids = data["track_ids"]
        pca_df = data["pca_df"]

        # Get image stack for this lineage
        stack = image_stacks[i]

        # Make sure time_point is valid
        valid_time_point = min(time_point, stack.shape[0] - 1)

        # Get the images for the selected time point
        phase_img = stack[valid_time_point, 0, z_slice, :]
        gfp_img = stack[valid_time_point, 1, z_slice, :]
        mcherry_img = stack[valid_time_point, 2, z_slice, :]

        # Create axes for images
        ax_phase = fig.add_subplot(gs[i, 0])
        ax_gfp = fig.add_subplot(gs[i, 1])
        ax_mcherry = fig.add_subplot(gs[i, 2])

        # Set background color for all axes
        ax_phase.set_facecolor("black")
        ax_gfp.set_facecolor("black")
        ax_mcherry.set_facecolor("black")

        # Display phase image
        ax_phase.imshow(
            phase_img, cmap="gray", vmin=phase_limits[0], vmax=phase_limits[1]
        )
        ax_phase.set_title(
            f"Phase - {fov_name}\nTrack {track_ids[0]}", color="white", fontsize=10
        )
        ax_phase.axis("off")

        # Create RGB image for GFP (green channel)
        gfp_rgb = np.zeros((*gfp_img.shape, 3))
        gfp_rgb[:, :, 1] = safe_norm(gfp_norm, gfp_img)  # Green channel
        ax_gfp.imshow(gfp_rgb)
        ax_gfp.set_title("GFP", color="white")
        ax_gfp.axis("off")

        # Create RGB image for mCherry (magenta = red + blue)
        mcherry_rgb = np.zeros((*mcherry_img.shape, 3))
        mcherry_rgb[:, :, 0] = safe_norm(mcherry_norm, mcherry_img)  # Red component
        mcherry_rgb[:, :, 2] = safe_norm(mcherry_norm, mcherry_img)  # Blue component
        ax_mcherry.imshow(mcherry_rgb)
        ax_mcherry.set_title("mCherry", color="white")
        ax_mcherry.axis("off")

        # Plot each PC
        for pc_idx, pc_col in enumerate(pc_columns):
            ax_pc = fig.add_subplot(gs[i, 3 + pc_idx])
            ax_pc.set_facecolor("black")

            # Plot time points
            time_points = np.arange(len(reference_pattern))

            # Plot reference PC
            ax_pc.plot(
                time_points,
                reference_pca[pc_col].values,
                "w-",
                linewidth=2.5,
                label="Reference",
            )

            # Plot lineage PC
            ax_pc.plot(
                time_points,
                pca_df[pc_col].values,
                "c--" if mode == "unaligned" else "r-",
                linewidth=1.5,
                label=mode.capitalize(),
            )

            # Highlight current time point
            if valid_time_point < len(time_points):
                ax_pc.plot(
                    valid_time_point,
                    pca_df[pc_col].iloc[valid_time_point],
                    "co" if mode == "unaligned" else "ro",
                    markersize=10,
                )

            ax_pc.set_title(f"{pc_col}", color="white")
            ax_pc.set_xlabel("Time Point", color="white")
            ax_pc.set_ylabel(f"{pc_col}", color="white")
            ax_pc.tick_params(colors="white")
            ax_pc.legend(facecolor="black", edgecolor="white")
            ax_pc.grid(True, alpha=0.3, color="gray")

    # Add main title
    plt.suptitle(
        f"{mode.capitalize()} Trajectories - Multiple Lineages (TP:{time_point})",
        fontsize=16,
        color="white",
        y=0.98,
    )

    # Adjust figure margins
    fig.subplots_adjust(
        left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.1, hspace=0.3
    )

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Multiple lineages PCA plot saved to {output_path}")
    return output_path


# %%
# Create the comparison plots for unaligned and aligned trajectories with PCs
if len(all_unaligned_stacks) > 0:
    # Create directory for output
    save_path = output_save_dir / "multiple_lineages_pc_plots"
    save_path.mkdir(parents=True, exist_ok=True)

    # Plot PC1 and PC2 (default)
    plot_multiple_lineages_pca(
        reference_pattern,
        top_n_aligned_cells,
        pca_df,
        all_unaligned_stacks,
        output_path=save_path / "20241107_SEC61B_unaligned_pc1_pc2.png",
        mode="unaligned",
        pc_columns=["PCA1", "PCA2"],
        num_lineages=3,
        time_point=20,
    )

    # Plot PC1, PC2, PC3 for unaligned
    plot_multiple_lineages_pca(
        reference_pattern,
        top_n_aligned_cells,
        pca_df,
        all_unaligned_stacks,
        output_path=save_path / "20241107_SEC61B_unaligned_pc1_pc2_pc3.png",
        mode="unaligned",
        pc_columns=["PCA1", "PCA2", "PCA3"],
        num_lineages=3,
        time_point=20,
    )

    # Plot PC1 and PC2 for aligned
    if len(all_aligned_stacks) > 0:
        plot_multiple_lineages_pca(
            reference_pattern,
            top_n_aligned_cells,
            pca_df,
            all_aligned_stacks,
            output_path=save_path / "20241107_SEC61B_aligned_pc1_pc2.png",
            mode="aligned",
            pc_columns=["PCA1", "PCA2"],
            num_lineages=3,
            time_point=20,
        )

        # Plot PC1, PC2, PC3, PC4 for aligned
        plot_multiple_lineages_pca(
            reference_pattern,
            top_n_aligned_cells,
            pca_df,
            all_aligned_stacks,
            output_path=save_path / "20241107_SEC61B_aligned_pc1_pc2_pc3_pc4.png",
            mode="aligned",
            pc_columns=["PCA1", "PCA2", "PCA3", "PCA4"],
            num_lineages=3,
            time_point=20,
        )


# %%
def create_multiple_lineages_pc_video(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_PCA_df: pd.DataFrame,
    image_stacks: np.ndarray,  # Either all_unaligned_stacks or all_aligned_stacks
    output_path: Path,
    mode: str = "unaligned",  # "unaligned" or "aligned"
    pc_columns: list = ["PCA1", "PCA2"],  # PCA components to plot
    num_lineages: int = 3,
    z_slice: int = 15,
    fps: int = 5,
    show_title: bool = True,
):
    """
    Create a video showing multiple lineages with phase/GFP/mCherry side by side,
    followed by PC components plots over time.

    Args:
        reference_pattern: The reference pattern embeddings
        top_aligned_cells: DataFrame with alignment information
        embeddings_PCA_df: DataFrame with pre-computed PCA components
        image_stacks: Array of image stacks (unaligned or aligned)
        output_path: Path to save the output video
        mode: "unaligned" or "aligned" to determine which trajectories to display
        pc_columns: List of PC columns to plot (default: ["PCA1", "PCA2"])
        num_lineages: Number of lineages to display (default: 3)
        z_slice: Z-slice to use for images (default: 15)
        fps: Frames per second for the video (default: 5)
        show_title: Whether to show the title in the video (default: True)
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec
    import io
    import cv2
    from pathlib import Path

    # Set dark theme
    plt.style.use("dark_background")

    # Limit to requested number of lineages
    num_lineages = min(num_lineages, len(top_aligned_cells), len(image_stacks))
    num_pcs = len(pc_columns)

    # Get max number of time points across all stacks
    max_frames = max([stack.shape[0] for stack in image_stacks[:num_lineages]])

    # Define contrast limits for each channel
    phase_limits = (-0.74, 0.4)
    gfp_limits = (106, 215)
    mcherry_limits = (106, 170)

    # Function to safely normalize values to [0,1] range
    def safe_norm(norm_func, data):
        normalized = norm_func(data)
        return np.clip(normalized, 0, 1)

    # Get reference lineage PCA data
    reference_fov = "/C/2/001000"
    reference_track_id = 129

    reference_pca = embeddings_PCA_df[
        (embeddings_PCA_df["fov_name"] == reference_fov)
        & (embeddings_PCA_df["track_id"] == reference_track_id)
    ].sort_values(by="t")

    # Get reference timepoints [8, 70]
    reference_start = 8
    reference_end = 70
    reference_pca = reference_pca[
        (reference_pca["t"] >= reference_start) & (reference_pca["t"] < reference_end)
    ].reset_index(drop=True)

    # Ensure reference has correct length
    reference_pca = reference_pca.iloc[: len(reference_pattern)].reset_index(drop=True)

    # Collect all trajectory data for each lineage
    lineage_data = []
    for i in range(num_lineages):
        # Get data for this lineage
        row = top_aligned_cells.iloc[i]
        fov_name = row["fov_name"]
        track_ids = ast.literal_eval(row["track_ids"])
        start_time = int(row["start_timepoint"])
        warp_path = ast.literal_eval(row["warp_path"])

        # Get lineage PCA data
        full_pca = (
            embeddings_PCA_df[
                (embeddings_PCA_df["fov_name"] == fov_name)
                & (embeddings_PCA_df["track_id"].isin(track_ids))
            ]
            .sort_values(by="t")
            .reset_index(drop=True)
        )

        # Extract trajectory based on mode
        if mode == "unaligned":
            # Extract unaligned window
            traj_pca = full_pca.iloc[
                start_time : start_time + len(reference_pattern)
            ].reset_index(drop=True)

            # Pad if needed
            if len(traj_pca) < len(reference_pattern):
                last_row = (
                    traj_pca.iloc[-1].copy()
                    if not traj_pca.empty
                    else pd.Series(dtype="float64")
                )
                for _ in range(len(reference_pattern) - len(traj_pca)):
                    traj_pca = pd.concat(
                        [traj_pca, pd.DataFrame([last_row])], ignore_index=True
                    )

            # Trim if needed
            traj_pca = traj_pca.iloc[: len(reference_pattern)].reset_index(drop=True)

        else:  # aligned mode
            # Create aligned trajectory using warping path
            traj_pca = pd.DataFrame(columns=full_pca.columns)

            # Map each reference timepoint to the corresponding lineage timepoint
            for ref_idx, query_idx in warp_path:
                lineage_idx = int(start_time + query_idx)
                if 0 <= lineage_idx < len(full_pca):
                    # Add a row to traj_pca at position ref_idx
                    if ref_idx >= len(traj_pca):
                        # If ref_idx is beyond current length, extend traj_pca
                        missing_rows = ref_idx - len(traj_pca)
                        if missing_rows > 0:
                            # Add empty rows
                            empty_rows = pd.DataFrame(
                                [pd.Series(dtype="float64")] * missing_rows,
                                columns=traj_pca.columns,
                            )
                            traj_pca = pd.concat(
                                [traj_pca, empty_rows], ignore_index=True
                            )
                        # Add the mapped row
                        traj_pca = pd.concat(
                            [traj_pca, pd.DataFrame([full_pca.iloc[lineage_idx]])],
                            ignore_index=True,
                        )
                    else:
                        # Replace existing row
                        traj_pca.iloc[ref_idx] = full_pca.iloc[lineage_idx]

            # Fill any remaining gaps with nearest values
            traj_pca = traj_pca.fillna(method="ffill").fillna(method="bfill")

            # Ensure correct length
            traj_pca = traj_pca.iloc[: len(reference_pattern)].reset_index(drop=True)

            # If still too short, pad with last value
            if len(traj_pca) < len(reference_pattern):
                last_row = (
                    traj_pca.iloc[-1].copy()
                    if not traj_pca.empty
                    else pd.Series(dtype="float64")
                )
                for _ in range(len(reference_pattern) - len(traj_pca)):
                    traj_pca = pd.concat(
                        [traj_pca, pd.DataFrame([last_row])], ignore_index=True
                    )

        lineage_data.append(
            {"fov_name": fov_name, "track_ids": track_ids, "pca_df": traj_pca}
        )

    # Create figure with rows for each lineage, columns for images and PCs
    fig = plt.figure(figsize=(20, 4 * num_lineages), facecolor="black")

    # Define column ratios - equal space for images and PCs
    column_widths = [1.5, 1.5, 1.5]  # For phase, GFP, mCherry
    column_widths.extend([2] * num_pcs)  # For each PC plot

    gs = fig.add_gridspec(
        num_lineages, 3 + num_pcs, wspace=0.2, hspace=0.3, width_ratios=column_widths
    )

    # Create normalizers for each channel
    phase_norm = Normalize(vmin=phase_limits[0], vmax=phase_limits[1])
    gfp_norm = Normalize(vmin=gfp_limits[0], vmax=gfp_limits[1])
    mcherry_norm = Normalize(vmin=mcherry_limits[0], vmax=mcherry_limits[1])

    # Setup for animation
    axes = {}
    image_objects = {}
    line_objects = {}
    highlight_objects = {}
    time_text = None

    # Use a list to store the first frame number for each lineage
    # This ensures we correctly handle lineages that may have gaps
    first_frame_nums = []

    # Initialize time text at the top
    if show_title:
        time_text = fig.text(
            0.5, 0.98, "", ha="center", va="top", fontsize=16, color="white"
        )

    # Create axes and initialize plots
    for i in range(num_lineages):
        axes[i] = {}
        image_objects[i] = {}
        line_objects[i] = {}
        highlight_objects[i] = {}

        data = lineage_data[i]
        fov_name = data["fov_name"]
        track_ids = data["track_ids"]

        # Get image stack for this lineage
        stack = image_stacks[i]

        # Store the appropriate frame number offset
        first_frame = 0  # Default offset
        first_frame_nums.append(first_frame)

        # Create axes for images
        ax_phase = fig.add_subplot(gs[i, 0])
        ax_gfp = fig.add_subplot(gs[i, 2])
        ax_mcherry = fig.add_subplot(gs[i, 1])

        # Set background color for all axes
        ax_phase.set_facecolor("black")
        ax_gfp.set_facecolor("black")
        ax_mcherry.set_facecolor("black")

        # Get images from first frame
        if stack.shape[0] > 0:
            phase_img = stack[0, 0, z_slice, :]
            gfp_img = stack[0, 1, z_slice, :]
            mcherry_img = stack[0, 2, z_slice, :]

            # Initialize phase image
            img_phase = ax_phase.imshow(
                phase_img, cmap="gray", vmin=phase_limits[0], vmax=phase_limits[1]
            )
            ax_phase.axis("off")

            # Initialize GFP image (green channel)
            gfp_rgb = np.zeros((*gfp_img.shape, 3))
            gfp_rgb[:, :, 1] = safe_norm(gfp_norm, gfp_img)  # Green channel
            img_gfp = ax_gfp.imshow(gfp_rgb)
            ax_gfp.axis("off")

            # Initialize mCherry image (magenta = red + blue)
            mcherry_rgb = np.zeros((*mcherry_img.shape, 3))
            mcherry_rgb[:, :, 0] = safe_norm(mcherry_norm, mcherry_img)  # Red
            mcherry_rgb[:, :, 2] = safe_norm(mcherry_norm, mcherry_img)  # Blue
            img_mcherry = ax_mcherry.imshow(mcherry_rgb)
            ax_mcherry.axis("off")

            # Store image objects for updating
            image_objects[i]["phase"] = img_phase
            image_objects[i]["gfp"] = img_gfp
            image_objects[i]["mcherry"] = img_mcherry

        # Create PC plots
        for pc_idx, pc_col in enumerate(pc_columns):
            ax_pc = fig.add_subplot(gs[i, 3 + pc_idx])
            ax_pc.set_facecolor("black")

            # Plot time points
            time_points = np.arange(len(reference_pattern))

            # Plot reference PC
            ax_pc.plot(
                time_points,
                reference_pca[pc_col].values,
                "w-",
                linewidth=2.5,
                label="Reference",
            )

            # Plot the complete lineage trajectory from the beginning
            ax_pc.plot(
                time_points,
                pca_df[pc_col].values,
                "c--" if mode == "unaligned" else "r-",
                linewidth=1.5,
                alpha=0.6,
                label=f"{mode.capitalize()} Full Path",
            )

            # Initialize highlight marker (will be updated in animation)
            (highlight,) = ax_pc.plot(
                [], [], "co" if mode == "unaligned" else "ro", markersize=10
            )

            # Store axes and line objects for updating
            axes[i][pc_col] = ax_pc
            highlight_objects[i][pc_col] = highlight

            # Configure axes
            ax_pc.set_title(f"{pc_col}", color="white")
            ax_pc.set_xlabel("Time Point", color="white")
            ax_pc.set_ylabel(f"{pc_col}", color="white")
            ax_pc.tick_params(colors="white")
            ax_pc.legend(facecolor="black", edgecolor="white")
            ax_pc.grid(True, alpha=0.3, color="gray")

            # Set fixed y-axis limits based on all data
            min_val = min(
                reference_pca[pc_col].min(),
                *[d["pca_df"][pc_col].min() for d in lineage_data],
            )
            max_val = max(
                reference_pca[pc_col].max(),
                *[d["pca_df"][pc_col].max() for d in lineage_data],
            )
            buffer = 0.1 * (max_val - min_val)
            ax_pc.set_ylim(min_val - buffer, max_val + buffer)

            # Set x-axis limit to match reference pattern
            ax_pc.set_xlim(-1, len(reference_pattern))

    # Define animation function
    def animate(frame):
        # Update title with frame number if title is shown
        if show_title:
            time_text.set_text(f"{mode.capitalize()} Trajectories - Frame {frame}")

        # Update each lineage
        for i in range(num_lineages):
            # Get data for this lineage
            data = lineage_data[i]
            pca_df = data["pca_df"]

            # Get image stack for this lineage
            stack = image_stacks[i]

            # Check if we have enough frames
            if frame < stack.shape[0]:
                # Update images
                if i in image_objects:
                    phase_img = stack[frame, 0, z_slice, :]
                    gfp_img = stack[frame, 1, z_slice, :]
                    mcherry_img = stack[frame, 2, z_slice, :]

                    # Update phase image
                    image_objects[i]["phase"].set_data(phase_img)

                    # Update GFP image (green channel)
                    gfp_rgb = np.zeros((*gfp_img.shape, 3))
                    gfp_rgb[:, :, 1] = safe_norm(gfp_norm, gfp_img)
                    image_objects[i]["gfp"].set_data(gfp_rgb)

                    # Update mCherry image (magenta = red + blue)
                    mcherry_rgb = np.zeros((*mcherry_img.shape, 3))
                    mcherry_rgb[:, :, 0] = safe_norm(mcherry_norm, mcherry_img)
                    mcherry_rgb[:, :, 2] = safe_norm(mcherry_norm, mcherry_img)
                    image_objects[i]["mcherry"].set_data(mcherry_rgb)

            # Update PC plots - just move the highlight dot
            if frame < len(pca_df):
                for pc_col in pc_columns:
                    # Get highlight object
                    highlight = highlight_objects[i][pc_col]

                    # Update highlight marker position
                    highlight.set_data([frame], [pca_df[pc_col].iloc[frame]])

        # Return all artists that were updated
        artists = []
        if show_title:
            artists.append(time_text)
        for i in range(num_lineages):
            if i in image_objects:
                artists.extend(list(image_objects[i].values()))
            for pc_col in pc_columns:
                artists.append(highlight_objects[i][pc_col])

        return artists

    # Create animation
    ani = animation.FuncAnimation(
        fig, animate, frames=max_frames, interval=1000 / fps, blit=True
    )

    # Save animation
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up writer
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist="Me"), bitrate=5000)

    # Save animation
    ani.save(output_path, writer=writer)

    # Close figure
    plt.close(fig)

    logger.info(f"Multiple lineages PC video saved to {output_path}")
    return output_path


# %%
# Create PC videos for both unaligned and aligned
# Create directory for output
save_path = output_save_dir / "multiple_lineages_pc_videos"
save_path.mkdir(parents=True, exist_ok=True)

# Create unaligned PC1/PC2 video
create_multiple_lineages_pc_video(
    reference_pattern,
    top_n_aligned_cells,
    pca_df,
    all_unaligned_stacks,
    output_path=save_path / "SEC61B_unaligned_trajectory_pc1_pc2.mp4",
    mode="unaligned",
    pc_columns=["PCA1", "PCA2"],
    num_lineages=3,
    fps=5,
)

# Create aligned PC1/PC2 video
create_multiple_lineages_pc_video(
    reference_pattern,
    top_n_aligned_cells,
    pca_df,
    all_aligned_stacks,
    output_path=save_path / "SEC61B_aligned_trajectory_pc1_pc2.mp4",
    mode="aligned",
    pc_columns=["PCA1", "PCA2"],
    num_lineages=3,
    fps=5,
    show_title=False,
)


# %%
