# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import read_zarr
from iohub import open_ome_zarr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from viscy.data.triplet import TripletDataset
from viscy.representation.pseudotime import CytoDtw

# %%
logger = logging.getLogger("viscy")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

"""
TODO
- We need to find a way to save the annotations, features and track information into one file.
- We need to standardize the naming convention. i.e The annotations fov_name is missing a / at the beginning.
- It would be nice to also select which will be the reference lineages and add that as a column.
- Figure out what is the best format to save the consensus lineage
- Does the consensus track generalize?
- There is a lot of fragmentation. Which tracking was used for the annotations? There is a script that unifies this but no record of which one was it. We can append these as extra columns

"""

# Configuration
NAPARI = True
if NAPARI:
    import os

    import napari

    os.environ["DISPLAY"] = ":1"
    viewer = napari.Viewer()

# File paths

# ANNOTATIONS
cell_cycle_annotations_denv_dict = {
    # "tomm20_cc_1":
    #     {'data_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/2-assemble/2024_11_21_A549_TOMM20_DENV.zarr",
    #     'fov_name': "/C/2/001000",
    #     'annotations_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/4-phenotyping/0-annotations/track_cell_state_annotation.csv",
    #     'features_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/4-phenotyping/1-predictions/phase_160patch_104ckpt_ver3max.zarr",
    #     'tracks_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/2-assemble/tracking.zarr",
    #     },
    # "tomm20_cc_2":
    #     {'data_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_21_A549_TOMM20_DENV/2-assemble/2024_11_21_A549_TOMM20_DENV.zarr",
    #     'fov_name': "/B/3/000001",
    #     'annotations_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_21_A549_TOMM20_DENV/4-phenotyping/0-annotations/track_cell_state_annotation.csv",
    #     'features_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_21_A549_TOMM20_DENV/4-phenotyping/1-predictions/phase_160patch_104ckpt_ver3max.zarr",
    #     # 'tracks_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_21_A549_TOMM20_DENV/2-assemble/tracking.zarr",
    #     },
    # "sec61b_cc_1":
    #     {'data_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/2-assemble/2024_11_07_A549_SEC61_DENV.zarr/B/3/001000",
    #     'annotations_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/4-phenotyping/0-annotation/track_cell_state_annotation.csv",
    #     'tracks_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/2-assemble/tracking.zarr",
    #     },
    # "sec61b_cc_2":
    #     {'data_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/2-assemble/2024_11_07_A549_SEC61_DENV.zarr/C/2/000001",
    #     'annotations_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/4-phenotyping/0-annotation/track_cell_state_annotation.csv",
    #     },
    "g3bp1_cc_1": {
        "data_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/train-test/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr",
        "annotations_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/cytospeak_annotations/2025_07_24_annotations.csv",
        "features_path": "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/cell_cycle/output/phase_160patch_104ckpt_ver3max.anndata",
        "fov_name": "C/1/001000",
    },
}
output_root = Path(
    "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/cell_cycle/output"
)
output_root.mkdir(parents=True, exist_ok=True)

# %%
color_dict = {
    "interphase": "blue",
    "mitosis": "orange",
}
ANNOTATION_CELL_CYCLE = "predicted_cellstate"

# Load each dataframe and find the lineages
key, cell_cycle_annotations_denv = next(iter(cell_cycle_annotations_denv_dict.items()))
cell_cycle_annotations_df = pd.read_csv(cell_cycle_annotations_denv["annotations_path"])
data_path = cell_cycle_annotations_denv["data_path"]
fov_name = cell_cycle_annotations_denv["fov_name"]
features_path = cell_cycle_annotations_denv["features_path"]

# Load AnnData directly
adata = read_zarr(features_path)
print("Loaded AnnData with shape:", adata.shape)
print("Available columns:", adata.obs.columns.tolist())

# Instantiate the CytoDtw object with AnnData
cytodtw = CytoDtw(adata)
feature_df = cytodtw.adata.obs

min_timepoints = 7
filtered_lineages = cytodtw.get_lineages(min_timepoints)
filtered_lineages = pd.DataFrame(filtered_lineages, columns=["fov_name", "track_id"])
logger.info(
    f"Found {len(filtered_lineages)} lineages with at least {min_timepoints} timepoints"
)

# %%
n_timepoints_before = min_timepoints // 2
n_timepoints_after = min_timepoints // 2
valid_annotated_examples = [
    {
        "fov_name": "A/2/001001",
        "track_id": [136, 137],
        "timepoints": (43 - n_timepoints_before, 43 + n_timepoints_after + 1),
        "annotations": ["interphase"] * (n_timepoints_before)
        + ["mitosis"]
        + ["interphase"] * (n_timepoints_after - 1),
        "weight": 1.0,
    },
    # {
    #     'fov_name': "C/1/001000",
    #     'track_id': [47,48],
    #     'timepoints': (45-n_timepoints_before, 45+n_timepoints_after+1),
    #     'annotations': ["interphase"] * (n_timepoints_before) + ["mitosis"] + ["interphase"] * (n_timepoints_after-1),
    #     'weight': 1.0
    # },
    {
        "fov_name": "C/1/000000",
        "track_id": [118, 119],
        "timepoints": (27 - n_timepoints_before, 27 + n_timepoints_after + 1),
        "annotations": ["interphase"] * (n_timepoints_before)
        + ["mitosis"]
        + ["interphase"] * (n_timepoints_after - 1),
        "weight": 1.0,
    },
    # {
    #     'fov_name': "C/1/001000",
    #     'track_id': [59,60],
    #     'timepoints': (52-n_timepoints_before, 52+n_timepoints_after+1),
    #     'annotations': ["interphase"] * (n_timepoints_before) + ["mitosis"] + ["interphase"] * (n_timepoints_after-1),
    #     'weight': 1.0
    # },
    {
        "fov_name": "C/1/001001",
        "track_id": [93, 94],
        "timepoints": (29 - n_timepoints_before, 29 + n_timepoints_after + 1),
        "annotations": ["interphase"] * (n_timepoints_before)
        + ["mitosis"]
        + ["interphase"] * (n_timepoints_after - 1),
        "weight": 1.0,
    },
]
# %%
# Extract all reference patterns
patterns = []
pattern_info = []
REFERENCE_TYPE = "features"
DTW_CONSTRAINT_TYPE = "sakoe_chiba"
DTW_BAND_WIDTH_RATIO = 0.3

for i, example in enumerate(valid_annotated_examples):
    pattern = cytodtw.get_reference_pattern(
        fov_name=example["fov_name"],
        track_id=example["track_id"],
        timepoints=example["timepoints"],
        reference_type=REFERENCE_TYPE,
    )
    patterns.append(pattern)
    pattern_info.append(
        {
            "index": i,
            "fov_name": example["fov_name"],
            "track_id": example["track_id"],
            "timepoints": example["timepoints"],
            "annotations": example["annotations"],
        }
    )

# Concatenate all patterns to fit PCA on full dataset
all_patterns_concat = np.vstack(patterns)

# %%
# Plot the sample patterns

# Fit PCA on all data
scaler = StandardScaler()
scaled_patterns = scaler.fit_transform(all_patterns_concat)
pca = PCA(n_components=3)
pca.fit(scaled_patterns)

# Create subplots for PC1, PC2, PC3 over time
n_patterns = len(patterns)
fig, axes = plt.subplots(n_patterns, 3, figsize=(12, 3 * n_patterns))
if n_patterns == 1:
    axes = axes.reshape(1, -1)

# Plot each pattern
for i, (pattern, info) in enumerate(zip(patterns, pattern_info)):
    # Transform this pattern to PC space
    scaled_pattern = scaler.transform(pattern)
    pc_pattern = pca.transform(scaled_pattern)

    # Create time axis
    time_axis = np.arange(len(pattern))

    # Plot PC1, PC2, PC3
    for pc_idx in range(3):
        ax = axes[i, pc_idx]

        # Plot PC trajectory with colorblind-friendly colors
        ax.plot(
            time_axis,
            pc_pattern[:, pc_idx],
            "o-",
            color="blue",
            linewidth=2,
            markersize=4,
        )

        # Color timepoints by annotation
        annotations = info["annotations"]
        for t, annotation in enumerate(annotations):
            if annotation == "mitosis":
                ax.axvline(t, color="orange", alpha=0.7, linestyle="--", linewidth=2)
                ax.scatter(t, pc_pattern[t, pc_idx], c="orange", s=50, zorder=5)

        # Formatting
        ax.set_xlabel("Time")
        ax.set_ylabel(f"PC{pc_idx + 1}")
        ax.set_title(
            f"Pattern {i + 1}: FOV {info['fov_name']}, Tracks {info['track_id']}\nPC{pc_idx + 1} over time"
        )
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Create consensus pattern if we have valid examples
if len(valid_annotated_examples) >= 2:
    consensus_result = cytodtw.create_consensus_reference_pattern(
        annotated_samples=valid_annotated_examples,
        reference_selection="median_length",
        aggregation_method="median",
        reference_type=REFERENCE_TYPE,
    )
    consensus_lineage = consensus_result["pattern"]
    consensus_annotations = consensus_result.get("annotations", None)
    consensus_metadata = consensus_result["metadata"]

    logger.info(f"Created consensus pattern with shape: {consensus_lineage.shape}")
    logger.info(f"Consensus method: {consensus_metadata['aggregation_method']}")
    logger.info(f"Reference pattern: {consensus_metadata['reference_pattern']}")
    if consensus_annotations:
        logger.info(f"Consensus annotations length: {len(consensus_annotations)}")
else:
    logger.warning("Not enough valid lineages found to create consensus pattern")

# %%
# Perform DTW analysis for each embedding method
alignment_results = {}
top_n = 30

name = "consensus_lineage"
consensus_lineage = cytodtw.consensus_data["pattern"]
# Find pattern matches
matches = cytodtw.get_matches(
    reference_pattern=consensus_lineage,
    lineages=filtered_lineages.to_numpy(),
    window_step=1,
    num_candidates=top_n,
    method="bernd_clifford",
    metric="cosine",
    save_path=output_root / f"{name}_matching_lineages_cosine.csv",
    reference_type=REFERENCE_TYPE,
    constraint_type=DTW_CONSTRAINT_TYPE,
    band_width_ratio=DTW_BAND_WIDTH_RATIO,
)

alignment_results[name] = matches
logger.info(f"Found {len(matches)} matches for {name}")
# %%
# Save matches
print(f"Saving matches to {output_root / f'{name}_matching_lineages_cosine.csv'}")
# cytodtw.save_consensus(output_root / f"{name}_consensus_lineage.pkl")
# Add consensus path to the df all rows
# Add a new column 'consensus_path' to the matches DataFrame, with the same value for all rows.
# This is useful for downstream analysis to keep track of the consensus pattern used for matching.
# Reference: pandas.DataFrame.assign
matches["consensus_path"] = str(output_root / f"{name}_consensus_lineage.pkl")
# Save the pkl
cytodtw.save_consensus(output_root / f"{name}_consensus_lineage.pkl")

matches.to_csv(output_root / f"{name}_matching_lineages_cosine.csv", index=False)
# %%
top_matches = matches.head(top_n)

# Use the new enhanced alignment dataframe method instead of manual alignment
enhanced_df = cytodtw.create_enhanced_alignment_dataframe(
    top_matches,
    consensus_lineage,
    alignment_name="cell_division",
    reference_type=REFERENCE_TYPE,
)

logger.info(f"Enhanced DataFrame created with {len(enhanced_df)} rows")
logger.info(f"Lineages: {enhanced_df['lineage_id'].nunique()} (including consensus)")
logger.info(
    f"Cell division aligned timepoints: {enhanced_df['dtw_cell_division_aligned'].sum()}/{len(enhanced_df)} ({100 * enhanced_df['dtw_cell_division_aligned'].mean():.1f}%)"
)
# PCA plotting and alignment visualization is now handled by the enhanced alignment dataframe method
logger.info("Cell division consensus analysis completed successfully!")
print(f"Enhanced DataFrame columns: {enhanced_df.columns.tolist()}")

# %%
# Prototype video alignment based on DTW matches

z_range = slice(0, 1)
initial_yx_patch_size = (192, 192)

positions = []
tracks_tables = []
images_plate = open_ome_zarr(data_path)

# Load matching positions
print(f"Loading positions for {len(top_matches)} FOV matches...")
matches_found = 0
for _, pos in images_plate.positions():
    pos_name = pos.zgroup.name
    pos_normalized = pos_name.lstrip("/")

    if pos_normalized in top_matches["fov_name"].values:
        positions.append(pos)
        matches_found += 1

        # Get ALL tracks for this FOV to ensure TripletDataset has complete access
        tracks_df = cytodtw.adata.obs[
            cytodtw.adata.obs["fov_name"] == pos_normalized
        ].copy()

        if len(tracks_df) > 0:
            tracks_df = tracks_df.dropna(subset=["x", "y"])
            tracks_df["x"] = tracks_df["x"].astype(int)
            tracks_df["y"] = tracks_df["y"].astype(int)
            tracks_tables.append(tracks_df)

            if matches_found == 1:
                processing_channels = pos.channel_names

print(
    f"Loaded {matches_found} positions with {sum(len(df) for df in tracks_tables)} total tracks"
)

# Create TripletDataset if we have valid positions
if len(positions) > 0 and len(tracks_tables) > 0:
    if "processing_channels" not in locals():
        processing_channels = positions[0].channel_names

    # Use all three channels for overlay visualization
    selected_channels = processing_channels  # Use all available channels
    print(
        f"Creating TripletDataset with {len(selected_channels)} channels: {selected_channels}"
    )

    dataset = TripletDataset(
        positions=positions,
        tracks_tables=tracks_tables,
        channel_names=selected_channels,
        initial_yx_patch_size=initial_yx_patch_size,
        z_range=z_range,
        fit=False,
        predict_cells=False,
        include_fov_names=None,
        include_track_ids=None,
        time_interval=1,
        return_negative=False,
    )
    print(f"TripletDataset created with {len(dataset.valid_anchors)} valid anchors")
else:
    print("Cannot create TripletDataset - no valid positions or tracks")
    dataset = None


# %%
# Simplified sequence alignment using existing DTW results
def get_aligned_image_sequences(dataset: TripletDataset, candidates_df: pd.DataFrame):
    """Get image sequences aligned to consensus timeline using DTW warp paths."""

    aligned_sequences = {}
    for idx, row in candidates_df.iterrows():
        fov_name = row["fov_name"]
        track_ids = row["track_ids"]
        warp_path = row["warp_path"]
        int(row["start_track_timepoint"]) if not pd.isna(
            row["start_track_timepoint"]
        ) else 0

        # Determine alignment length from warp path
        alignment_length = max(ref_idx for ref_idx, _ in warp_path) + 1

        # Find matching dataset indices
        matching_indices = []
        for dataset_idx in range(len(dataset.valid_anchors)):
            anchor_row = dataset.valid_anchors.iloc[dataset_idx]
            if (
                anchor_row["fov_name"] == fov_name
                and anchor_row["track_id"] in track_ids
            ):
                matching_indices.append(dataset_idx)

        if not matching_indices:
            logger.warning(
                f"No matching indices found for FOV {fov_name}, tracks {track_ids}"
            )
            continue

        # Get images and sort by time
        batch_data = dataset.__getitems__(matching_indices)

        # Extract individual images from batch
        images = []
        for i in range(len(matching_indices)):
            img_data = {
                "anchor": batch_data["anchor"][i],
                "index": batch_data["index"][i],
            }
            images.append(img_data)

        images.sort(key=lambda x: x["index"]["t"])
        time_to_image = {img["index"]["t"]: img for img in images}

        # Create warp_path mapping and align images
        # Note: query_idx is now actual t value, not relative index
        ref_to_query = {ref_idx: query_t for ref_idx, query_t in warp_path}
        aligned_images = [None] * alignment_length

        for ref_idx in range(alignment_length):
            if ref_idx in ref_to_query:
                query_time = ref_to_query[
                    ref_idx
                ]  # query_time is already actual t value
                if query_time in time_to_image:
                    aligned_images[ref_idx] = time_to_image[query_time]
                else:
                    # Find closest available time
                    available_times = list(time_to_image.keys())
                    if available_times:
                        closest_time = min(
                            available_times, key=lambda x: abs(x - query_time)
                        )
                        aligned_images[ref_idx] = time_to_image[closest_time]

        # Fill None values with nearest neighbor
        for i in range(alignment_length):
            if aligned_images[i] is None:
                for offset in range(1, alignment_length):
                    for direction in [-1, 1]:
                        neighbor_idx = i + direction * offset
                        if (
                            0 <= neighbor_idx < alignment_length
                            and aligned_images[neighbor_idx] is not None
                        ):
                            aligned_images[i] = aligned_images[neighbor_idx]
                            break
                    if aligned_images[i] is not None:
                        break

        aligned_sequences[idx] = {
            "aligned_images": aligned_images,
            "metadata": {
                "fov_name": fov_name,
                "track_ids": track_ids,
                "distance": row["distance"],
                "alignment_length": alignment_length,
            },
        }

    return aligned_sequences


# Get aligned sequences using consolidated function
aligned_sequences = get_aligned_image_sequences(dataset, top_matches)

logger.info(f"Retrieved {len(aligned_sequences)} aligned sequences")
for idx, seq in aligned_sequences.items():
    meta = seq["metadata"]
    index = seq["aligned_images"][0]["index"]
    logger.info(
        f"Track id {index['track_id']}: FOV {meta['fov_name']} aligned images, distance={meta['distance']:.3f}"
    )

# %%
# Load aligned sequences into napari
if NAPARI and len(aligned_sequences) > 0:
    import numpy as np

    for idx, seq_data in aligned_sequences.items():
        aligned_images = seq_data["aligned_images"]
        meta = seq_data["metadata"]
        index = seq_data["aligned_images"][0]["index"]

        if len(aligned_images) == 0:
            continue

        # Stack images into time series (T, C, Z, Y, X)
        image_stack = []
        for img_sample in aligned_images:
            img_tensor = img_sample["anchor"]  # Shape should be (Z, C, Y, X)
            img_np = img_tensor.cpu().numpy()
            image_stack.append(img_np)

        if len(image_stack) > 0:
            # Stack into (T, Z, C, Y, X) or (T, C, Z, Y, X)
            time_series = np.stack(image_stack, axis=0)

            # Add to napari viewer
            layer_name = f"track_id_{index['track_id']}_FOV_{meta['fov_name']}_dist_{meta['distance']:.3f}"
            viewer.add_image(
                time_series,
                name=layer_name,
                contrast_limits=(time_series.min(), time_series.max()),
            )
            logger.info(f"Added {layer_name} with shape {time_series.shape}")
# Enhanced DataFrame was already created above with PCA plotting - skip duplicate
logger.info(
    f"Cell division aligned timepoints: {enhanced_df['dtw_cell_division_aligned'].sum()}/{len(enhanced_df)} ({100 * enhanced_df['dtw_cell_division_aligned'].mean():.1f}%)"
)
logger.info(f"Columns: {list(enhanced_df.columns)}")

# Show sample of the enhanced DataFrame
print("\nSample of enhanced DataFrame:")
sample_df = enhanced_df[enhanced_df["lineage_id"] != -1].head(10)
display_cols = [
    "lineage_id",
    "track_id",
    "t",
    "dtw_cell_division_aligned",
    "dtw_cell_division_consensus_mapping",
    "PC1",
]
print(sample_df[display_cols].to_string())

# %%


# Clean function that works directly with enhanced DataFrame
def plot_concatenated_from_dataframe(
    df,
    alignment_name="cell_division",
    feature_columns=["PC1", "PC2", "PC3"],
    max_lineages=5,
    y_offset_step=2.0,
    aligned_scale=1.0,
    unaligned_scale=1.0,
):
    """
    Plot concatenated [DTW-aligned portion] + [unaligned portion] sequences
    using ONLY the enhanced DataFrame and alignment information stored in it.

    This function reconstructs the aligned portions using the consensus mapping
    information already stored in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Enhanced DataFrame with alignment information
    alignment_name : str
        Name of alignment to plot (e.g., "cell_division")
    feature_columns : list
        Feature columns to plot
    max_lineages : int
        Maximum number of lineages to display
    y_offset_step : float
        Vertical separation between lineages
    aligned_scale : float
        Scale factor for DTW-aligned portions (line width & marker size)
    unaligned_scale : float
        Scale factor for unaligned portions (line width & marker size)
    """
    import matplotlib.pyplot as plt

    # Calculate line widths and marker sizes based on separate scale factors
    aligned_linewidth = 5 * aligned_scale
    unaligned_linewidth = 2 * unaligned_scale
    aligned_markersize = 8 * aligned_scale
    unaligned_markersize = 4 * unaligned_scale

    # Dynamic column names based on alignment_name
    aligned_col = f"dtw_{alignment_name}_aligned"
    mapping_col = f"dtw_{alignment_name}_consensus_mapping"
    distance_col = f"dtw_{alignment_name}_distance"

    # Check if alignment columns exist
    if aligned_col not in df.columns:
        logger.error(f"Alignment '{alignment_name}' not found in DataFrame")
        return

    # Get consensus and lineages
    consensus_df = df[df["lineage_id"] == -1].sort_values("t").copy()
    lineages = df[df["lineage_id"] != -1]["lineage_id"].unique()[:max_lineages]

    if consensus_df.empty:
        logger.error("No consensus found in DataFrame")
        return

    consensus_length = len(consensus_df)

    # Create concatenated sequences for each lineage
    concatenated_lineages = {}

    for lineage_id in lineages:
        lineage_df = df[df["lineage_id"] == lineage_id].copy().sort_values("t")
        if lineage_df.empty:
            continue

        # Split into aligned and unaligned portions
        aligned_rows = lineage_df[lineage_df[aligned_col]].copy()
        unaligned_rows = lineage_df[~lineage_df[aligned_col]].copy()

        # Create consensus-length aligned portion using mapping information
        aligned_portion = {}  # consensus_idx -> feature_values

        for _, row in aligned_rows.iterrows():
            consensus_idx = row[mapping_col]
            if not pd.isna(consensus_idx):
                consensus_idx = int(consensus_idx)
                if 0 <= consensus_idx < consensus_length:
                    aligned_portion[consensus_idx] = {
                        col: row[col] for col in feature_columns
                    }

        # Fill gaps in aligned portion (interpolate missing consensus indices)
        if aligned_portion:
            filled_aligned = {}
            for i in range(consensus_length):
                if i in aligned_portion:
                    filled_aligned[i] = aligned_portion[i]
                else:
                    # Find nearest available index
                    available_indices = list(aligned_portion.keys())
                    if available_indices:
                        closest_idx = min(available_indices, key=lambda x: abs(x - i))
                        filled_aligned[i] = aligned_portion[closest_idx]
                    else:
                        # Use consensus values if no aligned portion available
                        consensus_row = consensus_df.iloc[i]
                        filled_aligned[i] = {
                            col: consensus_row[col] for col in feature_columns
                        }

            # Convert aligned portion to arrays
            aligned_arrays = {}
            for col in feature_columns:
                aligned_arrays[col] = np.array(
                    [filled_aligned[i][col] for i in range(consensus_length)]
                )
        else:
            # No aligned portion, use consensus as fallback
            aligned_arrays = {}
            for col in feature_columns:
                aligned_arrays[col] = consensus_df[col].values.copy()

        # Get unaligned portion (sorted by original time)
        unaligned_arrays = {}
        if not unaligned_rows.empty:
            unaligned_rows = unaligned_rows.sort_values("t")
            for col in feature_columns:
                unaligned_arrays[col] = unaligned_rows[col].values
        else:
            for col in feature_columns:
                unaligned_arrays[col] = np.array([])

        # Concatenate aligned + unaligned portions
        concatenated_arrays = {}
        for col in feature_columns:
            if len(unaligned_arrays[col]) > 0:
                concatenated_arrays[col] = np.concatenate(
                    [aligned_arrays[col], unaligned_arrays[col]]
                )
            else:
                concatenated_arrays[col] = aligned_arrays[col]

        # Store concatenated data
        concatenated_lineages[lineage_id] = {
            "concatenated": concatenated_arrays,
            "aligned_length": len(aligned_arrays[feature_columns[0]]),
            "unaligned_length": len(unaligned_arrays[feature_columns[0]]),
            "dtw_distance": lineage_df[distance_col].iloc[0]
            if not pd.isna(lineage_df[distance_col].iloc[0])
            else np.nan,
        }

    # Plotting
    n_features = len(feature_columns)
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 4 * n_features))
    if n_features == 1:
        axes = [axes]

    # Generate colors using a colormap that works for all scenarios
    cmap = plt.cm.get_cmap(
        "tab10"
        if len(concatenated_lineages) <= 10
        else "tab20"
        if len(concatenated_lineages) <= 20
        else "hsv"
    )
    colors = [
        cmap(i / max(len(concatenated_lineages), 1))
        for i in range(len(concatenated_lineages))
    ]

    for feat_idx, feat_col in enumerate(feature_columns):
        ax = axes[feat_idx]

        # Plot consensus (no offset)
        consensus_values = consensus_df[feat_col].values
        consensus_time = np.arange(len(consensus_values))
        ax.plot(
            consensus_time,
            consensus_values,
            "o-",
            color="black",
            linewidth=4,
            markersize=8,
            label=f"Consensus ({alignment_name})",
            alpha=0.9,
            zorder=5,
        )

        # Add consensus annotations if available
        if alignment_name == "cell_division" and "consensus_annotations" in globals():
            for t, annotation in enumerate(consensus_annotations):
                if annotation == "mitosis":
                    ax.axvline(
                        t,
                        color="orange",
                        alpha=0.7,
                        linestyle="--",
                        linewidth=2,
                        zorder=1,
                    )

        # Plot each concatenated lineage
        for lineage_idx, (lineage_id, data) in enumerate(concatenated_lineages.items()):
            # Remove the color limit - now we have enough colors

            y_offset = -(lineage_idx + 1) * y_offset_step
            color = colors[lineage_idx]

            # Get concatenated sequence values
            concat_values = data["concatenated"][feat_col] + y_offset
            time_axis = np.arange(len(concat_values))

            # Plot full concatenated sequence
            ax.plot(
                time_axis,
                concat_values,
                ".-",
                color=color,
                linewidth=unaligned_linewidth,
                markersize=unaligned_markersize,
                alpha=0.8,
                label=f"Lineage {lineage_id} (d={data['dtw_distance']:.3f})",
            )

            # Highlight aligned portion with thicker line
            aligned_length = data["aligned_length"]
            if aligned_length > 0:
                aligned_time = time_axis[:aligned_length]
                aligned_values = concat_values[:aligned_length]

                ax.plot(
                    aligned_time,
                    aligned_values,
                    "s-",
                    color=color,
                    linewidth=aligned_linewidth,
                    markersize=aligned_markersize,
                    alpha=0.9,
                    zorder=4,
                )

            # Mark boundary between aligned and unaligned
            if aligned_length > 0 and aligned_length < len(concat_values):
                ax.axvline(
                    aligned_length, color=color, alpha=0.5, linestyle=":", linewidth=1
                )

        # Formatting
        ax.set_xlabel("Concatenated Time: [DTW Aligned] + [Unaligned Continuation]")
        ax.set_ylabel(f"{feat_col} (vertically separated)")
        ax.set_title(
            f"{feat_col}: Concatenated {alignment_name.replace('_', ' ').title()} Trajectories"
        )
        ax.grid(True, alpha=0.3)

        if feat_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.suptitle(
        f"DataFrame-Based Concatenated Alignment: {alignment_name.replace('_', ' ').title()}\n"
        f"Thick lines = DTW-aligned portions, Dotted lines = segment boundaries",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    logger.info(f"\nConcatenated alignment summary for '{alignment_name}':")
    logger.info(f"Processed {len(concatenated_lineages)} lineages")
    for lineage_id, data in concatenated_lineages.items():
        logger.info(
            f"  Lineage {lineage_id}: A={data['aligned_length']} + U={data['unaligned_length']} = {data['aligned_length'] + data['unaligned_length']}, d={data['dtw_distance']:.3f}"
        )


# Plot using the clean DataFrame-only function
plot_concatenated_from_dataframe(
    enhanced_df,
    alignment_name="cell_division",
    feature_columns=["PC1", "PC2", "PC3"],
    max_lineages=15,
    aligned_scale=0.5,
    unaligned_scale=0.7,
)

# %%


def get_concatenated_image_sequences_from_dataframe(
    dataset, df, alignment_name="cell_division", max_lineages=5
):
    """
    Create concatenated [DTW-aligned portion] + [unaligned portion] image sequences
    using the enhanced DataFrame alignment information, similar to plot_concatenated_from_dataframe().

    Parameters
    ----------
    dataset : TripletDataset
        Dataset containing the images
    df : pd.DataFrame
        Enhanced DataFrame with alignment information
    alignment_name : str
        Name of alignment to use (e.g., "cell_division")
    max_lineages : int
        Maximum number of lineages to process

    Returns
    -------
    dict
        Dictionary mapping lineage_id to concatenated image sequences
        Each entry contains:
        - 'concatenated_images': List of concatenated image tensors
        - 'aligned_length': Number of DTW-aligned images
        - 'unaligned_length': Number of unaligned continuation images
        - 'metadata': Lineage metadata
    """

    # Dynamic column names based on alignment_name
    aligned_col = f"dtw_{alignment_name}_aligned"
    mapping_col = f"dtw_{alignment_name}_consensus_mapping"
    distance_col = f"dtw_{alignment_name}_distance"

    # Check if alignment columns exist
    if aligned_col not in df.columns:
        logger.error(f"Alignment '{alignment_name}' not found in DataFrame")
        return {}

    # Get consensus and lineages
    consensus_df = df[df["lineage_id"] == -1].sort_values("t").copy()
    lineages = df[df["lineage_id"] != -1]["lineage_id"].unique()[:max_lineages]

    if consensus_df.empty:
        logger.error("No consensus found in DataFrame")
        return {}

    consensus_length = len(consensus_df)
    concatenated_sequences = {}

    for lineage_id in lineages:
        lineage_df = df[df["lineage_id"] == lineage_id].copy().sort_values("t")
        if lineage_df.empty:
            continue

        # Get FOV name and track IDs for this lineage
        fov_name = lineage_df["fov_name"].iloc[0]
        track_ids = lineage_df["track_id"].unique()

        # Find matching dataset indices
        matching_indices = []
        for dataset_idx in range(len(dataset.valid_anchors)):
            anchor_row = dataset.valid_anchors.iloc[dataset_idx]
            if (
                anchor_row["fov_name"] == fov_name
                and anchor_row["track_id"] in track_ids
            ):
                matching_indices.append(dataset_idx)

        if not matching_indices:
            logger.warning(
                f"No matching indices found for lineage {lineage_id}, FOV {fov_name}, tracks {track_ids}"
            )
            continue

        # Get images and create time mapping
        batch_data = dataset.__getitems__(matching_indices)
        images = []
        for i in range(len(matching_indices)):
            img_data = {
                "anchor": batch_data["anchor"][i],
                "index": batch_data["index"][i],
            }
            images.append(img_data)

        images.sort(key=lambda x: x["index"]["t"])
        time_to_image = {img["index"]["t"]: img for img in images}

        # Split DataFrame into aligned and unaligned portions
        aligned_rows = lineage_df[lineage_df[aligned_col]].copy()
        unaligned_rows = lineage_df[~lineage_df[aligned_col]].copy()

        # Create consensus-length aligned portion using mapping information
        aligned_images = [None] * consensus_length

        for _, row in aligned_rows.iterrows():
            consensus_idx = row[mapping_col]
            timepoint = row["t"]

            if not pd.isna(consensus_idx) and timepoint in time_to_image:
                consensus_idx = int(consensus_idx)
                if 0 <= consensus_idx < consensus_length:
                    aligned_images[consensus_idx] = time_to_image[timepoint]

        # Fill gaps in aligned portion with nearest neighbor
        for i in range(consensus_length):
            if aligned_images[i] is None:
                # Find nearest available aligned image
                available_indices = [
                    j for j, img in enumerate(aligned_images) if img is not None
                ]
                if available_indices:
                    closest_idx = min(available_indices, key=lambda x: abs(x - i))
                    aligned_images[i] = aligned_images[closest_idx]
                else:
                    # Use first available image from time_to_image as fallback
                    if time_to_image:
                        aligned_images[i] = next(iter(time_to_image.values()))

        # Get unaligned continuation images (sorted by original time)
        unaligned_images = []
        if not unaligned_rows.empty:
            unaligned_rows = unaligned_rows.sort_values("t")
            for _, row in unaligned_rows.iterrows():
                timepoint = row["t"]
                if timepoint in time_to_image:
                    unaligned_images.append(time_to_image[timepoint])

        # Concatenate aligned + unaligned portions
        concatenated_images = aligned_images + unaligned_images

        # Store results
        concatenated_sequences[lineage_id] = {
            "concatenated_images": concatenated_images,
            "aligned_length": len(aligned_images),
            "unaligned_length": len(unaligned_images),
            "metadata": {
                "fov_name": fov_name,
                "track_ids": list(track_ids),
                "dtw_distance": lineage_df[distance_col].iloc[0]
                if not pd.isna(lineage_df[distance_col].iloc[0])
                else np.nan,
                "lineage_id": lineage_id,
            },
        }

    logger.info(
        f"Created concatenated sequences for {len(concatenated_sequences)} lineages"
    )
    for lineage_id, data in concatenated_sequences.items():
        logger.info(
            f"  Lineage {lineage_id}: A={data['aligned_length']} + U={data['unaligned_length']} = {len(data['concatenated_images'])}, d={data['metadata']['dtw_distance']:.3f}"
        )

    return concatenated_sequences


# %%
# Create concatenated image sequences

# Create concatenated image sequences using the DataFrame alignment information
if dataset is not None:
    concatenated_image_sequences = get_concatenated_image_sequences_from_dataframe(
        dataset, enhanced_df, alignment_name="cell_division", max_lineages=30
    )
else:
    print("Skipping image sequence creation - no valid dataset available")
    concatenated_image_sequences = {}

# Load concatenated sequences into napari
if NAPARI and dataset is not None and len(concatenated_image_sequences) > 0:
    import numpy as np

    for lineage_id, seq_data in concatenated_image_sequences.items():
        concatenated_images = seq_data["concatenated_images"]
        meta = seq_data["metadata"]
        aligned_length = seq_data["aligned_length"]
        unaligned_length = seq_data["unaligned_length"]

        if len(concatenated_images) == 0:
            continue

        # Stack images into time series (T, C, Z, Y, X)
        image_stack = []
        for img_sample in concatenated_images:
            if img_sample is not None:
                img_tensor = img_sample["anchor"]  # Shape should be (C, Z, Y, X)
                img_np = img_tensor.cpu().numpy()
                image_stack.append(img_np)

        if len(image_stack) > 0:
            # Stack into (T, C, Z, Y, X)
            time_series = np.stack(image_stack, axis=0)
            n_channels = time_series.shape[1]

            logger.info(
                f"Processing lineage {lineage_id} with {n_channels} channels, shape {time_series.shape}"
            )

            # Set up colormap based on number of channels
            if n_channels == 2:
                colormap = ["green", "magenta"]
            elif n_channels == 3:
                colormap = ["gray", "green", "magenta"]
            else:
                colormap = ["gray"] * n_channels  # Default fallback

            # Add each channel as a separate layer in napari
            for channel_idx in range(n_channels):
                # Extract single channel: (T, Z, Y, X)
                channel_data = time_series[:, channel_idx, :, :, :]

                # Get channel name if available
                channel_name = (
                    processing_channels[channel_idx]
                    if channel_idx < len(processing_channels)
                    else f"ch{channel_idx}"
                )

                layer_name = f"track_id_{meta['track_ids'][0]}_FOV_{meta['fov_name']}_dist_{meta['dtw_distance']:.3f}_{channel_name}"

                viewer.add_image(
                    channel_data,
                    name=layer_name,
                    contrast_limits=(channel_data.min(), channel_data.max()),
                    colormap=colormap[channel_idx],
                    blending="additive",
                )
                logger.info(
                    f"Added {channel_name} channel for lineage {lineage_id} with shape {channel_data.shape}"
                )
# %%
