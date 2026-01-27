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
from viscy.representation.pseudotime import (
    CytoDtw,
    align_embedding_patterns,
    get_aligned_image_sequences,
)

# FIXME: standardize the naming convention for the computed features columns. (i.e replace time_point with t)
# FIXME: merge the computed features and the features in AnnData object
# FIXME: the pipeline should take the Anndata objects instea of pd.Dataframes
# FIXME: aligned_df should be an Anndata object instead of pandas
# FIXME: generalize the merging to use the tracking Dictionary instead of hardcoding the column names
# FIXME: be able to load the csv from another file and align the new embeddings w.r.t to this.
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
# %%
# File paths for infection state analysis
#
# FIXME combine the annotations,computed features into 1 single file
perturbations_dict = {
    # 'denv': {
    #     'data_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/4-phenotyping/train-test/2024_11_21_A549_TOMM20_DENV.zarr",
    #     'annotations_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/4-phenotyping/0-annotations/track_cell_state_annotation.csv",
    #     'features_path': "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/infection_state/output/phase_160patch_104ckpt_ver3max.zarr",
    # },
    "zikv": {
        "data_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/train-test/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr",
        "annotations_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/0-annotations/track_cell_state_annotation.csv",
        "features_path_sensor": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/anndata_predictions/sensor_160patch_104ckpt_ver3max.zarr",
        "features_path_phase": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/anndata_predictions/phase_160patch_104ckpt_ver3max.zarr",
        "features_path_organelle": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/anndata_predictions/organelle_160patch_104ckpt_ver3max.zarr",
        "computed_features_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/quantify_remodeling/feature_list_all.csv",
        "segmentation_features_path": "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/organelle_segmentation/output/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV_segs_features.csv",
    },
}


ALIGN_TYPE = "infection_apoptotic"  # Options: "cell_division" or "infection_state" or "apoptosis"
ALIGNMENT_CHANNEL = "sensor"  # sensor,phase,organelle

output_root = Path(
    "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/infection_state/output/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"
)
output_root.mkdir(parents=True, exist_ok=True)

# FIXME: find a better logic to manage this
consensus_path = None
# consensus_path = "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/infection_state/output/2025_06_26_A549_G3BP1_ZIKV/consensus_lineage_infection_apoptotic_sensor.pkl"


# %%
color_dict = {
    "uninfected": "blue",
    "infected": "orange",
}


for key in perturbations_dict.keys():
    data_path = perturbations_dict[key]["data_path"]
    annotations_path = perturbations_dict[key]["annotations_path"]

    if ALIGNMENT_CHANNEL not in ["sensor", "phase", "organelle"]:
        raise ValueError(
            "ALIGNMENT_CHANNEL must be one of 'sensor', 'phase', or 'organelle'"
        )

    computed_features_path = perturbations_dict[key]["computed_features_path"]
    segmentation_features_path = perturbations_dict[key]["segmentation_features_path"]

    channel_types = ["sensor", "phase", "organelle"]
    features_paths = {}
    ad_features = {}

    for channel in channel_types:
        path_key = f"features_path_{channel}"
        if path_key in perturbations_dict[key]:
            path = perturbations_dict[key][path_key]
            features_paths[channel] = path
            ad_features[channel] = read_zarr(path)

    n_pca_components = 8
    scaler = StandardScaler()
    pca = PCA(n_components=n_pca_components)

    for channel, adata in ad_features.items():
        scaled_features = scaler.fit_transform(adata.X)
        pca_features = pca.fit_transform(scaled_features)

        adata.obsm["X_pca"] = pca_features

        logger.info(
            f"Computed PCA for {channel} channel: explained variance ratio = {pca.explained_variance_ratio_}"
        )

    ad_features_alignment = ad_features[ALIGNMENT_CHANNEL]

    print("Loaded AnnData with shape:", ad_features_alignment.shape)
    print("Available columns:", ad_features_alignment.obs.columns.tolist())

    logger.info(f"Processing dataset: {key}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Annotations path: {annotations_path}")
    logger.info(f"Alignment channel: {ALIGNMENT_CHANNEL}")
    logger.info(f"Features path for alignment: {features_paths[ALIGNMENT_CHANNEL]}")
    logger.info(f"Computed features path: {computed_features_path}")

    # Instantiate the CytoDtw object with AnnData
    cytodtw = CytoDtw(ad_features_alignment)
    feature_df = cytodtw.adata.obs
    break

min_timepoints = 20
filtered_lineages = cytodtw.get_lineages(min_timepoints)
filtered_lineages = pd.DataFrame(filtered_lineages, columns=["fov_name", "track_id"])
logger.info(
    f"Found {len(filtered_lineages)} lineages with at least {min_timepoints} timepoints"
)

# %%
# Annotations
n_timepoints_before = min_timepoints // 2
n_timepoints_after = min_timepoints // 2

# Annotations on the 2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV dataset
cell_div_infected_annotations = [
    {
        "fov_name": "A/2/001000",
        "track_id": [239],
        "timepoints": (25 - n_timepoints_before, 25 + n_timepoints_after),
        "annotations": ["uninfected"] * (n_timepoints_before)
        + ["infected"] * (n_timepoints_after),
        "weight": 1.0,
    },
    {
        "fov_name": "C/2/001001",
        "track_id": [120],
        "timepoints": (30 - n_timepoints_before, 30 + n_timepoints_after),
        "annotations": ["uninfected"] * (n_timepoints_before)
        + ["infected"] * (n_timepoints_after),
        "weight": 1.0,
    },
]
# apoptotic infected annotation from the 2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV dataset
apoptotic_infected_annotations = [
    {
        "fov_name": "B/2/001000",
        "track_id": [109],
        "timepoints": (25 - n_timepoints_before, 25 + n_timepoints_after),
        "annotations": ["uninfected"] * (n_timepoints_before)
        + ["infected"] * (n_timepoints_after),
        "weight": 1.0,
    },
    {
        "fov_name": "B/2/001000",
        "track_id": [77],
        "timepoints": (21 - n_timepoints_before, 21 + n_timepoints_after),
        "annotations": ["uninfected"] * (n_timepoints_before)
        + ["infected"] * (n_timepoints_after),
        "weight": 1.0,
    },
    # Dies but there is no infection. there is some that look infected from the phase but dont die.
    #  in A/2/001000 we see cells that never get infected and all around die
    #     {
    #     "fov_name": "A/2/000001",
    #     "track_id": [137],
    #     "timepoints": (21 - n_timepoints_before, 21 + n_timepoints_after),
    #     "annotations": ["uninfected"] * (n_timepoints_before)
    #     + ["infected"] * (n_timepoints_after),
    #     "weight": 1.0,
    # },
    # {
    #     "fov_name": "C/2/000001",
    #     "track_id": [40],
    #     "timepoints": (24 - n_timepoints_before, 24 + n_timepoints_after),
    #     "annotations": ["uninfected"] * (n_timepoints_before)
    #     + ["infected"] * (n_timepoints_after),
    #     "weight": 1.0,
    # },
    # {
    #     "fov_name": "C/2/001001",
    #     "track_id": [115],
    #     "timepoints": (21 - n_timepoints_before, 2 + n_timepoints_after),
    #     "annotations": ["uninfected"] * (n_timepoints_before)
    #     + ["infected"] * (n_timepoints_after),
    #     "weight": 1.0,
    # },
]

# Annotations on the 2024_11_21_A549_TOMM20_DENV dataset
infection_annotations = [
    {
        "fov_name": "C/2/001001",
        "track_id": [193],
        "timepoints": (31 - n_timepoints_before, 31 + n_timepoints_after),
        "annotations": ["uinfected"] * (n_timepoints_before)
        + ["infected"]
        + ["uninfected"] * (n_timepoints_after - 1),
        "weight": 1.0,
    },
    {
        "fov_name": "C/2/001000",
        "track_id": [66],
        "timepoints": (19 - n_timepoints_before, 19 + n_timepoints_after),
        "annotations": ["uninfected"] * (n_timepoints_before)
        + ["infected"]
        + ["uninfected"] * (n_timepoints_after - 1),
        "weight": 1.0,
    },
    {
        "fov_name": "C/2/001000",
        "track_id": [54],
        "timepoints": (27 - n_timepoints_before, 27 + n_timepoints_after),
        "annotations": ["uninfected"] * (n_timepoints_before)
        + ["infected"]
        + ["uninfected"] * (n_timepoints_after - 1),
        "weight": 1.0,
    },
    {
        "fov_name": "C/2/001000",
        "track_id": [53],
        "timepoints": (21 - n_timepoints_before, 21 + n_timepoints_after),
        "annotations": ["uninfected"] * (n_timepoints_before)
        + ["infected"]
        + ["uninfected"] * (n_timepoints_after - 1),
        "weight": 1.0,
    },
]

if ALIGN_TYPE == "infection_state":
    aligning_annotations = infection_annotations
elif "apoptotic" in ALIGN_TYPE:
    aligning_annotations = apoptotic_infected_annotations
else:
    NotImplementedError("Only infection_state alignment is implemented in this example")


# %%
# Extract all reference patterns
patterns = []
pattern_info = []
REFERENCE_TYPE = "features"
DTW_CONSTRAINT_TYPE = "sakoe_chiba"
DTW_BAND_WIDTH_RATIO = 0.3

for i, example in enumerate(aligning_annotations):
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
all_patterns_concat = np.vstack(patterns)

# %%
# Plot the sample patterns
scaler = StandardScaler()
scaled_patterns = scaler.fit_transform(all_patterns_concat)
pca = PCA(n_components=3)
pca.fit(scaled_patterns)

n_patterns = len(patterns)
fig, axes = plt.subplots(n_patterns, 3, figsize=(12, 3 * n_patterns))
if n_patterns == 1:
    axes = axes.reshape(1, -1)

for i, (pattern, info) in enumerate(zip(patterns, pattern_info)):
    scaled_pattern = scaler.transform(pattern)
    pc_pattern = pca.transform(scaled_pattern)
    time_axis = np.arange(len(pattern))

    for pc_idx in range(3):
        ax = axes[i, pc_idx]

        ax.plot(
            time_axis,
            pc_pattern[:, pc_idx],
            "o-",
            color="blue",
            linewidth=2,
            markersize=4,
        )

        annotations = info["annotations"]
        for t, annotation in enumerate(annotations):
            if annotation == "mitosis":
                ax.axvline(t, color="orange", alpha=0.7, linestyle="--", linewidth=2)
                ax.scatter(t, pc_pattern[t, pc_idx], c="orange", s=50, zorder=5)
            elif annotation == "infected":
                ax.axvline(t, color="red", alpha=0.5, linestyle="--", linewidth=1)
                ax.scatter(t, pc_pattern[t, pc_idx], c="red", s=30, zorder=5)
                break  # Only mark the first infection timepoint

        ax.set_xlabel("Time")
        ax.set_ylabel(f"PC{pc_idx + 1}")
        ax.set_title(
            f"Pattern {i + 1}: FOV {info['fov_name']}, Tracks {info['track_id']}\nPC{pc_idx + 1} over time"
        )
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Create consensus pattern
if consensus_path is not None and Path(consensus_path).exists():
    logger.info(f"Loading existing consensus from {consensus_path}")
    consensus_result = np.load(consensus_path, allow_pickle=True)
    cytodtw.consensus_data = consensus_result
else:
    consensus_result = cytodtw.create_consensus_reference_pattern(
        annotated_samples=aligning_annotations,
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


# %%
# Plot all aligned consensus patterns together to validate alignment
# The patterns need to be DTW-aligned to the consensus for proper visualization
# We'll align each pattern to the consensus using the same method used internally

aligned_patterns_list = []
aligned_annotations_list = []
for i, example in enumerate(aligning_annotations):
    # Extract pattern
    pattern = cytodtw.get_reference_pattern(
        fov_name=example["fov_name"],
        track_id=example["track_id"],
        timepoints=example["timepoints"],
        reference_type=REFERENCE_TYPE,
    )

    # Align to consensus
    if len(pattern) == len(consensus_lineage):
        # Already same length, likely the reference pattern
        aligned_patterns_list.append(pattern)
        aligned_annotations_list.append(example.get("annotations", None))
    else:
        # Align to consensus
        alignment_result = align_embedding_patterns(
            query_pattern=pattern,
            reference_pattern=consensus_lineage,
            metric="cosine",
            query_annotations=example.get("annotations", None),
            constraint_type=DTW_CONSTRAINT_TYPE,
            band_width_ratio=DTW_BAND_WIDTH_RATIO,
        )
        aligned_patterns_list.append(alignment_result["pattern"])
        aligned_annotations_list.append(alignment_result.get("annotations", None))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for pc_idx in range(3):
    ax = axes[pc_idx]

    # Transform each aligned pattern to PC space and plot
    for i, pattern in enumerate(aligned_patterns_list):
        scaled_ref = scaler.transform(pattern)
        pc_ref = pca.transform(scaled_ref)

        time_axis = np.arange(len(pc_ref))
        ax.plot(
            time_axis,
            pc_ref[:, pc_idx],
            "o-",
            label=f"Ref {i + 1}",
            alpha=0.7,
            linewidth=2,
            markersize=4,
        )

        # Mark infection timepoint for this aligned trajectory
        if aligned_annotations_list[i] and "infected" in aligned_annotations_list[i]:
            infection_t = aligned_annotations_list[i].index("infected")
            ax.axvline(
                infection_t, color="orange", alpha=0.4, linestyle="--", linewidth=1
            )

    # Overlay consensus pattern
    scaled_consensus = scaler.transform(consensus_lineage)
    pc_consensus = pca.transform(scaled_consensus)
    time_axis = np.arange(len(pc_consensus))
    ax.plot(
        time_axis,
        pc_consensus[:, pc_idx],
        "s-",
        color="black",
        linewidth=3,
        markersize=6,
        label="Consensus",
        zorder=10,
    )

    # Mark consensus infection timepoint with a thicker, more prominent line
    if consensus_annotations and "infected" in consensus_annotations:
        consensus_infection_t = consensus_annotations.index("infected")
        ax.axvline(
            consensus_infection_t,
            color="orange",
            alpha=0.9,
            linestyle="--",
            linewidth=2.5,
            label="Infection",
        )

    ax.set_xlabel("Aligned Time")
    ax.set_ylabel(f"PC{pc_idx + 1}")
    ax.set_title(f"PC{pc_idx + 1}: All DTW-Aligned References + Consensus")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle(
    "Consensus Validation: DTW-Aligned References + Computed Consensus", fontsize=14
)
plt.tight_layout()
plt.show()
logger.info("Plotted DTW-aligned consensus patterns for validation")

# %%
# Perform DTW analysis for each embedding method
alignment_results = {}
top_n = 30

name = f"consensus_lineage_{ALIGN_TYPE}_{ALIGNMENT_CHANNEL}"
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
matches["consensus_path"] = str(output_root / f"{name}.pkl")
cytodtw.save_consensus(output_root / f"{name}.pkl")
matches.to_csv(output_root / f"{name}_matching_lineages_cosine.csv", index=False)
# %%
top_matches = matches.head(top_n)

# Use the new alignment dataframe method instead of manual alignment
alignment_df = cytodtw.create_alignment_dataframe(
    top_matches,
    consensus_lineage,
    alignment_name=ALIGN_TYPE,
    reference_type=REFERENCE_TYPE,
)

logger.info(f"Enhanced DataFrame created with {len(alignment_df)} rows")
logger.info(f"Lineages: {alignment_df['lineage_id'].nunique()} (including consensus)")

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
# Get aligned sequences using consolidated function
if dataset is not None:

    def load_images_from_triplet_dataset(fov_name, track_ids):
        """Load images from TripletDataset for given FOV and track IDs."""
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
            return {}

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
        return {img["index"]["t"]: img for img in images}

    # Filter alignment_df to only aligned rows for loading just the aligned region
    alignment_col = f"dtw_{ALIGN_TYPE}_aligned"
    aligned_only_df = alignment_df[alignment_df[alignment_col]].copy()

    # Use filtered alignment_df since get_aligned_image_sequences expects 'track_id' column
    aligned_sequences = get_aligned_image_sequences(
        cytodtw_instance=cytodtw,
        df=aligned_only_df,
        alignment_name=ALIGN_TYPE,
        image_loader_fn=load_images_from_triplet_dataset,
        max_lineages=top_n,
    )
else:
    aligned_sequences = {}

logger.info(f"Retrieved {len(aligned_sequences)} aligned sequences")
for idx, seq in aligned_sequences.items():
    meta = seq["metadata"]
    # Handle both possible keys depending on return structure
    images_key = "aligned_images" if "aligned_images" in seq else "concatenated_images"
    if images_key in seq and len(seq[images_key]) > 0:
        index = seq[images_key][0]["index"]
        logger.info(
            f"Track id {index['track_id']}: FOV {meta['fov_name']} aligned images, distance={meta.get('distance', meta.get('dtw_distance', 'N/A')):.3f}"
        )

# %%
# # Load aligned sequences into napari (ALIGNED REGION ONLY)
# # Note: This loads only the aligned portion of the trajectory.
# # For complete trajectories (unaligned + aligned + unaligned), see the concatenated_image_sequences section below.
# if NAPARI and len(aligned_sequences) > 0:
#     import numpy as np

#     for idx, seq_data in aligned_sequences.items():
#         # Handle both possible keys depending on return structure
#         images_key = "aligned_images" if "aligned_images" in seq_data else "concatenated_images"

#         if images_key not in seq_data or len(seq_data[images_key]) == 0:
#             continue

#         aligned_images = seq_data[images_key]
#         meta = seq_data["metadata"]
#         index = aligned_images[0]["index"]

#         # Stack images into time series (T, C, Z, Y, X)
#         image_stack = []
#         for img_sample in aligned_images:
#             if img_sample is not None:
#                 img_tensor = img_sample["anchor"]  # Shape should be (Z, C, Y, X)
#                 img_np = img_tensor.cpu().numpy()
#                 image_stack.append(img_np)

#         if len(image_stack) > 0:
#             # Stack into (T, Z, C, Y, X) or (T, C, Z, Y, X)
#             time_series = np.stack(image_stack, axis=0)

#             # Add to napari viewer (prefix with ALIGNED_ to distinguish from full trajectories)
#             distance = meta.get('distance', meta.get('dtw_distance', 0.0))
#             layer_name = f"ALIGNED_track_id_{index['track_id']}_FOV_{meta['fov_name']}_dist_{distance:.3f}"
#             viewer.add_image(
#                 time_series,
#                 name=layer_name,
#                 contrast_limits=(time_series.min(), time_series.max()),
#             )
#             logger.info(
#                 f"Added {layer_name} with shape {time_series.shape} (aligned region only)"
#             )
# %%
# Enhanced DataFrame was already created above with PCA plotting - skip duplicate
logger.info(
    f"{ALIGN_TYPE.capitalize()} aligned timepoints: {alignment_df[f'dtw_{ALIGN_TYPE}_aligned'].sum()}/{len(alignment_df)} ({100 * alignment_df[f'dtw_{ALIGN_TYPE}_aligned'].mean():.1f}%)"
)
logger.info(f"Columns: {list(alignment_df.columns)}")

print("\nSample of enhanced DataFrame:")
sample_df = alignment_df[alignment_df["lineage_id"] != -1].head(10)
display_cols = [
    "lineage_id",
    "track_id",
    "t",
    f"dtw_{ALIGN_TYPE}_aligned",
    f"dtw_{ALIGN_TYPE}_consensus_mapping",
    "PC1",
]
print(sample_df[display_cols].to_string())


# Plot using the CytoDtw method
fig = cytodtw.plot_individual_lineages(
    alignment_df,
    alignment_name=ALIGN_TYPE,
    feature_columns=["PC1", "PC2", "PC3"],
    max_lineages=15,
    aligned_linewidth=2.5,
    unaligned_linewidth=1.4,
    y_offset_step=0,
)


# %%
# Create concatenated image sequences using the DataFrame alignment information
# Filter for infection wells only for specific organelles
fov_name_patterns = ["consensus", "B/2"]
filtered_alignment_df = alignment_df[
    alignment_df["fov_name"].str.contains("|".join(fov_name_patterns))
]
if dataset is not None:
    # Define TripletDataset-specific image loader
    def load_images_from_triplet_dataset(fov_name, track_ids):
        """Load images from TripletDataset for given FOV and track IDs."""
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
            return {}

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
        return {img["index"]["t"]: img for img in images}

    concatenated_image_sequences = get_aligned_image_sequences(
        cytodtw_instance=cytodtw,
        df=filtered_alignment_df,
        alignment_name=ALIGN_TYPE,
        image_loader_fn=load_images_from_triplet_dataset,
        max_lineages=30,
    )
else:
    print("Skipping image sequence creation - no valid dataset available")
    concatenated_image_sequences = {}

# Load concatenated sequences into napari (includes unaligned + aligned + unaligned timepoints)
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
                img_tensor = img_sample["anchor"]
                img_np = img_tensor.cpu().numpy()
                image_stack.append(img_np)

        if len(image_stack) > 0:
            time_series = np.stack(image_stack, axis=0)
            n_channels = time_series.shape[1]

            logger.info(
                f"Processing lineage {lineage_id} with {n_channels} channels, shape {time_series.shape}"
            )
            logger.info(
                f"  Aligned length: {aligned_length}, Unaligned length: {unaligned_length}, Total: {len(image_stack)}"
            )

            # Set up colormap based on number of channels
            # FIXME: This is hardcoded for specific datasets - improve logic as needed
            if n_channels == 2:
                colormap = ["green", "magenta"]
            elif n_channels == 3:
                colormap = ["gray", "green", "magenta"]
            else:
                colormap = ["gray"] * n_channels  # Default fallback

            # Add each channel as a separate layer in napari
            for channel_idx in range(n_channels):
                channel_data = time_series[:, channel_idx, :, :, :]
                channel_name = (
                    processing_channels[channel_idx]
                    if channel_idx < len(processing_channels)
                    else f"ch{channel_idx}"
                )
                # Indicate that this includes full trajectory (unaligned + aligned + unaligned)
                layer_name = f"FULL_track_id_{meta['track_ids'][0]}_FOV_{meta['fov_name']}_dist_{meta['dtw_distance']:.3f}_{channel_name}"

                viewer.add_image(
                    channel_data,
                    name=layer_name,
                    contrast_limits=(channel_data.min(), channel_data.max()),
                    colormap=colormap[channel_idx],
                    blending="additive",
                )
                logger.debug(
                    f"Added {channel_name} channel for lineage {lineage_id} with shape {channel_data.shape}"
                )
# %%
# Get the segmentation based features and compute per-cell aggregates
segmentation_features_df = pd.read_csv(segmentation_features_path)
segmentation_features_df["fov_name"] = segmentation_features_df["fov_name"].str.lstrip(
    "/"
)

# Compute per-cell mitochondria population statistics
segs_population_features = []
for (fov, track, t), group in segmentation_features_df.groupby(
    ["fov_name", "track_id", "t"]
):
    stats = {
        "fov_name": fov,
        "track_id": track,
        "t": t,
        # Count metrics
        "segs_count": len(group),
        # Area/volume metrics
        "segs_total_area": group["area"].sum(),
        "segs_mean_area": group["area"].mean(),
        "segs_std_area": group["area"].std(),
        "segs_median_area": group["area"].median(),
        # Shape metrics
        "segs_mean_eccentricity": group["eccentricity"].mean(),
        "segs_std_eccentricity": group["eccentricity"].std(),
        "segs_mean_solidity": group["solidity"].mean(),
        "segs_std_solidity": group["solidity"].std(),
        "segs_circularity_mean": group["circularity"].mean(),
        "segs_circularity_std": group["circularity"].std(),
        # Intensity metrics
        "segs_mean_intensity": group["mean_intensity"].mean(),
        "segs_std_intensity_across_mitos": group["mean_intensity"].std(),
        "segs_mean_max_intensity": group["max_intensity"].mean(),
        # Texture metrics (aggregated)
        "segs_mean_texture_contrast": group["texture_contrast"].mean(),
        "segs_mean_texture_homogeneity": group["texture_homogeneity"].mean(),
        # Frangi filter metrics (tubularity/network structure)
        "segs_mean_frangi_mean": group["frangi_mean_intensity"].mean(),
        "segs_mean_frangi_std": group["frangi_std_intensity"].mean(),
        # Shape diversity (coefficient of variation)
        "segs_area_cv": group["area"].std() / (group["area"].mean() + 1e-6),
        "segs_eccentricity_cv": group["eccentricity"].std()
        / (group["eccentricity"].mean() + 1e-6),
        "segs_solidity_cv": group["solidity"].std() / (group["solidity"].mean() + 1e-6),
        "segs_frangi_cv": group["frangi_mean_intensity"].std()
        / (group["frangi_mean_intensity"].mean() + 1e-6),
        "segs_circularity_cv": group["circularity"].std()
        / (group["circularity"].mean() + 1e-6),
    }
    segs_population_features.append(stats)

segs_population_df = pd.DataFrame(segs_population_features)

logger.info(
    f"Computed mitochondria population features for {len(segs_population_df)} (fov, track, t) combinations"
)
logger.info(
    f"Mitochondria population feature columns: {list(segs_population_df.columns)}"
)

# Load the computed features and PCs
computed_features_df = pd.read_csv(computed_features_path)
# Rename time_point to t for merging
computed_features_df = computed_features_df.rename(columns={"time_point": "t"})
# Remove the first forward slash from the fov_name
computed_features_df["fov_name"] = computed_features_df["fov_name"].str.lstrip("/")

# Population-based normalization to measure conserved remodeling states across cells
cf_of_interests = ["homogeneity", "contrast", "edge_density", "organelle_volume"]
percentile = 10

for cf in cf_of_interests:
    # Use population-wide baseline (same for all cells) to preserve absolute differences
    population_baseline = computed_features_df[cf].quantile(percentile / 100)
    computed_features_df[f"normalized_{cf}"] = (
        computed_features_df[cf] - population_baseline
    ) / (population_baseline + 1e-6)
# %%
# Merge the computed features and mitochondria population features
combined_features_df = computed_features_df.merge(
    segs_population_df, on=["fov_name", "track_id", "t"], how="left"
)

# Add PCs from each channel to the combined features
for channel, adata in ad_features.items():
    # Create a temporary dataframe with PCs from this channel
    pcs_df = adata.obs[["fov_name", "track_id", "t"]].copy()

    # Add PC columns with channel prefix
    for i in range(n_pca_components):
        pcs_df[f"{channel}_PC{i + 1}"] = adata.obsm["X_pca"][:, i]

    # Merge with combined features
    combined_features_df = combined_features_df.merge(
        pcs_df,
        on=["fov_name", "track_id", "t"],
        how="left",
    )
    logger.info(
        f"Added {n_pca_components} PCs from {channel} channel to combined features"
    )

# %%
# Create dataframe with uninfected cells (B/1) - no alignment needed
# Start from original tracking data for B/1 wells only
uninfected_features_df = cytodtw.adata.obs[
    cytodtw.adata.obs["fov_name"].str.contains("B/1")
].merge(
    combined_features_df,
    on=["fov_name", "track_id", "t", "x", "y"],
    how="left",
)

logger.info(
    f"Created uninfected_features_df with B/1 wells. Shape: {uninfected_features_df.shape}"
)
logger.info(f"Wells included: {sorted(uninfected_features_df['fov_name'].unique())}")

# %%
# Create filtered dataframe with only B/2 (infected) wells for specific analysis
align_n_comp_feat_df = filtered_alignment_df.merge(
    combined_features_df,
    on=["fov_name", "track_id", "t", "x", "y"],
    how="left",
)

logger.info(
    f"Created align_n_comp_feat_df with B/2 wells only. Shape: {align_n_comp_feat_df.shape}"
)

all_infected_tracking_df = cytodtw.adata.obs[
    cytodtw.adata.obs["fov_name"].str.contains("B/2")
][["fov_name", "track_id", "t", "x", "y"]].copy()

all_infected_features_df = all_infected_tracking_df.merge(
    combined_features_df,
    on=["fov_name", "track_id", "t", "x", "y"],
    how="left",
)

# %%
fig = cytodtw.plot_individual_lineages(
    align_n_comp_feat_df,
    alignment_name=ALIGN_TYPE,
    feature_columns=[
        "sensor_PC1",
        "homogeneity",
        "contrast",
        "edge_density",
        "segs_count",
        "segs_total_area",
        "segs_mean_area",
    ],
    max_lineages=8,
    aligned_linewidth=2.5,
    unaligned_linewidth=1.4,
    y_offset_step=0.0,
)
# %%
# Heatmap showing all tracks
fig = cytodtw.plot_global_trends(
    align_n_comp_feat_df,
    alignment_name=ALIGN_TYPE,
    plot_type="heatmap",
    cmap="RdBu",
    figsize=(12, 12),
    feature_columns=[
        # "sensor_PC1",
        # "sensor_PC2",
        # "sensor_PC3",
        # "phase_PC1",
        # "phase_PC2",
        # "phase_PC3",
        "organelle_PC1",
        "organelle_PC2",
        "organelle_PC3",
        # "homogeneity",
        "edge_density",
        "segs_count",
        "segs_total_area",
        "segs_mean_area",
        "segs_circularity_mean",
        "segs_mean_frangi_mean",
    ],
    max_lineages=8,
)


# %%
# Unified trajectory aggregation function
def aggregate_trajectory_by_time(
    df: pd.DataFrame,
    feature_columns: list,
) -> pd.DataFrame:
    """
    Unified function to aggregate trajectory data by timepoint.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-filtered dataframe with features and 't' column
    feature_columns : list
        Features to aggregate

    Returns
    -------
    pd.DataFrame
        Aggregated trajectory with columns: t, n_cells, {feature}_{stat}
        where stat is one of: mean, median, std, q25, q75
    """
    aggregated_data = []

    for t in sorted(df["t"].unique()):
        time_slice = df[df["t"] == t]

        row_data = {
            "t": t,
            "n_cells": len(time_slice),
        }

        # Compute statistics for each feature
        for feature in feature_columns:
            if feature not in time_slice.columns:
                # Feature doesn't exist - set all stats to NaN
                row_data[f"{feature}_median"] = np.nan
                row_data[f"{feature}_mean"] = np.nan
                row_data[f"{feature}_std"] = np.nan
                row_data[f"{feature}_q25"] = np.nan
                row_data[f"{feature}_q75"] = np.nan
                continue

            values = time_slice[feature].dropna()

            if len(values) == 0:
                # No valid values - set all stats to NaN
                row_data[f"{feature}_median"] = np.nan
                row_data[f"{feature}_mean"] = np.nan
                row_data[f"{feature}_std"] = np.nan
                row_data[f"{feature}_q25"] = np.nan
                row_data[f"{feature}_q75"] = np.nan
                continue

            # Compute all statistics
            row_data[f"{feature}_mean"] = values.mean()
            row_data[f"{feature}_median"] = values.median()
            row_data[f"{feature}_std"] = values.std()
            row_data[f"{feature}_q25"] = values.quantile(0.25)
            row_data[f"{feature}_q75"] = values.quantile(0.75)

        aggregated_data.append(row_data)

    result_df = pd.DataFrame(aggregated_data)

    logger.info(
        f"Aggregated {len(result_df)} timepoints from {len(df)} total observations"
    )

    return result_df


# Select features to compute common response for
common_response_features = [
    "organelle_PC1",
    "organelle_PC2",
    "organelle_PC3",
    "phase_PC1",
    "phase_PC2",
    "phase_PC3",
    "edge_density",
    # "organelle_volume",
    "segs_count",
    # "segs_total_area",
    "segs_mean_area",
    "segs_mean_eccentricity",
    # "segs_mean_texture_contrast",
    "segs_mean_frangi_mean",
    "segs_circularity_mean",
    "segs_circularity_cv",
    "segs_eccentricity_cv",
    "segs_area_cv",
]

# Compute common response from top N aligned cells
# First, select top N lineages by DTW distance
top_n_cells = 10
alignment_col = f"dtw_{ALIGN_TYPE}_aligned"

# Get aligned cells only
aligned_cells = align_n_comp_feat_df[align_n_comp_feat_df[alignment_col]].copy()

# Select top N lineages by distance
if "distance" in aligned_cells.columns and "lineage_id" in aligned_cells.columns:
    # Drop duplicates to get one row per lineage, then select N with smallest distance
    top_lineages = (
        aligned_cells.drop_duplicates("lineage_id")
        .nsmallest(top_n_cells, "distance")["lineage_id"]
        .tolist()
    )
    logger.info(
        f"Selected top {len(top_lineages)} lineages by DTW distance: {top_lineages}"
    )
else:
    top_lineages = aligned_cells["lineage_id"].unique()[:top_n_cells].tolist()
    logger.info(f"Selected {len(top_lineages)} lineages (no distance info)")

# Filter to top lineages - include ALL timepoints (aligned and unaligned)
top_cells_df = align_n_comp_feat_df[
    align_n_comp_feat_df["lineage_id"].isin(top_lineages)
].copy()

logger.info(
    f"Filtered to {len(top_cells_df)} observations from {len(top_lineages)} lineages"
)

# Aggregate using unified function
common_response_df = aggregate_trajectory_by_time(
    top_cells_df,
    feature_columns=common_response_features,
)

# Compute infection timepoint and aligned region for visualization
# These are derived from the alignment metadata BEFORE aggregation
infection_timepoint = None
aligned_region_bounds = None

if alignment_col in top_cells_df.columns:
    aligned_mask = top_cells_df[alignment_col]
    if aligned_mask.any():
        aligned_subset = top_cells_df[aligned_mask]

        # Aligned region: compute median aligned span across lineages
        # Each lineage has an aligned window; we want the typical/median window
        lineage_aligned_regions = []
        for lineage_id in aligned_subset["lineage_id"].unique():
            lineage_aligned = aligned_subset[aligned_subset["lineage_id"] == lineage_id]
            lineage_times = sorted(lineage_aligned["t"].unique())
            if len(lineage_times) > 0:
                lineage_aligned_regions.append((lineage_times[0], lineage_times[-1]))

        if len(lineage_aligned_regions) > 0:
            # Use median start and end across all lineages
            starts = [r[0] for r in lineage_aligned_regions]
            ends = [r[1] for r in lineage_aligned_regions]
            aligned_region_bounds = (int(np.median(starts)), int(np.median(ends)))

        # Infection timepoint: propagate consensus annotations via DTW alignment
        consensus_mapping_col = f"dtw_{ALIGN_TYPE}_consensus_mapping"
        if consensus_annotations and "infected" in consensus_annotations:
            if consensus_mapping_col in aligned_subset.columns:
                # For each aligned cell, look up its annotation from consensus
                # consensus_mapping tells us which position in the consensus pattern
                aligned_subset_copy = aligned_subset.copy()

                # Map consensus position to annotation
                def get_annotation(consensus_pos):
                    idx = int(round(consensus_pos))
                    if 0 <= idx < len(consensus_annotations):
                        return consensus_annotations[idx]
                    return None

                aligned_subset_copy["propagated_annotation"] = aligned_subset_copy[
                    consensus_mapping_col
                ].apply(get_annotation)

                # Find first appearance of "infected" for each lineage
                first_infected_times = []
                for lineage_id in aligned_subset_copy["lineage_id"].unique():
                    lineage_data = aligned_subset_copy[
                        aligned_subset_copy["lineage_id"] == lineage_id
                    ].sort_values("t")
                    infected_rows = lineage_data[
                        lineage_data["propagated_annotation"] == "infected"
                    ]
                    if len(infected_rows) > 0:
                        first_infected_times.append(infected_rows.iloc[0]["t"])

                if len(first_infected_times) > 0:
                    # Use mean of first infection timepoints across lineages
                    infection_timepoint = int(np.mean(first_infected_times))

logger.info(f"Infection timepoint: {infection_timepoint}")
logger.info(f"Aligned region: {aligned_region_bounds}")

# Debug: check if consensus_annotations is available
if consensus_annotations:
    logger.info(
        f"Consensus annotations available: 'infected' at position {consensus_annotations.index('infected') if 'infected' in consensus_annotations else 'NOT FOUND'}"
    )
else:
    logger.warning("consensus_annotations is None or empty!")

# Debug: check what columns are available in top_cells_df
logger.info(
    f"Available alignment columns in top_cells_df: {[c for c in top_cells_df.columns if 'dtw' in c.lower() or 'align' in c.lower()]}"
)


# %%
# Compute uninfected baseline from control wells (B/1)
# Filter to uninfected FOVs with sufficient track length
uninfected_fov_pattern = "B/1"
min_track_length = 20

uninfected_filtered = uninfected_features_df[
    uninfected_features_df["fov_name"].str.contains(uninfected_fov_pattern)
].copy()

# Filter by track length
track_lengths = uninfected_filtered.groupby(["fov_name", "track_id"]).size()
valid_tracks = track_lengths[track_lengths >= min_track_length].index

uninfected_filtered = uninfected_filtered[
    uninfected_filtered.set_index(["fov_name", "track_id"]).index.isin(valid_tracks)
].reset_index(drop=True)

logger.info(
    f"Filtered uninfected cells: {len(valid_tracks)} tracks with >= {min_track_length} timepoints"
)
logger.info(f"Total observations: {len(uninfected_filtered)}")

# Aggregate using unified function
uninfected_baseline_df = aggregate_trajectory_by_time(
    uninfected_filtered,
    feature_columns=common_response_features,
)

logger.info(f"Uninfected baseline shape: {uninfected_baseline_df.shape}")


# %%
# Compute global average of ALL infected cells (B/2) without alignment
# Filter to infected FOVs with sufficient track length
infected_fov_pattern = "B/2"

global_infected_filtered = all_infected_features_df[
    all_infected_features_df["fov_name"].str.contains(infected_fov_pattern)
].copy()

if len(global_infected_filtered) == 0:
    logger.warning(f"No cells found matching pattern '{infected_fov_pattern}'")
    global_infected_df = pd.DataFrame()
else:
    # Filter by track length - use (fov_name, track_id) since lineage_id may not exist
    track_lengths = global_infected_filtered.groupby(["fov_name", "track_id"]).size()
    valid_tracks = track_lengths[track_lengths >= min_track_length].index

    global_infected_filtered = global_infected_filtered[
        global_infected_filtered.set_index(["fov_name", "track_id"]).index.isin(
            valid_tracks
        )
    ].reset_index(drop=True)

    n_tracks = len(valid_tracks)
    logger.info(
        f"Filtered global infected: {n_tracks} tracks with >= {min_track_length} timepoints"
    )
    logger.info(f"Total observations: {len(global_infected_filtered)}")

    # Aggregate using unified function
    global_infected_df = aggregate_trajectory_by_time(
        global_infected_filtered,
        feature_columns=common_response_features,
    )

    logger.info(f"Global infected average shape: {global_infected_df.shape}")


# %%
# Compute baseline normalization values BEFORE plotting
def compute_baseline_normalization_values(
    infected_df: pd.DataFrame,
    uninfected_df: pd.DataFrame,
    global_infected_df: pd.DataFrame,
    feature_columns: list,
    n_baseline_timepoints: int = 10,
):
    """
    Compute baseline normalization values from the first n timepoints of each trajectory.

    Each trajectory (infected, uninfected, global infected) is normalized by its own
    baseline computed from its first n timepoints. This allows comparison of relative
    changes across different trajectories.

    Parameters
    ----------
    infected_df : pd.DataFrame
        Infected common response aggregated dataframe with 't' column
    uninfected_df : pd.DataFrame
        Uninfected baseline aggregated dataframe with 't' column
    global_infected_df : pd.DataFrame
        Global infected average aggregated dataframe with 't' column
    feature_columns : list
        Features to compute baselines for
    n_baseline_timepoints : int
        Number of initial timepoints to use as baseline (default: 10)

    Returns
    -------
    dict
        Baseline values for each feature and trajectory type
    """
    # Compute baseline normalization values from first n timepoints of each trajectory
    baseline_values = {}

    logger.info(
        f"Computing baseline from first {n_baseline_timepoints} timepoints of each trajectory"
    )

    for feature in feature_columns:
        median_col = f"{feature}_median"

        # Initialize baseline dict for this feature
        baseline_values[feature] = {}

        # Infected trajectory baseline - use first n timepoints
        if median_col in infected_df.columns:
            sorted_times = sorted(infected_df["t"].unique())
            baseline_times = sorted_times[:n_baseline_timepoints]
            baseline_mask = infected_df["t"].isin(baseline_times)
            baseline_vals = infected_df.loc[baseline_mask, median_col].dropna()
            if len(baseline_vals) > 0:
                baseline_values[feature]["infected"] = baseline_vals.mean()
                logger.debug(
                    f"  {feature} infected baseline: {baseline_vals.mean():.3f} (from times {min(baseline_times)}-{max(baseline_times)})"
                )
            else:
                baseline_values[feature]["infected"] = None

        # Uninfected trajectory baseline - use first n timepoints
        if median_col in uninfected_df.columns:
            sorted_times = sorted(uninfected_df["t"].unique())
            baseline_times = sorted_times[:n_baseline_timepoints]
            baseline_mask = uninfected_df["t"].isin(baseline_times)
            baseline_vals = uninfected_df.loc[baseline_mask, median_col].dropna()
            if len(baseline_vals) > 0:
                baseline_values[feature]["uninfected"] = baseline_vals.mean()
                logger.debug(
                    f"  {feature} uninfected baseline: {baseline_vals.mean():.3f} (from times {min(baseline_times)}-{max(baseline_times)})"
                )
            else:
                baseline_values[feature]["uninfected"] = None

        # Global infected trajectory baseline - use first n timepoints
        if global_infected_df is not None and median_col in global_infected_df.columns:
            sorted_times = sorted(global_infected_df["t"].unique())
            baseline_times = sorted_times[:n_baseline_timepoints]
            baseline_mask = global_infected_df["t"].isin(baseline_times)
            baseline_vals = global_infected_df.loc[baseline_mask, median_col].dropna()
            if len(baseline_vals) > 0:
                baseline_values[feature]["global"] = baseline_vals.mean()
                logger.debug(
                    f"  {feature} global infected baseline: {baseline_vals.mean():.3f} (from times {min(baseline_times)}-{max(baseline_times)})"
                )
            else:
                baseline_values[feature]["global"] = None

    return baseline_values


# Compute baseline values
baseline_normalization_values = compute_baseline_normalization_values(
    common_response_df,
    uninfected_baseline_df,
    global_infected_df,
    feature_columns=common_response_features,
    n_baseline_timepoints=int(np.floor(min_track_length * 0.75)),
)

logger.info(
    f"Computed baseline values for {len(baseline_normalization_values)} features"
)


# %%
def plot_binned_period_comparison(
    infected_df: pd.DataFrame,
    uninfected_df: pd.DataFrame,
    feature_columns: list,
    infection_time: int,
    baseline_values: dict = None,
    global_infected_df: pd.DataFrame = None,
    output_root: Path = None,
    figsize=(18, 14),
    plot_type: str = "line",
    add_stats: bool = True,
):
    """
    Plot binned period comparison showing fold-change across biological phases.

    Creates line plots or bar plots comparing infected vs uninfected trajectories across
    biologically meaningful periods (baseline, peri-infection, fragmentation, late/death).
    Each period's value is normalized to the baseline period to show fold-change.
    Includes statistical testing to identify significant differences.

    Parameters
    ----------
    infected_df : pd.DataFrame
        Infected common response aggregated dataframe
    uninfected_df : pd.DataFrame
        Uninfected baseline aggregated dataframe
    feature_columns : list
        Features to plot
    infection_time : int
        Infection timepoint (used to define period boundaries)
    baseline_values : dict, optional
        Pre-computed baseline normalization values
    global_infected_df : pd.DataFrame, optional
        Global average of all infected cells
    output_root : Path, optional
        Directory to save output figure
    figsize : tuple
        Figure size
    plot_type : str
        'line' for connected line plots or 'bar' for grouped bar plots (default: 'line')
    add_stats : bool
        If True, perform statistical testing and mark significant differences (default: True)
    """
    from scipy.stats import ttest_ind

    # Define biologically meaningful periods relative to infection
    periods = {
        "Baseline": (infection_time - 10, infection_time),
        "Peri-infection": (infection_time - 2, infection_time + 3),
        "Fragmentation": (infection_time + 5, infection_time + 15),
        "Late/Death": (infection_time + 15, infection_time + 25),
    }

    period_names = list(periods.keys())
    n_periods = len(periods)

    n_features = len(feature_columns)
    ncols = 3
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    # Colorblind-friendly palette
    colors = {
        "uninfected": "#1f77b4",  # blue
        "infected": "#ff7f0e",  # orange
        "global": "#2ca02c",  # green
    }

    # Store statistical results for logging
    stats_results = {}

    for idx, feature in enumerate(feature_columns):
        ax = axes[idx]

        median_col = f"{feature}_median"

        # Check if feature exists
        if (
            median_col not in infected_df.columns
            or median_col not in uninfected_df.columns
        ):
            ax.text(0.5, 0.5, f"{feature}\nno data", ha="center", va="center")
            ax.set_title(feature)
            continue

        # Check if CV/SEM feature (no normalization)
        is_cv_feature = feature.endswith("_cv") or feature.endswith("_sem")

        # Compute values for each period
        period_values = {"uninfected": [], "infected": [], "global": []}
        period_errors = {"uninfected": [], "infected": [], "global": []}

        # Baseline period values for normalization (use mean within baseline period)
        baseline_period = periods["Baseline"]

        def compute_baseline_value(df, feature_col):
            """Compute baseline value from baseline period."""
            mask = (df["t"] >= baseline_period[0]) & (df["t"] <= baseline_period[1])
            values = df.loc[mask, feature_col].dropna()
            return values.mean() if len(values) > 0 else None

        # Compute baseline for this feature from each trajectory
        uninfected_baseline = compute_baseline_value(uninfected_df, median_col)
        infected_baseline = compute_baseline_value(infected_df, median_col)
        global_baseline = None
        if global_infected_df is not None and median_col in global_infected_df.columns:
            global_baseline = compute_baseline_value(global_infected_df, median_col)

        # For each period, compute aggregate value and normalize
        for period_name, (t_start, t_end) in periods.items():
            # Uninfected
            mask = (uninfected_df["t"] >= t_start) & (uninfected_df["t"] <= t_end)
            values = uninfected_df.loc[mask, median_col].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()

                # Normalize to baseline if not CV feature
                if not is_cv_feature and uninfected_baseline is not None:
                    mean_val = mean_val / uninfected_baseline
                    std_val = std_val / (np.abs(uninfected_baseline) + 1e-6)

                period_values["uninfected"].append(mean_val)
                period_errors["uninfected"].append(std_val)
            else:
                period_values["uninfected"].append(np.nan)
                period_errors["uninfected"].append(np.nan)

            # Infected
            mask = (infected_df["t"] >= t_start) & (infected_df["t"] <= t_end)
            values = infected_df.loc[mask, median_col].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()

                if not is_cv_feature and infected_baseline is not None:
                    mean_val = mean_val / infected_baseline
                    std_val = std_val / (np.abs(infected_baseline) + 1e-6)

                period_values["infected"].append(mean_val)
                period_errors["infected"].append(std_val)
            else:
                period_values["infected"].append(np.nan)
                period_errors["infected"].append(np.nan)

            # Global infected
            if (
                global_infected_df is not None
                and median_col in global_infected_df.columns
            ):
                mask = (global_infected_df["t"] >= t_start) & (
                    global_infected_df["t"] <= t_end
                )
                values = global_infected_df.loc[mask, median_col].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()

                    if not is_cv_feature and global_baseline is not None:
                        mean_val = mean_val / global_baseline
                        std_val = std_val / (np.abs(global_baseline) + 1e-6)

                    period_values["global"].append(mean_val)
                    period_errors["global"].append(std_val)
                else:
                    period_values["global"].append(np.nan)
                    period_errors["global"].append(np.nan)

        # Statistical testing between infected and uninfected at each period
        p_values = []
        if add_stats:
            stats_results[feature] = {}
            for period_name, (t_start, t_end) in periods.items():
                # Get raw values (not aggregated medians) for statistical testing
                uninfected_mask = (uninfected_df["t"] >= t_start) & (
                    uninfected_df["t"] <= t_end
                )
                infected_mask = (infected_df["t"] >= t_start) & (
                    infected_df["t"] <= t_end
                )

                uninfected_vals = uninfected_df.loc[
                    uninfected_mask, median_col
                ].dropna()
                infected_vals = infected_df.loc[infected_mask, median_col].dropna()

                if len(uninfected_vals) >= 3 and len(infected_vals) >= 3:
                    _, p_val = ttest_ind(uninfected_vals, infected_vals)
                    p_values.append(p_val)
                    stats_results[feature][period_name] = p_val
                else:
                    p_values.append(np.nan)
                    stats_results[feature][period_name] = np.nan

        x = np.arange(n_periods)

        if plot_type == "line":
            # Line plot with error bars
            ax.errorbar(
                x,
                period_values["uninfected"],
                yerr=period_errors["uninfected"],
                label="Uninfected (B/1)",
                color=colors["uninfected"],
                marker="o",
                markersize=8,
                linewidth=2.5,
                capsize=4,
                capthick=2,
            )
            ax.errorbar(
                x,
                period_values["infected"],
                yerr=period_errors["infected"],
                label="Infected top-N (B/2)",
                color=colors["infected"],
                marker="s",
                markersize=8,
                linewidth=2.5,
                capsize=4,
                capthick=2,
            )

            if global_infected_df is not None:
                ax.errorbar(
                    x,
                    period_values["global"],
                    yerr=period_errors["global"],
                    label="All B/2 cells",
                    color=colors["global"],
                    marker="^",
                    markersize=7,
                    linewidth=2,
                    linestyle="--",
                    capsize=4,
                    capthick=1.5,
                    alpha=0.8,
                )

            # Mark significant differences with asterisks
            if add_stats and len(p_values) > 0:
                y_max = max(
                    [
                        max(
                            [v for v in period_values["uninfected"] if not np.isnan(v)]
                        ),
                        max([v for v in period_values["infected"] if not np.isnan(v)]),
                    ]
                )
                y_offset = 0.1 * (y_max - 1.0) if not is_cv_feature else 0.1 * y_max

                for i, p_val in enumerate(p_values):
                    if not np.isnan(p_val):
                        # Determine significance level
                        if p_val < 0.001:
                            marker = "***"
                        elif p_val < 0.01:
                            marker = "**"
                        elif p_val < 0.05:
                            marker = "*"
                        else:
                            marker = "ns"

                        if marker != "ns":
                            # Position text above the higher of the two values
                            max_val = max(
                                period_values["uninfected"][i],
                                period_values["infected"][i],
                            )
                            ax.text(
                                x[i],
                                max_val + y_offset,
                                marker,
                                ha="center",
                                va="bottom",
                                fontsize=10,
                                fontweight="bold",
                                color="black",
                            )

        else:  # bar plot
            width = 0.25

            ax.bar(
                x - width,
                period_values["uninfected"],
                width,
                label="Uninfected (B/1)",
                color=colors["uninfected"],
                yerr=period_errors["uninfected"],
                capsize=3,
            )
            ax.bar(
                x,
                period_values["infected"],
                width,
                label="Infected top-N (B/2)",
                color=colors["infected"],
                yerr=period_errors["infected"],
                capsize=3,
            )

            if global_infected_df is not None:
                ax.bar(
                    x + width,
                    period_values["global"],
                    width,
                    label="All B/2 cells",
                    color=colors["global"],
                    yerr=period_errors["global"],
                    capsize=3,
                    alpha=0.8,
                )

            # Mark significant differences with brackets
            if add_stats and len(p_values) > 0:
                y_max = max(
                    [
                        max(
                            [v for v in period_values["uninfected"] if not np.isnan(v)]
                        ),
                        max([v for v in period_values["infected"] if not np.isnan(v)]),
                    ]
                )
                y_offset = 0.1 * (y_max - 1.0) if not is_cv_feature else 0.1 * y_max

                for i, p_val in enumerate(p_values):
                    if not np.isnan(p_val) and p_val < 0.05:
                        if p_val < 0.001:
                            marker = "***"
                        elif p_val < 0.01:
                            marker = "**"
                        else:
                            marker = "*"

                        max_val = max(
                            period_values["uninfected"][i], period_values["infected"][i]
                        )
                        ax.text(
                            x[i],
                            max_val + y_offset,
                            marker,
                            ha="center",
                            va="bottom",
                            fontsize=10,
                            fontweight="bold",
                        )

        # Add horizontal line at 1.0 (no change from baseline)
        if not is_cv_feature:
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        ax.set_xlabel("Period")
        if is_cv_feature:
            ax.set_ylabel(f"{feature}\n(raw value)")
        else:
            ax.set_ylabel(f"{feature}\n(fold-change from baseline)")
        ax.set_title(feature)
        ax.set_xticks(x)
        ax.set_xticklabels(period_names, rotation=45, ha="right")
        # ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    # Create title with statistical note
    title = "Binned Period Comparison: Fold-Change Across Infection Phases"
    if add_stats:
        title += "\n(* p<0.05, ** p<0.01, *** p<0.001)"

    plt.suptitle(
        title,
        fontsize=14,
        y=1.00,
    )
    plt.tight_layout()

    if output_root is not None:
        save_path = output_root / f"binned_period_comparison_{ALIGN_TYPE}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved binned period comparison to {save_path}")

    plt.show()

    # Log summary in markdown format
    logger.info("\n## Binned Period Comparison Summary")
    logger.info(f"**Infection timepoint:** {infection_time}")
    logger.info("\n### Period Definitions")
    for period_name, (t_start, t_end) in periods.items():
        logger.info(f"- **{period_name}:** t={t_start} to t={t_end}")

    if add_stats and len(stats_results) > 0:
        logger.info("\n### Statistical Significance (t-tests)")
        logger.info(
            "Comparing infected top-N vs uninfected at each period. Significance levels: * p<0.05, ** p<0.01, *** p<0.001\n"
        )

        # Create markdown table
        logger.info(f"| Feature | {' | '.join(period_names)} |")
        logger.info(f"|---------|{'---------|-' * (len(period_names) - 1)}---------|\n")

        for feature, period_results in stats_results.items():
            sig_markers = []
            for period_name in period_names:
                p_val = period_results.get(period_name, np.nan)
                if np.isnan(p_val):
                    sig_markers.append("N/A")
                elif p_val < 0.001:
                    sig_markers.append(f"***({p_val:.3e})")
                elif p_val < 0.01:
                    sig_markers.append(f"**({p_val:.3f})")
                elif p_val < 0.05:
                    sig_markers.append(f"*({p_val:.3f})")
                else:
                    sig_markers.append(f"ns({p_val:.3f})")

            logger.info(f"| {feature} | {' | '.join(sig_markers)} |")

        logger.info("\nns = not significant (p >= 0.05)")


# %%
# Compare infected (aligned) vs uninfected (baseline) trajectories
def plot_infected_vs_uninfected_comparison(
    infected_df: pd.DataFrame,
    uninfected_df: pd.DataFrame,
    feature_columns: list,
    baseline_values: dict = None,
    infection_time: int = None,
    aligned_region: tuple = None,
    figsize=(18, 14),
    n_consecutive_divergence: int = 5,
    global_infected_df: pd.DataFrame = None,
    normalize_to_baseline: bool = True,
):
    """
    Plot comparison of infected (DTW-aligned) vs uninfected (baseline) trajectories.

    Shows where infected cells diverge from normal behavior (crossover points).
    Includes infection timepoint marker, aligned region highlighting, and consecutive divergence detection.

    NOTE: Features ending in '_cv' or '_sem' are plotted as raw values without baseline
    normalization, since CV and SEM are already relative/uncertainty metrics.

    Parameters
    ----------
    infected_df : pd.DataFrame
        Infected common response with 'time', 'is_aligned', and 'aligned_time' columns
    uninfected_df : pd.DataFrame
        Uninfected baseline
    feature_columns : list
        Features to plot
    baseline_values : dict, optional
        Pre-computed baseline normalization values for each feature and trajectory
    infection_time : int, optional
        Pre-computed infection timepoint in raw time
    aligned_region : tuple, optional
        Pre-computed aligned region boundaries (start, end)
    figsize : tuple
        Figure size
    n_consecutive_divergence : int
        Number of consecutive timepoints required to confirm divergence (default: 5)
    global_infected_df : pd.DataFrame, optional
        Global average of ALL infected cells without alignment (if provided, will be plotted)
    normalize_to_baseline : bool
        If True, normalize all trajectories to pre-infection baseline showing fold-change (default: True).
        CV and SEM features (ending in '_cv' or '_sem') are always plotted as raw values regardless of this setting.
    """
    from scipy.interpolate import interp1d

    n_features = len(feature_columns)
    ncols = 3
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    # Colorblind-friendly palette: blue (uninfected) vs orange (infected)
    uninfected_color = "#1f77b4"  # blue
    infected_color = "#ff7f0e"  # orange

    # Use pre-computed baseline values or initialize empty dict
    if baseline_values is None:
        baseline_values = {}
        logger.warning("No baseline_values provided - plots will not be normalized")

    for idx, feature in enumerate(feature_columns):
        ax = axes[idx]

        median_col = f"{feature}_median"
        q25_col = f"{feature}_q25"
        q75_col = f"{feature}_q75"

        # Check if data exists
        if (
            median_col not in infected_df.columns
            or median_col not in uninfected_df.columns
        ):
            ax.text(0.5, 0.5, f"{feature}\nno data", ha="center", va="center")
            ax.set_title(feature)
            continue

        # Highlight aligned region first (background layer)
        if aligned_region is not None:
            ax.axvspan(
                aligned_region[0],
                aligned_region[1],
                alpha=0.1,
                color="gray",
                label="Aligned region",
                zorder=0,
            )

        # Plot uninfected baseline
        uninfected_time = uninfected_df["t"].values
        uninfected_median = uninfected_df[median_col].values
        uninfected_q25 = uninfected_df[q25_col].values
        uninfected_q75 = uninfected_df[q75_col].values

        # Check if this is a CV or SEM feature (skip normalization for relative/uncertainty metrics)
        is_cv_feature = feature.endswith("_cv") or feature.endswith("_sem")

        # Apply baseline normalization if requested (but skip for CV features)
        if (
            normalize_to_baseline
            and not is_cv_feature
            and feature in baseline_values
            and baseline_values[feature]["uninfected"] is not None
        ):
            baseline = baseline_values[feature]["uninfected"]
            uninfected_median = (uninfected_median - baseline) / (
                np.abs(baseline) + 1e-6
            )
            uninfected_q25 = (uninfected_q25 - baseline) / (np.abs(baseline) + 1e-6)
            uninfected_q75 = (uninfected_q75 - baseline) / (np.abs(baseline) + 1e-6)

        ax.plot(
            uninfected_time,
            uninfected_median,
            color=uninfected_color,
            linewidth=2.5,
            label="Uninfected (B/1)",
            linestyle="-",
        )
        ax.fill_between(
            uninfected_time,
            uninfected_q25,
            uninfected_q75,
            color=uninfected_color,
            alpha=0.2,
        )

        # Plot infected aligned response
        infected_time = infected_df["t"].values
        infected_median = infected_df[median_col].values
        infected_q25 = infected_df[q25_col].values
        infected_q75 = infected_df[q75_col].values

        # Apply baseline normalization if requested (but skip for CV features)
        if (
            normalize_to_baseline
            and not is_cv_feature
            and feature in baseline_values
            and baseline_values[feature]["infected"] is not None
        ):
            baseline = baseline_values[feature]["infected"]
            infected_median = (infected_median - baseline) / (np.abs(baseline) + 1e-6)
            infected_q25 = (infected_q25 - baseline) / (np.abs(baseline) + 1e-6)
            infected_q75 = (infected_q75 - baseline) / (np.abs(baseline) + 1e-6)

        ax.plot(
            infected_time,
            infected_median,
            color=infected_color,
            linewidth=2.5,
            label="Infected top-N aligned (B/2)",
            linestyle="-",
        )
        ax.fill_between(
            infected_time,
            infected_q25,
            infected_q75,
            color=infected_color,
            alpha=0.2,
        )

        # Plot global infected average (all B/2 cells, no alignment) if provided
        if global_infected_df is not None and median_col in global_infected_df.columns:
            global_time = global_infected_df["t"].values
            global_median = global_infected_df[median_col].values
            global_q25 = (
                global_infected_df[q25_col].values
                if q25_col in global_infected_df.columns
                else None
            )
            global_q75 = (
                global_infected_df[q75_col].values
                if q75_col in global_infected_df.columns
                else None
            )

            # Apply baseline normalization if requested (but skip for CV features)
            if (
                normalize_to_baseline
                and not is_cv_feature
                and feature in baseline_values
                and baseline_values[feature]["global"] is not None
            ):
                baseline = baseline_values[feature]["global"]
                global_median = (global_median - baseline) / (np.abs(baseline) + 1e-6)
                if global_q25 is not None:
                    global_q25 = (global_q25 - baseline) / (np.abs(baseline) + 1e-6)
                if global_q75 is not None:
                    global_q75 = (global_q75 - baseline) / (np.abs(baseline) + 1e-6)

            ax.plot(
                global_time,
                global_median,
                color="#15ba10",  # red
                linewidth=2,
                label="All B/2 cells (no alignment)",
                linestyle="--",
                alpha=0.8,
            )
            if global_q25 is not None and global_q75 is not None:
                ax.fill_between(
                    global_time,
                    global_q25,
                    global_q75,
                    color="#15ba10",
                    alpha=0.15,
                )

        # Mark infection timepoint
        if infection_time is not None:
            ax.axvline(
                infection_time,
                color="red",
                linestyle="-",
                alpha=0.8,
                linewidth=2.5,
                label="Infection event",
                zorder=5,
            )

        # Find consecutive divergence points
        if len(uninfected_median) > 0 and n_consecutive_divergence > 0:
            uninfected_std = np.nanstd(uninfected_median)

            # Interpolate uninfected to match infected timepoints for comparison
            if len(uninfected_time) > 1 and len(infected_time) > 1:
                # Only interpolate within the range of uninfected data
                min_t = max(uninfected_time.min(), infected_time.min())
                max_t = min(uninfected_time.max(), infected_time.max())

                if min_t < max_t:
                    interp_func = interp1d(
                        uninfected_time,
                        uninfected_median,
                        kind="linear",
                        fill_value="extrapolate",
                    )

                    # Find timepoints where infected is significantly different
                    # Constrain to only timepoints AFTER infection
                    overlap_mask = (infected_time >= min_t) & (infected_time <= max_t)
                    if infection_time is not None:
                        overlap_mask = overlap_mask & (infected_time >= infection_time)

                    overlap_times = infected_time[overlap_mask]
                    overlap_infected = infected_median[overlap_mask]
                    overlap_uninfected = interp_func(overlap_times)

                    divergence = np.abs(overlap_infected - overlap_uninfected)
                    threshold = 1.5 * uninfected_std

                    divergent_mask = divergence > threshold

                    # Find consecutive divergence streaks
                    if np.any(divergent_mask):
                        # Find runs of consecutive True values
                        consecutive_start = None
                        consecutive_count = 0

                        for i, is_divergent in enumerate(divergent_mask):
                            if is_divergent:
                                if consecutive_start is None:
                                    consecutive_start = i
                                consecutive_count += 1

                                # Check if we've reached the required consecutive count
                                if consecutive_count >= n_consecutive_divergence:
                                    # Mark the first divergence point
                                    first_divergence = overlap_times[consecutive_start]
                                    ax.axvline(
                                        first_divergence,
                                        color="red",
                                        linestyle="--",
                                        alpha=0.6,
                                        linewidth=2,
                                        label=f"Divergence (t={first_divergence:.0f})",
                                        zorder=4,
                                    )
                                    break  # Only mark the first sustained divergence
                            else:
                                # Reset counter if streak breaks
                                consecutive_start = None
                                consecutive_count = 0

        ax.set_xlabel("Time")
        # Update y-axis label based on whether CV/SEM feature or normalized
        if feature.endswith("_cv"):
            ax.set_ylabel(f"{feature}\n(raw CV)")
        elif feature.endswith("_sem"):
            ax.set_ylabel(f"{feature}\n(raw SEM)")
        elif normalize_to_baseline and feature in baseline_values:
            ax.set_ylabel(f"{feature}\n(fold-change from baseline)")
        else:
            ax.set_ylabel(feature)
        ax.set_title(feature)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(
        output_root / f"infected_vs_uninfected_comparison_{ALIGN_TYPE}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


# Plot comparison
plot_infected_vs_uninfected_comparison(
    common_response_df,
    uninfected_baseline_df,
    feature_columns=common_response_features,
    baseline_values=baseline_normalization_values,
    infection_time=infection_timepoint,
    aligned_region=aligned_region_bounds,
    figsize=(18, 14),
    # global_infected_df=global_infected_df,
    normalize_to_baseline=True,
)
# %%
# Plot binned period comparison with statistical testing
plot_binned_period_comparison(
    common_response_df,
    uninfected_baseline_df,
    feature_columns=common_response_features,
    infection_time=infection_timepoint,
    baseline_values=baseline_normalization_values,
    # global_infected_df=global_infected_df,
    output_root=output_root,
    figsize=(12, 24),
    plot_type="line",  # Use line plots to show trends across periods
    add_stats=True,  # Include statistical testing with significance markers
)
# %%
# Plot PC/PHATE for all the cells grayed out and the top N aligned cells highlighted with fancyarrows
# Shows unaligned + aligned + unaligned timepoints like in the time-series plots
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 10))
if (
    "organelle_PC1" in combined_features_df.columns
    and "organelle_PC2" in combined_features_df.columns
):
    # Highlight top N aligned cells - include ALL timepoints (unaligned + aligned + unaligned)
    top_aligned_cells = align_n_comp_feat_df[
        align_n_comp_feat_df["lineage_id"].isin(top_lineages[:3])
    ]

    # Filter the top_n_aligned_cells to only include wells B/1 and B/2
    top_n_aligned_cells = top_aligned_cells[
        top_aligned_cells["fov_name"].str.contains("B/1|B/2")
    ]
    # Filter combined features to only include wells B/1 and B/2
    filter_combined_features = combined_features_df[
        combined_features_df["fov_name"].str.contains("B/1|B/2")
    ]
    ax.scatter(
        filter_combined_features["organelle_PC1"],
        filter_combined_features["organelle_PC2"],
        color="lightgray",
        alpha=0.3,
        s=10,
        label="All cells",
        zorder=1,
    )

    # Get colormap for lineages
    n_lineages = len(top_n_aligned_cells["lineage_id"].unique())
    colors = cm.tab10(np.linspace(0, 1, n_lineages))

    # Column name for alignment status
    alignment_col = f"dtw_{ALIGN_TYPE}_aligned"

    # One color per track with temporal arrows, showing aligned vs unaligned timepoints
    for idx, lineage_id in enumerate(top_n_aligned_cells["lineage_id"].unique()):
        lineage_data = top_n_aligned_cells[
            top_n_aligned_cells["lineage_id"] == lineage_id
        ].sort_values("t")

        color = colors[idx]

        # Split into aligned and unaligned portions
        if alignment_col in lineage_data.columns:
            aligned_data = lineage_data[lineage_data[alignment_col]]
            unaligned_data = lineage_data[not lineage_data[alignment_col]]
        else:
            aligned_data = lineage_data
            unaligned_data = pd.DataFrame()

        # Plot unaligned timepoints (pre/post alignment) with smaller, more transparent markers
        if len(unaligned_data) > 0:
            n_unaligned = len(unaligned_data)
            alphas_unaligned = np.linspace(0.3, 0.5, n_unaligned)

            for i, (_, row) in enumerate(unaligned_data.iterrows()):
                ax.scatter(
                    row["organelle_PC1"],
                    row["organelle_PC2"],
                    color=color,
                    alpha=alphas_unaligned[i],
                    s=15,  # Smaller size for unaligned
                    zorder=2,
                    edgecolors="gray",
                    linewidths=0.3,
                    marker="s",  # Square marker for unaligned
                )

        # Plot aligned timepoints with larger, more prominent markers
        if len(aligned_data) > 0:
            n_aligned = len(aligned_data)
            alphas_aligned = np.linspace(0.5, 1.0, n_aligned)

            for i, (_, row) in enumerate(aligned_data.iterrows()):
                ax.scatter(
                    row["organelle_PC1"],
                    row["organelle_PC2"],
                    color=color,
                    alpha=alphas_aligned[i],
                    s=40,  # Larger size for aligned
                    zorder=3,
                    edgecolors="white",
                    linewidths=0.8,
                    marker="o",  # Circle marker for aligned
                )

        # Add fancy arrows connecting ALL consecutive timepoints
        for i in range(len(lineage_data) - 1):
            row_start = lineage_data.iloc[i]
            row_end = lineage_data.iloc[i + 1]

            # Check if this arrow is within aligned region
            is_aligned_arrow = False
            if alignment_col in row_start.index and alignment_col in row_end.index:
                is_aligned_arrow = row_start[alignment_col] and row_end[alignment_col]

            # Create arrow with different styles for aligned vs unaligned
            arrow = FancyArrowPatch(
                (row_start["organelle_PC1"], row_start["organelle_PC2"]),
                (row_end["organelle_PC1"], row_end["organelle_PC2"]),
                arrowstyle="->,head_width=0.4,head_length=0.4",
                color=color,
                alpha=0.7 if is_aligned_arrow else 0.3,
                linewidth=2.0 if is_aligned_arrow else 1.0,
                linestyle="-" if is_aligned_arrow else "--",
                zorder=2,
            )
            ax.add_patch(arrow)

        # Mark first timepoint with a star
        first_row = lineage_data.iloc[0]
        ax.scatter(
            first_row["organelle_PC1"],
            first_row["organelle_PC2"],
            marker="*",
            s=300,
            color=color,
            edgecolors="black",
            linewidths=1.5,
            zorder=4,
            label=f"Track {lineage_id}",
        )

    # Add legend elements for aligned vs unaligned
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="Aligned timepoints",
            markeredgecolor="white",
            markeredgewidth=0.8,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markersize=6,
            label="Unaligned timepoints",
            markeredgecolor="gray",
            markeredgewidth=0.3,
        ),
    ]

    ax.set_xlabel("Organelle PC1")
    ax.set_ylabel("Organelle PC2")
    ax.set_title(
        "PCA of Organelle Channel: Complete Temporal Trajectories\n(Unaligned + Aligned + Unaligned)"
    )

    # Combine legend elements
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        fontsize=8,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_root / f"trajectory_plot_{ALIGN_TYPE}.png", dpi=150, bbox_inches="tight"
    )
    plt.show()
# %%
