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

from viscy.data.triplet import INDEX_COLUMNS, TripletDataset
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
        "features_path_phase": "/hpc/pr"
        "ojects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/anndata_predictions/phase_160patch_104ckpt_ver3max.zarr",
        "features_path_organelle": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/anndata_predictions/organelle_160patch_104ckpt_ver3max.zarr",
        "computed_features_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/quantify_remodeling/feature_list_all.csv",
    },
}

ALIGN_TYPE = "infection_apoptotic"  # Options: "cell_division" or "infection_state" or "apoptosis"
ALIGNMENT_CHANNEL = "sensor"  # sensor,phase,organelle

output_root = Path(
    "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/infection_state/output/2025_06_26_A549_G3BP1_ZIKV"
)
output_root.mkdir(parents=True, exist_ok=True)

# FIXME: find a better logic to manage this
consensus_path = None
# consensus_path = "/path/to/specific/consensus_lineage.pkl"

# If no explicit path provided, check for default consensus file
if consensus_path is None:
    default_consensus = (
        output_root / f"consensus_lineage_{ALIGN_TYPE}_{ALIGNMENT_CHANNEL}.pkl"
    )
    if default_consensus.exists():
        consensus_path = default_consensus

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

min_timepoints = 15
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
        "timepoints": (25 - n_timepoints_before, 25 + n_timepoints_after + 1),
        "annotations": ["uninfected"] * (n_timepoints_before)
        + ["infected"] * (n_timepoints_after // 2 + 1),
        "weight": 1.0,
    },
    {
        "fov_name": "C/2/001001",
        "track_id": [120],
        "timepoints": (30 - n_timepoints_before, 30 + n_timepoints_after + 1),
        "annotations": ["uninfected"] * (n_timepoints_before)
        + ["infected"] * (n_timepoints_after // 2 + 1),
        "weight": 1.0,
    },
]

apoptotic_infected_annotations = [
    {
        "fov_name": "B/2/001000",
        "track_id": [109],
        "timepoints": (25 - n_timepoints_before, 25 + n_timepoints_after + 1),
        "annotations": ["uninfected"] * (n_timepoints_before)
        + ["infected"] * (n_timepoints_after // 2 + 1),
        "weight": 1.0,
    },
    {
        "fov_name": "B/2/001000",
        "track_id": [77],
        "timepoints": (21 - n_timepoints_before, 21 + n_timepoints_after + 1),
        "annotations": ["uninfected"] * (n_timepoints_before)
        + ["infected"] * (n_timepoints_after // 2 + 1),
        "weight": 1.0,
    },
    # Dies but there is no infection. there is some that look infected from the phase but dont die.
    #  in A/2/001000 we see cells that never get infected and all around die
    #     {
    #     "fov_name": "A/2/000001",
    #     "track_id": [137],
    #     "timepoints": (21 - n_timepoints_before, 21 + n_timepoints_after + 1),
    #     "annotations": ["uninfected"] * (n_timepoints_before)
    #     + ["infected"] * (n_timepoints_after // 2 + 1),
    #     "weight": 1.0,
    # },
    {
        "fov_name": "C/2/000001",
        "track_id": [40],
        "timepoints": (24 - n_timepoints_before, 24 + n_timepoints_after + 1),
        "annotations": ["uninfected"] * (n_timepoints_before)
        + ["infected"] * (n_timepoints_after // 2 + 1),
        "weight": 1.0,
    },
    # {
    #     "fov_name": "C/2/001001",
    #     "track_id": [115],
    #     "timepoints": (21 - n_timepoints_before, 2 + n_timepoints_after + 1),
    #     "annotations": ["uninfected"] * (n_timepoints_before)
    #     + ["infected"] * (n_timepoints_after // 2 + 1),
    #     "weight": 1.0,
    # },
]

# Annotations on the 2024_11_21_A549_TOMM20_DENV dataset
infection_annotations = [
    {
        "fov_name": "C/2/001001",
        "track_id": [193],
        "timepoints": (31 - n_timepoints_before, 31 + n_timepoints_after + 1),
        "annotations": ["uinfected"] * (n_timepoints_before)
        + ["infected"]
        + ["uninfected"] * (n_timepoints_after - 1),
        "weight": 1.0,
    },
    {
        "fov_name": "C/2/001000",
        "track_id": [66],
        "timepoints": (19 - n_timepoints_before, 19 + n_timepoints_after + 1),
        "annotations": ["uninfected"] * (n_timepoints_before)
        + ["infected"]
        + ["uninfected"] * (n_timepoints_after - 1),
        "weight": 1.0,
    },
    {
        "fov_name": "C/2/001000",
        "track_id": [54],
        "timepoints": (27 - n_timepoints_before, 27 + n_timepoints_after + 1),
        "annotations": ["uninfected"] * (n_timepoints_before)
        + ["infected"]
        + ["uninfected"] * (n_timepoints_after - 1),
        "weight": 1.0,
    },
    {
        "fov_name": "C/2/001000",
        "track_id": [53],
        "timepoints": (21 - n_timepoints_before, 21 + n_timepoints_after + 1),
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
# Concatenate all patterns to fit PCA on full dataset
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

        ax.set_xlabel("Time")
        ax.set_ylabel(f"PC{pc_idx+1}")
        ax.set_title(
            f'Pattern {i+1}: FOV {info["fov_name"]}, Tracks {info["track_id"]}\nPC{pc_idx+1} over time'
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
            label=f"Ref {i+1}",
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
    ax.set_ylabel(f"PC{pc_idx+1}")
    ax.set_title(f"PC{pc_idx+1}: All DTW-Aligned References + Consensus")
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
print(f'Saving matches to {output_root / f"{name}_matching_lineages_cosine.csv"}')
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
logger.info(
    f"Cell division aligned timepoints: {alignment_df[f'dtw_{ALIGN_TYPE}_aligned'].sum()}/{len(alignment_df)} ({100*alignment_df[f'dtw_{ALIGN_TYPE}_aligned'].mean():.1f}%)"
)

# PCA plotting and alignment visualization is now handled by the enhanced alignment dataframe method
logger.info("Cell division consensus analysis completed successfully!")
print(f"Enhanced DataFrame columns: {alignment_df.columns.tolist()}")

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

    aligned_sequences = get_aligned_image_sequences(
        cytodtw_instance=cytodtw,
        df=top_matches,
        alignment_name=ALIGN_TYPE,
        image_loader_fn=load_images_from_triplet_dataset,
        max_lineages=None,
    )
else:
    aligned_sequences = {}

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
    f"{ALIGN_TYPE.capitalize()} aligned timepoints: {alignment_df[f'dtw_{ALIGN_TYPE}_aligned'].sum()}/{len(alignment_df)} ({100*alignment_df[f'dtw_{ALIGN_TYPE}_aligned'].mean():.1f}%)"
)
logger.info(f"Columns: {list(alignment_df.columns)}")

# Show sample of the enhanced DataFrame
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
cytodtw.plot_individual_lineages(
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
                logger.debug(
                    f"Added {channel_name} channel for lineage {lineage_id} with shape {channel_data.shape}"
                )
# %%
# Load the computer features and PCs

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
# merge with the enhanced dataframe
align_n_comp_feat_df = filtered_alignment_df.merge(
    computed_features_df,
    on=["fov_name", "track_id", "t", "x", "y"],
    how="left",
)

# Add PCs from each channel to the alignment dataframe
for channel, adata in ad_features.items():
    # Create a temporary dataframe with PCs from this channel
    pcs_df = adata.obs[["fov_name", "track_id", "t"]].copy()

    # Add PC columns with channel prefix
    for i in range(n_pca_components):
        pcs_df[f"{channel}_PC{i+1}"] = adata.obsm["X_pca"][:, i]

    # Merge with alignment dataframe
    align_n_comp_feat_df = align_n_comp_feat_df.merge(
        pcs_df,
        on=["fov_name", "track_id", "t"],
        how="left",
    )

    logger.info(
        f"Added {n_pca_components} PCs from {channel} channel to alignment dataframe"
    )
# %%
cytodtw.plot_individual_lineages(
    align_n_comp_feat_df,
    alignment_name=ALIGN_TYPE,
    feature_columns=[
        "sensor_PC1",
        "homogeneity",
        "contrast",
        "edge_density",
        "organelle_volume",
    ],
    max_lineages=10,
    aligned_linewidth=2.5,
    unaligned_linewidth=1.4,
    y_offset_step=0.0,
)
# %%
# Heatmap showing all tracks
cytodtw.plot_global_trends(
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
        "homogeneity",
        "contrast",
        "edge_density",
        "organelle_volume",
    ],
    max_lineages=10,
)

# %%
