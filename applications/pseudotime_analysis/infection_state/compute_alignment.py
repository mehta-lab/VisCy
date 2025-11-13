# %%
import logging
import pickle
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
    compute_hpi_from_absolute_time,
    create_synchronized_warped_sequences,
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
TIME_INTERVAL_MINUTES = 30  # Frame sampling interval in minutes

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
        # "segmentation_features_path": "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/organelle_segmentation/output/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV_mito_features.csv",
        "segmentation_features_path": "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/organelle_segmentation/output/train_test_mito_seg_2/train_test_mito_seg_2_mito_features_nellie.csv",
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

min_timepoints = 10
filtered_lineages = cytodtw.get_lineages(min_timepoints)
filtered_lineages = pd.DataFrame(filtered_lineages, columns=["fov_name", "track_id"])
logger.info(
    f"Found {len(filtered_lineages)} lineages with at least {min_timepoints} timepoints"
)

# %%
# TODO: cleanup annotations
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
DTW_BAND_WIDTH_RATIO = 0.4

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
# Create consensus pattern
# If loading existing consensus, skip pattern plotting and go directly to alignment
if consensus_path is not None and Path(consensus_path).exists():
    logger.info(f"Loading existing consensus from {consensus_path}")
    consensus_result = np.load(consensus_path, allow_pickle=True)
    cytodtw.consensus_data = consensus_result
    logger.info("Skipping pattern plotting - using existing consensus")
else:
    # Plot the sample patterns when creating new consensus
    logger.info("Creating new consensus - plotting sample patterns")
    fig = cytodtw.plot_sample_patterns(
        annotated_samples=aligning_annotations,
        reference_type=REFERENCE_TYPE,
        n_pca_components=3,
    )
    plt.show()

    # Create consensus pattern
    consensus_result = cytodtw.create_consensus_reference_pattern(
        annotated_samples=aligning_annotations,
        reference_selection="median_length",
        aggregation_method="median",
        reference_type=REFERENCE_TYPE,
    )

    # Plot consensus validation
    logger.info("Plotting consensus validation")
    fig = cytodtw.plot_consensus_validation(
        annotated_samples=aligning_annotations,
        reference_type=REFERENCE_TYPE,
        metric="cosine",
        constraint_type=DTW_CONSTRAINT_TYPE,
        band_width_ratio=DTW_BAND_WIDTH_RATIO,
        n_pca_components=3,
    )
    plt.show()

# Extract consensus data for use in alignment
consensus_lineage = consensus_result["pattern"]
consensus_annotations = consensus_result.get("annotations", None)
consensus_metadata = consensus_result["metadata"]

logger.info(f"Consensus pattern shape: {consensus_lineage.shape}")
logger.info(f"Consensus method: {consensus_metadata['aggregation_method']}")
logger.info(f"Reference pattern: {consensus_metadata['reference_pattern']}")
if consensus_annotations:
    logger.info(f"Consensus annotations length: {len(consensus_annotations)}")

# Extract raw infection timepoint from consensus annotations
raw_infection_timepoint = None
if consensus_annotations is not None and "infected" in consensus_annotations:
    consensus_infection_idx = consensus_annotations.index("infected")
    # For apoptotic infections, find the reference cell's infection timepoint
    # We'll update this after alignment with the top-1 cell's actual timepoint
    logger.info(
        f"Consensus infection marker at index {consensus_infection_idx} in consensus space"
    )

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
# Filtering and creating one with just the top n matches
alignment_df = cytodtw.create_alignment_dataframe(
    matches,
    consensus_lineage,
    alignment_name=ALIGN_TYPE,
    reference_type=REFERENCE_TYPE,
)

logger.info(f"Enhanced DataFrame created with {len(alignment_df)} rows")
logger.info(f"Lineages: {alignment_df['lineage_id'].nunique()} (including consensus)")

# Extract reference cell info from alignment_df
distance_col = f"dtw_{ALIGN_TYPE}_distance"
consensus_mapping_col = f"dtw_{ALIGN_TYPE}_consensus_mapping"

# Find reference cell: cell with minimum DTW distance (NaN distances are automatically skipped)
reference_lineage_id = alignment_df.loc[
    alignment_df[distance_col].idxmin(), "lineage_id"
]
reference_cell_rows = alignment_df[
    alignment_df["lineage_id"] == reference_lineage_id
].copy()

# Build reference cell info
reference_cell_info = {
    "fov_name": reference_cell_rows.iloc[0]["fov_name"],
    "track_ids": reference_cell_rows["track_id"].unique().tolist(),
    "dtw_distance": reference_cell_rows.iloc[0][distance_col],
    "lineage_id": reference_lineage_id,
}

# Map consensus infection index to raw timepoint
matching_row = reference_cell_rows[
    reference_cell_rows[consensus_mapping_col] == consensus_infection_idx
]
raw_infection_timepoint = matching_row.iloc[0]["t"] if len(matching_row) > 0 else None
reference_cell_info["raw_infection_timepoint"] = raw_infection_timepoint

logger.info(
    f"Reference cell (top-1 match): FOV={reference_cell_info['fov_name']}, "
    f"lineage_id={reference_lineage_id}, "
    f"track_ids={reference_cell_info['track_ids']}, "
    f"distance={reference_cell_info['dtw_distance']:.4f}"
)

if raw_infection_timepoint is not None:
    logger.info(
        f"Mapped consensus infection (idx={consensus_infection_idx}) to raw timepoint t={raw_infection_timepoint}"
    )
else:
    logger.warning(
        f"Could not map consensus infection index {consensus_infection_idx} to raw timepoint for reference cell"
    )

# %%
# Prototype video alignment based on DTW matches
z_range = slice(0, 1)
initial_yx_patch_size = (192, 192)
# Top matches should be unique fov_name and lineage_id combinations
top_matches = alignment_df.drop_duplicates(subset=["fov_name", "lineage_id"]).head(
    top_n
)

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
        max_lineages=10,
    )
else:
    print("Skipping image sequence creation - no valid dataset available")
    concatenated_image_sequences = {}

# Load concatenated sequences into napari using WARPED SYNCHRONIZED coordinates
if NAPARI and dataset is not None and len(concatenated_image_sequences) > 0:
    import numpy as np

    logger.info("\n" + "=" * 70)
    logger.info("Creating synchronized warped sequences for napari visualization")
    logger.info("=" * 70)

    # Create synchronized warped sequences with HPI metadata
    warped_result = create_synchronized_warped_sequences(
        concatenated_image_sequences=concatenated_image_sequences,
        alignment_df=filtered_alignment_df,
        alignment_name=ALIGN_TYPE,
        consensus_infection_idx=consensus_infection_idx,
        time_interval_minutes=TIME_INTERVAL_MINUTES,
    )

    warped_sequences = warped_result["warped_sequences"]
    alignment_shifts = warped_result["alignment_shifts"]
    warped_metadata = warped_result["warped_metadata"]

    # Log HPI mapping information
    logger.info("\n" + "=" * 70)
    logger.info("Alignment Shifts and HPI Metadata")
    logger.info("=" * 70)
    for lineage_id, shift_info in alignment_shifts.items():
        logger.info(f"\nLineage {lineage_id}:")
        logger.info(
            f"  FOV: {shift_info['fov_name']}, tracks: {shift_info['track_ids']}"
        )
        logger.info(
            f"  first_aligned_t (pseudotime 0): {shift_info['first_aligned_t']}"
        )
        logger.info(
            f"  infection_t_abs (biological event): {shift_info['infection_t_abs']}"
        )
        logger.info(f"  shift (t_viz = t_abs + shift): {shift_info['shift']}")
        logger.info(
            f"  infection_offset_in_viz: {shift_info['infection_offset_in_viz']} frames from pseudotime 0"
        )

        # Example HPI calculation
        example_t = shift_info["infection_t_abs"] + 5
        example_hpi = compute_hpi_from_absolute_time(
            t_abs=example_t,
            alignment_shifts=alignment_shifts,
            lineage_id=lineage_id,
            time_interval_minutes=TIME_INTERVAL_MINUTES,
        )
        logger.info(
            f"  Example: t_abs={example_t} → HPI={example_hpi:.2f} hours post infection"
        )

    # Load into napari using warped coordinates
    logger.info("\n" + "=" * 70)
    logger.info("Loading warped sequences into napari")
    logger.info("=" * 70)

    aligned_start_idx = warped_metadata["aligned_region_start_idx"]
    aligned_end_idx = warped_metadata["aligned_region_end_idx"]

    logger.info(
        f"Warped time structure: {warped_metadata['total_warped_length']} total frames"
    )
    logger.info(f"  Frames [0 to {aligned_start_idx - 1}]: Before aligned region")
    logger.info(
        f"  Frames [{aligned_start_idx} to {aligned_end_idx - 1}]: ALIGNED REGION (synchronized!)"
    )
    logger.info(f"  Frames [{aligned_end_idx} to end]: After aligned region")
    logger.info("  Black frames = zero-padding (cell has no data at this warped time)")

    for lineage_id, warped_time_series in warped_sequences.items():
        meta = concatenated_image_sequences[lineage_id]["metadata"]
        n_channels = warped_time_series.shape[1]

        logger.info(
            f"Lineage {lineage_id}: {warped_time_series.shape[0]} warped frames, {n_channels} channels"
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
            channel_data = warped_time_series[:, channel_idx, :, :, :]
            channel_name = (
                processing_channels[channel_idx]
                if channel_idx < len(processing_channels)
                else f"ch{channel_idx}"
            )
            # Use WARPED prefix to indicate warped/synchronized time coordinates
            layer_name = f"WARPED_track_id_{meta['track_ids'][0]}_FOV_{meta['fov_name']}_dist_{meta['dtw_distance']:.3f}_{channel_name}"

            viewer.add_image(
                channel_data,
                name=layer_name,
                contrast_limits=(channel_data.min(), channel_data.max()),
                colormap=colormap[channel_idx],
                blending="additive",
            )
            logger.debug(f"Added {channel_name} channel for lineage {lineage_id}")

        # Create shapes layer with circle markers for aligned region
        img_height = warped_time_series.shape[3]  # y dimension
        img_width = warped_time_series.shape[4]  # x dimension
        circle_center_y = img_height * 0.1  # 10% from top
        circle_center_x = img_width * 0.1  # 10% from left
        circle_radius = min(img_height, img_width) * 0.05  # 5% of smaller dimension

        # Create circles for each aligned frame in warped time
        shapes_data_warped = []
        for frame_idx in range(aligned_start_idx, aligned_end_idx):
            # Ellipse defined by 4 corners of bounding box: [t, z, y, x]
            ellipse = np.array(
                [
                    [
                        frame_idx,
                        0,
                        circle_center_y - circle_radius,
                        circle_center_x - circle_radius,
                    ],  # top-left
                    [
                        frame_idx,
                        0,
                        circle_center_y - circle_radius,
                        circle_center_x + circle_radius,
                    ],  # top-right
                    [
                        frame_idx,
                        0,
                        circle_center_y + circle_radius,
                        circle_center_x + circle_radius,
                    ],  # bottom-right
                    [
                        frame_idx,
                        0,
                        circle_center_y + circle_radius,
                        circle_center_x - circle_radius,
                    ],  # bottom-left
                ]
            )
            shapes_data_warped.append(ellipse)

        if len(shapes_data_warped) > 0:
            # Add shapes layer with cyan circles marking aligned frames
            viewer.add_shapes(
                shapes_data_warped,
                shape_type="ellipse",
                edge_width=3,
                edge_color="cyan",
                face_color="transparent",
                name=f"WARPED_ALIGNED_MARKER_track_id_{meta['track_ids'][0]}_FOV_{meta['fov_name']}",
                ndim=4,
            )
            logger.info(
                f"Added alignment marker for lineage {lineage_id} "
                f"(frames {aligned_start_idx} to {aligned_end_idx - 1} marked with cyan circles)"
            )

    logger.info("\n" + "=" * 70)
    logger.info("Napari visualization complete!")
    logger.info("=" * 70)
    logger.info("All cells are synchronized:")
    logger.info(
        f"  - Frame {aligned_start_idx} = START of aligned region (pseudotime 0)"
    )
    logger.info(
        f"  - Frame {aligned_start_idx + consensus_infection_idx} = INFECTION EVENT"
    )
    logger.info(f"  - Frame {aligned_end_idx - 1} = END of aligned region")
    logger.info("\nVisual indicators:")
    logger.info("  - Cyan circles in top-left corner = ALIGNED frames (synchronized!)")
    logger.info("  - Black frames = zero-padding (cell has no data at that time)")
    logger.info("\nScrub through frames to see synchronized biological progression!")
    logger.info("=" * 70)
# %%
if segmentation_features_path is not None or Path(segmentation_features_path).exists():
    # Get the segmentation based features and compute per-cell aggregates
    segmentation_features_df = pd.read_csv(segmentation_features_path)
    segmentation_features_df["fov_name"] = segmentation_features_df[
        "fov_name"
    ].str.lstrip("/")

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
            # "segs_mean_intensity": group["mean_intensity"].mean(),
            # "segs_std_intensity_across_mitos": group["mean_intensity"].std(),
            # "segs_mean_max_intensity": group["max_intensity"].mean(),
            # Texture metrics (aggregated)
            # "segs_mean_texture_contrast": group["texture_contrast"].mean(),
            # "segs_mean_texture_homogeneity": group["texture_homogeneity"].mean(),
            # Frangi filter metrics (tubularity/network structure)
            "segs_mean_frangi_mean": group["frangi_mean_intensity"].mean(),
            "segs_mean_frangi_std": group["frangi_std_intensity"].mean(),
            # Shape diversity (coefficient of variation)
            "segs_area_cv": group["area"].std() / (group["area"].mean() + 1e-6),
            "segs_eccentricity_cv": group["eccentricity"].std()
            / (group["eccentricity"].mean() + 1e-6),
            "segs_solidity_cv": group["solidity"].std()
            / (group["solidity"].mean() + 1e-6),
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


# %%
if segmentation_features_path is not None and Path(segmentation_features_path).exists():
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
# Merge alignment_df with combined_features_df to create master features dataframe
master_features_df = alignment_df.merge(
    combined_features_df,
    on=["fov_name", "track_id", "t", "x", "y"],
    how="outer",  # Use outer to keep all tracking data, not just aligned
)

logger.info(f"Created master features dataframe. Shape: {master_features_df.shape}")
logger.info(f"Columns: {list(master_features_df.columns)}")

# Save master features dataframe
output_path = output_root / f"master_features_{ALIGN_TYPE}_{ALIGNMENT_CHANNEL}.csv"
master_features_df.to_csv(output_path, index=False)
logger.info(f"Saved master features dataframe to {output_path}")

# Save alignment metadata (including warped visualization metadata if available)
metadata = {
    "consensus_pattern": consensus_lineage,
    "consensus_annotations": consensus_annotations,
    "consensus_metadata": consensus_metadata,
    "reference_cell_info": reference_cell_info,  # Top-1 cell's full trajectory info
    "raw_infection_timepoint": raw_infection_timepoint,  # Infection timepoint in raw data space
    "infection_timepoint": raw_infection_timepoint,  # Backward compatibility
    "aligned_region_bounds": None,  # Will be computed in visualization script
    "alignment_type": ALIGN_TYPE,
    "alignment_channel": ALIGNMENT_CHANNEL,
    "time_interval_minutes": TIME_INTERVAL_MINUTES,  # For HPI conversion
}

# Add warped visualization metadata if napari visualization was run
if NAPARI and "alignment_shifts" in locals():
    metadata["alignment_shifts"] = alignment_shifts
    metadata["warped_metadata"] = warped_metadata
    logger.info("Added warped visualization metadata to save file")

metadata_path = output_root / f"alignment_metadata_{ALIGN_TYPE}_{ALIGNMENT_CHANNEL}.pkl"
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)
logger.info(f"Saved alignment metadata to {metadata_path}")

logger.info("\n## Pipeline Complete!")
logger.info("To visualize results, run visualize_alignment.py with the saved outputs:")

# %%
