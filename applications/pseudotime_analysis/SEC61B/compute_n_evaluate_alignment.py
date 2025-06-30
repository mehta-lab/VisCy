# %%
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.dtw import find_pattern_matches, identify_lineages
from glob import glob

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
# %%
# Alignment to infection state

annotations_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/4-phenotyping/0-annotation/track_infection_annotation_C_2_000001.csv"
)
data_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/2-assemble/2024_11_07_A549_SEC61_DENV.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/2-assemble/tracking.zarr"
)
dynaclr_features_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/4-phenotyping/prediction_infection/2chan_192patch_100ckpt_timeAware_ntxent_rerun.zarr"
)
openphenom_features_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/4-phenotyping/dtw_evaluation/OpenPhenom/20241107_sensor_n_phase_openphenom_2.zarr"
)

output_root = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/4-phenotyping/dtw_evaluation"
)
output_root.mkdir(parents=True, exist_ok=True)

# %%
dynaclr_features_dataset = read_embedding_dataset(dynaclr_features_path)
openphenom_features_dataset = read_embedding_dataset(openphenom_features_path)

dynaclr_feature_df = (
    dynaclr_features_dataset["sample"].to_dataframe().reset_index(drop=True)
)
openphenom_feature_df = (
    openphenom_features_dataset["sample"].to_dataframe().reset_index(drop=True)
)

input_annotations_df = pd.read_csv(annotations_path)
# annotations_fov_id = "/C/2/000001"
annotations_fov_id = None

# TODO: A bit clunky since some functions take xarray and others take dataframes
features_dict = {
    "dynaclr": dynaclr_feature_df,
    "openphenom": openphenom_feature_df,
}

embeddings_dict = {
    "dynaclr": dynaclr_features_dataset,
    "openphenom": openphenom_features_dataset,
}
lineages_dict = {}

min_timepoints = 30
for name, embeddings_dataset in embeddings_dict.items():
    lineages_dict[f"{name}_lineages"] = identify_lineages(
        embeddings_dataset, min_timepoints=min_timepoints
    )
    logger.info(
        f"Found {len(lineages_dict[f'{name}_lineages'])} {name} lineages with at least {min_timepoints} timepoints"
    )

# Filter lineages to only include those from the annotations fov
filtered_lineages = {}


for name, lineages in lineages_dict.items():
    filtered_lineages[name] = []
    for fov_id, track_ids in lineages:
        if fov_id == annotations_fov_id:
            filtered_lineages[name].append((fov_id, track_ids))
        elif annotations_fov_id is None and fov_id.startswith("/C/2"):
            filtered_lineages[name].append((fov_id, track_ids))
    logger.info(
        f"Found {len(filtered_lineages[name])} {name} lineages from the annotations fov"
    )


# %%
# Condition to align:
CONDITION_TO_ALIGN = "infection"

if CONDITION_TO_ALIGN == "infection":
    reference_lineage_fov = "/C/2/000001"
    reference_lineage_track_id = [138]
    reference_timepoints = [10, 70]

elif CONDITION_TO_ALIGN == "cell_division":
    reference_lineage_fov = "/C/2/000001"
    reference_lineage_track_id = [107, 108, 109]
    reference_timepoints = [25, 70]

# Get the reference pattern for each model
reference_patterns = {}

for name, lineages in filtered_lineages.items():
    base_name = name.split("_")[0]
    embeddings_dataset = embeddings_dict[base_name]

    # Debug: Print available lineages for this model
    logger.info(f"Available {base_name} lineages:")
    for fov_id, track_ids in lineages:
        logger.info(f"  FOV: {fov_id}, track_ids: {track_ids}")

    # Try to find the exact reference lineage
    reference_found = False
    for fov_id, track_ids in lineages:
        if fov_id == reference_lineage_fov and all(
            track_id in track_ids for track_id in reference_lineage_track_id
        ):
            # Construct reference pattern by concatenating individual track embeddings
            track_embeddings_list = []
            for track_id in reference_lineage_track_id:
                track_embeddings = embeddings_dataset.sel(
                    sample=(fov_id, track_id)
                ).features.values
                track_embeddings_list.append(track_embeddings)

            # Concatenate all track embeddings
            reference_pattern = np.concatenate(track_embeddings_list, axis=0)
            reference_pattern = reference_pattern[
                reference_timepoints[0] : reference_timepoints[1]
            ]
            reference_patterns[base_name] = reference_pattern
            reference_found = True
            logger.info(
                f"Found exact reference for {base_name}: {fov_id}, {reference_lineage_track_id}"
            )
            break

    # If exact reference not found, use the first available lineage as reference
    if not reference_found:
        logger.warning(
            f"Exact reference not found for {base_name}, using first available lineage"
        )
        if lineages:
            fov_id, track_ids = lineages[0]
            # Use the first track in the lineage as reference
            first_track_id = track_ids[0]
            track_embeddings = embeddings_dataset.sel(
                sample=(fov_id, first_track_id)
            ).features.values

            # Use the same time range as the original reference
            if len(track_embeddings) >= reference_timepoints[1]:
                reference_pattern = track_embeddings[
                    reference_timepoints[0] : reference_timepoints[1]
                ]
                reference_patterns[base_name] = reference_pattern
                logger.info(
                    f"Using fallback reference for {base_name}: {fov_id}, {first_track_id}"
                )
            else:
                logger.error(
                    f"Track {first_track_id} in {base_name} doesn't have enough timepoints"
                )
                continue

# Validate that we have reference patterns for all models
for base_name in embeddings_dict.keys():
    if base_name not in reference_patterns:
        raise ValueError(f"Reference pattern not found for {base_name} model")

# Add debugging information
for base_name, pattern in reference_patterns.items():
    logger.info(f"{base_name} reference pattern shape: {pattern.shape}")
    logger.info(
        f"{base_name} reference pattern timepoints: {reference_timepoints[0]} to {reference_timepoints[1]}"
    )
    logger.info(f"{base_name} reference pattern length: {len(pattern)}")

    # Validate that reference pattern has the expected shape (timepoints, features)
    if len(pattern.shape) != 2:
        raise ValueError(
            f"{base_name} reference pattern should be 2D (timepoints, features), got shape {pattern.shape}"
        )

# %%
METRIC = "cosine"
alignment_results = {}
for name, lineages in filtered_lineages.items():
    base_name = name.split("_")[0]
    for fov_id, track_ids in lineages:
        embeddings = embeddings_dict[base_name]
        # Find all matches to the reference pattern
        all_match_positions = find_pattern_matches(
            reference_patterns[base_name],
            lineages,
            embeddings,
            window_step=2,
            num_candidates=55,
            method="bernd_clifford",
            save_path=str(output_root / f"{name}_matching_lineages_{METRIC}.csv"),
            metric=METRIC,
            n_jobs=15,
            show_inner_progress=True,
        )
        alignment_results[name] = all_match_positions

# %%
# Align the lineages to the reference pattern
# Extract aligned embeddings for each model
aligned_embeddings = {}
for model_name, match_positions_df in alignment_results.items():
    base_name = model_name.split("_")[0]
    embeddings_dataset = embeddings_dict[base_name]

    # Initialize the model's dictionary if it doesn't exist
    if model_name not in aligned_embeddings:
        aligned_embeddings[model_name] = {}

    # Process each row in the DataFrame
    for idx, row in match_positions_df.iterrows():
        logger.debug(f"Processing lineage {idx}")
        logger.debug(f"Lineage data: {row.to_dict()}")
        warp_path = row["warp_path"]
        start_time = int(row["start_timepoint"])
        end_time = int(row["end_timepoint"])
        fov_name = row.get("fov_name")
        track_ids = row.get("track_ids")

        # Extract embeddings for this lineage
        lineage_embeddings_list = []
        lineage_pc_components_list = []  # Changed: store individual PC components
        lineage_phate_components_list = []  # Changed: store individual PHATE components

        for track_id in track_ids:
            try:
                # Extract the full track data
                track_data = embeddings_dataset.sel(sample=(fov_name, track_id))

                # Use full embeddings for alignment (as before)
                track_embeddings = track_data.features.values

                # Extract PC and PHATE components separately for analysis
                track_pc_components = {}  # Changed: store as dictionary
                track_phate_components = {}  # Changed: store as dictionary

                # Add all available PCA components dynamically
                pc_components = [
                    str(col) for col in track_data.coords if str(col).startswith("PC")
                ]
                pc_components.sort(key=lambda x: int(x[3:]))  # Sort by PCA number

                for pc_col in pc_components:
                    pc_values = track_data[pc_col].values
                    track_pc_components[pc_col] = (
                        pc_values  # Changed: store individually
                    )

                # Add and process dynamically PHATE components
                phate_components = [
                    str(col)
                    for col in track_data.coords
                    if str(col).startswith("PHATE")
                ]
                phate_components.sort(key=lambda x: int(x[5:]))  # Sort by PHATE number

                for phate_col in phate_components:
                    phate_values = track_data[phate_col].values
                    track_phate_components[phate_col] = (
                        phate_values  # Changed: store individually
                    )

                lineage_embeddings_list.append(track_embeddings)
                lineage_pc_components_list.append(track_pc_components)
                lineage_phate_components_list.append(track_phate_components)

                logger.debug(
                    f"Extracted {track_embeddings.shape} full embeddings for track {track_id}"
                )
                logger.debug(f"  PC components: {list(track_pc_components.keys())}")
                logger.debug(
                    f"  PHATE components: {list(track_phate_components.keys())}"
                )

            except Exception as e:
                logger.warning(
                    f"Could not extract embeddings for {fov_name}, {track_id}: {e}"
                )
                continue

        # Concatenate all track embeddings for this lineage
        lineage_embeddings = np.concatenate(lineage_embeddings_list, axis=0)

        # Changed: Combine PC and PHATE components across tracks
        lineage_pc_components = {}
        lineage_phate_components = {}

        # Get all unique PC component names
        all_pc_names = set()
        for track_components in lineage_pc_components_list:
            all_pc_names.update(track_components.keys())

        # Concatenate each PC component separately
        for pc_name in sorted(all_pc_names):
            pc_values_list = []
            for track_components in lineage_pc_components_list:
                if pc_name in track_components:
                    pc_values_list.append(track_components[pc_name])
                else:
                    # If this track doesn't have this PC component, pad with zeros
                    pc_values_list.append(
                        np.zeros_like(lineage_embeddings_list[0][:, 0])
                    )
            lineage_pc_components[pc_name] = np.concatenate(pc_values_list, axis=0)

        # Get all unique PHATE component names
        all_phate_names = set()
        for track_components in lineage_phate_components_list:
            all_phate_names.update(track_components.keys())

        # Concatenate each PHATE component separately
        for phate_name in sorted(all_phate_names):
            phate_values_list = []
            for track_components in lineage_phate_components_list:
                if phate_name in track_components:
                    phate_values_list.append(track_components[phate_name])
                else:
                    # If this track doesn't have this PHATE component, pad with zeros
                    phate_values_list.append(
                        np.zeros_like(lineage_embeddings_list[0][:, 0])
                    )
            lineage_phate_components[phate_name] = np.concatenate(
                phate_values_list, axis=0
            )

        # Extract the query sequence using the time range
        # Add bounds checking
        if (
            start_time < 0
            or end_time > len(lineage_embeddings)
            or start_time >= end_time
        ):
            logger.warning(
                f"Invalid time range for lineage {idx}: start={start_time}, end={end_time}, length={len(lineage_embeddings)}"
            )
            continue

        # Get the query sequences of the embeddings and projections
        query_sequence = lineage_embeddings[start_time:end_time]

        # Changed: Extract query sequences for each PC/PHATE component separately
        query_pc_components = {}
        for pc_name, pc_values in lineage_pc_components.items():
            query_pc_components[pc_name] = pc_values[start_time:end_time]

        query_phate_components = {}
        for phate_name, phate_values in lineage_phate_components.items():
            query_phate_components[phate_name] = phate_values[start_time:end_time]

        logger.debug(f"Query sequence shape: {query_sequence.shape}")
        logger.debug(f"Query PC components: {list(query_pc_components.keys())}")
        logger.debug(f"Query PHATE components: {list(query_phate_components.keys())}")

        # Apply warp path to align embeddings
        aligned_embeddings_sequence = []
        aligned_pc_components = {
            name: [] for name in query_pc_components.keys()
        }  # Changed
        aligned_phate_components = {
            name: [] for name in query_phate_components.keys()
        }  # Changed
        reference_timepoints_aligned = []
        query_timepoints_aligned = []

        for ref_idx, query_idx in warp_path:
            # Get the embedding at the query timepoint
            if query_idx < len(query_sequence):
                aligned_embeddings_sequence.append(query_sequence[query_idx])

                # Get PC components if available
                for pc_name, pc_sequence in query_pc_components.items():
                    if query_idx < len(pc_sequence):
                        aligned_pc_components[pc_name].append(pc_sequence[query_idx])
                    else:
                        # Pad with zeros if index out of bounds
                        aligned_pc_components[pc_name].append(0.0)

                # Get PHATE components if available
                for phate_name, phate_sequence in query_phate_components.items():
                    if query_idx < len(phate_sequence):
                        aligned_phate_components[phate_name].append(
                            phate_sequence[query_idx]
                        )
                    else:
                        # Pad with zeros if index out of bounds
                        aligned_phate_components[phate_name].append(0.0)

                # Calculate actual timepoints
                ref_t = reference_timepoints[0] + ref_idx
                query_t = start_time + query_idx

                reference_timepoints_aligned.append(ref_t)
                query_timepoints_aligned.append(query_t)
            else:
                logger.warning(
                    f"Query index {query_idx} out of bounds for sequence length {len(query_sequence)}"
                )

        # Convert to numpy arrays
        aligned_embeddings_sequence = np.array(aligned_embeddings_sequence)
        reference_timepoints_aligned = np.array(reference_timepoints_aligned)
        query_timepoints_aligned = np.array(query_timepoints_aligned)

        # Convert PC and PHATE sequences to arrays (keeping them separate)
        aligned_pc_components_arrays = {}
        for pc_name, pc_sequence in aligned_pc_components.items():
            aligned_pc_components_arrays[pc_name] = np.array(pc_sequence)

        aligned_phate_components_arrays = {}
        for phate_name, phate_sequence in aligned_phate_components.items():
            aligned_phate_components_arrays[phate_name] = np.array(phate_sequence)

        # Store results
        _fov_name = fov_name.replace("/", "_")

        lineage_key = f"lineage{_fov_name}_{track_ids[0]}"
        aligned_embeddings[model_name][lineage_key] = {
            "aligned_embeddings": aligned_embeddings_sequence,
            "aligned_pc_components": aligned_pc_components_arrays,  # Changed: individual components
            "aligned_phate_components": aligned_phate_components_arrays,  # Changed: individual components
            "reference_timepoints": reference_timepoints_aligned,
            "query_timepoints": query_timepoints_aligned,
            "warp_path": warp_path,
            "fov_id": fov_name,
            "track_ids": track_ids,
            "start_time": start_time,
            "end_time": end_time,
        }

        logger.info(
            f"Aligned {model_name} lineage {lineage_key}: "
            f"shape={aligned_embeddings_sequence.shape}, "
            f"PC components={list(aligned_pc_components_arrays.keys())}, "
            f"PHATE components={list(aligned_phate_components_arrays.keys())}, "
            f"timepoints={len(aligned_embeddings_sequence)}"
        )


# %%
# %%
def extract_individual_components_from_existing_data(aligned_embeddings_dict):
    """
    Extract individual PC and PHATE components from the existing concatenated structure.
    This is a workaround if the data wasn't re-processed with the new structure.
    """
    updated_aligned_embeddings = {}

    for model_name, lineages_dict in aligned_embeddings_dict.items():
        updated_aligned_embeddings[model_name] = {}

        for lineage_key, lineage_data in lineages_dict.items():
            updated_lineage_data = lineage_data.copy()

            # Extract individual PC components if they exist as concatenated array
            if "aligned_pc_embeddings" in lineage_data:
                pc_embeddings = lineage_data["aligned_pc_embeddings"]
                if pc_embeddings.size > 0 and len(pc_embeddings.shape) == 2:
                    # Assume each column is a different PC component
                    pc_components = {}
                    for i in range(pc_embeddings.shape[1]):
                        pc_components[f"PCA{i+1}"] = pc_embeddings[:, i]
                    updated_lineage_data["aligned_pc_components"] = pc_components
                    print(
                        f"Extracted {len(pc_components)} PC components for {lineage_key}"
                    )

            # Extract individual PHATE components if they exist as concatenated array
            if "aligned_phate_embeddings" in lineage_data:
                phate_embeddings = lineage_data["aligned_phate_embeddings"]
                if phate_embeddings.size > 0 and len(phate_embeddings.shape) == 2:
                    # Assume each column is a different PHATE component
                    phate_components = {}
                    for i in range(phate_embeddings.shape[1]):
                        phate_components[f"PHATE{i+1}"] = phate_embeddings[:, i]
                    updated_lineage_data["aligned_phate_components"] = phate_components
                    print(
                        f"Extracted {len(phate_components)} PHATE components for {lineage_key}"
                    )

            updated_aligned_embeddings[model_name][lineage_key] = updated_lineage_data

    return updated_aligned_embeddings


# Apply the extraction to your existing data
aligned_embeddings = extract_individual_components_from_existing_data(
    aligned_embeddings
)

# Now you can plot individual components
for model_name, lineages in aligned_embeddings.items():
    base_name = model_name.split("_")[0]
    plt.figure(figsize=(15, 10))

    # Check what components are available
    sample_lineage = next(iter(lineages.values()))
    available_pc_components = list(
        sample_lineage.get("aligned_pc_components", {}).keys()
    )
    available_phate_components = list(
        sample_lineage.get("aligned_phate_components", {}).keys()
    )

    print(f"\n{model_name} - Available PC components: {available_pc_components}")
    print(f"{model_name} - Available PHATE components: {available_phate_components}")

    # Plot PC components
    if available_pc_components:
        n_pc = len(available_pc_components)
        for i, pc_name in enumerate(
            available_pc_components[:4]
        ):  # Plot first 4 PC components
            plt.subplot(2, 4, i + 1)
            for lineage_key, lineage_data in lineages.items():
                if (
                    "aligned_pc_components" in lineage_data
                    and pc_name in lineage_data["aligned_pc_components"]
                ):
                    pc_values = lineage_data["aligned_pc_components"][pc_name]
                    plt.plot(pc_values, alpha=0.7)
            plt.title(f"{model_name} - {pc_name}")
            plt.xlabel("Aligned Time")
            plt.ylabel(f"{pc_name} Value")

    # Plot PHATE components
    if available_phate_components:
        n_phate = len(available_phate_components)
        for i, phate_name in enumerate(
            available_phate_components[:4]
        ):  # Plot first 4 PHATE components
            plt.subplot(2, 4, i + 5)
            for lineage_key, lineage_data in lineages.items():
                if (
                    "aligned_phate_components" in lineage_data
                    and phate_name in lineage_data["aligned_phate_components"]
                ):
                    phate_values = lineage_data["aligned_phate_components"][phate_name]
                    plt.plot(phate_values, alpha=0.7)
            plt.title(f"{model_name} - {phate_name}")
            plt.xlabel("Aligned Time")
            plt.ylabel(f"{phate_name} Value")

    plt.tight_layout()
    plt.show()

# %%
# Display the top alignments for each model in napari
# Cache
import os
import napari
from viscy.data.triplet import TripletDataModule
from tqdm import tqdm

os.environ["DISPLAY"] = ":1"
viewer = napari.Viewer()
# %%

YX_PATCH_SIZE = 128
TOP_N_ALIGNED_CELLS = 20
z_range = (0, 1)
channels_to_display = ["Phase3D", "raw mCherry EX561 EM600-37"]
# %%
all_lineage_images = {}

# Cache the unaligned images first for each model
for model_name, lineages in alignment_results.items():
    model_name = model_name.split("_")[0]
    top_n_aligned_cells = lineages[:TOP_N_ALIGNED_CELLS]
    all_lineage_images[model_name] = []
    for idx, row in tqdm(
        top_n_aligned_cells.iterrows(), total=len(top_n_aligned_cells)
    ):
        fov_name = row["fov_name"]
        track_ids = row["track_ids"]

        data_module = TripletDataModule(
            data_path=str(data_path),  # Convert Path to string
            tracks_path=str(tracks_path),  # Convert Path to string
            include_fov_names=[fov_name] * len(track_ids),
            include_track_ids=track_ids,
            source_channel=channels_to_display,
            z_range=z_range,
            initial_yx_patch_size=(YX_PATCH_SIZE, YX_PATCH_SIZE),
            final_yx_patch_size=(YX_PATCH_SIZE, YX_PATCH_SIZE),
            batch_size=1,
            num_workers=16,
            normalizations=[],
            predict_cells=True,
        )
        data_module.setup("predict")
        img_tczyx = []
        for batch in data_module.predict_dataloader():
            images = batch["anchor"].numpy()[0]
            indices = batch["index"]
            t_idx = indices["t"].tolist()
            # Take the middle z-slice
            z_idx = images.shape[1] // 2
            C, Z, Y, X = images.shape
            image_out = np.zeros((C, 1, Y, X), dtype=np.float32)
            for c_idx, channel in enumerate(channels_to_display):
                if channel in ["Phase3D", "DIC", "BF"]:
                    image_out[c_idx] = images[c_idx, z_idx]
                else:
                    image_out[c_idx] = np.max(images[c_idx], axis=0)
            img_tczyx.append(image_out)
        img_tczyx = np.array(img_tczyx)
        all_lineage_images[model_name].append(img_tczyx)

# Save the unaligned images
with open(output_root / "all_unaligned_images.pkl", "wb") as f:
    pickle.dump(all_lineage_images, f)

# %%
aligned_images = {}
unaligned_images = {}
unaligned_wrt_startpoint_images = {}
# Align the images to the corresponding warps (dynaclr and openphenom)
for model_name, lineages in alignment_results.items():
    model_name = model_name.split("_")[0]
    reference_pattern = reference_patterns[model_name]
    reference_pattern_length = len(reference_pattern)
    top_n_aligned_cells = lineages[:TOP_N_ALIGNED_CELLS]

    logger.info(f"Aligning {model_name} with {len(top_n_aligned_cells)} cells")

    # Initialize the model_name key as an empty dictionary
    aligned_images[model_name] = []
    unaligned_images[model_name] = []
    unaligned_wrt_startpoint_images[model_name] = []

    for idx, (_, row) in enumerate(top_n_aligned_cells.iterrows()):
        fov_name = row["fov_name"]
        track_ids = row["track_ids"]
        warp_path = row["warp_path"]
        start_time = row["start_timepoint"]
        unaligned_stack = all_lineage_images[model_name][idx]

        _aligned_stack = np.zeros(
            (reference_pattern_length, *unaligned_stack.shape[1:])
        )
        _unaligned_stack = np.zeros(
            (reference_pattern_length, *unaligned_stack.shape[1:])
        )
        _unaligned_wrt_startpoint_stack = np.zeros(
            (reference_pattern_length, *unaligned_stack.shape[1:])
        )
        for ref_idx in range(reference_pattern_length):
            matches = [(i, q) for i, q in warp_path if i == ref_idx]
            ref_startpoint = int(ref_idx + start_time)
            _unaligned_stack[ref_idx] = unaligned_stack[ref_idx]
            _unaligned_wrt_startpoint_stack[ref_idx] = unaligned_stack[ref_startpoint]
            if matches:
                match = matches[0]
                query_idx = match[1]
                lineage_idx = int(start_time + query_idx)
                if 0 <= lineage_idx < unaligned_stack.shape[0]:
                    _aligned_stack[ref_idx] = unaligned_stack[lineage_idx]
                else:
                    # Find nearest valid timepoint if out of bounds
                    nearest_idx = min(max(0, lineage_idx), len(unaligned_stack) - 1)
                    _aligned_stack[ref_idx] = unaligned_stack[nearest_idx]
            else:
                # If no direct match, find closest reference timepoint in warping path
                logger.warning(f"No match found for ref idx: {ref_idx}")
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
                        if 0 <= lineage_idx < unaligned_stack.shape[0]:
                            _aligned_stack[ref_idx] = unaligned_stack[lineage_idx]
                        else:
                            # Bound to valid range
                            nearest_idx = min(
                                max(0, lineage_idx), len(unaligned_stack) - 1
                            )
                            _aligned_stack[ref_idx] = unaligned_stack[nearest_idx]

        # Convert to numpy array after collecting all aligned images for this model
        aligned_images[model_name].append(_aligned_stack)
        unaligned_images[model_name].append(_unaligned_stack)
        unaligned_wrt_startpoint_images[model_name].append(
            _unaligned_wrt_startpoint_stack
        )
# Save the aligned images for each model
with open(output_root / "aligned_images.pkl", "wb") as f:
    pickle.dump(aligned_images, f)
with open(output_root / "unaligned_images.pkl", "wb") as f:
    pickle.dump(unaligned_images, f)
with open(output_root / "unaligned_wrt_startpoint_images.pkl", "wb") as f:
    pickle.dump(unaligned_wrt_startpoint_images, f)
# %%
# Load the unaligned images
with open(output_root / "unaligned_images.pkl", "rb") as f:
    unaligned_images = pickle.load(f)
# Load the aligned images
with open(output_root / "aligned_images.pkl", "rb") as f:
    aligned_images = pickle.load(f)

# %%
TOP_N_ALIGNED_CELLS = 5
clims_mcherry = (104, 164)
viewer.grid.shape = (-1, TOP_N_ALIGNED_CELLS)
viewer.grid.stride = 1
viewer.grid.enabled = True
screen_frames = [0, 20, 59]
output_path = output_root / "movies" / "screenshots"
output_path.mkdir(parents=True, exist_ok=True)

# %%
viewer.layers.clear()
for idx, aligned_img in enumerate(aligned_images["dynaclr"]):
    if idx < TOP_N_ALIGNED_CELLS:
        viewer.add_image(
            aligned_img[:, 1],
            name=f"dynaclr_aligned_{idx}",
            colormap="magenta",
            contrast_limits=clims_mcherry,
        )
for idx, unaligned_img in enumerate(unaligned_images["dynaclr"]):
    if idx < TOP_N_ALIGNED_CELLS:
        viewer.add_image(
            unaligned_img[:, 1],
            name=f"dynaclr_unaligned_{idx}",
            colormap="magenta",
            contrast_limits=clims_mcherry,
        )
viewer.reset_view()


for frame in screen_frames:
    viewer.dims.current_step = (frame, 0, 0, 0)
    viewer.screenshot(
        str(
            output_path
            / f"dynaclr_alignment_comparison_{frame}_top_{TOP_N_ALIGNED_CELLS}.png"
        )
    )

# %%
# openphenom
viewer.layers.clear()

for idx, aligned_img in enumerate(aligned_images["openphenom"]):
    if idx < TOP_N_ALIGNED_CELLS:
        viewer.add_image(
            aligned_img[:, 1],
            name=f"openphenom_aligned_{idx}",
            colormap="magenta",
            contrast_limits=clims_mcherry,
        )

for idx, unaligned_img in enumerate(unaligned_images["openphenom"]):
    if idx < TOP_N_ALIGNED_CELLS:
        viewer.add_image(
            unaligned_img[:, 1],
            name=f"openphenom_unaligned_{idx}",
            colormap="magenta",
            contrast_limits=clims_mcherry,
        )
viewer.reset_view()
for frame in screen_frames:
    viewer.dims.current_step = (frame, 0, 0, 0)
    viewer.screenshot(
        str(
            output_path
            / f"openphenom_alignment_comparison_{frame}_top_{TOP_N_ALIGNED_CELLS}.png"
        )
    )


# %%
def plot_component_trajectories(
    aligned_embeddings,
    alignment_results,
    delta_t=1,
    component_type="aligned_pc_components",
    components_to_plot=["PC1", "PC2", "PC3", "PC4"],
    top_n=5,
    figsize=None,
    title_prefix="",
    output_path=None,
):
    """
    Modular function to plot specific components (PC or PHATE) for top N aligned cells.
    Layout automatically adapts: up to 4 components per row, creates new rows as needed.

    Parameters:
    -----------
    aligned_embeddings : dict
        Dictionary containing aligned embeddings for each model
    alignment_results : dict
        Dictionary containing DTW alignment results
    delta_t : int
        Time step between frames to plot
    component_type : str
        Either "aligned_pc_components" or "aligned_phate_components"
    components_to_plot : list
        List of component names to plot (e.g., ["PC1", "PC2", "PC3", "PC4"])
    top_n : int
        Number of top-aligned cells to plot
    figsize : tuple or None
        Figure size, if None will auto-calculate based on layout
    title_prefix : str
        Prefix for plot titles
    output_path : str or None
        Path to save the plots. If None, plots are only displayed.
    """

    for model_name, lineages in aligned_embeddings.items():

        dtw_results = alignment_results[model_name]
        top_alignments = dtw_results.nsmallest(top_n, "distance")

        # Get lineage keys for top aligned cells
        lineage_keys_to_plot = []
        for idx, row in top_alignments.iterrows():
            fov_name = row.get("fov_name", "")
            track_ids = row.get("track_ids", [])
            if track_ids:
                _fov_name = fov_name.replace("/", "_")
                lineage_key = f"lineage{_fov_name}_{track_ids[0]}"
                if lineage_key in lineages:
                    lineage_keys_to_plot.append(lineage_key)

        print(
            f"Found {len(lineage_keys_to_plot)} matching lineages to plot for {model_name}"
        )

        if not lineage_keys_to_plot:
            print(f"No matching lineages found for {model_name}")
            continue

        # Check what components are available
        sample_lineage = lineages[lineage_keys_to_plot[0]]
        available_components = list(sample_lineage.get(component_type, {}).keys())

        print(f"{model_name} - Available {component_type}: {available_components}")

        # Filter components to plot based on availability
        components_to_plot_filtered = [
            comp for comp in components_to_plot if comp in available_components
        ]

        if not components_to_plot_filtered:
            print(f"No requested components found in {model_name}")
            continue

        # Auto-calculate layout: max 4 components per row
        n_components = len(components_to_plot_filtered)
        n_cols = min(4, n_components)
        n_rows = (n_components + 3) // 4  # Ceiling division

        if figsize is None:
            figsize = (4 * n_cols, 4 * n_rows)

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Handle different subplot configurations
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Define colors for the top N cells
        colors = plt.cm.tab10(np.linspace(0, 1, top_n))

        # Plot each component
        for i, component_name in enumerate(components_to_plot_filtered):
            ax = axes[i]

            for j, lineage_key in enumerate(lineage_keys_to_plot):
                lineage_data = lineages[lineage_key]
                if (
                    component_type in lineage_data
                    and component_name in lineage_data[component_type]
                ):

                    component_values = lineage_data[component_type][component_name]
                    timepoints = np.arange(0, len(component_values) * delta_t, delta_t)
                    ax.plot(
                        timepoints,
                        component_values,
                        alpha=0.8,
                        color=colors[j],
                        linewidth=2,
                        label=f"Cell {j+1}",
                    )

            ax.set_title(f"{component_name}")
            ax.set_xlabel("Aligned Time")
            ax.set_ylabel(f"{component_name} Value")
            ax.grid(True, alpha=0.3)

            # Add legend only to first subplot
            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        # Hide unused subplots if any
        for i in range(n_components, len(axes)):
            axes[i].set_visible(False)

        # Set main title
        component_type_display = (
            component_type.replace("aligned_", "").replace("_components", "").upper()
        )
        plt.suptitle(
            f"{model_name.upper()} - {component_type_display} Components\n"
            f"Top {top_n} Best Aligned Cells {title_prefix}",
            fontsize=14,
            y=0.98,
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        # Save plot if output_path is provided
        if output_path is not None:
            import os

            os.makedirs(output_path, exist_ok=True)

            # Create filename
            component_type_short = component_type.replace("aligned_", "").replace(
                "_components", ""
            )
            components_str = "_".join(components_to_plot_filtered)
            filename = f"{model_name}_{component_type_short}_{components_str}_top{top_n}_individual.pdf"
            filepath = os.path.join(output_path, filename)

            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved plot to: {filepath}")

        plt.show()


def plot_component_trajectories_with_stats(
    aligned_embeddings,
    alignment_results,
    component_type="aligned_pc_components",
    components_to_plot=["PC1", "PC2", "PC3", "PC4"],
    top_n=5,
    figsize=None,
    title_prefix="",
    output_path=None,
    colors=None,
):
    """
    Modular function to plot mean ± std for specific components for top N aligned cells.
    Layout automatically adapts: up to 4 components per row, creates new rows as needed.
    Uses different colors for different models.

    Parameters:
    -----------
    aligned_embeddings : dict
        Dictionary containing aligned embeddings for each model
    alignment_results : dict
        Dictionary containing DTW alignment results
    component_type : str
        Either "aligned_pc_components" or "aligned_phate_components"
    components_to_plot : list
        List of component names to plot (e.g., ["PC1", "PC2", "PC3", "PC4"])
    top_n : int
        Number of top-aligned cells to plot
    figsize : tuple or None
        Figure size, if None will auto-calculate based on layout
    title_prefix : str
        Prefix for plot titles
    output_path : str or None
        Path to save the plots. If None, plots are only displayed.
    colors : list or None
        List of colors, one for each model in the same order as aligned_embeddings.
        If None, uses default colors: ['steelblue', 'crimson', 'forestgreen', 'orange', 'purple', 'brown', 'pink', 'gray']
        Examples: ['blue', 'red'], ['#1f77b4', '#ff7f0e'], [(0.2, 0.4, 0.8), (0.8, 0.2, 0.2)]
    """

    # Default colors if none provided
    if colors is None:
        default_colors = [
            "steelblue",
            "crimson",
            "forestgreen",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
        ]
    else:
        default_colors = colors

    # Get model names in order
    model_names = list(aligned_embeddings.keys())

    for i, (model_name, lineages) in enumerate(aligned_embeddings.items()):

        # Assign color for this model
        plot_color = default_colors[
            i % len(default_colors)
        ]  # Cycle through colors if more models than colors

        print(
            f"Using color '{plot_color}' for {model_name} (model {i+1}/{len(model_names)})"
        )

        dtw_results = alignment_results[model_name]
        top_alignments = dtw_results.nsmallest(top_n, "distance")

        # Get lineage keys for top aligned cells
        lineage_keys_to_plot = []
        for idx, row in top_alignments.iterrows():
            fov_name = row.get("fov_name", "")
            track_ids = row.get("track_ids", [])
            if track_ids:
                _fov_name = fov_name.replace("/", "_")
                lineage_key = f"lineage{_fov_name}_{track_ids[0]}"
                if lineage_key in lineages:
                    lineage_keys_to_plot.append(lineage_key)

        if not lineage_keys_to_plot:
            continue

        # Check available components
        sample_lineage = lineages[lineage_keys_to_plot[0]]
        available_components = list(sample_lineage.get(component_type, {}).keys())
        components_to_plot_filtered = [
            comp for comp in components_to_plot if comp in available_components
        ]

        if not components_to_plot_filtered:
            continue

        # Auto-calculate layout: max 4 components per row
        n_components = len(components_to_plot_filtered)
        n_cols = min(4, n_components)
        n_rows = (n_components + 3) // 4  # Ceiling division

        if figsize is None:
            figsize = (4 * n_cols, 4 * n_rows)

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Handle different subplot configurations
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Plot each component with mean ± std
        for j, component_name in enumerate(components_to_plot_filtered):
            ax = axes[j]

            # Collect values for this component across top N cells
            component_values_list = []
            for lineage_key in lineage_keys_to_plot:
                lineage_data = lineages[lineage_key]
                if (
                    component_type in lineage_data
                    and component_name in lineage_data[component_type]
                ):
                    component_values = lineage_data[component_type][component_name]
                    component_values_list.append(component_values)

            if component_values_list:
                # Find common length and calculate statistics
                min_length = min(len(vals) for vals in component_values_list)
                truncated_values = [vals[:min_length] for vals in component_values_list]
                values_array = np.array(truncated_values)

                mean_values = np.mean(values_array, axis=0)
                std_values = np.std(values_array, axis=0)
                timepoints = np.arange(min_length)

                # Plot mean line with std band using assigned color
                ax.plot(
                    timepoints,
                    mean_values,
                    "-",
                    color=plot_color,
                    linewidth=2,
                    label=f"{component_name} Mean",
                )
                ax.fill_between(
                    timepoints,
                    mean_values - std_values,
                    mean_values + std_values,
                    alpha=0.3,
                    color=plot_color,
                    label=f"{component_name} ±1σ",
                )

            ax.set_title(f"{component_name}")
            ax.set_xlabel("Aligned Time")
            ax.set_ylabel(f"{component_name} Value")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Hide unused subplots if any
        for j in range(n_components, len(axes)):
            axes[j].set_visible(False)

        # Set main title
        component_type_display = (
            component_type.replace("aligned_", "").replace("_components", "").upper()
        )
        plt.suptitle(
            f"{model_name.upper()} - {component_type_display} Components (Mean ± Std)\n"
            f"Top {top_n} Best Aligned Cells {title_prefix}",
            fontsize=14,
            y=0.98,
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        # Save plot if output_path is provided
        if output_path is not None:
            import os

            os.makedirs(output_path, exist_ok=True)

            # Create filename
            component_type_short = component_type.replace("aligned_", "").replace(
                "_components", ""
            )
            components_str = "_".join(components_to_plot_filtered)
            filename = f"{model_name}_{component_type_short}_{components_str}_top{top_n}_stats.pdf"
            filepath = os.path.join(output_path, filename)

            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved plot to: {filepath}")

        plt.show()


# Usage examples with different color schemes:
print("Using default colors (steelblue for DynaCLR, crimson for OpenPhenom)...")
plot_component_trajectories(
    aligned_embeddings,
    alignment_results,
    component_type="aligned_pc_components",
    components_to_plot=["PCA1", "PCA2", "PCA3", "PCA4"],
    top_n=5,
    title_prefix="(Mean ± Std)",
    output_path=output_root / "plots",
)

print("Using custom colors...")
plot_component_trajectories_with_stats(
    aligned_embeddings,
    alignment_results,
    component_type="aligned_pc_components",
    components_to_plot=["PCA1", "PCA2", "PCA3", "PCA4"],
    top_n=30,
    title_prefix="(Mean ± Std)",
    output_path=output_root / "plots",
    colors=["blue", "red"],  # Custom colors for each model
)

print("Using hex colors...")
plot_component_trajectories_with_stats(
    aligned_embeddings,
    alignment_results,
    component_type="aligned_phate_components",
    components_to_plot=["PHATE1", "PHATE2"],
    top_n=50,
    title_prefix="(Mean ± Std)",
    output_path=output_root / "plots",
    colors=["#1f77b4", "#ff7f0e"],  # Matplotlib default blue and orange
)


# %%
