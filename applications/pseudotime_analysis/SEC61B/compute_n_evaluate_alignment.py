# %%
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.dtw import find_pattern_matches, identify_lineages

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
dynaclr_features_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/4-phenotyping/prediction_infection/2chan_192patch_100ckpt_timeAware_ntxent_rerun.zarr"
)
openphenom_features_path = Path(
    "/home/eduardo.hirata/repos/viscy/applications/benchmarking/DynaCLR/OpenPhenom/openphenom_sec61b_n_phase_all.zarr"
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
annotations_fov_id = "/C/2/000001"

# TODO: A bit clunky since some functions take xarray and others take dataframes
features_dict = {
    "dynaclr": dynaclr_feature_df,
    # "openphenom": openphenom_feature_df,
}

embeddings_dict = {
    "dynaclr": dynaclr_features_dataset,
    # "openphenom": openphenom_features_dataset,
}
lineages = {}

min_timepoints = 20
for name, embeddings_dataset in embeddings_dict.items():
    lineages[f"{name}_lineages"] = identify_lineages(
        embeddings_dataset, min_timepoints=min_timepoints
    )
    logger.info(
        f"Found {len(lineages[f'{name}_lineages'])} {name} lineages with at least {min_timepoints} timepoints"
    )

# Filter lineages to only include those from the annotations fov
filtered_lineages = {}
for name, lineages in lineages.items():
    filtered_lineages[name] = []
    for fov_id, track_ids in lineages:
        if fov_id == annotations_fov_id:
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

# Debug: Check embedding dimensions from both datasets
logger.info(f"DynaCLR feature dimension: {dynaclr_features_dataset.features.shape}")
logger.info(
    f"OpenPhenom feature dimension: {openphenom_features_dataset.features.shape}"
)

# Check if the datasets have different feature dimensions
if (
    dynaclr_features_dataset.features.shape[-1]
    != openphenom_features_dataset.features.shape[-1]
):
    logger.warning(
        f"Feature dimension mismatch: DynaCLR={dynaclr_features_dataset.features.shape[-1]}, OpenPhenom={openphenom_features_dataset.features.shape[-1]}"
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
            window_step_fraction=0.1,
            num_candidates=4,
            method="bernd_clifford",
            save_path=str(output_root / f"{name}_matching_lineages_{METRIC}.csv"),
            metric=METRIC,
        )
        alignment_results[name] = all_match_positions
