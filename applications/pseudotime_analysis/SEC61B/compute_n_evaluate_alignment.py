# %%
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.dtw import find_pattern_matches, identify_lineages

# Create a custom logger for just this script
logger = logging.getLogger("viscy")
logger.setLevel(logging.DEBUG)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add a console handler specifically for this logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
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
annotations_fov_id = "/C/2/000001"

# TODO: A bit clunky since some functions take xarray and others take dataframes
features_dict = {
    "dynaclr": dynaclr_feature_df,
    "openphenom": openphenom_feature_df,
}

embeddings_dict = {
    "dynaclr": dynaclr_features_dataset,
    "openphenom": openphenom_features_dataset,
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
            num_candidates=20,
            method="bernd_clifford",
            save_path=str(output_root / f"{name}_matching_lineages_{METRIC}.csv"),
            metric=METRIC,
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
        lineage_pc_embeddings_list = []
        lineage_phate_embeddings_list = []

        for track_id in track_ids:
            try:
                # Extract the full track data
                track_data = embeddings_dataset.sel(sample=(fov_name, track_id))

                # Use full embeddings for alignment (as before)
                track_embeddings = track_data.features.values

                # Extract PC and PHATE components separately for analysis
                pc_features = []
                phate_features = []

                # Add all available PCA components dynamically
                pc_components = [
                    str(col) for col in track_data.coords if str(col).startswith("PCA")
                ]
                pc_components.sort(key=lambda x: int(x[3:]))  # Sort by PCA number

                for pc_col in pc_components:
                    pc_values = track_data[pc_col].values
                    pc_features.append(pc_values)

                # add and process dynamically PHATE components
                phate_components = [
                    str(col)
                    for col in track_data.coords
                    if str(col).startswith("PHATE")
                ]
                phate_components.sort(key=lambda x: int(x[5:]))  # Sort by PHATE number

                for phate_col in phate_components:
                    phate_values = track_data[phate_col].values
                    phate_features.append(phate_values)

                # Stack PC and PHATE features separately
                track_pc_embeddings = (
                    np.column_stack(pc_features)
                    if pc_features
                    else np.empty((len(track_data.t), 0))
                )
                track_phate_embeddings = (
                    np.column_stack(phate_features)
                    if phate_features
                    else np.empty((len(track_data.t), 0))
                )

                lineage_embeddings_list.append(track_embeddings)
                lineage_pc_embeddings_list.append(track_pc_embeddings)
                lineage_phate_embeddings_list.append(track_phate_embeddings)

                logger.debug(
                    f"Extracted {track_embeddings.shape} full embeddings for track {track_id}"
                )
                logger.debug(f"  PC components: {track_pc_embeddings.shape}")
                logger.debug(f"  PHATE components: {track_phate_embeddings.shape}")

            except Exception as e:
                logger.warning(
                    f"Could not extract embeddings for {fov_id}, {track_id}: {e}"
                )
                continue
        # Concatenate all track embeddings for this lineage
        lineage_embeddings = np.concatenate(lineage_embeddings_list, axis=0)
        lineage_pc_embeddings = (
            np.concatenate(lineage_pc_embeddings_list, axis=0)
            if lineage_pc_embeddings_list
            else np.empty((0, 0))
        )
        lineage_phate_embeddings = (
            np.concatenate(lineage_phate_embeddings_list, axis=0)
            if lineage_phate_embeddings_list
            else np.empty((0, 0))
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
        query_pc_sequence = lineage_pc_embeddings[start_time:end_time]
        query_phate_sequence = lineage_phate_embeddings[start_time:end_time]

        logger.debug(f"Query sequence shape: {query_sequence.shape}")
        logger.debug(f"Query PC sequence shape: {query_pc_sequence.shape}")
        logger.debug(f"Query PHATE sequence shape: {query_phate_sequence.shape}")

        # Apply warp path to align embeddings
        aligned_embeddings_sequence = []
        aligned_pc_embeddings_sequence = []
        aligned_phate_embeddings_sequence = []
        reference_timepoints_aligned = []
        query_timepoints_aligned = []

        # Debug: Print warp path structure
        logger.debug(
            f"Warp path type: {type(warp_path)}, length: {len(warp_path) if hasattr(warp_path, '__len__') else 'N/A'}"
        )
        logger.debug(
            f"First few warp path entries: {warp_path[:3] if hasattr(warp_path, '__len__') else warp_path}"
        )

        for ref_idx, query_idx in warp_path:
            # Debug: Print indices
            logger.debug(
                f"Processing warp path entry: ref_idx={ref_idx}, query_idx={query_idx}"
            )

            # Get the embedding at the query timepoint
            if query_idx < len(query_sequence):
                aligned_embeddings_sequence.append(query_sequence[query_idx])

                # Get PC embeddings if available
                if query_pc_sequence.size > 0 and query_idx < len(query_pc_sequence):
                    aligned_pc_embeddings_sequence.append(query_pc_sequence[query_idx])
                elif query_pc_sequence.size > 0:
                    # Pad with zeros if index out of bounds
                    aligned_pc_embeddings_sequence.append(
                        np.zeros(query_pc_sequence.shape[1])
                    )

                # Get PHATE embeddings if available
                if query_phate_sequence.size > 0 and query_idx < len(
                    query_phate_sequence
                ):
                    aligned_phate_embeddings_sequence.append(
                        query_phate_sequence[query_idx]
                    )
                elif query_phate_sequence.size > 0:
                    # Pad with zeros if index out of bounds
                    aligned_phate_embeddings_sequence.append(
                        np.zeros(query_phate_sequence.shape[1])
                    )

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

        # Convert PC and PHATE sequences to arrays
        if aligned_pc_embeddings_sequence:
            aligned_pc_embeddings_sequence = np.array(aligned_pc_embeddings_sequence)
        else:
            aligned_pc_embeddings_sequence = np.empty((0, 0))

        if aligned_phate_embeddings_sequence:
            aligned_phate_embeddings_sequence = np.array(
                aligned_phate_embeddings_sequence
            )
        else:
            aligned_phate_embeddings_sequence = np.empty((0, 0))

        # Store results
        _fov_name = fov_id.replace("/", "_")

        lineage_key = f"lineage{_fov_name}_{track_ids[0]}"
        aligned_embeddings[model_name][lineage_key] = {
            "aligned_embeddings": aligned_embeddings_sequence,
            "aligned_pc_embeddings": aligned_pc_embeddings_sequence,
            "aligned_phate_embeddings": aligned_phate_embeddings_sequence,
            "reference_timepoints": reference_timepoints_aligned,
            "query_timepoints": query_timepoints_aligned,
            "warp_path": warp_path,
            "fov_id": fov_id,
            "track_ids": track_ids,
            "original_start_time": start_time,
            "original_end_time": end_time,
        }

        logger.info(
            f"Aligned {model_name} lineage {lineage_key}: "
            f"shape={aligned_embeddings_sequence.shape}, "
            f"timepoints={len(aligned_embeddings_sequence)}"
        )
# %%
# Plot the aligned PHATE vs time for each model
for model_name, lineages in aligned_embeddings.items():
    base_name = model_name.split("_")[0]
    plt.figure(figsize=(10, 6))
    for lineage_key, lineage_data in lineages.items():
        phate_embeddings = lineage_data["aligned_phate_embeddings"]
        plt.plot(phate_embeddings)

    plt.show()


# %%
def measure_alignment_accuracy(lineages_dict, output_dir, reference_lineage_key=None):
    """
    Measure how well sequences are aligned by comparing variance at each timepoint.
    Uses full embeddings for alignment accuracy measurement, with separate analysis
    of PC and PHATE components for detailed comparison.

    Parameters:
        aligned_embeddings_dict (dict): Dictionary of aligned embeddings
        output_dir (str): Directory to save the alignment metrics

    Returns:
        dict: Alignment metrics including mean and std per timepoint for PC/PHATE components
    """
    from sklearn.metrics import r2_score
    from sklearn.metrics.pairwise import cosine_similarity

    logger.info("measuring the alignment accuracy of the aligned embeddings")
    alignment_metrics = {}

    # Collect all aligned sequences for this model
    aligned_sequences = []
    aligned_pc_sequences = []
    aligned_phate_sequences = []
    sequence_info = []

    reference_pc = None
    reference_phate = None
    if reference_lineage_key is not None and reference_lineage_key in lineages_dict:
        reference_pc = lineages_dict[reference_lineage_key].get("aligned_pc_embeddings")
        reference_phate = lineages_dict[reference_lineage_key].get(
            "aligned_phate_embeddings"
        )
        logger.info(
            f"Using reference lineage '{reference_lineage_key}' for similarity metrics."
        )
    else:
        logger.warning(
            f"Reference lineage key '{reference_lineage_key}' not found — skipping R²/cosine similarity."
        )

    for lineage_key, lineage_data in lineages_dict.items():
        embeddings = lineage_data["aligned_embeddings"]
        pc_embeddings = lineage_data.get("aligned_pc_embeddings", np.empty((0, 0)))
        phate_embeddings = lineage_data.get(
            "aligned_phate_embeddings", np.empty((0, 0))
        )

        if len(embeddings) > 0:
            aligned_sequences.append(embeddings)
            if pc_embeddings.size > 0:
                aligned_pc_sequences.append(pc_embeddings)
            if phate_embeddings.size > 0:
                aligned_phate_sequences.append(phate_embeddings)

            sequence_info.append(
                {
                    "lineage_key": lineage_key,
                    "length": len(embeddings),
                    "dimensions": embeddings.shape[1],
                    "pc_dimensions": (
                        pc_embeddings.shape[1] if pc_embeddings.size > 0 else 0
                    ),
                    "phate_dimensions": (
                        phate_embeddings.shape[1] if phate_embeddings.size > 0 else 0
                    ),
                }
            )
    min_length = min(len(seq) for seq in aligned_sequences)

    pc_stats = {}  # For plotting: mean and std per timepoint per PC component
    truncated_pc_sequences = [seq[:min_length] for seq in aligned_pc_sequences]

    # Calculate mean and std per timepoint for each PC component
    pc_sequences_array = np.array(
        truncated_pc_sequences
    )  # Shape: (n_sequences, timepoints, n_pc_components)
    pc_mean_per_timepoint = np.mean(
        pc_sequences_array, axis=0
    )  # Shape: (timepoints, n_pc_components)
    pc_std_per_timepoint = np.std(
        pc_sequences_array, axis=0
    )  # Shape: (timepoints, n_pc_components)
    r2_pc_scores = []
    cosine_pc_scores = []
    if reference_pc is not None:
        for seq in truncated_pc_sequences:
            min_len = min(len(seq), len(reference_pc))
            if min_len > 1 and seq.shape[1] == reference_pc.shape[1]:
                r2_pc_scores.append(
                    r2_score(
                        reference_pc[:min_len],
                        seq[:min_len],
                        multioutput="variance_weighted",
                    )
                )
                cosine_pc_scores.append(
                    np.mean(
                        [
                            cosine_similarity(reference_pc[i : i + 1], seq[i : i + 1])[
                                0, 0
                            ]
                            for i in range(min_len)
                        ]
                    )
                )

    # Store plotting data for each PC component
    for pc_idx in range(pc_mean_per_timepoint.shape[1]):
        logger.info(
            f"PC component {pc_idx+1} shape: {pc_mean_per_timepoint[:, pc_idx].shape}"
        )
        pc_stats[f"PC{pc_idx+1}"] = {
            "timepoints": np.arange(min_length),
            "mean": pc_mean_per_timepoint[:, pc_idx],
            "std": pc_std_per_timepoint[:, pc_idx],
        }

    # Analyze PHATE components separately if available
    phate_stats = {}
    min_phate_length = min(len(seq) for seq in aligned_phate_sequences)
    truncated_phate_sequences = [
        seq[:min_phate_length] for seq in aligned_phate_sequences
    ]

    # Calculate mean and std per timepoint for each PHATE component
    phate_sequences_array = np.array(
        truncated_phate_sequences
    )  # Shape: (n_sequences, timepoints, n_phate_components)
    phate_mean_per_timepoint = np.mean(
        phate_sequences_array, axis=0
    )  # Shape: (timepoints, n_phate_components)
    phate_std_per_timepoint = np.std(
        phate_sequences_array, axis=0
    )  # Shape: (timepoints, n_phate_components)

    # Store plotting data for each PHATE component
    for phate_idx in range(phate_mean_per_timepoint.shape[1]):
        phate_stats[f"PHATE{phate_idx+1}"] = {
            "timepoints": np.arange(min_phate_length),
            "mean": phate_mean_per_timepoint[:, phate_idx],
            "std": phate_std_per_timepoint[:, phate_idx],
        }
        logger.info(
            f"PHATE component {phate_idx+1} mean: {phate_stats[f'PHATE{phate_idx+1}']['mean']}"
        )
        logger.info(
            f"PHATE component {phate_idx+1} std: {phate_stats[f'PHATE{phate_idx+1}']['std']}"
        )

    r2_phate_scores, cosine_phate_scores = [], []
    if reference_phate is not None:
        for seq in truncated_phate_sequences:
            if (
                seq.ndim == 2
                and reference_phate.ndim == 2
                and seq.shape[1] == reference_phate.shape[1]
            ):
                min_len = min(seq.shape[0], reference_phate.shape[0])
                if min_len > 1:
                    r2 = r2_score(
                        reference_phate[:min_len],
                        seq[:min_len],
                        multioutput="variance_weighted",
                    )
                    cos_sim = np.mean(
                        [
                            cosine_similarity(
                                reference_phate[i : i + 1], seq[i : i + 1]
                            )[0, 0]
                            for i in range(min_len)
                        ]
                    )
                    r2_phate_scores.append(r2)
                    cosine_phate_scores.append(cos_sim)
                else:
                    logger.warning(f"Skipping {lineage_key} due to too-short overlap.")
            else:
                logger.warning(
                    f"Shape mismatch — {lineage_key}: {seq.shape} vs {reference_phate.shape}"
                )

    # Store metrics
    alignment_metrics = {
        "pc_stats": pc_stats,
        "phate_stats": phate_stats,
        "num_sequences": len(aligned_sequences),
        "common_length": min_length,
        "dimensions": aligned_sequences[0].shape[1],
        "r2_pc_scores": r2_pc_scores,
        "cosine_pc_scores": cosine_pc_scores,
        "r2_phate_scores": r2_phate_scores,
        "cosine_phate_scores": cosine_phate_scores,
    }

    # save the alignment metrics to a pickle file
    with open(output_dir / "alignment_metrics.pkl", "wb") as f:
        pickle.dump(alignment_metrics, f)

    return alignment_metrics


# %%
# Run alignment accuracy measurement
alignment_metrics = {}
for model_name, lineages in aligned_embeddings.items():
    base_name = model_name.split("_")[0]
    output_metrics_path = output_root / f"{base_name}"
    output_metrics_path.mkdir(parents=True, exist_ok=True)

    # FIXME there is probably a better way to do this
    reference_key = "lineage_C_2_000001_138"
    alignment_metrics[model_name] = measure_alignment_accuracy(
        lineages,
        output_metrics_path,
        reference_lineage_key=reference_key,
    )

for model_name, metrics in alignment_metrics.items():
    logger.info(f"Model: {model_name}")
    base_name = model_name.split("_")[0]
    logger.info(f"Mean R² (PHATE): {np.mean(metrics['r2_phate_scores']):.4f}")
    logger.info(
        f"Mean cosine similarity (PHATE): {np.mean(metrics['cosine_phate_scores']):.4f}"
    )
    logger.info(f"Mean R² (PC): {np.mean(metrics['r2_pc_scores']):.4f}")
    logger.info(
        f"Mean cosine similarity (PC): {np.mean(metrics['cosine_pc_scores']):.4f}"
    )


# %%
# Plot the alignment metrics with standard deviation bands
def plot_alignment_metrics(phate_stats, output_dir, delta_t=1):
    for element in phate_stats.keys():
        logger.info(f"Plotting {element} with std bands")
        plt.figure(figsize=(10, 6))

        timepoints = phate_stats[element]["timepoints"] * delta_t
        mean_values = phate_stats[element]["mean"]
        std_values = phate_stats[element]["std"]

        # Plot mean line
        plt.plot(timepoints, mean_values, "b-", linewidth=2, label=f"{element} Mean")

        # Add std bands (mean ± std)
        plt.fill_between(
            timepoints,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.3,
            color="blue",
            label=f"{element} ±1σ",
        )

        plt.xlabel("Time (min)")
        plt.ylabel(f"{element} Component Value")
        plt.title(f"{element} alignment over time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{element}.png")
        plt.close()


# %%
# Plot the alignment metrics with standard deviation bands
for model_name, lineages in alignment_metrics.items():
    logger.info(f"Model: {model_name}")
    base_name = model_name.split("_")[0]
    pc_stats = lineages["pc_stats"]
    plot_alignment_metrics(pc_stats, output_root / f"{base_name}", delta_t=10)
    logger.debug(f"PC stats: {pc_stats}")
    phate_stats = lineages["phate_stats"]
    plot_alignment_metrics(phate_stats, output_root / f"{base_name}", delta_t=10)
    logger.debug(f"PHATE stats: {phate_stats}")
# %%
# Plot the PC and PHATE components for each model side by side
dynaclr_phate_stats = alignment_metrics["dynaclr_lineages"]["phate_stats"]
dynaclr_pc_stats = alignment_metrics["dynaclr_lineages"]["pc_stats"]
openphenom_phate_stats = alignment_metrics["openphenom_lineages"]["phate_stats"]
openphenom_pc_stats = alignment_metrics["openphenom_lineages"]["pc_stats"]

# Plot dynaclr vs openphenom phate components side by side
for phate_idx in range(len(dynaclr_phate_stats.keys())):
    plt.figure(figsize=(10, 6))
    plt.plot(
        dynaclr_phate_stats[f"PHATE{phate_idx+1}"]["timepoints"],
        dynaclr_phate_stats[f"PHATE{phate_idx+1}"]["mean"],
        "b-",
        linewidth=2,
        label=f"dynaclr PHATE{phate_idx+1} Mean",
    )
    plt.fill_between(
        dynaclr_phate_stats[f"PHATE{phate_idx+1}"]["timepoints"],
        dynaclr_phate_stats[f"PHATE{phate_idx+1}"]["mean"]
        - dynaclr_phate_stats[f"PHATE{phate_idx+1}"]["std"],
        dynaclr_phate_stats[f"PHATE{phate_idx+1}"]["mean"]
        + dynaclr_phate_stats[f"PHATE{phate_idx+1}"]["std"],
        alpha=0.3,
        color="blue",
        label=f"dynaclr PHATE{phate_idx+1} ±1σ",
    )
    plt.plot(
        openphenom_phate_stats[f"PHATE{phate_idx+1}"]["timepoints"],
        openphenom_phate_stats[f"PHATE{phate_idx+1}"]["mean"],
        "r-",
        linewidth=2,
        label=f"openphenom PHATE{phate_idx+1} Mean",
    )
    plt.fill_between(
        openphenom_phate_stats[f"PHATE{phate_idx+1}"]["timepoints"],
        openphenom_phate_stats[f"PHATE{phate_idx+1}"]["mean"]
        - openphenom_phate_stats[f"PHATE{phate_idx+1}"]["std"],
        openphenom_phate_stats[f"PHATE{phate_idx+1}"]["mean"]
        + openphenom_phate_stats[f"PHATE{phate_idx+1}"]["std"],
        alpha=0.3,
        color="red",
        label=f"openphenom PHATE{phate_idx+1} ±1σ",
    )
    plt.legend()
    plt.savefig(output_root / f"dynaclr_vs_openphenom_phate{phate_idx+1}.png")
    plt.show()
    plt.close()

# Plot dynaclr vs openphenom pc components side by side
for pc_idx in range(len(dynaclr_pc_stats.keys())):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].plot(
        dynaclr_pc_stats[f"PC{pc_idx+1}"]["timepoints"],
        dynaclr_pc_stats[f"PC{pc_idx+1}"]["mean"],
        "b-",
        linewidth=2,
        label=f"dynaclr PC{pc_idx+1} Mean",
    )
    ax[0].fill_between(
        dynaclr_pc_stats[f"PC{pc_idx+1}"]["timepoints"],
        dynaclr_pc_stats[f"PC{pc_idx+1}"]["mean"]
        - dynaclr_pc_stats[f"PC{pc_idx+1}"]["std"],
        dynaclr_pc_stats[f"PC{pc_idx+1}"]["mean"]
        + dynaclr_pc_stats[f"PC{pc_idx+1}"]["std"],
        alpha=0.3,
        color="blue",
        label=f"dynaclr PC{pc_idx+1} ±1σ",
    )
    ax[1].plot(
        openphenom_pc_stats[f"PC{pc_idx+1}"]["timepoints"],
        openphenom_pc_stats[f"PC{pc_idx+1}"]["mean"],
        "r-",
        linewidth=2,
        label=f"openphenom PC{pc_idx+1} Mean",
    )
    ax[1].fill_between(
        openphenom_pc_stats[f"PC{pc_idx+1}"]["timepoints"],
        openphenom_pc_stats[f"PC{pc_idx+1}"]["mean"]
        - openphenom_pc_stats[f"PC{pc_idx+1}"]["std"],
        openphenom_pc_stats[f"PC{pc_idx+1}"]["mean"]
        + openphenom_pc_stats[f"PC{pc_idx+1}"]["std"],
        alpha=0.3,
        color="red",
        label=f"openphenom PC{pc_idx+1} ±1σ",
    )
    ax[0].legend()
    ax[0].set_xlabel("Time (min)")
    ax[1].legend()
    ax[1].set_xlabel("Time (min)")
    plt.savefig(output_root / f"dynaclr_vs_openphenom_pc{pc_idx+1}.png")
    plt.show()
    plt.close()

# %%
# PHATE
avg_std_dynaclr = np.mean(
    [v["std"] for v in alignment_metrics["dynaclr_lineages"]["phate_stats"].values()]
)
avg_std_openphenom = np.mean(
    [v["std"] for v in alignment_metrics["openphenom_lineages"]["phate_stats"].values()]
)

logger.info(f"Average std of dynaclr: {avg_std_dynaclr}")
logger.info(f"Average std of openphenom: {avg_std_openphenom}")

# %%
# Display the top alignments for each model in napari
# Cach
import os
import napari
from viscy.data.triplet import TripletDataModule

os.environ["DISPLAY"] = ":1"


YX_PATCH_SIZE = 128
z_range = (0, 1)
channels_to_display = ["Phase3D", "DIC", "BF"]

top_n_aligned_cells = 1

image_cache = {}
for i, row in top_n_aligned_cells.iterrows():
    data_module = TripletDataModule(
        data_path=data_path,
        tracks_path=tracks_path,
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
    image_cache[f"{fov_name[1:].replace('/', '_')}_track_{track_ids[0]}"] = img_tczyx

# %%
