# %%
import ast
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plotting_utils import (
    find_pattern_matches,
    identify_lineages,
    plot_pc_trajectories,
)
from tqdm import tqdm

from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import read_embedding_dataset

logger = logging.getLogger("viscy")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")  # Simplified format
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


NAPARI = True
if NAPARI:
    import os

    import napari

    os.environ["DISPLAY"] = ":1"
    viewer = napari.Viewer()
# %%
# Organelle and Phate aligned to infection

input_data_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_DENV.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/4-track-gt/2024_11_07_A549_SEC61_ZIKV_DENV_2_cropped.zarr"
)
infection_annotations_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/4-phenotyping/0-annotation/combined_annotations_n_tracks_infection.csv"
)

pretrain_features_root = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/prediction_pretrained_models"
)
# Phase n organelle
# dynaclr_features_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/predictions/timeAware_2chan__ntxent_192patch_70ckpt_rev7_GT.zarr"

# pahe n sensor
dynaclr_features_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/3-phenotyping/predictions_infection/2chan_192patch_100ckpt_timeAware_ntxent_GT.zarr"

output_root = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/figure/SEC61B/model_comparison"
)


# Load embeddings
imagenet_features_path = (
    pretrain_features_root / "ImageNet/20241107_sensor_n_phase_imagenet.zarr"
)
openphenom_features_path = (
    pretrain_features_root / "OpenPhenom/20241107_sensor_n_phase_openphenom.zarr"
)

dynaclr_embeddings = read_embedding_dataset(dynaclr_features_path)
imagenet_embeddings = read_embedding_dataset(imagenet_features_path)
openphenom_embeddings = read_embedding_dataset(openphenom_features_path)

# Load infection annotations
infection_annotations_df = pd.read_csv(infection_annotations_path)
infection_annotations_df["fov_name"] = "/C/2/000001"

process_embeddings = [
    (dynaclr_embeddings, "dynaclr"),
    (imagenet_embeddings, "imagenet"),
    (openphenom_embeddings, "openphenom"),
]


output_root.mkdir(parents=True, exist_ok=True)
# %%
feature_df = dynaclr_embeddings["sample"].to_dataframe().reset_index(drop=True)

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
# Aligning condition embeddings to infection
# OPTION 1: Use the infection annotations to find the reference lineage
reference_lineage_fov = "/C/2/001000"
reference_lineage_track_id = [129]
reference_timepoints = [8, 70]  # sensor rellocalization and partial remodelling

# Option 2: from the filtered lineages find one from FOV C/2/000001
reference_lineage_fov = "/C/2/000001"
for fov_id, track_ids in filtered_lineages:
    if reference_lineage_fov == fov_id:
        break
reference_lineage_track_id = track_ids
reference_timepoints = [8, 70]  # sensor rellocalization and partial remodelling

# %%
# Dictionary to store alignment results for comparison
alignment_results = {}

for embeddings, name in process_embeddings:
    # Get the reference pattern from the current embedding space
    reference_pattern = None
    reference_lineage = []
    for fov_id, track_ids in filtered_lineages:
        if fov_id == reference_lineage_fov and all(
            track_id in track_ids for track_id in reference_lineage_track_id
        ):
            logger.info(
                f"Found reference pattern for {fov_id} {reference_lineage_track_id} using {name} embeddings"
            )
            reference_pattern = embeddings.sel(
                sample=(fov_id, reference_lineage_track_id)
            ).features.values
            reference_lineage.append(reference_pattern)
            break
    if reference_pattern is None:
        logger.info(f"Reference pattern not found for {name} embeddings. Skipping.")
        continue
    reference_pattern = np.concatenate(reference_lineage)
    reference_pattern = reference_pattern[
        reference_timepoints[0] : reference_timepoints[1]
    ]

    # Find all matches to the reference pattern
    metric = "cosine"
    all_match_positions = find_pattern_matches(
        reference_pattern,
        filtered_lineages,
        embeddings,
        window_step_fraction=0.1,
        num_candidates=4,
        method="bernd_clifford",
        save_path=output_root / f"{name}_matching_lineages_{metric}.csv",
        metric=metric,
    )

    # Store results for later comparison
    alignment_results[name] = all_match_positions

# Visualize warping paths in PC space instead of raw embedding dimensions
for name, match_positions in alignment_results.items():
    if match_positions is not None and not match_positions.empty:
        # Call the new function from plotting_utils
        plot_pc_trajectories(
            reference_lineage_fov=reference_lineage_fov,
            reference_lineage_track_id=reference_lineage_track_id,
            reference_timepoints=reference_timepoints,
            match_positions=match_positions,
            embeddings_dataset=next(
                emb for emb, emb_name in process_embeddings if emb_name == name
            ),
            filtered_lineages=filtered_lineages,
            name=name,
            save_path=output_root / f"{name}_pc_lineage_alignment.png",
        )


# %%
# Compare DTW performance between embedding methods

# Create a DataFrame to collect the alignment statistics for comparison
match_data = []
for name, match_positions in alignment_results.items():
    if match_positions is not None and not match_positions.empty:
        for i, row in match_positions.head(10).iterrows():  # Take top 10 matches
            warping_path = (
                ast.literal_eval(row["warp_path"])
                if isinstance(row["warp_path"], str)
                else row["warp_path"]
            )
            match_data.append(
                {
                    "model": name,
                    "match_position": row["start_timepoint"],
                    "dtw_distance": row["distance"],
                    "path_skewness": row["skewness"],
                    "path_length": len(warping_path),
                }
            )

comparison_df = pd.DataFrame(match_data)

# Create visualizations to compare alignment quality
plt.figure(figsize=(12, 10))

# 1. Compare DTW distances
plt.subplot(2, 2, 1)
sns.boxplot(x="model", y="dtw_distance", data=comparison_df)
plt.title("DTW Distance by Model")
plt.ylabel("DTW Distance (lower is better)")

# 2. Compare path skewness
plt.subplot(2, 2, 2)
sns.boxplot(x="model", y="path_skewness", data=comparison_df)
plt.title("Path Skewness by Model")
plt.ylabel("Skewness (lower is better)")

# 3. Compare path lengths
plt.subplot(2, 2, 3)
sns.boxplot(x="model", y="path_length", data=comparison_df)
plt.title("Warping Path Length by Model")
plt.ylabel("Path Length")

# 4. Scatterplot of distance vs skewness
plt.subplot(2, 2, 4)
scatter = sns.scatterplot(
    x="dtw_distance", y="path_skewness", hue="model", data=comparison_df
)
plt.title("DTW Distance vs Path Skewness")
plt.xlabel("DTW Distance")
plt.ylabel("Path Skewness")
plt.legend(title="Model")

plt.tight_layout()
plt.savefig(output_root / "dtw_alignment_comparison.png", dpi=300)
plt.close()

# %%
# Analyze warping path step patterns for better understanding of alignment quality

# Step pattern analysis
step_pattern_counts = {
    name: {"diagonal": 0, "horizontal": 0, "vertical": 0, "total": 0}
    for name in alignment_results.keys()
}

for name, match_positions in alignment_results.items():
    if match_positions is not None and not match_positions.empty:
        # Get the top match
        top_match = match_positions.iloc[0]
        path = (
            ast.literal_eval(top_match["warp_path"])
            if isinstance(top_match["warp_path"], str)
            else top_match["warp_path"]
        )

        # Count step types
        for i in range(1, len(path)):
            prev_i, prev_j = path[i - 1]
            curr_i, curr_j = path[i]

            step_i = curr_i - prev_i
            step_j = curr_j - prev_j

            if step_i == 1 and step_j == 1:
                step_pattern_counts[name]["diagonal"] += 1
            elif step_i == 1 and step_j == 0:
                step_pattern_counts[name]["vertical"] += 1
            elif step_i == 0 and step_j == 1:
                step_pattern_counts[name]["horizontal"] += 1

            step_pattern_counts[name]["total"] += 1

# Convert to percentages
for name in step_pattern_counts:
    total = step_pattern_counts[name]["total"]
    if total > 0:
        for key in ["diagonal", "horizontal", "vertical"]:
            step_pattern_counts[name][key] = (
                step_pattern_counts[name][key] / total
            ) * 100

# Visualize step pattern distributions
step_df = pd.DataFrame(
    {
        "model": [name for name in step_pattern_counts.keys() for _ in range(3)],
        "step_type": ["diagonal", "horizontal", "vertical"] * len(step_pattern_counts),
        "percentage": [
            step_pattern_counts[name]["diagonal"] for name in step_pattern_counts.keys()
        ]
        + [
            step_pattern_counts[name]["horizontal"]
            for name in step_pattern_counts.keys()
        ]
        + [
            step_pattern_counts[name]["vertical"] for name in step_pattern_counts.keys()
        ],
    }
)

plt.figure(figsize=(10, 6))
sns.barplot(x="model", y="percentage", hue="step_type", data=step_df)
plt.title("Step Pattern Distribution in Warping Paths")
plt.ylabel("Percentage (%)")
plt.savefig(output_root / "step_pattern_distribution.png", dpi=300)
plt.close()

# %%
# Find all matches to the reference pattern
MODEL = "openphenom"
alignment_df_path = output_root / f"{MODEL}_matching_lineages_cosine.csv"
alignment_df = pd.read_csv(alignment_df_path)

# Get the top N aligned cells

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
top_aligned_cells = alignment_df.head(5)
napari_viewer = viewer if NAPARI else None
# Plot the aligned and unaligned stacks
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
if NAPARI:
    for idx, row in tqdm(
        top_aligned_cells.reset_index().iterrows(),
        total=len(top_aligned_cells),
        desc="Plotting aligned and unaligned stacks",
    ):
        fov_name = row["fov_name"]
        # track_ids = ast.literal_eval(row["track_ids"])
        track_ids = row["track_ids"]

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
# Evaluate model performance based on infection state warping accuracy
# Check unique infection status values
unique_infection_statuses = infection_annotations_df["infection_status"].unique()
logger.info(f"Unique infection status values: {unique_infection_statuses}")

# If "infected" is not in the unique values, this could explain zero precision/recall
if "infected" not in unique_infection_statuses:
    logger.warning('The label "infected" is not found in the infection_status column!')
    logger.info(f"Using these values instead: {unique_infection_statuses}")

    # If we need to map values, we could do it here
    if len(unique_infection_statuses) >= 2:
        logger.info(
            f'Will treat "{unique_infection_statuses[1]}" as "infected" for metrics calculation'
        )
        infection_target_value = unique_infection_statuses[1]
    else:
        infection_target_value = unique_infection_statuses[0]
else:
    infection_target_value = "infected"

logger.info(f'Using "{infection_target_value}" as positive class for F1 calculation')

# Check if the reference track is in the annotations
logger.info(
    f"Looking for infection annotations for reference lineage: {reference_lineage_fov}, tracks: {reference_lineage_track_id}"
)
print(f"Sample of infection_annotations_df: {infection_annotations_df.head()}")

reference_infection_states = {}
for track_id in reference_lineage_track_id:
    reference_annotations = infection_annotations_df[
        (infection_annotations_df["fov_name"] == reference_lineage_fov)
        & (infection_annotations_df["track_id"] == track_id)
    ]

    # Add annotations for this reference track
    annotation_count = len(reference_annotations)
    logger.info(f"Found {annotation_count} annotations for track {track_id}")
    if annotation_count > 0:
        print(
            f"Sample annotations for track {track_id}: {reference_annotations.head()}"
        )

    for _, row in reference_annotations.iterrows():
        reference_infection_states[row["t"]] = row["infection_status"]

if reference_infection_states:
    logger.info(
        f"Total reference timepoints with infection status: {len(reference_infection_states)}"
    )
    reference_t_range = range(reference_timepoints[0], reference_timepoints[1])
    reference_gt_states = [
        reference_infection_states.get(t, "unknown") for t in reference_t_range
    ]
    logger.info(f"Reference track infection states: {reference_gt_states[:5]}...")

    # Evaluate warping accuracy for each model
    model_performance = []

    for name, match_positions in alignment_results.items():
        if match_positions is not None and not match_positions.empty:
            total_correct = 0
            total_predictions = 0
            true_positives = 0
            false_positives = 0
            false_negatives = 0

            # Analyze top alignments for this model
            alignment_details = []
            for i, row in match_positions.head(10).iterrows():
                fov_name = row["fov_name"]
                track_ids = row[
                    "track_ids"
                ]  # This is already a list of track IDs for the lineage
                warp_path = (
                    ast.literal_eval(row["warp_path"])
                    if isinstance(row["warp_path"], str)
                    else row["warp_path"]
                )
                start_time = int(row["start_timepoint"])

                # Get annotations for all tracks in this lineage
                track_infection_states = {}
                for track_id in track_ids:
                    track_annotations = infection_annotations_df[
                        (infection_annotations_df["fov_name"] == fov_name)
                        & (infection_annotations_df["track_id"] == track_id)
                    ]

                    # Add annotations for this track to the combined dictionary
                    for _, annotation_row in track_annotations.iterrows():
                        # Use t + track-specific offset if needed to handle timepoint overlaps between tracks
                        track_infection_states[annotation_row["t"]] = annotation_row[
                            "infection_status"
                        ]

                # Only proceed if we found annotations for at least one track
                if track_infection_states:
                    # For each reference timepoint, check if the warped timepoint maintains the infection state
                    track_correct = 0
                    track_predictions = 0
                    track_tp = 0
                    track_fp = 0
                    track_fn = 0

                    for ref_idx, query_idx in warp_path:
                        # Map to actual timepoints
                        ref_t = reference_timepoints[0] + ref_idx
                        query_t = start_time + query_idx

                        # Get ground truth infection states
                        ref_state = reference_infection_states.get(ref_t, "unknown")
                        query_state = track_infection_states.get(query_t, "unknown")

                        # Skip unknown states
                        if ref_state != "unknown" and query_state != "unknown":
                            track_predictions += 1

                            # Count correct alignments
                            if ref_state == query_state:
                                track_correct += 1

                            # Calculate F1 score components for "infected" state
                            if (
                                ref_state == infection_target_value
                                and query_state == infection_target_value
                            ):
                                track_tp += 1
                            elif (
                                ref_state != infection_target_value
                                and query_state == infection_target_value
                            ):
                                track_fp += 1
                            elif (
                                ref_state == infection_target_value
                                and query_state != infection_target_value
                            ):
                                track_fn += 1

                    # Calculate track-specific metrics
                    if track_predictions > 0:
                        track_accuracy = track_correct / track_predictions
                        track_precision = (
                            track_tp / (track_tp + track_fp)
                            if (track_tp + track_fp) > 0
                            else 0
                        )
                        track_recall = (
                            track_tp / (track_tp + track_fn)
                            if (track_tp + track_fn) > 0
                            else 0
                        )
                        track_f1 = (
                            2
                            * (track_precision * track_recall)
                            / (track_precision + track_recall)
                            if (track_precision + track_recall) > 0
                            else 0
                        )

                        alignment_details.append(
                            {
                                "fov_name": fov_name,
                                "track_ids": track_ids,
                                "accuracy": track_accuracy,
                                "precision": track_precision,
                                "recall": track_recall,
                                "f1_score": track_f1,
                                "correct": track_correct,
                                "total": track_predictions,
                            }
                        )

                        # Add to model totals
                        total_correct += track_correct
                        total_predictions += track_predictions
                        true_positives += track_tp
                        false_positives += track_fp
                        false_negatives += track_fn

            # Calculate metrics
            accuracy = total_correct / total_predictions if total_predictions > 0 else 0
            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            # Store alignment details for this model
            if alignment_details:
                alignment_details_df = pd.DataFrame(alignment_details)
                print(f"\nDetailed alignment results for {name}:")
                print(alignment_details_df)
                alignment_details_df.to_csv(
                    output_root / f"{name}_alignment_details.csv", index=False
                )

            model_performance.append(
                {
                    "model": name,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "total_predictions": total_predictions,
                }
            )

    # Create performance DataFrame and visualize
    performance_df = pd.DataFrame(model_performance)
    print(performance_df)

    # Plot performance metrics
    plt.figure(figsize=(12, 8))

    # Accuracy plot
    plt.subplot(2, 2, 1)
    sns.barplot(x="model", y="accuracy", data=performance_df)
    plt.title("Infection State Warping Accuracy")
    plt.ylabel("Accuracy")

    # Precision plot
    plt.subplot(2, 2, 2)
    sns.barplot(x="model", y="precision", data=performance_df)
    plt.title("Precision for Infected State")
    plt.ylabel("Precision")

    # Recall plot
    plt.subplot(2, 2, 3)
    sns.barplot(x="model", y="recall", data=performance_df)
    plt.title("Recall for Infected State")
    plt.ylabel("Recall")

    # F1 score plot
    plt.subplot(2, 2, 4)
    sns.barplot(x="model", y="f1_score", data=performance_df)
    plt.title("F1 Score for Infected State")
    plt.ylabel("F1 Score")

    plt.tight_layout()
    # plt.savefig(output_root / "infection_state_warping_performance.png", dpi=300)
    # plt.close()
else:
    logger.warning("Reference track annotations not found in infection_annotations_df")

# %%
