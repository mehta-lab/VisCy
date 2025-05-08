# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotting_utils import (
    align_image_stacks,
    create_consensus_embedding,
    find_pattern_matches,
    plot_reference_aligned_average,
    plot_reference_vs_full_lineages,
)

# from dtaidistance import dtw
# from dtaidistance import dtw_visualisation as dtwvis
from scipy.spatial.distance import cdist

from utils import (
    dtw_with_matrix,
    filter_lineages_by_timepoints,
    identify_lineages,
    path_skew,
)
from viscy.representation.embedding_writer import read_embedding_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def validate_alignment(
    dtw_distance, warping_path, ref_len, query_len, max_distance=5.0, max_skew=0.8
):
    """
    Validate a DTW alignment based on distance and path skewness.

    Args:
        dtw_distance: The DTW distance value
        warping_path: The warping path as a list of (ref_idx, query_idx) tuples
        ref_len: Length of the reference sequence
        query_len: Length of the query sequence
        max_distance: Maximum allowed DTW distance (normalized)
        max_skew: Maximum allowed path skewness (0-1)

    Returns:
        (is_valid, message): Tuple with boolean validation result and reason
    """
    if dtw_distance > max_distance:
        return (
            False,
            f"No valid alignment (too dissimilar, distance={dtw_distance:.2f})",
        )

    skewness = path_skew(warping_path, ref_len, query_len)
    if skewness > max_skew:
        return False, f"Path too skewed (skew={skewness:.2f}), likely bad alignment"

    return (
        True,
        f"Good alignment found (distance={dtw_distance:.2f}, skew={skewness:.2f})",
    )


def plot_dtw_distances(
    distances, labels=None, title="DTW Distances from Reference Track"
):
    """Plot DTW distances from reference track to other tracks"""
    plt.figure(figsize=(12, 6))
    x = range(len(distances))
    plt.bar(x, distances)

    if labels:
        plt.xticks(x, labels, rotation=90, fontsize=8)
    else:
        plt.xticks(x, rotation=90)

    plt.xlabel("Track Index")
    plt.ylabel("DTW Distance")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()


def get_annotations_for_lineage(fov_id, track_ids, annotations_df):
    """
    Get combined annotations for all tracks in a lineage.

    Args:
        fov_id: FOV ID of the lineage
        track_ids: Track IDs in the lineage
        annotations_df: DataFrame containing annotations

    Returns:
        numpy array of annotations
    """
    lineage_annotations = []

    # Fix fov_id format if needed
    if not fov_id.startswith("/"):
        fov_id = "/" + fov_id

    for track_id in track_ids:
        # Get annotations for this track
        track_annotations = annotations_df[
            (annotations_df["fov_name"] == fov_id)
            & (annotations_df["track_id"] == track_id)
        ].sort_values("t")

        # Extract mitosis/interphase annotations (binary)
        if len(track_annotations) > 0:
            phases = track_annotations["division"].values
            lineage_annotations.extend(phases)

    return np.array(lineage_annotations)


def plot_multiple_alignments(
    alignments_data, reference_embeddings, dataset, n_cols=3, max_plots=12
):
    """
    Plot multiple lineage alignments in a single figure with subplots.

    Args:
        alignments_data: DataFrame of alignment results from align_all_lineages_to_reference
        reference_embeddings: Reference lineage embeddings
        dataset: Dataset containing embeddings
        n_cols: Number of columns in the subplot grid
        max_plots: Maximum number of plots to show

    Returns:
        matplotlib figure object
    """
    # Filter to valid alignments
    valid_alignments = alignments_data[alignments_data["is_valid"] == True]
    n_valid = min(len(valid_alignments), max_plots)

    if n_valid == 0:
        logger.warning("No valid alignments to plot")
        return None

    # Calculate grid dimensions
    n_rows = (n_valid + n_cols - 1) // n_cols

    # Create figure
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3))

    for i, (index, result) in enumerate(valid_alignments.iloc[:max_plots].iterrows()):
        if i >= max_plots:
            break

        # Get embeddings for this lineage
        fov_id = result["fov_id"]
        track_ids = result["track_ids"]

        lineage_embeddings = []
        for track_id in track_ids:
            try:
                track_embeddings = dataset.sel(
                    sample=("/" + fov_id, track_id)
                ).features.values
                lineage_embeddings.append(track_embeddings)
            except KeyError:
                continue

        if not lineage_embeddings:
            continue

        lineage_emb = np.concatenate(lineage_embeddings, axis=0)

        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        # Compute distance matrix
        dist_matrix = cdist(reference_embeddings, lineage_emb, metric="euclidean")

        # Plot distance matrix
        im = ax.imshow(dist_matrix, origin="lower", aspect="auto", cmap="viridis")

        # Plot warping path
        best_path = result["warping_path"]
        ax.plot([y for x, y in best_path], [x for x, y in best_path], "r-", linewidth=1)

        # Mark division events if available
        # Check if column exists and isn't NaN (for list/array columns, check if it's not None)
        has_reference_divisions = (
            "reference_division_indices" in result
            and result["reference_division_indices"] is not None
        )
        has_query_divisions = (
            "query_division_indices" in result
            and result["query_division_indices"] is not None
        )

        if has_reference_divisions and has_query_divisions:
            ref_div = result["reference_division_indices"]
            query_div = result["query_division_indices"]

            # Mark reference divisions with horizontal lines
            for div_idx in ref_div:
                ax.axhline(
                    y=div_idx, color="blue", linestyle="--", alpha=0.5, linewidth=0.8
                )

            # Mark query divisions with vertical lines
            for div_idx in query_div:
                ax.axvline(
                    x=div_idx, color="green", linestyle="--", alpha=0.5, linewidth=0.8
                )

        # Add title and legend
        match_rate_str = ""
        if (
            "iou_match_rate" in result
            and result["iou_match_rate"] is not None
            and not pd.isna(result["iou_match_rate"])
        ):
            match_rate_str = f", IoU: {result['iou_match_rate']:.2f}"
        elif (
            "match_rate" in result
            and result["match_rate"] is not None
            and not pd.isna(result["match_rate"])
        ):
            match_rate_str = f", Match: {result.get('match_rate', 0):.2f}"

        title = f"Lineage {result['lineage_label']}\n"
        title += f"Dist: {result['distance']:.2f}{match_rate_str}"
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Query", fontsize=7)
        ax.set_ylabel("Reference", fontsize=7)
        ax.tick_params(axis="both", which="major", labelsize=6)

    plt.tight_layout()
    return fig


def plot_division_events_summary(alignments_data, reference_embeddings, max_plots=20):
    """
    Plot a summary of division events across multiple lineages.

    Args:
        alignments_data: DataFrame of alignment results
        reference_embeddings: Reference embeddings
        max_plots: Maximum number of lineages to include

    Returns:
        matplotlib figure
    """
    # Filter to alignments with division information
    valid_mask = alignments_data["is_valid"] == True

    # For list/array columns, we need to check if they're not None rather than using pd.isna()
    has_division_info = alignments_data.apply(
        lambda row: row.get("reference_division_indices") is not None
        and isinstance(row["reference_division_indices"], list)
        and len(row["reference_division_indices"]) > 0,
        axis=1,
    )

    div_mask = valid_mask & has_division_info
    div_alignments = alignments_data[div_mask]

    if len(div_alignments) == 0:
        logger.warning("No alignments with division information to plot")
        return None

    # Sort by match rate if available
    if "iou_match_rate" in div_alignments.columns:
        not_na_iou = div_alignments["iou_match_rate"].notna()
        if not_na_iou.any():
            sorted_alignments = (
                div_alignments[not_na_iou]
                .sort_values("iou_match_rate", ascending=False)
                .iloc[:max_plots]
            )
        else:
            sorted_alignments = div_alignments.iloc[:max_plots]
    else:
        sorted_alignments = div_alignments.iloc[:max_plots]

    n_alignments = len(sorted_alignments)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, n_alignments * 0.4)))

    # Create a timeline based on reference embeddings length
    max_len = len(reference_embeddings)
    timeline = np.arange(max_len)

    # Plot division events for each lineage
    y_positions = []
    y_labels = []

    for i, (index, result) in enumerate(sorted_alignments.iterrows()):
        y_pos = i + 1
        y_positions.append(y_pos)
        y_labels.append(result["lineage_label"])

        # Reference division indices
        ref_div = result["reference_division_indices"]

        # Plot reference division events
        for div_idx in ref_div:
            if div_idx < max_len:
                ax.scatter(div_idx, y_pos, color="red", marker="X", s=120, alpha=0.8)

        # Plot query division indices mapped to reference timeline
        has_query_divisions = (
            "query_divisions_on_ref_timeline" in result
            and result["query_divisions_on_ref_timeline"] is not None
        )
        if has_query_divisions:
            query_div_on_ref = result["query_divisions_on_ref_timeline"]

            for div_idx in query_div_on_ref:
                if div_idx < max_len:
                    ax.scatter(
                        div_idx, y_pos, color="blue", marker="|", s=150, alpha=0.8
                    )

            # Highlight where reference and query divisions match
            matching_divisions = set(ref_div) & set(query_div_on_ref)
            for div_idx in matching_divisions:
                if div_idx < max_len:
                    # Draw a circle around matching divisions
                    ax.scatter(
                        div_idx,
                        y_pos,
                        color="purple",
                        marker="o",
                        s=180,
                        facecolors="none",
                        linewidth=2,
                        alpha=0.7,
                    )

    # Set labels and title
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Timeline (Reference)")
    ax.set_ylabel("Lineage FOV")
    ax.set_title("Division Events Across Aligned Lineages")

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="X",
            color="red",
            label="Reference Division",
            markerfacecolor="red",
            markersize=10,
            linestyle="none",
        ),
        Line2D(
            [0],
            [0],
            marker="|",
            color="blue",
            label="Query Division",
            markerfacecolor="blue",
            markersize=12,
            linestyle="none",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="purple",
            label="Matching Division",
            markerfacecolor="none",
            markersize=10,
            linestyle="none",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    # Grid lines
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig


def load_track_images(input_data_path, tracks_path, fov_id, track_id):
    """
    Load images for a specific track.

    Args:
        input_data_path: Path to input data
        tracks_path: Path to tracking data
        fov_id: FOV ID
        track_id: Track ID

    Returns:
        Numpy array of images for the track
    """
    from viscy.data.triplet import TripletDataModule

    # Initialize the data module
    data_module = TripletDataModule(
        data_path=input_data_path,
        tracks_path=tracks_path,
        source_channel=["Phase3D"],  # Adjust channels as needed
        z_range=[10, 30],
        initial_yx_patch_size=(192, 192),
        final_yx_patch_size=(192, 192),
        batch_size=1,
        num_workers=4,
        predict_cells=True,
        include_fov_names=[fov_id],
        include_track_ids=[track_id],
    )
    data_module.setup("predict")

    # Get the images for the track
    track_images = []
    for batch in data_module.predict_dataloader():
        image = batch["anchor"].numpy()[0]
        track_images.append(image)

    return np.array(track_images)


# %%
# Align all lineages to a reference and identify division events


def align_all_lineages_to_reference(
    reference_embeddings,
    reference_fov,
    reference_tracks,
    all_lineages,
    dataset,
    annotation_path,
    max_distance=5.0,
    max_skew=0.8,
    identify_divisions=True,
):
    """
    Align all lineages to a reference lineage and identify important events like cell division.

    Args:
        reference_embeddings: Embeddings of the reference lineage
        reference_fov: FOV ID of the reference lineage
        reference_tracks: Track IDs in the reference lineage
        all_lineages: List of lineages to align (list of (fov_id, [track_ids]) tuples)
        dataset: Dataset containing the embeddings
        annotation_path: Path to the annotations file
        max_distance: Maximum allowed DTW distance
        max_skew: Maximum allowed path skewness
        identify_divisions: Whether to identify division events in aligned lineages

    Returns:
        DataFrame with alignment results for each lineage, indexed by lineage_id
    """
    logger.info(
        f"Starting alignment of all lineages to reference lineage (FOV: {reference_fov}, tracks: {reference_tracks})"
    )

    annotations_df = pd.read_csv(annotation_path)
    annotations_df["fov_name"] = "/" + annotations_df["fov ID"]

    reference_annotations = None
    if identify_divisions:
        reference_annotations = get_annotations_for_lineage(
            reference_fov, reference_tracks, annotations_df
        )

        if len(reference_annotations) != len(reference_embeddings):
            logger.warning(
                f"Reference annotations count ({len(reference_annotations)}) "
                + f"does not match embeddings count ({len(reference_embeddings)})"
            )
            if len(reference_annotations) > len(reference_embeddings):
                reference_annotations = reference_annotations[
                    : len(reference_embeddings)
                ]
            else:
                reference_annotations = np.pad(
                    reference_annotations,
                    (0, len(reference_embeddings) - len(reference_annotations)),
                    constant_values=-1,
                )

    alignment_results = {}

    for lineage_idx, (fov_id, track_ids) in enumerate(all_lineages):
        logger.info(
            f"Aligning lineage {lineage_idx+1}/{len(all_lineages)}: FOV {fov_id}, tracks: {track_ids}"
        )

        lineage_embeddings = []
        for track_id in track_ids:
            try:
                track_embeddings = dataset.sel(
                    sample=("/" + fov_id, track_id)
                ).features.values
                lineage_embeddings.append(track_embeddings)
            except KeyError:
                logger.warning(
                    f"Could not find embeddings for track {track_id} in FOV {fov_id}"
                )
                continue

        if not lineage_embeddings:
            logger.warning(f"No valid embeddings found for lineage in FOV {fov_id}")
            continue

        lineage_emb = np.concatenate(lineage_embeddings, axis=0)

        dist_matrix = cdist(reference_embeddings, lineage_emb, metric="euclidean")
        dtw_distance, warping_matrix, best_path = dtw_with_matrix(dist_matrix)

        is_valid, validation_message = validate_alignment(
            dtw_distance,
            best_path,
            len(reference_embeddings),
            len(lineage_emb),
            max_distance=max_distance,
            max_skew=max_skew,
        )

        result = {
            "lineage_idx": lineage_idx,
            "fov_id": "/" + fov_id,
            "track_ids": track_ids,
            "distance": dtw_distance,
            "is_valid": is_valid,
            "validation_message": validation_message,
            "warping_path": best_path,
        }

        if identify_divisions and is_valid and reference_annotations is not None:
            lineage_annotations = get_annotations_for_lineage(
                fov_id, track_ids, annotations_df
            )

            if len(lineage_annotations) != len(lineage_emb):
                if len(lineage_annotations) > len(lineage_emb):
                    lineage_annotations = lineage_annotations[: len(lineage_emb)]
                else:
                    lineage_annotations = np.pad(
                        lineage_annotations,
                        (0, len(lineage_emb) - len(lineage_annotations)),
                        constant_values=-1,
                    )

            # Create aligned versions of the annotations
            aligned_ref_annotations = np.ones(len(lineage_annotations)) * -1
            aligned_query_annotations = np.ones(len(reference_annotations)) * -1

            for ref_idx, query_idx in best_path:
                if 0 <= ref_idx < len(reference_annotations) and 0 <= query_idx < len(
                    lineage_annotations
                ):
                    aligned_ref_annotations[query_idx] = reference_annotations[ref_idx]
                    aligned_query_annotations[ref_idx] = lineage_annotations[query_idx]

            # Identify division events in reference and aligned lineage
            division_indices_ref = np.where(reference_annotations == 1)[0]
            division_indices_query = np.where(lineage_annotations == 1)[0]

            # Map reference division events to query timeline
            aligned_division_indices = []
            for ref_div_idx in division_indices_ref:
                # Find the corresponding query indices in the warping path
                query_indices = [
                    q_idx for r_idx, q_idx in best_path if r_idx == ref_div_idx
                ]
                if query_indices:
                    aligned_division_indices.extend(query_indices)

            # Map query divisions to reference timeline
            query_divisions_on_ref_timeline = []
            for query_div_idx in division_indices_query:
                # Find corresponding reference indices in the warping path
                ref_indices = [
                    r_idx for r_idx, q_idx in best_path if q_idx == query_div_idx
                ]
                if ref_indices:
                    query_divisions_on_ref_timeline.extend(ref_indices)

            # Calculate match rates
            valid_indices_query = aligned_ref_annotations != -1
            valid_indices_ref = aligned_query_annotations != -1

            match_rate_query = 0
            match_rate_ref = 0
            match_rate = 0

            if np.any(valid_indices_query):
                match_rate_query = np.mean(
                    aligned_ref_annotations[valid_indices_query]
                    == lineage_annotations[valid_indices_query]
                )

            if np.any(valid_indices_ref):
                match_rate_ref = np.mean(
                    reference_annotations[valid_indices_ref]
                    == aligned_query_annotations[valid_indices_ref]
                )

            if np.any(valid_indices_query) and np.any(valid_indices_ref):
                match_rate = (match_rate_query + match_rate_ref) / 2

            # Calculate match rate using IoU (Intersection over Union)
            ref_div_set = set(division_indices_ref.tolist())
            query_div_on_ref_set = set(query_divisions_on_ref_timeline)

            intersection = len(ref_div_set.intersection(query_div_on_ref_set))
            union = len(ref_div_set.union(query_div_on_ref_set))

            iou_match_rate = 0
            if union > 0:
                iou_match_rate = intersection / union

            # Update result with division information and match rates
            result.update(
                {
                    "reference_division_indices": division_indices_ref.tolist(),
                    "query_division_indices": division_indices_query.tolist(),
                    "aligned_division_indices": aligned_division_indices,
                    "query_divisions_on_ref_timeline": query_divisions_on_ref_timeline,
                    "match_rate_query": match_rate_query,
                    "match_rate_ref": match_rate_ref,
                    "match_rate": match_rate,
                    "iou_match_rate": iou_match_rate,
                }
            )

        # Generate a unique lineage ID combining FOV and track IDs
        lineage_id = f"{fov_id}_{'-'.join(map(str, track_ids))}"
        alignment_results[lineage_id] = result

    # Convert the results dictionary to a DataFrame
    results_df = pd.DataFrame.from_dict(alignment_results, orient="index")

    # Add a "lineage_label" column that combines FOV and first track ID for display
    results_df["lineage_label"] = results_df.apply(
        lambda row: f"{row['fov_id']}-{row['track_ids'][0] if row['track_ids'] else 'unknown'}",
        axis=1,
    )

    logger.info(f"Completed alignment of {len(results_df)} lineages")
    return results_df


# %% Main entry point for script execution

if __name__ == "__main__":
    test_data_embedding_path = Path(
        "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_7mins.zarr"
    )
    test_data_timeaware_embeddings = read_embedding_dataset(test_data_embedding_path)

    annotation_path = "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/fixed_test_annotations.csv"

    # Get all lineages
    lineages = identify_lineages(annotation_path)
    logger.info(f"Found {len(lineages)} distinct lineages")

    # Filter lineages with fewer than 10 timepoints
    filtered_lineages = filter_lineages_by_timepoints(
        lineages, annotation_path, min_timepoints=25
    )
    logger.info(f"After filtering, {len(filtered_lineages)} lineages remain")

    # Print details of filtered lineages
    for i, (fov_id, tracks) in enumerate(filtered_lineages, 1):
        logger.info(f"Lineage {i-1}: FOV {fov_id}, {len(tracks)} tracks: {tracks}")

    # %% Select a reference lineage containing cell division
    reference_lineage_idx = 26
    reference_fov = filtered_lineages[reference_lineage_idx][0]
    reference_tracks = filtered_lineages[reference_lineage_idx][1]
    logger.info(
        f"Reference lineage: {reference_lineage_idx}, FOV: {reference_fov}, tracks: {reference_tracks}"
    )
    reference_embeddings = None
    for track in reference_tracks:
        track_embeddings = test_data_timeaware_embeddings.sel(
            sample=("/" + reference_fov, track)
        ).features.values
        if reference_embeddings is None:
            reference_embeddings = track_embeddings
        else:
            reference_embeddings = np.vstack((reference_embeddings, track_embeddings))
    # Save a copy of filtered lineages and remove reference
    filtered_lineages.pop(reference_lineage_idx)
    all_filtered_lineages = filtered_lineages.copy()
    # %%
    logger.info("Aligning all lineages to the reference")
    all_alignment_results = align_all_lineages_to_reference(
        reference_embeddings,
        reference_fov,
        reference_tracks,
        all_filtered_lineages,
        test_data_timeaware_embeddings,
        annotation_path,
        max_distance=100.0,
        max_skew=0.8,
    )

    # %% Analyze alignment results
    valid_alignments = all_alignment_results[all_alignment_results["is_valid"] == True]
    invalid_alignments = all_alignment_results[
        all_alignment_results["is_valid"] == False
    ]

    logger.info(f"Total lineages processed: {len(all_alignment_results)}")
    logger.info(
        f"Valid alignments: {len(valid_alignments)} ({len(valid_alignments)/len(all_alignment_results):.2%})"
    )
    logger.info(
        f"Invalid alignments: {len(invalid_alignments)} ({len(invalid_alignments)/len(all_alignment_results):.2%})"
    )

    # Analyze match rates for valid alignments with division info
    if len(valid_alignments) > 0:
        division_mask = ~valid_alignments["iou_match_rate"].isna()
        if division_mask.any():
            match_rates = valid_alignments.loc[division_mask, "iou_match_rate"]
            logger.info(f"Average IoU match rate: {match_rates.mean():.2%}")
            logger.info(f"Median IoU match rate: {match_rates.median():.2%}")

    # Find lineages where we successfully aligned division events
    has_query_divisions = valid_alignments.apply(
        lambda row: "query_division_indices" in row
        and row["query_division_indices"] is not None
        and isinstance(row["query_division_indices"], list)
        and len(row["query_division_indices"]) > 0,
        axis=1,
    )

    lineages_with_divisions = valid_alignments[has_query_divisions]

    logger.info(f"Lineages with division events: {len(lineages_with_divisions)}")

    # Plot all valid alignments in a grid
    logger.info("Plotting all valid alignments in a grid")
    grid_fig = plot_multiple_alignments(
        all_alignment_results,
        reference_embeddings,
        test_data_timeaware_embeddings,
        n_cols=4,
        max_plots=16,
    )

    # Plot comparison of division events on reference timeline
    logger.info("Creating division event comparison visualization")

    # Find lineages that have both reference and query divisions
    has_ref_divisions = valid_alignments.apply(
        lambda row: "reference_division_indices" in row
        and row["reference_division_indices"] is not None
        and isinstance(row["reference_division_indices"], list),
        axis=1,
    )

    has_query_divisions = valid_alignments.apply(
        lambda row: "query_divisions_on_ref_timeline" in row
        and row["query_divisions_on_ref_timeline"] is not None
        and isinstance(row["query_divisions_on_ref_timeline"], list),
        axis=1,
    )

    has_any_divisions = valid_alignments.apply(
        lambda row: (
            row.get("reference_division_indices") is not None
            and isinstance(row["reference_division_indices"], list)
            and len(row["reference_division_indices"]) > 0
        )
        or (
            row.get("query_divisions_on_ref_timeline") is not None
            and isinstance(row["query_divisions_on_ref_timeline"], list)
            and len(row["query_divisions_on_ref_timeline"]) > 0
        ),
        axis=1,
    )

    div_comparison_mask = has_ref_divisions & has_query_divisions & has_any_divisions
    div_comparison_lineages = valid_alignments[div_comparison_mask]

    if len(div_comparison_lineages) > 0:
        # Create a figure for comparing division events on reference timeline
        n_lineages = min(len(div_comparison_lineages), 5)  # Show top 5
        fig, axs = plt.subplots(
            n_lineages, 1, figsize=(12, n_lineages * 2), sharex=True
        )

        # If only one lineage, make axs iterable
        if n_lineages == 1:
            axs = [axs]

        # Sort by IoU match rate
        if (
            "iou_match_rate" in div_comparison_lineages.columns
            and not div_comparison_lineages["iou_match_rate"].isna().all()
        ):
            sorted_div_lineages = div_comparison_lineages.sort_values(
                "iou_match_rate", ascending=False
            ).iloc[:n_lineages]
        else:
            sorted_div_lineages = div_comparison_lineages.iloc[:n_lineages]

        max_len = len(reference_embeddings)

        for i, (index, result) in enumerate(sorted_div_lineages.iterrows()):
            # Get reference and query division indices
            ref_div = result["reference_division_indices"]
            query_div_on_ref = result["query_divisions_on_ref_timeline"]

            # Create timeline indicators
            timeline = np.arange(max_len)
            ref_indicator = np.zeros(max_len)
            query_indicator = np.zeros(max_len)

            for idx in ref_div:
                if idx < max_len:
                    ref_indicator[idx] = 1

            for idx in query_div_on_ref:
                if idx < max_len:
                    query_indicator[idx] = 0.8

            # Plot on the same timeline
            axs[i].step(
                timeline,
                ref_indicator,
                where="post",
                color="blue",
                alpha=0.7,
                label="Reference",
            )
            axs[i].step(
                timeline,
                query_indicator,
                where="post",
                color="green",
                alpha=0.7,
                label="Query",
            )

            # Highlight matches and mismatches
            match_indicator = np.zeros(max_len)
            mismatch_indicator = np.zeros(max_len)

            # Precise match (exact same index)
            for idx in set(ref_div) & set(query_div_on_ref):
                if idx < max_len:
                    match_indicator[idx] = 1.5

            # Mismatches: ref divisions with no nearby query, and query with no nearby ref
            mismatch_ref = [idx for idx in ref_div if idx not in query_div_on_ref]
            mismatch_query = [idx for idx in query_div_on_ref if idx not in ref_div]

            for idx in mismatch_ref + mismatch_query:
                if idx < max_len:
                    mismatch_indicator[idx] = 1.2

            # Plot matches and mismatches
            if np.any(match_indicator > 0):
                axs[i].scatter(
                    np.where(match_indicator > 0)[0],
                    match_indicator[match_indicator > 0],
                    color="purple",
                    marker="*",
                    s=100,
                    alpha=0.7,
                    label="Match",
                )

            if np.any(mismatch_indicator > 0):
                axs[i].scatter(
                    np.where(mismatch_indicator > 0)[0],
                    mismatch_indicator[mismatch_indicator > 0],
                    color="red",
                    marker="x",
                    s=80,
                    alpha=0.7,
                    label="Mismatch",
                )

            # Add title and legend
            match_rate_str = ""
            if (
                "iou_match_rate" in result
                and result["iou_match_rate"] is not None
                and not pd.isna(result["iou_match_rate"])
            ):
                match_rate_str = f", IoU: {result['iou_match_rate']:.2f}"
            elif (
                "match_rate" in result
                and result["match_rate"] is not None
                and not pd.isna(result["match_rate"])
            ):
                match_rate_str = f", Match: {result.get('match_rate', 0):.2f}"

            axs[i].set_title(f"Lineage {result['lineage_label']}{match_rate_str}")
            axs[i].set_ylim(-0.1, 2.0)
            axs[i].set_yticks([])

            # Show legend only on first plot
            if i == 0:
                axs[i].legend(loc="upper right")

        plt.xlabel("Reference Timeline")

        # Apply tight layout before adding the suptitle
        plt.tight_layout()

        # Add more space at the top for the suptitle and increase spacing between subplots
        plt.subplots_adjust(top=0.9, hspace=0.4)

        # Add the suptitle after layout adjustments
        plt.suptitle(
            "Division Events Comparison (aligned to reference timeline)", y=0.98
        )

        # Save as a figure
        div_comparison_fig = fig

    # Plot summary of division events
    if len(lineages_with_divisions) > 0:
        logger.info("Plotting division events summary")
        div_summary_fig = plot_division_events_summary(
            all_alignment_results, reference_embeddings, max_plots=20
        )

    # Example: Print details of top 5 alignments by match rate
    if len(valid_alignments) > 0 and "iou_match_rate" in valid_alignments.columns:
        # Filter to rows with valid IoU match rates
        has_iou_rate = valid_alignments.apply(
            lambda row: "iou_match_rate" in row
            and row["iou_match_rate"] is not None
            and not pd.isna(row["iou_match_rate"]),
            axis=1,
        )

        if has_iou_rate.any():
            top_alignments = (
                valid_alignments[has_iou_rate]
                .sort_values("iou_match_rate", ascending=False)
                .head(5)
            )

            logger.info("\nTop 5 alignments by IoU match rate:")
            for idx, row in top_alignments.iterrows():
                lineage_label = row["lineage_label"]

                logger.info(
                    f"Lineage {lineage_label}: IoU match rate {row['iou_match_rate']:.2%}, DTW distance: {row['distance']:.2f}"
                )

                has_div_indices = (
                    "query_division_indices" in row
                    and row["query_division_indices"] is not None
                    and isinstance(row["query_division_indices"], list)
                )

                if has_div_indices:
                    logger.info(
                        f"  - Reference divisions at: {row['reference_division_indices']}"
                    )
                    logger.info(
                        f"  - Query divisions at: {row['query_division_indices']}"
                    )
                    logger.info(
                        f"  - Query divisions on ref timeline: {row['query_divisions_on_ref_timeline']}"
                    )

    # Find pattern matches using the pattern matching approach from plotting_utils
    reference_timepoints = [0, 50]  # Example timepoints to use as reference pattern
    reference_pattern = reference_embeddings[
        reference_timepoints[0] : reference_timepoints[1]
    ]

    all_match_positions = find_pattern_matches(
        reference_pattern,
        all_filtered_lineages,
        test_data_timeaware_embeddings,
        window_step_fraction=0.25,
        num_candidates=3,
        # save_path="./pattern_matching_results.csv",
    )

    # Get the top N aligned cells
    n_cells = 5
    top_n_aligned_cells = all_match_positions.head(n_cells)

    # Create consensus embedding using the imported function
    consensus_embedding = create_consensus_embedding(
        reference_pattern,
        top_n_aligned_cells,
        test_data_timeaware_embeddings,
    )

    # Add visualization using the imported plotting functions
    plot_reference_aligned_average(
        reference_pattern,
        top_n_aligned_cells,
        test_data_timeaware_embeddings,
        save_path="./reference_aligned_average.png",
    )

    plot_reference_vs_full_lineages(
        reference_pattern,
        top_n_aligned_cells,
        test_data_timeaware_embeddings,
        save_path="./reference_vs_full_lineages.png",
    )

    # Save the consensus embedding
    np.save("./consensus_embedding.npy", consensus_embedding)

    # Save the original alignment results as well
    all_alignment_results.to_csv("./ALFI_test_dataset_alignment_results.csv")

# %%
