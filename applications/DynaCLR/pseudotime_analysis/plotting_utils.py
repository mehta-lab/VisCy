# Plotting utils
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from viscy.data.triplet import TripletDataModule

logger = logging.getLogger(__name__)


def plot_reference_aligned_average(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_dataset: xr.Dataset,
    save_path: str | None = None,
) -> np.ndarray:
    """
    Plot the reference embedding, aligned embeddings, and average aligned embedding.

    Args:
        reference_pattern: The reference pattern embeddings
        top_aligned_cells: DataFrame with alignment information
        embeddings_dataset: Dataset containing embeddings
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(15, 10))

    # Get the reference pattern embeddings
    reference_embeddings = reference_pattern

    # Calculate average aligned embeddings
    all_aligned_embeddings = []
    for idx, row in top_aligned_cells.iterrows():
        fov_name = row["fov_name"]
        track_ids = row["track_ids"]
        warp_path = row["warp_path"]
        start_time = int(row["start_timepoint"])

        # Reconstruct the concatenated lineage
        lineages = []
        track_offsets = {}  # To keep track of where each track starts in the concatenated array
        current_offset = 0

        for track_id in track_ids:
            track_embeddings = embeddings_dataset.sel(
                sample=(fov_name, track_id)
            ).features.values
            track_offsets[track_id] = current_offset
            current_offset += len(track_embeddings)
            lineages.append(track_embeddings)

        lineage_embeddings = np.concatenate(lineages, axis=0)

        # Create aligned embeddings using the warping path
        aligned_embeddings = np.zeros(
            (len(reference_pattern), lineage_embeddings.shape[1]),
            dtype=lineage_embeddings.dtype,
        )

        # Create mapping from reference to lineage
        ref_to_lineage = {}
        for ref_idx, query_idx in warp_path:
            lineage_idx = int(start_time + query_idx)
            if 0 <= lineage_idx < len(lineage_embeddings):
                ref_to_lineage[ref_idx] = lineage_idx

        # Fill aligned embeddings
        for ref_idx in range(len(reference_pattern)):
            if ref_idx in ref_to_lineage:
                aligned_embeddings[ref_idx] = lineage_embeddings[
                    ref_to_lineage[ref_idx]
                ]
            elif ref_to_lineage:
                closest_ref_idx = min(
                    ref_to_lineage.keys(), key=lambda x: abs(x - ref_idx)
                )
                aligned_embeddings[ref_idx] = lineage_embeddings[
                    ref_to_lineage[closest_ref_idx]
                ]

        all_aligned_embeddings.append(aligned_embeddings)

    # Calculate average aligned embeddings
    average_aligned_embeddings = np.mean(all_aligned_embeddings, axis=0)

    # Plot dimension 0
    plt.subplot(2, 1, 1)
    # Plot reference pattern
    plt.plot(
        range(len(reference_embeddings)),
        reference_embeddings[:, 0],
        label="Reference",
        color="black",
        linewidth=3,
    )

    # Plot each aligned embedding
    for i, aligned_embeddings in enumerate(all_aligned_embeddings):
        plt.plot(
            range(len(aligned_embeddings)),
            aligned_embeddings[:, 0],
            label=f"Aligned {i}",
            alpha=0.4,
            linestyle="--",
        )

    # Plot average aligned embedding
    plt.plot(
        range(len(average_aligned_embeddings)),
        average_aligned_embeddings[:, 0],
        label="Average Aligned",
        color="red",
        linewidth=2,
    )

    plt.title("Dimension 0: Reference, Aligned, and Average Embeddings")
    plt.xlabel("Reference Time Index")
    plt.ylabel("Embedding Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot dimension 1
    plt.subplot(2, 1, 2)
    # Plot reference pattern
    plt.plot(
        range(len(reference_embeddings)),
        reference_embeddings[:, 1],
        label="Reference",
        color="black",
        linewidth=3,
    )

    # Plot each aligned embedding
    for i, aligned_embeddings in enumerate(all_aligned_embeddings):
        plt.plot(
            range(len(aligned_embeddings)),
            aligned_embeddings[:, 1],
            label=f"Aligned {i}",
            alpha=0.4,
            linestyle="--",
        )

    # Plot average aligned embedding
    plt.plot(
        range(len(average_aligned_embeddings)),
        average_aligned_embeddings[:, 1],
        label="Average Aligned",
        color="red",
        linewidth=2,
    )

    plt.title("Dimension 1: Reference, Aligned, and Average Embeddings")
    plt.xlabel("Reference Time Index")
    plt.ylabel("Embedding Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

    return average_aligned_embeddings


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

    # Then plot each lineage with the matched section highlighted
    for i, (_, row) in enumerate(top_aligned_cells.iterrows()):
        fov_name = row["fov_name"]
        track_ids = row["track_ids"]
        warp_path = row["warp_path"]
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

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def align_and_average_embeddings(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_dataset: xr.Dataset,
    use_median: bool = False,
) -> np.ndarray:
    """
    Align embeddings from multiple lineages to a reference pattern and compute their average.

    Args:
        reference_pattern: The reference pattern embeddings
        top_aligned_cells: DataFrame with alignment information
        embeddings_dataset: Dataset containing embeddings
        use_median: If True, use median instead of mean for averaging

    Returns:
        The average (or median) aligned embeddings
    """
    all_aligned_embeddings = []

    for idx, row in top_aligned_cells.iterrows():
        fov_name = row["fov_name"]
        track_ids = row["track_ids"]
        warp_path = row["warp_path"]
        start_time = int(row["start_timepoint"])

        # Reconstruct the concatenated lineage
        lineages = []
        track_offsets = {}  # To keep track of where each track starts in the concatenated array
        current_offset = 0

        for track_id in track_ids:
            track_embeddings = embeddings_dataset.sel(
                sample=(fov_name, track_id)
            ).features.values
            track_offsets[track_id] = current_offset
            current_offset += len(track_embeddings)
            lineages.append(track_embeddings)

        lineage_embeddings = np.concatenate(lineages, axis=0)

        # Create aligned embeddings using the warping path
        aligned_segment = np.zeros_like(reference_pattern)

        # Map each reference timepoint to the corresponding lineage timepoint
        ref_to_lineage = {}
        for ref_idx, query_idx in warp_path:
            lineage_idx = int(start_time + query_idx)
            if 0 <= lineage_idx < len(lineage_embeddings):
                ref_to_lineage[ref_idx] = lineage_idx
                aligned_segment[ref_idx] = lineage_embeddings[lineage_idx]

        # Fill in missing values by using the closest available reference index
        for ref_idx in range(len(reference_pattern)):
            if ref_idx not in ref_to_lineage and ref_to_lineage:
                closest_ref_idx = min(
                    ref_to_lineage.keys(), key=lambda x: abs(x - ref_idx)
                )
                aligned_segment[ref_idx] = lineage_embeddings[
                    ref_to_lineage[closest_ref_idx]
                ]

        all_aligned_embeddings.append(aligned_segment)

    all_aligned_embeddings = np.array(all_aligned_embeddings)

    # Compute average or median
    if use_median:
        return np.median(all_aligned_embeddings, axis=0)
    else:
        return np.mean(all_aligned_embeddings, axis=0)


def align_image_stacks(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    input_data_path: Path,
    tracks_path: Path,
    source_channels: list[str],
    yx_patch_size: tuple[int, int] = (192, 192),
    z_range: tuple[int, int] = (0, 1),
    view_ref_sector_only: bool = True,
    napari_viewer=None,
) -> tuple[list, list]:
    """
    Align image stacks from multiple lineages to a reference pattern.

    Args:
        reference_pattern: The reference pattern embeddings
        top_aligned_cells: DataFrame with alignment information
        input_data_path: Path to the input data
        tracks_path: Path to the tracks data
        source_channels: List of channels to include
        yx_patch_size: Patch size for images
        z_range: Z-range to include
        view_ref_sector_only: If True, only show the section that matches the reference pattern
        napari_viewer: Optional napari viewer for visualization

    Returns:
        Tuple of (all_lineage_images, all_aligned_stacks)
    """
    from tqdm import tqdm

    all_lineage_images = []
    all_aligned_stacks = []

    for idx, row in tqdm(
        top_aligned_cells.iterrows(),
        total=len(top_aligned_cells),
        desc="Aligning images",
    ):
        fov_name = row["fov_name"]
        track_ids = row["track_ids"]
        warp_path = row["warp_path"]
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

            # Map each reference timepoint to the corresponding lineage timepoint
            for ref_idx in range(len(reference_pattern)):
                # Find matches in warping path for this reference index
                matches = [(i, q) for i, q in warp_path if i == ref_idx]

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
            if napari_viewer:
                napari_viewer.add_image(
                    aligned_stack,
                    name=f"Aligned_{fov_name}_track_{track_ids[0]}",
                    channel_axis=1,
                )
        else:
            # View the whole lineage shifted by the start time
            start_idx = int(start_time)
            aligned_stack = lineage_images[start_idx:]
            all_aligned_stacks.append(aligned_stack)
            if napari_viewer:
                napari_viewer.add_image(
                    aligned_stack,
                    name=f"Aligned_{fov_name}_track_{track_ids[0]}",
                    channel_axis=1,
                )

    return all_lineage_images, all_aligned_stacks


def find_pattern_matches(
    reference_pattern: np.ndarray,
    filtered_lineages: list[tuple[str, list[int]]],
    embeddings_dataset: xr.Dataset,
    window_step_fraction: float = 0.25,
    num_candidates: int = 3,
    max_distance: float = float("inf"),
    max_skew: float = 0.8,  # Add skewness parameter
    save_path: str | None = None,
    method: str = "bernd_clifford",
    normalize: bool = True,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """
    Find the best matches of a reference pattern in multiple lineages using DTW.

    Args:
        reference_pattern: The reference pattern embeddings
        filtered_lineages: List of lineages to search in (fov_name, track_ids)
        embeddings_dataset: Dataset containing embeddings
        window_step_fraction: Fraction of reference pattern length to use as window step
        num_candidates: Number of best candidates to consider per lineage
        max_distance: Maximum distance threshold to consider a match
        max_skew: Maximum allowed path skewness (0-1, where 0=perfect diagonal)
        save_path: Optional path to save the results CSV
        method: DTW method to use - 'bernd_clifford' (from utils.py) or 'dtai' (dtaidistance library)

    Returns:
        DataFrame with match positions and distances
    """
    from tqdm import tqdm

    # Calculate window step based on reference pattern length
    window_step = max(1, int(len(reference_pattern) * window_step_fraction))

    all_match_positions = {
        "fov_name": [],
        "track_ids": [],
        "distance": [],
        "skewness": [],  # Add skewness to results
        "warp_path": [],
        "start_timepoint": [],
        "end_timepoint": [],
    }

    for i, (fov_name, track_ids) in tqdm(
        enumerate(filtered_lineages),
        total=len(filtered_lineages),
        desc="Finding pattern matches",
    ):
        print(f"Finding pattern matches for {fov_name} with track ids: {track_ids}")
        # Reconstruct the concatenated lineage
        lineages = []
        track_offsets = {}  # To keep track of where each track starts in the concatenated array
        current_offset = 0

        for track_id in track_ids:
            track_embeddings = embeddings_dataset.sel(
                sample=(fov_name, track_id)
            ).features.values
            track_offsets[track_id] = current_offset
            current_offset += len(track_embeddings)
            lineages.append(track_embeddings)

        lineage_embeddings = np.concatenate(lineages, axis=0)

        # Find best matches using the selected DTW method
        if method == "bernd_clifford":
            matches_df = find_best_match_dtw_bernd_clifford(
                lineage_embeddings,
                reference_pattern=reference_pattern,
                num_candidates=num_candidates,
                window_step=window_step,
                max_distance=max_distance,
                max_skew=max_skew,
                normalize=normalize,
                metric=metric,
            )
        else:
            matches_df = find_best_match_dtw(
                lineage_embeddings,
                reference_pattern=reference_pattern,
                num_candidates=num_candidates,
                window_step=window_step,
                max_distance=max_distance,
                max_skew=max_skew,
                normalize=normalize,
            )

        if not matches_df.empty:
            # Get the best match (first row of the sorted DataFrame)
            best_match = matches_df.iloc[0]
            best_pos = best_match["position"]
            best_path = best_match["path"]
            best_dist = best_match["distance"]
            best_skew = best_match["skewness"]

            all_match_positions["fov_name"].append(fov_name)
            all_match_positions["track_ids"].append(track_ids)
            all_match_positions["distance"].append(best_dist)
            all_match_positions["skewness"].append(best_skew)
            all_match_positions["warp_path"].append(best_path)
            all_match_positions["start_timepoint"].append(best_pos)
            all_match_positions["end_timepoint"].append(
                best_pos + len(reference_pattern)
            )
        else:
            all_match_positions["fov_name"].append(fov_name)
            all_match_positions["track_ids"].append(track_ids)
            all_match_positions["distance"].append(None)
            all_match_positions["skewness"].append(None)
            all_match_positions["warp_path"].append(None)
            all_match_positions["start_timepoint"].append(None)
            all_match_positions["end_timepoint"].append(None)

    # Convert to DataFrame and drop rows with no matches
    all_match_positions = pd.DataFrame(all_match_positions)
    all_match_positions = all_match_positions.dropna()

    # Sort by distance (primary) and skewness (secondary)
    all_match_positions = all_match_positions.sort_values(
        by=["distance", "skewness"], ascending=[True, True]
    )

    # Save to CSV if path is provided
    if save_path:
        all_match_positions.to_csv(save_path, index=False)

    return all_match_positions


def find_best_match_dtw(
    lineage: np.ndarray,
    reference_pattern: np.ndarray,
    num_candidates: int = 5,
    window_step: int = 5,
    max_distance: float = float("inf"),
    max_skew: float = 0.8,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Find the best matches in a lineage using DTW with dtaidistance.

    Args:
        lineage: The lineage to search (t,embeddings).
        reference_pattern: The pattern to search for (t,embeddings).
        num_candidates: The number of candidates to return.
        window_step: The step size for the window.
        max_distance: Maximum distance threshold to consider a match.
        max_skew: Maximum allowed path skewness (0-1).

    Returns:
        DataFrame with position, warping_path, distance, and skewness for the best matches
    """
    from dtaidistance.dtw_ndim import warping_path

    from utils import path_skew

    dtw_results = []
    n_windows = len(lineage) - len(reference_pattern) + 1

    if n_windows <= 0:
        return pd.DataFrame(columns=["position", "path", "distance", "skewness"])

    for i in range(0, n_windows, window_step):
        window = lineage[i : i + len(reference_pattern)]
        path, dist = warping_path(
            reference_pattern,
            window,
            include_distance=True,
        )
        if normalize:
            # Normalize by path length to match bernd_clifford method
            dist = dist / len(path)
        # Calculate skewness using the utils function
        skewness = path_skew(path, len(reference_pattern), len(window))

        if dist <= max_distance and skewness <= max_skew:
            dtw_results.append(
                {"position": i, "path": path, "distance": dist, "skewness": skewness}
            )

    # Convert to DataFrame
    results_df = pd.DataFrame(dtw_results)

    # Sort by distance first (primary) and then by skewness (secondary)
    if not results_df.empty:
        results_df = results_df.sort_values(by=["distance", "skewness"]).head(
            num_candidates
        )

    return results_df


def find_best_match_dtw_bernd_clifford(
    lineage: np.ndarray,
    reference_pattern: np.ndarray,
    num_candidates: int = 5,
    window_step: int = 5,
    normalize: bool = True,
    max_distance: float = float("inf"),
    max_skew: float = 0.8,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """
    Find the best matches in a lineage using DTW with the utils.py implementation.

    Args:
        lineage: The lineage to search (t,embeddings).
        reference_pattern: The pattern to search for (t,embeddings).
        num_candidates: The number of candidates to return.
        window_step: The step size for the window.
        max_distance: Maximum distance threshold to consider a match.
        max_skew: Maximum allowed path skewness (0-1).

    Returns:
        DataFrame with position, warping_path, distance, and skewness for the best matches
    """
    from scipy.spatial.distance import cdist

    from utils import dtw_with_matrix, path_skew

    dtw_results = []
    n_windows = len(lineage) - len(reference_pattern) + 1

    if n_windows <= 0:
        return pd.DataFrame(columns=["position", "path", "distance", "skewness"])

    for i in range(0, n_windows, window_step):
        window = lineage[i : i + len(reference_pattern)]

        # Create distance matrix
        distance_matrix = cdist(reference_pattern, window, metric=metric)

        # Apply DTW using utils.py implementation
        distance, _, path = dtw_with_matrix(distance_matrix, normalize=normalize)

        # Calculate skewness
        skewness = path_skew(path, len(reference_pattern), len(window))

        # Only add if both distance and skewness pass thresholds
        if distance <= max_distance and skewness <= max_skew:
            logger.debug(
                f"Found match at {i} with distance {distance} and skewness {skewness}"
            )
            dtw_results.append(
                {
                    "position": i,
                    "path": path,
                    "distance": distance,
                    "skewness": skewness,
                }
            )

    # Convert to DataFrame
    results_df = pd.DataFrame(dtw_results)

    # Sort by distance first (primary) and then by skewness (secondary)
    if not results_df.empty:
        results_df = results_df.sort_values(by=["distance", "skewness"]).head(
            num_candidates
        )

    return results_df


def create_consensus_embedding(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_dataset: xr.Dataset,
) -> np.ndarray:
    """
    Create a consensus embedding from multiple aligned embeddings using
    a weighted approach based on DTW distances.

    Args:
        reference_pattern: The reference pattern embeddings
        top_aligned_cells: DataFrame with alignment information
        embeddings_dataset: Dataset containing embeddings

    Returns:
        The consensus embedding
    """
    all_aligned_embeddings = []
    distances = []

    for idx, row in top_aligned_cells.iterrows():
        fov_name = row["fov_name"]
        track_ids = row["track_ids"]
        warp_path = row["warp_path"]
        start_time = int(row["start_timepoint"])
        distance = row["distance"]

        # Get lineage embeddings (similar to align_and_average_embeddings)
        lineages = []
        for track_id in track_ids:
            track_embeddings = embeddings_dataset.sel(
                sample=(fov_name, track_id)
            ).features.values
            lineages.append(track_embeddings)

        lineage_embeddings = np.concatenate(lineages, axis=0)

        # Create aligned embeddings using the warping path
        aligned_segment = np.zeros_like(reference_pattern)

        # Map each reference timepoint to the corresponding lineage timepoint
        ref_to_lineage = {}
        for ref_idx, query_idx in warp_path:
            lineage_idx = int(start_time + query_idx)
            if 0 <= lineage_idx < len(lineage_embeddings):
                ref_to_lineage[ref_idx] = lineage_idx
                aligned_segment[ref_idx] = lineage_embeddings[lineage_idx]

        # Fill in missing values
        for ref_idx in range(len(reference_pattern)):
            if ref_idx not in ref_to_lineage and ref_to_lineage:
                closest_ref_idx = min(
                    ref_to_lineage.keys(), key=lambda x: abs(x - ref_idx)
                )
                aligned_segment[ref_idx] = lineage_embeddings[
                    ref_to_lineage[closest_ref_idx]
                ]

        all_aligned_embeddings.append(aligned_segment)
        distances.append(distance)

    all_aligned_embeddings = np.array(all_aligned_embeddings)

    # Convert distances to weights (smaller distance = higher weight)
    weights = 1.0 / (
        np.array(distances) + 1e-10
    )  # Add small epsilon to avoid division by zero
    weights = weights / np.sum(weights)  # Normalize weights

    # Create weighted consensus
    consensus_embedding = np.zeros_like(reference_pattern)
    for i, aligned_embedding in enumerate(all_aligned_embeddings):
        consensus_embedding += weights[i] * aligned_embedding

    return consensus_embedding


def identify_lineages(
    tracking_df: pd.DataFrame, return_both_branches: bool = False
) -> list[tuple[str, list[int]]]:
    """
    Identifies all distinct lineages in the cell tracking data, following only
    one branch after each division event.

    Args:
        annotations_path: Path to the annotations CSV file

    Returns:
        A list of tuples, where each tuple contains (fov_id, [track_ids])
        representing a single branch lineage within a single FOV
    """
    # Read the CSV file

    # Process each FOV separately to handle repeated track_ids
    all_lineages = []

    # Group by FOV
    for fov_id, fov_df in tracking_df.groupby("fov_name"):
        # Create a dictionary to map tracks to their parents within this FOV
        child_to_parent = {}

        # Group by track_id and get the first row for each track to find its parent
        for track_id, track_group in fov_df.groupby("track_id"):
            first_row = track_group.iloc[0]
            parent_track_id = first_row["parent_track_id"]

            if parent_track_id != -1:
                child_to_parent[track_id] = parent_track_id

        # Find root tracks (those without parents or with parent_track_id = -1)
        all_tracks = set(fov_df["track_id"].unique())
        child_tracks = set(child_to_parent.keys())
        root_tracks = all_tracks - child_tracks

        # Additional validation for root tracks
        root_tracks = set()
        for track_id in all_tracks:
            track_data = fov_df[fov_df["track_id"] == track_id]
            # Check if it's truly a root track
            if (
                track_data.iloc[0]["parent_track_id"] == -1
                or track_data.iloc[0]["parent_track_id"] not in all_tracks
            ):
                root_tracks.add(track_id)

        # Build a parent-to-children mapping
        parent_to_children = {}
        for child, parent in child_to_parent.items():
            if parent not in parent_to_children:
                parent_to_children[parent] = []
            parent_to_children[parent].append(child)

        # Function to get all branches from each parent
        def get_all_branches(track_id):
            branches = []
            current_branch = [track_id]

            if track_id in parent_to_children:
                # For each child, get all their branches
                for child in parent_to_children[track_id]:
                    child_branches = get_all_branches(child)
                    # Add current track to start of each child branch
                    for branch in child_branches:
                        branches.append(current_branch + branch)
            else:
                # If no children, return just this track
                branches.append(current_branch)

            return branches

        # Build lineages starting from root tracks within this FOV
        for root_track in root_tracks:
            # Get all branches from this root
            lineage_tracks = get_all_branches(root_track)
            if return_both_branches:
                for branch in lineage_tracks:
                    all_lineages.append((fov_id, branch))
            else:
                all_lineages.append((fov_id, lineage_tracks[0]))

    return all_lineages


def plot_pc_trajectories(
    reference_lineage_fov: str,
    reference_lineage_track_id: list[int],
    reference_timepoints: list[int],
    match_positions: pd.DataFrame,
    embeddings_dataset: xr.Dataset,
    filtered_lineages: list[tuple[str, list[int]]],
    name: str,
    save_path: Path,
):
    """
    Visualize warping paths in PC space, comparing reference pattern with aligned lineages.

    Args:
        reference_lineage_fov: FOV name for the reference lineage
        reference_lineage_track_id: Track ID for the reference lineage
        reference_timepoints: Time range [start, end] to use from reference
        match_positions: DataFrame with alignment matches
        embeddings_dataset: Dataset with embeddings
        filtered_lineages: List of lineages to search in (fov_name, track_ids)
        name: Name of the embedding model
        save_path: Path to save the figure
    """
    import ast

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Get reference pattern
    ref_pattern = None
    for fov_id, track_ids in filtered_lineages:
        if fov_id == reference_lineage_fov and all(
            track_id in track_ids for track_id in reference_lineage_track_id
        ):
            ref_pattern = embeddings_dataset.sel(
                sample=(fov_id, reference_lineage_track_id)
            ).features.values
            break

    if ref_pattern is None:
        logger.info(
            f"Reference pattern not found for {name}. Skipping PC trajectory plot."
        )
        return

    ref_pattern = np.concatenate([ref_pattern])
    ref_pattern = ref_pattern[reference_timepoints[0] : reference_timepoints[1]]

    # Get top matches
    top_n_aligned_cells = match_positions.head(5)

    # Compute PCA directly with sklearn
    # Scale the input data
    scaler = StandardScaler()
    ref_pattern_scaled = scaler.fit_transform(ref_pattern)

    # Create and fit PCA model
    pca_model = PCA(n_components=2, random_state=42)
    pca_ref = pca_model.fit_transform(ref_pattern_scaled)

    # Create a figure to display the results
    plt.figure(figsize=(15, 15))

    # Plot the reference pattern PCs
    plt.subplot(len(top_n_aligned_cells) + 1, 2, 1)
    plt.plot(
        range(len(pca_ref)),
        pca_ref[:, 0],
        label="Reference PC1",
        color="black",
        linewidth=2,
    )
    plt.title(f"{name} - Reference Pattern - PC1")
    plt.xlabel("Time Index")
    plt.ylabel("PC1 Value")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(len(top_n_aligned_cells) + 1, 2, 2)
    plt.plot(
        range(len(pca_ref)),
        pca_ref[:, 1],
        label="Reference PC2",
        color="black",
        linewidth=2,
    )
    plt.title(f"{name} - Reference Pattern - PC2")
    plt.xlabel("Time Index")
    plt.ylabel("PC2 Value")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Then plot each lineage with the matched section highlighted
    for i, (_, row) in enumerate(top_n_aligned_cells.iterrows()):
        fov_name = row["fov_name"]
        track_ids = row["track_ids"]
        if isinstance(track_ids, str):
            track_ids = ast.literal_eval(track_ids)
        warp_path = row["warp_path"]
        if isinstance(warp_path, str):
            warp_path = ast.literal_eval(warp_path)
        start_time = row["start_timepoint"]
        distance = row["distance"]

        # Get the full lineage embeddings
        lineage_embeddings = []
        for track_id in track_ids:
            try:
                track_emb = embeddings_dataset.sel(
                    sample=(fov_name, track_id)
                ).features.values
                lineage_embeddings.append(track_emb)
            except KeyError:
                pass

        if not lineage_embeddings:
            continue

        lineage_embeddings = np.concatenate(lineage_embeddings, axis=0)

        # Transform lineage embeddings using the same PCA model
        # Scale first using the same scaler
        lineage_scaled = scaler.transform(lineage_embeddings)
        pca_lineage = pca_model.transform(lineage_scaled)

        # Create a subplot for PC1
        plt.subplot(len(top_n_aligned_cells) + 1, 2, 2 * i + 3)

        # Plot the full lineage PC1
        plt.plot(
            range(len(pca_lineage)),
            pca_lineage[:, 0],
            label="Full Lineage",
            color="blue",
            alpha=0.7,
        )

        # Highlight the matched section
        matched_indices = set()
        for _, query_idx in warp_path:
            lineage_idx = (
                int(start_time) + query_idx if not pd.isna(start_time) else query_idx
            )
            if 0 <= lineage_idx < len(pca_lineage):
                matched_indices.add(lineage_idx)

        matched_indices = sorted(list(matched_indices))
        if matched_indices:
            plt.plot(
                matched_indices,
                [pca_lineage[idx, 0] for idx in matched_indices],
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
                min(pca_lineage[:, 0]),
                f"Start: {min(matched_indices)}",
                color="red",
                fontsize=10,
            )
            plt.text(
                max(matched_indices),
                min(pca_lineage[:, 0]),
                f"End: {max(matched_indices)}",
                color="red",
                fontsize=10,
            )

        plt.title(f"Lineage {i} ({fov_name}) Track {track_ids[0]} - PC1")
        plt.xlabel("Lineage Time")
        plt.ylabel("PC1 Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Create a subplot for PC2
        plt.subplot(len(top_n_aligned_cells) + 1, 2, 2 * i + 4)

        # Plot the full lineage PC2
        plt.plot(
            range(len(pca_lineage)),
            pca_lineage[:, 1],
            label="Full Lineage",
            color="green",
            alpha=0.7,
        )

        # Highlight the matched section
        if matched_indices:
            plt.plot(
                matched_indices,
                [pca_lineage[idx, 1] for idx in matched_indices],
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
                min(pca_lineage[:, 1]),
                f"Start: {min(matched_indices)}",
                color="red",
                fontsize=10,
            )
            plt.text(
                max(matched_indices),
                min(pca_lineage[:, 1]),
                f"End: {max(matched_indices)}",
                color="red",
                fontsize=10,
            )

        plt.title(f"Lineage {i} ({fov_name}) - PC2")
        plt.xlabel("Lineage Time")
        plt.ylabel("PC2 Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
