# Plotting utils
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from viscy.data.triplet import TripletDataModule


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
        track_offsets = (
            {}
        )  # To keep track of where each track starts in the concatenated array
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
        track_offsets = (
            {}
        )  # To keep track of where each track starts in the concatenated array
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
        start_time = int(row["start_timepoint"])  # Ensure start_time is an integer
        end_time = int(row["end_timepoint"])  # Ensure end_time is an integer

        # Initialize the data module
        data_module = TripletDataModule(
            data_path=input_data_path,
            tracks_path=tracks_path,
            source_channel=["Phase3D", "raw GFP EX488 EM525-45"],
            z_range=[10, 30],
            initial_yx_patch_size=(192, 192),
            final_yx_patch_size=(192, 192),
            batch_size=1,
            num_workers=12,
            predict_cells=True,
            include_fov_names=[fov_name] * len(track_ids),
            include_track_ids=track_ids,
        )
        data_module.setup("predict")

        # Get the images and timepoints for the lineage
        lineage_images = []
        for batch in data_module.predict_dataloader():
            image = batch["anchor"].numpy()[0]
            lineage_images.append(image)

        lineage_images = np.array(lineage_images)
        all_lineage_images.append(lineage_images)

        # Create an aligned stack based on the warping path
        if view_ref_sector_only:
            aligned_stack = np.zeros(
                (len(reference_pattern),) + lineage_images.shape[-4:],
                dtype=lineage_images.dtype,
            )

            # Create a mapping from reference indices to lineage indices
            ref_to_lineage = {}
            for ref_idx, query_idx in warp_path:
                lineage_idx = int(start_time + query_idx)
                if 0 <= lineage_idx < len(lineage_images):
                    ref_to_lineage[ref_idx] = lineage_idx

            # Fill the aligned stack, ensuring every reference index is covered
            for ref_idx in range(len(reference_pattern)):
                if ref_idx in ref_to_lineage:
                    aligned_stack[ref_idx] = lineage_images[ref_to_lineage[ref_idx]]
                else:
                    # Handle missing indices - find the closest available reference index
                    if ref_to_lineage:
                        closest_ref_idx = min(
                            ref_to_lineage.keys(), key=lambda x: abs(x - ref_idx)
                        )
                        aligned_stack[ref_idx] = lineage_images[
                            ref_to_lineage[closest_ref_idx]
                        ]

            all_aligned_stacks.append(aligned_stack)
            if napari_viewer:
                napari_viewer.add_image(
                    aligned_stack,
                    name=f"Aligned_{fov_name}_track_{track_ids[0]}",
                    channel_axis=1,
                )
        else:
            # View the whole lineage shifted by the start time
            # Ensure start_time is an integer for slicing
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
    save_path: str | None = None,
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
        save_path: Optional path to save the results CSV

    Returns:
        DataFrame with match positions and distances
    """
    from dtaidistance.dtw_ndim import warping_path
    from tqdm import tqdm

    # Calculate window step based on reference pattern length
    window_step = max(1, int(len(reference_pattern) * window_step_fraction))

    all_match_positions = {
        "fov_name": [],
        "track_ids": [],
        "distance": [],
        "warp_path": [],
        "start_timepoint": [],
        "end_timepoint": [],
    }

    for i, (fov_name, track_ids) in tqdm(
        enumerate(filtered_lineages),
        total=len(filtered_lineages),
        desc="Finding pattern matches",
    ):
        # Reconstruct the concatenated lineage
        lineages = []
        track_offsets = (
            {}
        )  # To keep track of where each track starts in the concatenated array
        current_offset = 0

        for track_id in track_ids:
            track_embeddings = embeddings_dataset.sel(
                sample=(fov_name, track_id)
            ).features.values
            track_offsets[track_id] = current_offset
            current_offset += len(track_embeddings)
            lineages.append(track_embeddings)

        lineage_embeddings = np.concatenate(lineages, axis=0)

        # Find best matches using DTW
        best_matches = find_best_match_dtw(
            lineage_embeddings,
            reference_pattern=reference_pattern,
            num_candidates=num_candidates,
            window_step=window_step,
            max_distance=max_distance,
        )

        if len(best_matches) > 0:
            best_pos, best_path, best_dist = best_matches[0]
            all_match_positions["fov_name"].append(fov_name)
            all_match_positions["track_ids"].append(track_ids)
            all_match_positions["distance"].append(best_dist)
            all_match_positions["warp_path"].append(best_path)
            all_match_positions["start_timepoint"].append(best_pos)
            all_match_positions["end_timepoint"].append(
                best_pos + len(reference_pattern)
            )
        else:
            all_match_positions["fov_name"].append(fov_name)
            all_match_positions["track_ids"].append(track_ids)
            all_match_positions["distance"].append(None)
            all_match_positions["warp_path"].append(None)
            all_match_positions["start_timepoint"].append(None)
            all_match_positions["end_timepoint"].append(None)

    # Convert to DataFrame and drop rows with no matches
    all_match_positions = pd.DataFrame(all_match_positions)
    all_match_positions = all_match_positions.dropna()

    # Sort by distance (best matches first)
    all_match_positions = all_match_positions.sort_values(by="distance", ascending=True)

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
) -> list[tuple[int, list, float]]:
    """
    Find the best matches in a lineage using DTW.

    Args:
        lineage: The lineage to search (t,embeddings).
        reference_pattern: The pattern to search for (t,embeddings).
        num_candidates: The number of candidates to return.
        window_step: The step size for the window.
        max_distance: Maximum distance threshold to consider a match.

    Returns:
        List of tuples (position, warping_path, distance) for the best matches
    """
    from dtaidistance.dtw_ndim import warping_path

    dtw_results = []
    n_windows = len(lineage) - len(reference_pattern) + 1

    if n_windows <= 0:
        return []

    for i in range(0, n_windows, window_step):
        window = lineage[i : i + len(reference_pattern)]
        path, dist = warping_path(
            window,
            reference_pattern,
            include_distance=True,
        )
        if dist <= max_distance:
            dtw_results.append((i, path, dist))

    sorted_dtw_results = sorted(dtw_results, key=lambda x: x[2])[:num_candidates]

    return sorted_dtw_results


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
