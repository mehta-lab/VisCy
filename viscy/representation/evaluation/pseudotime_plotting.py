import ast
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from viscy.data.triplet import TripletDataModule


def plot_reference_aligned_average(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_dataset: xr.Dataset,
    save_path: str | None = None,
) -> np.ndarray:
    """Plot the reference embedding, aligned embeddings, and average aligned embedding.
    
    Parameters
    ----------
    reference_pattern : np.ndarray
        The reference pattern embeddings
    top_aligned_cells : pd.DataFrame
        DataFrame with alignment information
    embeddings_dataset : xr.Dataset
        Dataset containing embeddings
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    np.ndarray
        Average aligned embeddings
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
        for track_id in track_ids:
            track_embeddings = embeddings_dataset.sel(
                sample=(fov_name, track_id)
            ).features.values
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
        color="orange",  # Changed from red for colorblind friendly
        linewidth=2,
    )

    plt.title("Dimension 0: Reference, Aligned, and Average Embeddings")
    plt.xlabel("Reference Time Index")
    plt.ylabel("Embedding Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot dimension 1
    plt.subplot(2, 1, 2)
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
        color="orange",  # Changed from red for colorblind friendly
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
) -> None:
    """Visualize where the reference pattern matches in each full lineage.
    
    Parameters
    ----------
    reference_pattern : np.ndarray
        The reference pattern embeddings
    top_aligned_cells : pd.DataFrame
        DataFrame with alignment information
    embeddings_dataset : xr.Dataset
        Dataset containing embeddings
    save_path : str, optional
        Path to save the figure
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
                "o-",
                color="orange",  # Changed from red for colorblind friendly
                label=f"Matched Section (DTW dist={distance:.2f})",
                linewidth=2,
            )

            # Add vertical lines to mark the start and end of the matched section
            plt.axvline(x=min(matched_indices), color="orange", linestyle="--", alpha=0.5)
            plt.axvline(x=max(matched_indices), color="orange", linestyle="--", alpha=0.5)

            # Add text labels
            plt.text(
                min(matched_indices),
                min(lineage_embeddings[:, 0]),
                f"Start: {min(matched_indices)}",
                color="orange",
                fontsize=10,
            )
            plt.text(
                max(matched_indices),
                min(lineage_embeddings[:, 0]),
                f"End: {max(matched_indices)}",
                color="orange",
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
            color="blue",  # Changed from green for consistency
            alpha=0.7,
        )

        # Highlight the matched section
        if matched_indices:
            plt.plot(
                matched_indices,
                [lineage_embeddings[idx, 1] for idx in matched_indices],
                "o-",
                color="orange",  # Changed from red for colorblind friendly
                label=f"Matched Section (DTW dist={distance:.2f})",
                linewidth=2,
            )

            # Add vertical lines to mark the start and end of the matched section
            plt.axvline(x=min(matched_indices), color="orange", linestyle="--", alpha=0.5)
            plt.axvline(x=max(matched_indices), color="orange", linestyle="--", alpha=0.5)

            # Add text labels
            plt.text(
                min(matched_indices),
                min(lineage_embeddings[:, 1]),
                f"Start: {min(matched_indices)}",
                color="orange",
                fontsize=10,
            )
            plt.text(
                max(matched_indices),
                min(lineage_embeddings[:, 1]),
                f"End: {max(matched_indices)}",
                color="orange",
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
    """Visualize warping paths in PC space, comparing reference pattern with aligned lineages.
    
    Parameters
    ----------
    reference_lineage_fov : str
        FOV name for the reference lineage
    reference_lineage_track_id : list[int]
        Track ID for the reference lineage
    reference_timepoints : list[int]
        Time range [start, end] to use from reference
    match_positions : pd.DataFrame
        DataFrame with alignment matches
    embeddings_dataset : xr.Dataset
        Dataset with embeddings
    filtered_lineages : list[tuple[str, list[int]]]
        List of lineages to search in (fov_name, track_ids)
    name : str
        Name of the embedding model
    save_path : Path
        Path to save the figure
    """
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
        print(f"Reference pattern not found for {name}. Skipping PC trajectory plot.")
        return

    ref_pattern = np.concatenate([ref_pattern])
    ref_pattern = ref_pattern[reference_timepoints[0] : reference_timepoints[1]]

    # Get top matches
    top_n_aligned_cells = match_positions.head(5)

    # Compute PCA directly with sklearn
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
                "o-",
                color="orange",  # Changed from red for colorblind friendly
                label=f"Matched Section (DTW dist={distance:.2f})",
                linewidth=2,
            )

            # Add vertical lines to mark the start and end of the matched section
            plt.axvline(x=min(matched_indices), color="orange", linestyle="--", alpha=0.5)
            plt.axvline(x=max(matched_indices), color="orange", linestyle="--", alpha=0.5)

            # Add text labels
            plt.text(
                min(matched_indices),
                min(pca_lineage[:, 0]),
                f"Start: {min(matched_indices)}",
                color="orange",
                fontsize=10,
            )
            plt.text(
                max(matched_indices),
                min(pca_lineage[:, 0]),
                f"End: {max(matched_indices)}",
                color="orange",
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
            color="blue",  # Changed from green for consistency
            alpha=0.7,
        )

        # Highlight the matched section
        if matched_indices:
            plt.plot(
                matched_indices,
                [pca_lineage[idx, 1] for idx in matched_indices],
                "o-",
                color="orange",  # Changed from red for colorblind friendly
                label=f"Matched Section (DTW dist={distance:.2f})",
                linewidth=2,
            )

            # Add vertical lines to mark the start and end of the matched section
            plt.axvline(x=min(matched_indices), color="orange", linestyle="--", alpha=0.5)
            plt.axvline(x=max(matched_indices), color="orange", linestyle="--", alpha=0.5)

            # Add text labels
            plt.text(
                min(matched_indices),
                min(pca_lineage[:, 1]),
                f"Start: {min(matched_indices)}",
                color="orange",
                fontsize=10,
            )
            plt.text(
                max(matched_indices),
                min(pca_lineage[:, 1]),
                f"End: {max(matched_indices)}",
                color="orange",
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


def align_and_average_embeddings(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_dataset: xr.Dataset,
    use_median: bool = False,
) -> np.ndarray:
    """Align embeddings from multiple lineages to a reference pattern and compute their average.
    
    Parameters
    ----------
    reference_pattern : np.ndarray
        The reference pattern embeddings
    top_aligned_cells : pd.DataFrame
        DataFrame with alignment information
    embeddings_dataset : xr.Dataset
        Dataset containing embeddings
    use_median : bool
        If True, use median instead of mean for averaging
        
    Returns
    -------
    np.ndarray
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
    """Align image stacks from multiple lineages to a reference pattern.
    
    Parameters
    ----------
    reference_pattern : np.ndarray
        The reference pattern embeddings
    top_aligned_cells : pd.DataFrame
        DataFrame with alignment information
    input_data_path : Path
        Path to the input data
    tracks_path : Path
        Path to the tracks data
    source_channels : list[str]
        List of channels to include
    yx_patch_size : tuple[int, int]
        Patch size for images
    z_range : tuple[int, int]
        Z-range to include
    view_ref_sector_only : bool
        If True, only show the section that matches the reference pattern
    napari_viewer : optional
        Optional napari viewer for visualization
        
    Returns
    -------
    tuple[list, list]
        Tuple of (all_lineage_images, all_aligned_stacks)
    """
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


def create_consensus_embedding(
    reference_pattern: np.ndarray,
    top_aligned_cells: pd.DataFrame,
    embeddings_dataset: xr.Dataset,
) -> np.ndarray:
    """Create a consensus embedding from multiple aligned embeddings using weighted approach.
    
    Parameters
    ----------
    reference_pattern : np.ndarray
        The reference pattern embeddings
    top_aligned_cells : pd.DataFrame
        DataFrame with alignment information
    embeddings_dataset : xr.Dataset
        Dataset containing embeddings
        
    Returns
    -------
    np.ndarray
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

        # Get lineage embeddings
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