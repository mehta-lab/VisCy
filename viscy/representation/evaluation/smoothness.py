from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

from viscy.representation.evaluation.clustering import (
    compare_time_offset,
    pairwise_distance_matrix,
    rank_nearest_neighbors,
    select_block,
)


def compute_piece_wise_distance(
    features_df: pd.DataFrame,
    cross_dist: NDArray,
    rank_fractions: NDArray,
    groupby: list[str] = ["fov_name", "track_id"],
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Computing the piece-wise distance and rank difference
    - Get the off diagonal per block and compute the mode
    - The blocks are not square, so we need to get the off diagonal elements
    - Get the 1 and 99 percentile of the off diagonal per block

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing the features
    cross_dist : NDArray
        Cross-distance matrix
    rank_fractions : NDArray
        Rank fractions
    groupby : list[str], optional
        Columns to group by, by default ["fov_name", "track_id"]

    Returns
    -------
    piece_wise_dissimilarity_per_track : list
        Piece-wise dissimilarity per track
    piece_wise_rank_difference_per_track : list
        Piece-wise rank difference per track
    """
    piece_wise_dissimilarity_per_track = []
    piece_wise_rank_difference_per_track = []
    for _, subdata in features_df.groupby(groupby):
        if len(subdata) > 1:
            indices = subdata.index.values
            single_track_dissimilarity = select_block(cross_dist, indices)
            single_track_rank_fraction = select_block(rank_fractions, indices)
            piece_wise_dissimilarity = compare_time_offset(
                single_track_dissimilarity, time_offset=1
            )
            piece_wise_rank_difference = compare_time_offset(
                single_track_rank_fraction, time_offset=1
            )
            piece_wise_dissimilarity_per_track.append(piece_wise_dissimilarity)
            piece_wise_rank_difference_per_track.append(piece_wise_rank_difference)
    return piece_wise_dissimilarity_per_track, piece_wise_rank_difference_per_track


def find_distribution_peak(
    data: np.ndarray, method: Literal["histogram", "kde_robust"] = "kde_robust"
) -> float:
    """Find the peak of a distribution

    Parameters
    ----------
    data: np.ndarray
        The data to find the peak of
    method: Literal["histogram", "kde_robust"], optional
        The method to use to find the peak, by default "kde_robust"

    Returns
    -------
    float: The peak of the distribution (highest peak if multiple)
    """
    if method == "histogram":
        # Simple histogram-based peak finding
        hist, bin_edges = np.histogram(data, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        peaks, properties = find_peaks(
            hist, height=np.max(hist) * 0.1
        )  # 10% of max height
        if len(peaks) == 0:
            return bin_centers[np.argmax(hist)]  # Fallback to global max
        # Return peak with highest density
        peak_heights = properties["peak_heights"]
        return bin_centers[peaks[np.argmax(peak_heights)]]

    elif method == "kde_robust":
        # More robust KDE approach
        kde = gaussian_kde(data)
        x_range = np.linspace(np.min(data), np.max(data), 1000)
        kde_vals = kde(x_range)
        peaks, properties = find_peaks(kde_vals, height=np.max(kde_vals) * 0.1)
        if len(peaks) == 0:
            return x_range[np.argmax(kde_vals)]  # Fallback to global max
        # Return peak with highest KDE value
        peak_heights = properties["peak_heights"]
        return x_range[peaks[np.argmax(peak_heights)]]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'histogram' or 'kde_robust'.")


def compute_embeddings_smoothness(
    features_ad: ad.AnnData,
    distance_metric: Literal["cosine", "euclidean"] = "cosine",
    time_offsets: list[int] = [1],
    use_optimized: bool = True,
    verbose: bool = False,
) -> tuple[dict, dict, list[list[float]]]:
    """
    Compute the smoothness statistics of embeddings

    Parameters
    --------
    features_ad : ad.AnnData
        AnnData object containing features with .obs having 'fov_name', 'track_id', and 't' columns
    distance_metric : Literal["cosine", "euclidean"], optional
        Distance metric to use, by default "cosine"
    time_offsets : list[int], optional
        Temporal offsets to compute (e.g., [1] for t→t+1, [1,2,3] for t→t+1,t+2,t+3)
        Distances from all offsets are aggregated together, by default [1]
    use_optimized : bool, optional
        Use memory-optimized computation that avoids full pairwise distance matrix.
        Recommended for large datasets (>50K samples). By default True
    verbose : bool, optional
        Print progress messages, by default False

    Returns
    -------
    stats : dict
        Dictionary containing metrics including:
        - adjacent_frame_mean: Mean of adjacent frame dissimilarity
        - adjacent_frame_std: Standard deviation of adjacent frame dissimilarity
        - adjacent_frame_median: Median of adjacent frame dissimilarity
        - adjacent_frame_peak: Peak of adjacent frame distribution
        - random_frame_mean: Mean of random sampling dissimilarity
        - random_frame_std: Standard deviation of random sampling dissimilarity
        - random_frame_median: Median of random sampling dissimilarity
        - random_frame_peak: Peak of random sampling distribution
        - smoothness_score: Score of smoothness (lower is better)
        - dynamic_range: Difference between random and adjacent peaks (higher is better)
    distributions : dict
        Dictionary containing distributions including:
        - adjacent_frame_distribution: Full distribution of adjacent frame dissimilarities
        - random_frame_distribution: Full distribution of random sampling dissimilarities
    piecewise_distance_per_track : list[list[float]]
        Piece-wise distance per track

    Notes
    -----
    Memory optimization: When use_optimized=True, this function avoids creating the full
    pairwise distance matrix (N×N), which can require 100+ GB for large datasets (>100K samples).
    Instead, it computes only temporal neighbor distances within tracks and samples random pairs
    on-demand, reducing memory usage to ~1GB. For datasets with <50K samples, both approaches
    work, but optimized is still recommended for consistency.
    """
    features = features_ad.X
    scaled_features = StandardScaler().fit_transform(features)
    features_df = features_ad.obs.reset_index(drop=True)

    if use_optimized:
        # Memory-optimized computation: avoid full pairwise distance matrix
        if verbose:
            print(
                f"Computing temporal neighbor distances (offsets: {time_offsets}) per track..."
            )

        # Compute temporal neighbor distances per track (memory efficient)
        adjacent_distances = []
        piecewise_distance_per_track = []

        for _, subdata in features_df.groupby(["fov_name", "track_id"]):
            if len(subdata) > 1:
                indices = subdata.index.values
                track_features = scaled_features[indices]
                track_distances = []

                # Compute distances for each time offset
                for offset in time_offsets:
                    for i in range(len(track_features) - offset):
                        dist = cdist(
                            track_features[i : i + 1],
                            track_features[i + offset : i + offset + 1],
                            metric=distance_metric,
                        )[0, 0]
                        adjacent_distances.append(dist)
                        if offset == 1:  # Only collect per-track for offset=1
                            track_distances.append(dist)

                if track_distances:
                    piecewise_distance_per_track.append(track_distances)

        adjacent_distances = np.array(adjacent_distances)
        n_adjacent = len(adjacent_distances)

        if verbose:
            print(f"Computed {n_adjacent:,} adjacent frame distances")

        if n_adjacent == 0:
            raise ValueError(
                "No adjacent frame distances found. Dataset may not have tracks with multiple timepoints."
            )

        # Sample random pairs (same number as adjacent distances) in batches
        if verbose:
            print("Sampling random pairs for baseline...")

        n_samples = len(scaled_features)
        n_random_samples = n_adjacent
        batch_size = 10000
        random_distances = []

        np.random.seed(42)
        for batch_start in range(0, n_random_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_random_samples)
            batch_n = batch_end - batch_start

            i_indices = np.random.randint(0, n_samples, size=batch_n)
            j_indices = np.random.randint(0, n_samples, size=batch_n)

            # Avoid diagonal
            diagonal_mask = i_indices == j_indices
            while diagonal_mask.any():
                j_indices[diagonal_mask] = np.random.randint(
                    0, n_samples, size=diagonal_mask.sum()
                )
                diagonal_mask = i_indices == j_indices

            # Compute distances for this batch
            for i, j in zip(i_indices, j_indices):
                dist = cdist(
                    scaled_features[i : i + 1],
                    scaled_features[j : j + 1],
                    metric=distance_metric,
                )[0, 0]
                random_distances.append(dist)

        random_distances = np.array(random_distances)

        if verbose:
            print(f"Computed {len(random_distances):,} random pair distances")

        all_piecewise_distances = adjacent_distances
        sampled_values = random_distances

    else:
        # Original computation using full pairwise distance matrix
        if verbose:
            print("Computing full pairwise distance matrix...")

        cross_dist = pairwise_distance_matrix(scaled_features, metric=distance_metric)
        rank_fractions = rank_nearest_neighbors(cross_dist, normalize=True)

        # Compute piece-wise distance and rank difference
        piecewise_distance_per_track, _ = compute_piece_wise_distance(
            features_df, cross_dist, rank_fractions
        )

        all_piecewise_distances = np.concatenate(piecewise_distance_per_track)

        # Random sampling values in the distance matrix with same size as adjacent frame measurements
        n_samples = len(all_piecewise_distances)
        np.random.seed(42)
        i_indices = np.random.randint(0, len(cross_dist), size=n_samples)
        j_indices = np.random.randint(0, len(cross_dist), size=n_samples)

        diagonal_mask = i_indices == j_indices
        while diagonal_mask.any():
            j_indices[diagonal_mask] = np.random.randint(
                0, len(cross_dist), size=diagonal_mask.sum()
            )
            diagonal_mask = i_indices == j_indices
        sampled_values = cross_dist[i_indices, j_indices]

    # Compute the peaks of both distributions using KDE
    adjacent_peak = find_distribution_peak(all_piecewise_distances, method="kde_robust")
    random_peak = find_distribution_peak(sampled_values, method="kde_robust")
    smoothness_score = np.mean(all_piecewise_distances) / np.mean(sampled_values)
    dynamic_range = random_peak - adjacent_peak

    stats = {
        "adjacent_frame_mean": float(np.mean(all_piecewise_distances)),
        "adjacent_frame_std": float(np.std(all_piecewise_distances)),
        "adjacent_frame_median": float(np.median(all_piecewise_distances)),
        "adjacent_frame_peak": float(adjacent_peak),
        "random_frame_mean": float(np.mean(sampled_values)),
        "random_frame_std": float(np.std(sampled_values)),
        "random_frame_median": float(np.median(sampled_values)),
        "random_frame_peak": float(random_peak),
        "smoothness_score": float(smoothness_score),
        "dynamic_range": float(dynamic_range),
    }
    distributions = {
        "adjacent_frame_distribution": all_piecewise_distances,
        "random_frame_distribution": sampled_values,
    }

    if verbose:
        for key, value in stats.items():
            print(f"{key}: {value}")

    return stats, distributions, piecewise_distance_per_track
