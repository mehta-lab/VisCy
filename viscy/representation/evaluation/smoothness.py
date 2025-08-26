from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.clustering import (
    compare_time_offset,
    pairwise_distance_matrix,
    rank_nearest_neighbors,
    select_block,
)


def compute_piece_wise_distance(
    features_df: pd.DataFrame, cross_dist: NDArray, rank_fractions: NDArray,groupby:list[str] = ["fov_name", "track_id"]
)->tuple[list[list[float]], list[list[float]]]:
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


def find_distribution_peak(data: np.ndarray, method: Literal["histogram", "kde_robust"] = "kde_robust") -> float:
    """ Find the peak of a distribution
    
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
    if method == 'histogram':
        # Simple histogram-based peak finding
        hist, bin_edges = np.histogram(data, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        peaks, properties = find_peaks(hist, height=np.max(hist) * 0.1)  # 10% of max height
        if len(peaks) == 0:
            return bin_centers[np.argmax(hist)]  # Fallback to global max
        # Return peak with highest density
        peak_heights = properties['peak_heights']
        return bin_centers[peaks[np.argmax(peak_heights)]]

    elif method == 'kde_robust':
        # More robust KDE approach
        kde = gaussian_kde(data)
        x_range = np.linspace(np.min(data), np.max(data), 1000)
        kde_vals = kde(x_range)
        peaks, properties = find_peaks(kde_vals, height=np.max(kde_vals) * 0.1)
        if len(peaks) == 0:
            return x_range[np.argmax(kde_vals)]  # Fallback to global max
        # Return peak with highest KDE value
        peak_heights = properties['peak_heights']
        return x_range[peaks[np.argmax(peak_heights)]]
     


def compute_embeddings_smoothness(
    prediction_path: Path,
    distance_metric: Literal["cosine", "euclidean"] = "cosine",
    verbose: bool = False,
) -> tuple[dict, dict, list[list[float]]]:
    """
    Compute the smoothness statistics of embeddings

    Parameters
    ----------
    prediction_path: Path to the embedding dataset
    distance_metric: Distance metric to use, by default "cosine"

    Returns:
    -------
    stats: dict: Dictionary containing metrics including:
        - adj_frame_mean: Mean of adjacent frame dissimilarity
        - adj_frame_std: Standard deviation of adjacent frame dissimilarity
        - adj_frame_median: Median of adjacent frame dissimilarity
        - adj_frame_peak: Peak of adjacent frame distribution
        - adj_frame_p99: 99th percentile of adjacent frame dissimilarity
        - adj_frame_p1: 1st percentile of adjacent frame dissimilarity
        - adj_frame_distribution: Full distribution of adjacent frame dissimilarities
        - random_frame_mean: Mean of random sampling dissimilarity
        - random_frame_std: Standard deviation of random sampling dissimilarity
        - random_frame_median: Median of random sampling dissimilarity
        - random_frame_peak: Peak of random sampling distribution
        - random_frame_distribution: Full distribution of random sampling dissimilarities
        - dynamic_range: Difference between random and adjacent peaks
    distributions: dict: Dictionary containing distributions including:
        - adjacent_frame_distribution: Full distribution of adjacent frame dissimilarities
        - random_frame_distribution: Full distribution of random sampling dissimilarities
    piecewise_distance_per_track: list[list[float]]
        Piece-wise distance per track
    """

    # Read the dataset
    embeddings = read_embedding_dataset(prediction_path)
    features = embeddings["features"]
    scaled_features = StandardScaler().fit_transform(features.values)

    # Compute the distance matrix
    cross_dist = pairwise_distance_matrix(scaled_features, metric=distance_metric)
    rank_fractions = rank_nearest_neighbors(cross_dist, normalize=True)

    # Compute piece-wise distance and rank difference
    features_df = features["sample"].to_dataframe().reset_index(drop=True)
    piecewise_distance_per_track, _ = (
        compute_piece_wise_distance(features_df, cross_dist, rank_fractions)
    )

    all_piecewise_distances = np.concatenate(piecewise_distance_per_track)

    # p99_piece_wise_distance = np.array(
    #     [np.percentile(track, 99) for track in piecewise_distance_per_track]
    # )
    # p1_percentile_piece_wise_distance = np.array(
    #     [np.percentile(track, 1) for track in piecewise_distance_per_track]
    # )

    # Random sampling values in the distance matrix with same size as adjacent frame measurements
    n_samples = len(all_piecewise_distances)
    # Avoid sampling the diagonal elements
    np.random.seed(42)
    i_indices = np.random.randint(0, len(cross_dist), size=n_samples)
    j_indices = np.random.randint(0, len(cross_dist), size=n_samples)
    
    diagonal_mask = i_indices == j_indices
    while diagonal_mask.any():
        j_indices[diagonal_mask] = np.random.randint(0, len(cross_dist),
    size=diagonal_mask.sum())
        diagonal_mask = i_indices == j_indices
    sampled_values = cross_dist[i_indices, j_indices]

    # Compute the peaks of both distributions using KDE
    adjacent_peak = find_distribution_peak(all_piecewise_distances, method="kde_robust")
    random_peak = find_distribution_peak(sampled_values, method="kde_robust")
    dynamic_range = random_peak - adjacent_peak

    stats = {
        "adjacent_frame_mean": float(np.mean(all_piecewise_distances)),
        "adjacent_frame_std": float(np.std(all_piecewise_distances)),
        "adjacent_frame_median": float(np.median(all_piecewise_distances)),
        "adjacent_frame_peak": float(adjacent_peak),
        # "adjacent_frame_p99": p99_piece_wise_distance,
        # "adjacent_frame_p1": p1_percentile_piece_wise_distance,
        # "adjacent_frame_distribution": all_piecewise_distances,
        "random_frame_mean": float(np.mean(sampled_values)),
        "random_frame_std": float(np.std(sampled_values)),
        "random_frame_median": float(np.median(sampled_values)),
        "random_frame_peak": float(random_peak),
        # "random_frame_distribution": sampled_values,
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

