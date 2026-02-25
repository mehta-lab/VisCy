from typing import Literal

import anndata as ad
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler


def find_distribution_peak(data: np.ndarray, method: Literal["histogram", "kde_robust"] = "kde_robust") -> float:
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
        hist, bin_edges = np.histogram(data, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        peaks, properties = find_peaks(hist, height=np.max(hist) * 0.1)
        if len(peaks) == 0:
            return bin_centers[np.argmax(hist)]
        peak_heights = properties["peak_heights"]
        return bin_centers[peaks[np.argmax(peak_heights)]]

    elif method == "kde_robust":
        kde = gaussian_kde(data)
        x_range = np.linspace(np.min(data), np.max(data), 1000)
        kde_vals = kde(x_range)
        peaks, properties = find_peaks(kde_vals, height=np.max(kde_vals) * 0.1)
        if len(peaks) == 0:
            return x_range[np.argmax(kde_vals)]
        peak_heights = properties["peak_heights"]
        return x_range[peaks[np.argmax(peak_heights)]]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'histogram' or 'kde_robust'.")


def compute_embeddings_smoothness(
    features_ad: ad.AnnData,
    distance_metric: Literal["cosine", "euclidean"] = "cosine",
    time_offsets: list[int] = [1],
    verbose: bool = False,
) -> tuple[dict, dict, list[list[float]]]:
    """
    Compute the smoothness statistics of embeddings.

    Computes temporal neighbor distances per track and compares against
    random pair distances, without building the full N x N pairwise
    distance matrix.

    Parameters
    ----------
    features_ad : ad.AnnData
        AnnData object containing features with .obs having
        'fov_name', 'track_id', and 't' columns.
    distance_metric : Literal["cosine", "euclidean"], optional
        Distance metric to use, by default "cosine".
    time_offsets : list[int], optional
        Temporal offsets to compute (e.g., [1] for t->t+1).
        Distances from all offsets are aggregated together, by default [1].
    verbose : bool, optional
        Print progress messages, by default False.

    Returns
    -------
    stats : dict
        Dictionary containing smoothness metrics.
    distributions : dict
        Dictionary containing adjacent and random frame distributions.
    piecewise_distance_per_track : list[list[float]]
        Piece-wise distance per track.
    """
    features = features_ad.X
    scaled_features = StandardScaler().fit_transform(features)
    features_df = features_ad.obs.reset_index(drop=True)

    if verbose:
        print(f"Computing temporal neighbor distances (offsets: {time_offsets}) per track...")

    adjacent_distances = []
    piecewise_distance_per_track = []

    for _, subdata in features_df.groupby(["fov_name", "track_id"]):
        if len(subdata) > 1:
            indices = subdata.index.values
            track_features = scaled_features[indices]
            track_distances = []

            for offset in time_offsets:
                for i in range(len(track_features) - offset):
                    dist = cdist(
                        track_features[i : i + 1],
                        track_features[i + offset : i + offset + 1],
                        metric=distance_metric,
                    )[0, 0]
                    adjacent_distances.append(dist)
                    if offset == 1:
                        track_distances.append(dist)

            if track_distances:
                piecewise_distance_per_track.append(track_distances)

    adjacent_distances = np.array(adjacent_distances)
    n_adjacent = len(adjacent_distances)

    if verbose:
        print(f"Computed {n_adjacent:,} adjacent frame distances")

    if n_adjacent == 0:
        raise ValueError("No adjacent frame distances found. Dataset may not have tracks with multiple timepoints.")

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

        diagonal_mask = i_indices == j_indices
        while diagonal_mask.any():
            j_indices[diagonal_mask] = np.random.randint(0, n_samples, size=diagonal_mask.sum())
            diagonal_mask = i_indices == j_indices

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

    # Compute the peaks of both distributions using KDE
    adjacent_peak = find_distribution_peak(adjacent_distances, method="kde_robust")
    random_peak = find_distribution_peak(random_distances, method="kde_robust")
    smoothness_score = np.mean(adjacent_distances) / np.mean(random_distances)
    dynamic_range = random_peak - adjacent_peak

    stats = {
        "adjacent_frame_mean": float(np.mean(adjacent_distances)),
        "adjacent_frame_std": float(np.std(adjacent_distances)),
        "adjacent_frame_median": float(np.median(adjacent_distances)),
        "adjacent_frame_peak": float(adjacent_peak),
        "random_frame_mean": float(np.mean(random_distances)),
        "random_frame_std": float(np.std(random_distances)),
        "random_frame_median": float(np.median(random_distances)),
        "random_frame_peak": float(random_peak),
        "smoothness_score": float(smoothness_score),
        "dynamic_range": float(dynamic_range),
    }
    distributions = {
        "adjacent_frame_distribution": adjacent_distances,
        "random_frame_distribution": random_distances,
    }

    if verbose:
        for key, value in stats.items():
            print(f"{key}: {value}")

    return stats, distributions, piecewise_distance_per_track
