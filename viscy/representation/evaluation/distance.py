from collections import defaultdict
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_similarity_cell(embedding_dataset, fov_name, track_id):
    """Extract embeddings and calculate cosine similarities for a specific cell"""
    filtered_data = embedding_dataset.where(
        (embedding_dataset["fov_name"] == fov_name)
        & (embedding_dataset["track_id"] == track_id),
        drop=True,
    )
    features = filtered_data["features"].values  # (sample, features)
    time_points = filtered_data["t"].values  # (sample,)
    first_time_point_embedding = features[0].reshape(1, -1)
    cosine_similarities = cosine_similarity(
        first_time_point_embedding, features
    ).flatten()
    return time_points, cosine_similarities.tolist()


def normalize_embeddings(
    embeddings: np.ndarray,
    strategy: Literal["per_feature", "per_embedding", "per_dataset"] = "per_feature",
) -> np.ndarray:
    """Normalize embeddings using different strategies.

    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape (n_samples, n_features) containing embeddings
    strategy : str
        Normalization strategy:
        - "per_feature": z-score each feature across all samples
        - "per_embedding": normalize each embedding vector to unit norm
        - "per_dataset": z-score entire dataset (across all features and samples)

    Returns
    -------
    np.ndarray
        Normalized embeddings with same shape as input
    """
    if strategy == "per_feature":
        # Normalize each feature independently
        return (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)
    elif strategy == "per_embedding":
        # Normalize each embedding to unit norm
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
    elif strategy == "per_dataset":
        # Normalize entire dataset
        return (embeddings - np.mean(embeddings)) / np.std(embeddings)
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")


def compute_displacement(
    embedding_dataset,
    max_tau: int = 10,
    distance_metric: Literal[
        "euclidean", "euclidean_squared", "cosine", "cosine_dissimilarity"
    ] = "euclidean_squared",
    embedding_coords: Literal["UMAP", "PHATE", None] = None,
    normalize: Literal["per_feature", "per_embedding", "per_dataset", None] = None,
) -> Dict[int, List[float]]:
    """Compute the displacement or mean square displacement (MSD) of embeddings.

    For each time difference τ, computes either:
    - |r(t + τ) - r(t)|² for squared Euclidean (MSD)
    - |r(t + τ) - r(t)| for Euclidean
    - cos_sim(r(t + τ), r(t)) for cosine
    for all particles and initial times t.

    Parameters
    ----------
    embedding_dataset : xarray.Dataset
        Dataset containing embeddings and metadata
    max_tau : int
        Maximum time difference to compute displacement for
    distance_metric : str
        The metric to use for computing distances between embeddings.
        Valid options are:
        - "euclidean": Euclidean distance (L2 norm)
        - "euclidean_squared": Squared Euclidean distance (for MSD, default)
        - "cosine": Cosine similarity
        - "cosine_dissimilarity": 1 - cosine similarity
    embedding_coords : str or None
        Which embedding coordinates to use for distance computation:
        - None: Use original features from dataset (default)
        - "UMAP": Use UMAP coordinates (UMAP1, UMAP2)
        - "PHATE": Use PHATE coordinates (PHATE1, PHATE2)
    normalize : str or None
        Normalization strategy to apply to embeddings before computing distances:
        - None: No normalization (default)
        - "per_feature": z-score each feature across all samples
        - "per_embedding": normalize each embedding vector to unit norm
        - "per_dataset": z-score entire dataset (across all features and samples)

    Returns
    -------
    Dict[int, List[float]]
        Dictionary mapping τ to list of displacements for all particles and initial times
    """
    # Get data from dataset
    fov_names = embedding_dataset["fov_name"].values
    track_ids = embedding_dataset["track_id"].values
    timepoints = embedding_dataset["t"].values

    # Get embeddings based on specified coordinates
    if embedding_coords == "UMAP":
        embeddings = np.vstack(
            (embedding_dataset["UMAP1"].values, embedding_dataset["UMAP2"].values)
        ).T
    elif embedding_coords == "PHATE":
        embeddings = np.vstack(
            (embedding_dataset["PHATE1"].values, embedding_dataset["PHATE2"].values)
        ).T
    else:
        embeddings = embedding_dataset["features"].values

    # Normalize embeddings if requested
    if normalize is not None:
        embeddings = normalize_embeddings(embeddings, strategy=normalize)

    # Initialize results dictionary
    displacement_per_tau = {tau: [] for tau in range(1, max_tau + 1)}

    # Get unique tracks using a set of tuples
    unique_tracks = set(zip(fov_names, track_ids))

    # Process each track
    for fov_name, track_id in unique_tracks:
        # Get sorted track data
        mask = (fov_names == fov_name) & (track_ids == track_id)
        times = timepoints[mask]
        track_embeddings = embeddings[mask]

        # Sort by time
        time_order = np.argsort(times)
        times = times[time_order]
        track_embeddings = track_embeddings[time_order]

        # Process each time point
        for t_idx, t in enumerate(times[:-1]):
            current_embedding = track_embeddings[t_idx]

            # Check each tau
            for tau in range(1, max_tau + 1):
                future_time = t + tau
                # Since times are sorted, we can use searchsorted
                future_idx = np.searchsorted(times, future_time)

                if future_idx < len(times) and times[future_idx] == future_time:
                    future_embedding = track_embeddings[future_idx]

                    if distance_metric in ["cosine", "cosine_dissimilarity"]:
                        dot_product = np.dot(current_embedding, future_embedding)
                        norms = np.linalg.norm(current_embedding) * np.linalg.norm(
                            future_embedding
                        )
                        similarity = dot_product / norms
                        displacement = (
                            1 - similarity
                            if distance_metric == "cosine_dissimilarity"
                            else similarity
                        )
                    else:  # Euclidean metrics
                        diff_squared = np.sum(
                            (current_embedding - future_embedding) ** 2
                        )
                        displacement = (
                            diff_squared
                            if distance_metric == "euclidean_squared"
                            else np.sqrt(diff_squared)
                        )

                    displacement_per_tau[tau].append(displacement)

    return displacement_per_tau


def compute_displacement_statistics(
    displacement_per_tau: Dict[int, List[float]]
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Compute mean and standard deviation of displacements for each tau.

    Parameters
    ----------
    displacement_per_tau : Dict[int, List[float]]
        Dictionary mapping τ to list of displacements

    Returns
    -------
    Tuple[Dict[int, float], Dict[int, float]]
        Tuple of (mean_displacements, std_displacements) where each is a
        dictionary mapping τ to the statistic
    """
    mean_displacement_per_tau = {
        tau: np.mean(displacements)
        for tau, displacements in displacement_per_tau.items()
    }
    std_displacement_per_tau = {
        tau: np.std(displacements)
        for tau, displacements in displacement_per_tau.items()
    }
    return mean_displacement_per_tau, std_displacement_per_tau


def compute_dynamic_range(mean_displacement_per_tau):
    """
    Compute the dynamic range as the difference between the maximum
    and minimum mean displacement per τ.

    Parameters:
    mean_displacement_per_tau: dict with τ as key and mean displacement as value

    Returns:
    float: dynamic range (max displacement - min displacement)
    """
    displacements = list(mean_displacement_per_tau.values())
    return max(displacements) - min(displacements)


def compute_rms_per_track(embedding_dataset):
    """
    Compute RMS of the time derivative of embeddings per track.

    Parameters:
    embedding_dataset : xarray.Dataset
        The dataset containing embeddings, timepoints, fov_name, and track_id.

    Returns:
    list: A list of RMS values, one for each track.
    """
    fov_names = embedding_dataset["fov_name"].values
    track_ids = embedding_dataset["track_id"].values
    timepoints = embedding_dataset["t"].values
    embeddings = embedding_dataset["features"].values

    cell_identifiers = np.array(
        list(zip(fov_names, track_ids)),
        dtype=[("fov_name", "O"), ("track_id", "int64")],
    )
    unique_cells = np.unique(cell_identifiers)

    rms_values = []

    for cell in unique_cells:
        fov_name = cell["fov_name"]
        track_id = cell["track_id"]
        indices = np.where((fov_names == fov_name) & (track_ids == track_id))[0]
        cell_timepoints = timepoints[indices]
        cell_embeddings = embeddings[indices]

        if len(cell_embeddings) < 2:
            continue

        sorted_indices = np.argsort(cell_timepoints)
        cell_embeddings = cell_embeddings[sorted_indices]
        differences = np.diff(cell_embeddings, axis=0)

        if differences.shape[0] == 0:
            continue

        norms = np.linalg.norm(differences, axis=1)
        rms = np.sqrt(np.mean(norms**2))
        rms_values.append(rms)

    return rms_values


def calculate_normalized_euclidean_distance_cell(embedding_dataset, fov_name, track_id):
    filtered_data = embedding_dataset.where(
        (embedding_dataset["fov_name"] == fov_name)
        & (embedding_dataset["track_id"] == track_id),
        drop=True,
    )
    features = filtered_data["features"].values  # (sample, features)
    time_points = filtered_data["t"].values  # (sample,)
    normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
    first_time_point_embedding = normalized_features[0].reshape(1, -1)
    euclidean_distances = np.linalg.norm(
        first_time_point_embedding - normalized_features, axis=1
    )
    return time_points, euclidean_distances.tolist()
