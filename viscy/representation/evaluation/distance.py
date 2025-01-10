from collections import defaultdict
from typing import Dict, List, Literal, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


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


def calculate_euclidian_distance_cell(embedding_dataset, fov_name, track_id):
    """Extract embeddings and calculate euclidean distances for a specific cell"""
    filtered_data = embedding_dataset.where(
        (embedding_dataset["fov_name"] == fov_name)
        & (embedding_dataset["track_id"] == track_id),
        drop=True,
    )
    features = filtered_data["features"].values  # (sample, features)
    time_points = filtered_data["t"].values  # (sample,)
    first_time_point_embedding = features[0].reshape(1, -1)
    euclidean_distances = np.linalg.norm(first_time_point_embedding - features, axis=1)
    return time_points, euclidean_distances.tolist()


def compute_displacement(
    embedding_dataset,
    distance_metric: Literal["euclidean_squared", "cosine"] = "euclidean_squared",
    max_delta_t: int = None,
) -> Dict[int, List[float]]:
    """Compute displacements between embeddings at different time differences.

    For each time difference τ, computes distances between embeddings of the same cell
    separated by τ timepoints. Supports multiple distance metrics.

    Parameters
    ----------
    embedding_dataset : xarray.Dataset
        Dataset containing embeddings and metadata with the following variables:
        - features: (N, D) array of embeddings
        - fov_name: (N,) array of field of view names
        - track_id: (N,) array of cell track IDs
        - t: (N,) array of timepoints
    distance_metric : str, optional
        The metric to use for computing distances between embeddings.
        Valid options are:
        - "euclidean_squared": Squared Euclidean distance (default)
        - "cosine": Cosine similarity
    max_delta_t : int, optional
        Maximum time difference τ to compute displacements for.
        If None, uses the maximum possible time difference in the dataset.

    Returns
    -------
    Dict[int, List[float]]
        Dictionary mapping time difference τ to list of displacements.
        Each displacement value represents the distance between a pair of
        embeddings from the same cell separated by τ timepoints.
    """

    # Get data from dataset
    fov_names = embedding_dataset["fov_name"].values
    track_ids = embedding_dataset["track_id"].values
    timepoints = embedding_dataset["t"].values
    embeddings = embedding_dataset["features"].values

    # Check if max_delta_t is provided, otherwise use the maximum timepoint
    if max_delta_t is None:
        max_delta_t = timepoints.max()

    displacement_per_delta_t = defaultdict(list)
    # Process each sample
    for i in tqdm(range(len(fov_names)), desc="Processing FOVs"):
        fov_name = fov_names[i]
        track_id = track_ids[i]
        current_time = timepoints[i]
        current_embedding = embeddings[i].reshape(1, -1)

        # Compute displacements for each delta t
        for delta_t in range(1, max_delta_t + 1):
            future_time = current_time + delta_t
            matching_indices = np.where(
                (fov_names == fov_name)
                & (track_ids == track_id)
                & (timepoints == future_time)
            )[0]

            if len(matching_indices) == 1:
                if distance_metric == "euclidean_squared":
                    future_embedding = embeddings[matching_indices[0]].reshape(1, -1)
                    displacement = np.sum((current_embedding - future_embedding) ** 2)
                elif distance_metric == "cosine":
                    future_embedding = embeddings[matching_indices[0]].reshape(1, -1)
                    displacement = cosine_similarity(
                        current_embedding, future_embedding
                    )
                displacement_per_delta_t[delta_t].append(displacement)
    return dict(displacement_per_delta_t)


def compute_displacement_statistics(
    displacement_per_delta_t: Dict[int, List[float]]
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Compute mean and standard deviation of displacements for each delta_t.

    Parameters
    ----------
    displacement_per_delta_t : Dict[int, List[float]]
        Dictionary mapping τ to list of displacements

    Returns
    -------
    Tuple[Dict[int, float], Dict[int, float]]
        Tuple of (mean_displacements, std_displacements) where each is a
        dictionary mapping τ to the statistic
    """
    mean_displacement_per_delta_t = {
        delta_t: np.mean(displacements)
        for delta_t, displacements in displacement_per_delta_t.items()
    }
    std_displacement_per_delta_t = {
        delta_t: np.std(displacements)
        for delta_t, displacements in displacement_per_delta_t.items()
    }
    return mean_displacement_per_delta_t, std_displacement_per_delta_t


def compute_dynamic_range(mean_displacement_per_delta_t):
    """
    Compute the dynamic range as the difference between the maximum
    and minimum mean displacement per τ.

    Parameters:
    mean_displacement_per_delta_t: dict with τ as key and mean displacement as value

    Returns:
    float: dynamic range (max displacement - min displacement)
    """
    displacements = list(mean_displacement_per_delta_t.values())
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
