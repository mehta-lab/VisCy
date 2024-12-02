from collections import defaultdict

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


def compute_displacement(
    embedding_dataset,
    max_tau=10,
    use_cosine=False,
    use_dissimilarity=False,
    use_umap=False,
    return_mean_std=False,
):
    """Compute the norm of differences between embeddings at t and t + tau"""
    fov_names = embedding_dataset["fov_name"].values
    track_ids = embedding_dataset["track_id"].values
    timepoints = embedding_dataset["t"].values

    if use_umap:
        embeddings = np.vstack(
            (embedding_dataset["UMAP1"].values, embedding_dataset["UMAP2"].values)
        ).T
    else:
        embeddings = embedding_dataset["features"].values

    displacement_per_tau = defaultdict(list)

    for i in range(len(fov_names)):
        fov_name = fov_names[i]
        track_id = track_ids[i]
        current_time = timepoints[i]
        current_embedding = embeddings[i].reshape(1, -1)

        for tau in range(1, max_tau + 1):
            future_time = current_time + tau
            matching_indices = np.where(
                (fov_names == fov_name)
                & (track_ids == track_id)
                & (timepoints == future_time)
            )[0]

            if len(matching_indices) == 1:
                future_embedding = embeddings[matching_indices[0]].reshape(1, -1)

                if use_cosine:
                    similarity = cosine_similarity(current_embedding, future_embedding)[
                        0
                    ][0]
                    displacement = 1 - similarity if use_dissimilarity else similarity
                else:
                    displacement = np.sum((current_embedding - future_embedding) ** 2)

                displacement_per_tau[tau].append(displacement)

    if return_mean_std:
        mean_displacement_per_tau = {
            tau: np.mean(displacements)
            for tau, displacements in displacement_per_tau.items()
        }
        std_displacement_per_tau = {
            tau: np.std(displacements)
            for tau, displacements in displacement_per_tau.items()
        }
        return mean_displacement_per_tau, std_displacement_per_tau

    return displacement_per_tau


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
