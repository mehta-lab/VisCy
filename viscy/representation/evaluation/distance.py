from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_similarity_cell(embedding_dataset, fov_name, track_id):
    """
    Calculate cosine similarities for a specific cell over time.

    Parameters:
    embedding_dataset : xarray.Dataset
        The dataset containing embeddings, timepoints, fov_name, and track_id.
    fov_name : str
        Field of view name to identify the specific cell.
    track_id : int
        Track ID to identify the specific cell.

    Returns:
    tuple
        - time_points (array): Array of time points for the cell.
        - cosine_similarities (list): Cosine similarities between the embedding
          at the first time point and each subsequent time point.
    """
    # Filter the dataset for the specific infected cell
    filtered_data = embedding_dataset.where(
        (embedding_dataset["fov_name"] == fov_name)
        & (embedding_dataset["track_id"] == track_id),
        drop=True,
    )

    # Extract the feature embeddings and time points
    features = filtered_data["features"].values  # (sample, features)
    time_points = filtered_data["t"].values  # (sample,)

    # Get the first time point's embedding
    first_time_point_embedding = features[0].reshape(1, -1)

    # Calculate cosine similarity between each time point and the first time point
    cosine_similarities = []
    for i in range(len(time_points)):
        similarity = cosine_similarity(
            first_time_point_embedding, features[i].reshape(1, -1)
        )
        cosine_similarities.append(similarity[0][0])

    return time_points, cosine_similarities


def compute_displacement_mean_std(
    embedding_dataset, max_tau=10, use_cosine=False, use_dissimilarity=False
):
    """
    Compute the mean and standard deviation of displacements between embeddings at time t and t + tau for each tau.

    Parameters:
    embedding_dataset : xarray.Dataset
        The dataset containing embeddings, timepoints, fov_name, and track_id.
    max_tau : int, optional
        The maximum tau value to compute displacements for. Default is 10.
    use_cosine : bool, optional
        If True, compute cosine similarity instead of Euclidean distance. Default is False.
    use_dissimilarity : bool, optional
        If True and use_cosine is True, compute cosine dissimilarity (1 - similarity).
        Default is False.

    Returns:
    tuple
        - mean_displacement_per_tau (dict): Mean displacement for each tau.
        - std_displacement_per_tau (dict): Standard deviation of displacements for each tau.
    """
    # Get the arrays of (fov_name, track_id, t, and embeddings)
    fov_names = embedding_dataset["fov_name"].values
    track_ids = embedding_dataset["track_id"].values
    timepoints = embedding_dataset["t"].values
    embeddings = embedding_dataset["features"].values

    # Dictionary to store displacements for each tau
    displacement_per_tau = defaultdict(list)

    # Iterate over all entries in the dataset
    for i in range(len(fov_names)):
        fov_name = fov_names[i]
        track_id = track_ids[i]
        current_time = timepoints[i]
        current_embedding = embeddings[i]

        # For each time point t, compute displacements for t + tau
        for tau in range(1, max_tau + 1):
            future_time = current_time + tau

            # Find if future_time exists for the same (fov_name, track_id)
            matching_indices = np.where(
                (fov_names == fov_name)
                & (track_ids == track_id)
                & (timepoints == future_time)
            )[0]

            if len(matching_indices) == 1:
                # Get the embedding at t + tau
                future_embedding = embeddings[matching_indices[0]]

                if use_cosine:
                    # Compute cosine similarity
                    similarity = cosine_similarity(
                        current_embedding.reshape(1, -1),
                        future_embedding.reshape(1, -1),
                    )[0][0]
                    # Choose whether to use similarity or dissimilarity
                    if use_dissimilarity:
                        displacement = 1 - similarity  # Cosine dissimilarity
                    else:
                        displacement = similarity  # Cosine similarity
                else:
                    # Compute the Euclidean distance, elementwise square on difference
                    displacement = np.sum((current_embedding - future_embedding) ** 2)

                # Store the displacement for the given tau
                displacement_per_tau[tau].append(displacement)

    # Compute mean and std displacement for each tau by averaging the displacements
    mean_displacement_per_tau = {
        tau: np.mean(displacements)
        for tau, displacements in displacement_per_tau.items()
    }
    std_displacement_per_tau = {
        tau: np.std(displacements)
        for tau, displacements in displacement_per_tau.items()
    }

    return mean_displacement_per_tau, std_displacement_per_tau


def compute_displacement(
    embedding_dataset,
    max_tau=10,
    use_cosine=False,
    use_dissimilarity=False,
    use_umap=False,
):
    """
    Compute the displacements between embeddings at time t and t + tau for each tau.

    Parameters:
    embedding_dataset : xarray.Dataset
        The dataset containing embeddings, timepoints, fov_name, and track_id.
    max_tau : int, optional
        The maximum tau value to compute displacements for. Default is 10.
    use_cosine : bool, optional
        If True, compute cosine similarity instead of Euclidean distance. Default is False.
    use_dissimilarity : bool, optional
        If True and use_cosine is True, compute cosine dissimilarity (1 - similarity).
        Default is False.
    use_umap : bool, optional
        If True, use UMAP embeddings instead of feature embeddings. Default is False.

    Returns:
    dict
        A dictionary where the key is tau and the value is a list of displacements
        for all cells at that tau.
    """
    # Get the arrays of (fov_name, track_id, t, and embeddings)
    fov_names = embedding_dataset["fov_name"].values
    track_ids = embedding_dataset["track_id"].values
    timepoints = embedding_dataset["t"].values

    if use_umap:
        umap1 = embedding_dataset["UMAP1"].values
        umap2 = embedding_dataset["UMAP2"].values
        embeddings = np.vstack((umap1, umap2)).T
    else:
        embeddings = embedding_dataset["features"].values

    # Dictionary to store displacements for each tau
    displacement_per_tau = defaultdict(list)

    # Iterate over all entries in the dataset
    for i in range(len(fov_names)):
        fov_name = fov_names[i]
        track_id = track_ids[i]
        current_time = timepoints[i]
        current_embedding = embeddings[i]

        # For each time point t, compute displacements for t + tau
        for tau in range(1, max_tau + 1):
            future_time = current_time + tau

            # Find if future_time exists for the same (fov_name, track_id)
            matching_indices = np.where(
                (fov_names == fov_name)
                & (track_ids == track_id)
                & (timepoints == future_time)
            )[0]

            if len(matching_indices) == 1:
                # Get the embedding at t + tau
                future_embedding = embeddings[matching_indices[0]]

                if use_cosine:
                    # Compute cosine similarity
                    similarity = cosine_similarity(
                        current_embedding.reshape(1, -1),
                        future_embedding.reshape(1, -1),
                    )[0][0]
                    # Choose whether to use similarity or dissimilarity
                    if use_dissimilarity:
                        displacement = 1 - similarity  # Cosine dissimilarity
                    else:
                        displacement = similarity  # Cosine similarity
                else:
                    # Compute the Euclidean distance, elementwise square on difference
                    displacement = np.sum((current_embedding - future_embedding) ** 2)

                # Store the displacement for the given tau
                displacement_per_tau[tau].append(displacement)

    return displacement_per_tau


def calculate_normalized_euclidean_distance_cell(embedding_dataset, fov_name, track_id):
    """
    Calculate the normalized Euclidean distance between the embedding at the first time point and each subsequent time point for a specific cell.

    Parameters:
    embedding_dataset : xarray.Dataset
        The dataset containing embeddings, timepoints, fov_name, and track_id.
    fov_name : str
        Field of view name to identify the specific cell.
    track_id : int
        Track ID to identify the specific cell.

    Returns:
    tuple
        - time_points (array): Array of time points for the cell.
        - euclidean_distances (list): Normalized Euclidean distances between the embedding
          at the first time point and each subsequent time point.
    """
    filtered_data = embedding_dataset.where(
        (embedding_dataset["fov_name"] == fov_name)
        & (embedding_dataset["track_id"] == track_id),
        drop=True,
    )

    features = filtered_data["features"].values  # (sample, features)
    time_points = filtered_data["t"].values  # (sample,)

    normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # Get the first time point's normalized embedding
    first_time_point_embedding = normalized_features[0].reshape(1, -1)

    euclidean_distances = []
    for i in range(len(time_points)):
        distance = np.linalg.norm(
            first_time_point_embedding - normalized_features[i].reshape(1, -1)
        )
        euclidean_distances.append(distance)

    return time_points, euclidean_distances


def compute_displacement_mean_std_full(embedding_dataset, max_tau=10):
    """
    Compute the mean and standard deviation of displacements between embeddings at time t and t + tau for all cells.

    Parameters:
    embedding_dataset : xarray.Dataset
        The dataset containing embeddings, timepoints, fov_name, and track_id.
    max_tau : int, optional
        The maximum tau value to compute displacements for. Default is 10.

    Returns:
    tuple
        - mean_displacement_per_tau (dict): Mean displacement for each tau.
        - std_displacement_per_tau (dict): Standard deviation of displacements for each tau.
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

    displacement_per_tau = defaultdict(list)

    for cell in unique_cells:
        fov_name = cell["fov_name"]
        track_id = cell["track_id"]

        indices = np.where((fov_names == fov_name) & (track_ids == track_id))[0]

        cell_timepoints = timepoints[indices]
        cell_embeddings = embeddings[indices]

        sorted_indices = np.argsort(cell_timepoints)
        cell_timepoints = cell_timepoints[sorted_indices]
        cell_embeddings = cell_embeddings[sorted_indices]

        for i in range(len(cell_timepoints)):
            current_time = cell_timepoints[i]
            current_embedding = cell_embeddings[i]

            current_embedding = current_embedding / np.linalg.norm(current_embedding)

            for tau in range(0, max_tau + 1):
                future_time = current_time + tau

                future_index = np.where(cell_timepoints == future_time)[0]

                if len(future_index) >= 1:
                    future_embedding = cell_embeddings[future_index[0]]
                    future_embedding = future_embedding / np.linalg.norm(
                        future_embedding
                    )

                    distance = np.linalg.norm(current_embedding - future_embedding)

                    displacement_per_tau[tau].append(distance)

    mean_displacement_per_tau = {
        tau: np.mean(displacements)
        for tau, displacements in displacement_per_tau.items()
    }
    std_displacement_per_tau = {
        tau: np.std(displacements)
        for tau, displacements in displacement_per_tau.items()
    }

    return mean_displacement_per_tau, std_displacement_per_tau


# Function to compute metrics for dynamic range and smoothness
def compute_dynamic_smoothness_metrics(mean_displacement_per_tau):
    """
    Compute dynamic range and smoothness metrics for displacement curves.

    Parameters:
    mean_displacement_per_tau: dict with tau as key and mean displacement as value

    Returns:
    tuple: (dynamic_range, smoothness)
        - dynamic_range: max displacement - min displacement
        - smoothness: RMS of second differences of normalized curve
    """
    taus = np.array(sorted(mean_displacement_per_tau.keys()))
    displacements = np.array([mean_displacement_per_tau[tau] for tau in taus])

    dynamic_range = np.max(displacements) - np.min(displacements)

    if np.max(displacements) != np.min(displacements):
        displacements_normalized = (displacements - np.min(displacements)) / (
            np.max(displacements) - np.min(displacements)
        )
    else:
        displacements_normalized = displacements - np.min(
            displacements
        )  # Handle constant case

    first_diff = np.diff(displacements_normalized)

    second_diff = np.diff(first_diff)

    # Compute RMS of second differences as smoothness metric
    # Lower values indicate smoother curves
    smoothness = np.sqrt(np.mean(second_diff**2))

    return dynamic_range, smoothness
