from collections import defaultdict
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.clustering import (
    compare_time_offset,
    pairwise_distance_matrix,
    rank_nearest_neighbors,
    select_block,
)


def calculate_distance_cell(
    embedding_dataset,
    fov_name,
    track_id,
    metric: Literal["cosine", "euclidean", "normalized_euclidean"] = "cosine",
):
    """
    Calculate distances between a cell's first timepoint embedding and all its subsequent embeddings.

    This function extracts embeddings for a specific cell (identified by fov_name and track_id)
    and calculates the distance between its first timepoint embedding and all subsequent timepoints
    using the specified distance metric.

    Parameters
    ----------
    embedding_dataset : xarray.Dataset
        Dataset containing the embeddings and metadata. Must have dimensions for 'features',
        'fov_name', 'track_id', and 't' (time).
    fov_name : str
        Field of view name to identify the specific imaging area.
    track_id : int
        Track ID of the cell to analyze.
    metric : {'cosine', 'euclidean', 'normalized_euclidean'}, default='cosine'
        Distance metric to use for calculations:
        - 'cosine': Cosine similarity between embeddings
        - 'euclidean': Standard Euclidean distance
        - 'normalized_euclidean': Euclidean distance between L2-normalized embeddings

    Returns
    -------
    time_points : numpy.ndarray
        Array of time points corresponding to the calculated distances.
    distances : list
        List of distances between the first timepoint embedding and each subsequent
        timepoint embedding, calculated using the specified metric.

    Notes
    -----
    For 'normalized_euclidean', embeddings are L2-normalized before distance calculation.
    Cosine similarity results in values between -1 and 1, where 1 indicates identical
    direction, 0 indicates orthogonality, and -1 indicates opposite directions.
    Euclidean distances are always non-negative.

    Examples
    --------
    >>> times, distances = calculate_distance_cell(dataset, "FOV1", 1, metric="cosine")
    >>> times, distances = calculate_distance_cell(dataset, "FOV1", 1, metric="euclidean")
    """
    filtered_data = embedding_dataset.where(
        (embedding_dataset["fov_name"] == fov_name)
        & (embedding_dataset["track_id"] == track_id),
        drop=True,
    )
    features = filtered_data["features"].values  # (sample, features)
    time_points = filtered_data["t"].values  # (sample,)

    if metric == "normalized_euclidean":
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

    first_time_point_embedding = features[0].reshape(1, -1)

    if metric == "cosine":
        distances = cosine_similarity(first_time_point_embedding, features).flatten()
    else:  # both euclidean and normalized_euclidean use norm
        distances = np.linalg.norm(first_time_point_embedding - features, axis=1)

    return time_points, distances.tolist()


def compute_displacement(
    embedding_dataset,
    distance_metric: Literal["euclidean_squared", "cosine"] = "euclidean_squared",
) -> dict[int, list[float]]:
    """Compute the displacement or mean square displacement (MSD) of embeddings.

    For each time difference τ, computes either:
    - |r(t + τ) - r(t)|² for squared Euclidean (MSD)
    - cos_sim(r(t + τ), r(t)) for cosine
    for all particles and initial times t.

    Parameters
    ----------
    embedding_dataset : xarray.Dataset
        Dataset containing embeddings and metadata
    distance_metric : str
        The metric to use for computing distances between embeddings.
        Valid options are:
        - "euclidean": Euclidean distance (L2 norm)
        - "euclidean_squared": Squared Euclidean distance (for MSD, default)
        - "cosine": Cosine similarity
        - "cosine_dissimilarity": 1 - cosine similarity

    Returns
    -------
    dict[int, list[float]]
        Dictionary mapping τ to list of displacements for all particles and initial times
    """
    # Get unique tracks efficiently using pandas operations
    unique_tracks_df = (
        embedding_dataset[["fov_name", "track_id"]].to_dataframe().drop_duplicates()
    )

    # Get data from dataset
    fov_names = embedding_dataset["fov_name"].values
    track_ids = embedding_dataset["track_id"].values
    timepoints = embedding_dataset["t"].values
    embeddings = embedding_dataset["features"].values

    # Initialize results dictionary with empty lists
    displacement_per_tau = defaultdict(list)

    # Process each track
    for fov_name, track_id in zip(
        unique_tracks_df["fov_name"], unique_tracks_df["track_id"]
    ):
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

            # Check all possible future time points
            for future_idx, future_time in enumerate(
                times[t_idx + 1 :], start=t_idx + 1
            ):
                tau = future_time - t
                future_embedding = track_embeddings[future_idx]

                if distance_metric in ["cosine"]:
                    dot_product = np.dot(current_embedding, future_embedding)
                    norms = np.linalg.norm(current_embedding) * np.linalg.norm(
                        future_embedding
                    )
                    similarity = dot_product / norms
                    displacement = similarity
                else:  # Euclidean metrics
                    diff_squared = np.sum((current_embedding - future_embedding) ** 2)
                    displacement = diff_squared
                displacement_per_tau[int(tau)].append(displacement)

    return dict(displacement_per_tau)


def compute_displacement_statistics(
    displacement_per_tau: dict[int, list[float]],
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute mean and standard deviation of displacements for each tau.

    Parameters
    ----------
    displacement_per_tau : dict[int, list[float]]
        Dictionary mapping τ to list of displacements

    Returns
    -------
    tuple[dict[int, float], dict[int, float]]
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


def find_distribution_peak(data: np.ndarray) -> float:
    """
    Find the peak (mode) of a distribution using kernel density estimation.

    Args:
        data: Array of values to find the peak for

    Returns:
        float: The x-value where the peak occurs
    """
    kde = gaussian_kde(data)
    # Find the peak (maximum) of the KDE
    result = minimize_scalar(
        lambda x: -kde(x), bounds=(np.min(data), np.max(data)), method="bounded"
    )
    return result.x


def compute_piece_wise_dissimilarity(
    features_df: pd.DataFrame, cross_dist: NDArray, rank_fractions: NDArray
):
    """
    Computing the smoothness and dynamic range
    - Get the off diagonal per block and compute the mode
    - The blocks are not square, so we need to get the off diagonal elements
    - Get the 1 and 99 percentile of the off diagonal per block
    """
    piece_wise_dissimilarity_per_track = []
    piece_wise_rank_difference_per_track = []
    for name, subdata in features_df.groupby(["fov_name", "track_id"]):
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


def compute_embedding_distances(
    prediction_path: Path,
    output_path: Path,
    distance_metric: Literal["cosine", "euclidean", "normalized_euclidean"] = "cosine",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute and save pairwise distances between embeddings.

    Parameters
    ----------
    prediction_path : Path
        Path to the embedding dataset
    output_path : Path
        name of saved CSV file
    distance_metric : str, optional
        Distance metric to use for computing distances between embeddings
    verbose : bool, optional
        If True, plots the distance matrix visualization

    Returns
    -------
    pd.DataFrame
        DataFrame containing the adjacent frame and random sampling distances
    """
    # Read the dataset
    embeddings = read_embedding_dataset(prediction_path)
    features = embeddings["features"]

    if distance_metric != "euclidean":
        features = StandardScaler().fit_transform(features.values)

    # Compute the distance matrix
    cross_dist = pairwise_distance_matrix(features, metric=distance_metric)

    # Normalize by sqrt of embedding dimension if using euclidean distance
    if distance_metric == "euclidean":
        cross_dist /= np.sqrt(features.shape[1])

    if verbose:
        # Plot the distance matrix
        plt.figure(figsize=(10, 10))
        plt.imshow(cross_dist, cmap="viridis")
        plt.colorbar(label=f"{distance_metric.capitalize()} Distance")
        plt.title(f"{distance_metric.capitalize()} Distance Matrix")
        plt.tight_layout()
        plt.show()

    rank_fractions = rank_nearest_neighbors(cross_dist, normalize=True)

    # Compute piece-wise dissimilarity and rank difference
    features_df = features["sample"].to_dataframe().reset_index(drop=True)
    piece_wise_dissimilarity_per_track, piece_wise_rank_difference_per_track = (
        compute_piece_wise_dissimilarity(features_df, cross_dist, rank_fractions)
    )

    all_dissimilarity = np.concatenate(piece_wise_dissimilarity_per_track)

    # Random sampling values in the dissimilarity matrix
    n_samples = len(all_dissimilarity)
    random_indices = np.random.randint(0, len(cross_dist), size=(n_samples, 2))
    sampled_values = cross_dist[random_indices[:, 0], random_indices[:, 1]]

    # Create and save DataFrame
    distributions_df = pd.DataFrame(
        {
            "adjacent_frame": pd.Series(all_dissimilarity),
            "random_sampling": pd.Series(sampled_values),
        }
    )

    csv_path = output_path
    distributions_df.to_csv(csv_path, index=False)

    return distributions_df


def analyze_and_plot_distances(
    distributions_df: pd.DataFrame,
    output_file_path: Optional[str],
    overwrite: bool = False,
) -> dict:
    """
    Analyze distance distributions and create visualization plots.

    Parameters
    ----------
    distributions_df : pd.DataFrame
        DataFrame containing 'adjacent_frame' and 'random_sampling' columns
    output_file_path : str, optional
        Path to save the plot ideally with a .pdf extension. Uses `plt.savefig()`
    overwrite : bool, default=False
        If True, overwrites existing files

    Returns
    -------
    dict
        Dictionary containing computed metrics including means, standard deviations,
        medians, peaks, and dynamic range of the distributions
    """
    # Compute statistics
    adjacent_dist = distributions_df["adjacent_frame"].values
    random_dist = distributions_df["random_sampling"].values

    # Compute peaks
    adjacent_peak = float(find_distribution_peak(adjacent_dist))
    random_peak = float(find_distribution_peak(random_dist))
    dynamic_range = float(random_peak - adjacent_peak)

    metrics = {
        "dissimilarity_mean": float(np.mean(adjacent_dist)),
        "dissimilarity_std": float(np.std(adjacent_dist)),
        "dissimilarity_median": float(np.median(adjacent_dist)),
        "dissimilarity_peak": adjacent_peak,
        "dissimilarity_p99": float(np.percentile(adjacent_dist, 99)),
        "dissimilarity_p1": float(np.percentile(adjacent_dist, 1)),
        "random_mean": float(np.mean(random_dist)),
        "random_std": float(np.std(random_dist)),
        "random_median": float(np.median(random_dist)),
        "random_peak": random_peak,
        "dynamic_range": dynamic_range,
    }

    # Create plot
    fig = plt.figure()
    sns.histplot(
        data=distributions_df,
        x="adjacent_frame",
        bins=30,
        kde=True,
        color="cyan",
        alpha=0.5,
        stat="density",
    )
    sns.histplot(
        data=distributions_df,
        x="random_sampling",
        bins=30,
        kde=True,
        color="red",
        alpha=0.5,
        stat="density",
    )
    plt.xlabel("Cosine Dissimilarity")
    plt.ylabel("Density")
    plt.axvline(x=adjacent_peak, color="cyan", linestyle="--", alpha=0.8)
    plt.axvline(x=random_peak, color="red", linestyle="--", alpha=0.8)
    plt.tight_layout()
    plt.legend(["Adjacent Frame", "Random Sample", "Adjacent Peak", "Random Peak"])
    if output_file_path.exists() and not overwrite:
        raise FileExistsError(
            f"File {output_file_path} already exists and overwrite=False"
        )
    fig.savefig(output_file_path, dpi=600)
    plt.show()

    return metrics
