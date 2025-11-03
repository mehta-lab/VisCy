from collections import defaultdict

import numpy as np
import xarray as xr
from sklearn.metrics.pairwise import cosine_similarity

from viscy.representation.evaluation.clustering import (
    compare_time_offset,
    pairwise_distance_matrix,
)


def calculate_cosine_similarity_cell(embedding_dataset, fov_name, track_id):
    """Extract embeddings and calculate cosine similarities for a specific cell"""
    filtered_data = embedding_dataset.where(
        (embedding_dataset["fov_name"] == fov_name)
        & (embedding_dataset["track_id"] == track_id),
        drop=True,
    )
    features = filtered_data["features"].values
    time_points = filtered_data["t"].values
    first_time_point_embedding = features[0].reshape(1, -1)
    cosine_similarities = cosine_similarity(
        first_time_point_embedding, features
    ).flatten()
    cosine_similarities = np.clip(cosine_similarities, -1.0, 1.0)
    return time_points, cosine_similarities.tolist()


def compute_track_displacement(
    embedding_dataset: xr.Dataset,
    distance_metric: str = "cosine",
) -> dict[int, list[float]]:
    """
    Compute Mean Squared Displacement using pairwise distance matrix.

    Parameters
    ----------
    embedding_dataset : xr.Dataset
        Dataset containing embeddings and metadata
    distance_metric : str
        Distance metric to use. Default is cosine.
        See for other supported distance metrics.
        https://github.com/scipy/scipy/blob/main/scipy/spatial/distance.py

    Returns
    -------
    dict[int, list[float]]
        Dictionary mapping time lag τ to list of squared displacements
    """

    unique_tracks_df = (
        embedding_dataset[["fov_name", "track_id"]].to_dataframe().drop_duplicates()
    )

    displacement_per_tau = defaultdict(list)

    for fov_name, track_id in zip(
        unique_tracks_df["fov_name"], unique_tracks_df["track_id"]
    ):
        # Filter data for this track
        track_data = embedding_dataset.where(
            (embedding_dataset["fov_name"] == fov_name)
            & (embedding_dataset["track_id"] == track_id),
            drop=True,
        )

        # Sort by time
        time_order = np.argsort(track_data["t"].values)
        times = track_data["t"].values[time_order]
        track_embeddings = track_data["features"].values[time_order]

        # Compute pairwise distance matrix
        distance_matrix = pairwise_distance_matrix(
            track_embeddings, metric=distance_metric
        )

        # Extract displacements using diagonal offsets
        n_timepoints = len(times)
        for time_offset in range(1, n_timepoints):
            diagonal_displacements = compare_time_offset(distance_matrix, time_offset)

            for i, displacement in enumerate(diagonal_displacements):
                tau = int(times[i + time_offset] - times[i])
                displacement_per_tau[tau].append(displacement)

    return dict(displacement_per_tau)
