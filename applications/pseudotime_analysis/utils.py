from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def load_annotation(
    embedding_dataset: xr.Dataset,
    track_csv_path: str,
    name: str,
    categories: dict | None = None,
) -> pd.Series:
    """
    Load annotations from a CSV file and map them to the dataset.
    """
    annotation = pd.read_csv(track_csv_path)
    annotation["fov_name"] = "/" + annotation["fov ID"]

    embedding_index = pd.MultiIndex.from_arrays(
        [
            embedding_dataset["fov_name"].values,
            embedding_dataset["id"].values,
            embedding_dataset["t"].values,
            embedding_dataset["track_id"].values,
        ],
        names=["fov_name", "id", "t", "track_id"],
    )

    annotation = annotation.set_index(["fov_name", "id", "t", "track_id"])
    selected = annotation.reindex(embedding_index)[name]

    if categories:
        if -1 in selected.values and -1 not in categories:
            categories = categories.copy()
            categories[-1] = np.nan
        selected = selected.map(categories)

    return selected


def identify_lineages(annotations_path: Path) -> list[tuple[str, list[int]]]:
    """
    Identifies all distinct lineages in the cell tracking data, following only
    one branch after each division event.

    Args:
        annotations_path: Path to the annotations CSV file

    Returns:
        A list of tuples, where each tuple contains (fov_id, [track_ids])
        representing a single branch lineage within a single FOV
    """
    # Read the CSV file
    df = pd.read_csv(annotations_path)

    # Ensure column names are consistent
    if "fov ID" in df.columns and "fov_id" not in df.columns:
        df["fov_id"] = df["fov ID"]

    # Process each FOV separately to handle repeated track_ids
    all_lineages = []

    # Group by FOV
    for fov_id, fov_df in df.groupby("fov_id"):
        # Create a dictionary to map tracks to their parents within this FOV
        child_to_parent = {}

        # Group by track_id and get the first row for each track to find its parent
        for track_id, track_group in fov_df.groupby("track_id"):
            first_row = track_group.iloc[0]
            parent_track_id = first_row["parent_track_id"]

            if parent_track_id != -1:
                child_to_parent[track_id] = parent_track_id

        # Find root tracks (those without parents or with parent_track_id = -1)
        all_tracks = set(fov_df["track_id"].unique())
        child_tracks = set(child_to_parent.keys())
        root_tracks = all_tracks - child_tracks

        # Build a parent-to-children mapping
        parent_to_children = {}
        for child, parent in child_to_parent.items():
            if parent not in parent_to_children:
                parent_to_children[parent] = []
            parent_to_children[parent].append(child)

        # Function to get a single branch from each parent
        # We'll choose the first child in the list (arbitrary choice)
        def get_single_branch(track_id):
            branch = [track_id]
            if track_id in parent_to_children:
                # Choose only the first child (you could implement other selection criteria)
                first_child = parent_to_children[track_id][0]
                branch.extend(get_single_branch(first_child))
            return branch

        # Build lineages starting from root tracks within this FOV
        for root_track in root_tracks:
            # Get a single branch from this root
            lineage_tracks = get_single_branch(root_track)
            all_lineages.append((fov_id, lineage_tracks))

    return all_lineages


def path_skew(warping_path, ref_len, query_len):
    """
    Calculate the skewness of a DTW warping path.

    Args:
        warping_path: List of tuples (ref_idx, query_idx) representing the warping path
        ref_len: Length of the reference sequence
        query_len: Length of the query sequence

    Returns:
        A skewness metric between 0 and 1, where:
        - 0 means perfectly diagonal path (ideal alignment)
        - 1 means completely skewed (worst alignment)
    """
    # Convert path to numpy array for easier manipulation
    path_array = np.array(warping_path)

    # Calculate "ideal" diagonal indices
    diagonal_x = np.linspace(0, ref_len - 1, len(warping_path))
    diagonal_y = np.linspace(0, query_len - 1, len(warping_path))
    diagonal_path = np.column_stack((diagonal_x, diagonal_y))

    # Calculate distances from points on the warping path to the diagonal
    # Normalize based on max possible distance (corner to diagonal)
    max_distance = max(ref_len, query_len)

    # Calculate distance from each point to the diagonal
    distances = []
    for i, (x, y) in enumerate(path_array):
        # Find the closest point on the diagonal
        dx, dy = diagonal_path[i]
        # Simple Euclidean distance
        dist = np.sqrt((x - dx) ** 2 + (y - dy) ** 2)
        distances.append(dist)

    # Average normalized distance as skewness metric
    skew = np.mean(distances) / max_distance

    return skew


def dtw_with_matrix(distance_matrix, normalize=True):
    """
    Compute DTW using a pre-computed distance matrix.

    Args:
        distance_matrix: Pre-computed distance matrix between two sequences
        normalize: Whether to normalize the distance by path length (default: True)

    Returns:
        dtw_distance: The DTW distance
        warping_matrix: The accumulated cost matrix
        best_path: The optimal warping path
    """
    n, m = distance_matrix.shape

    # Initialize the accumulated cost matrix
    warping_matrix = np.full((n, m), np.inf)
    warping_matrix[0, 0] = distance_matrix[0, 0]

    # Fill the first column and row
    for i in range(1, n):
        warping_matrix[i, 0] = warping_matrix[i - 1, 0] + distance_matrix[i, 0]
    for j in range(1, m):
        warping_matrix[0, j] = warping_matrix[0, j - 1] + distance_matrix[0, j]

    # Fill the rest of the matrix
    for i in range(1, n):
        for j in range(1, m):
            warping_matrix[i, j] = distance_matrix[i, j] + min(
                warping_matrix[i - 1, j],  # insertion
                warping_matrix[i, j - 1],  # deletion
                warping_matrix[i - 1, j - 1],  # match
            )

    # Backtrack to find the optimal path
    i, j = n - 1, m - 1
    path = [(i, j)]

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_cost = min(
                warping_matrix[i - 1, j],
                warping_matrix[i, j - 1],
                warping_matrix[i - 1, j - 1],
            )

            if min_cost == warping_matrix[i - 1, j - 1]:
                i, j = i - 1, j - 1
            elif min_cost == warping_matrix[i - 1, j]:
                i -= 1
            else:
                j -= 1

        path.append((i, j))

    path.reverse()

    # Get the DTW distance (bottom-right cell)
    dtw_distance = warping_matrix[n - 1, m - 1]

    # Normalize by path length if requested
    if normalize:
        dtw_distance = dtw_distance / len(path)

    return dtw_distance, warping_matrix, path


# %%
def filter_lineages_by_timepoints(lineages, annotation_path, min_timepoints=10):
    """
    Filter lineages that have fewer than min_timepoints total timepoints.

    Args:
        lineages: List of tuples (fov_id, [track_ids])
        annotation_path: Path to the annotations CSV file
        min_timepoints: Minimum number of timepoints required

    Returns:
        List of filtered lineages
    """
    # Read the annotations file
    df = pd.read_csv(annotation_path)

    # Ensure column names are consistent
    if "fov ID" in df.columns and "fov_id" not in df.columns:
        df["fov_id"] = df["fov ID"]

    filtered_lineages = []

    for fov_id, track_ids in lineages:
        # Get all rows for this lineage
        lineage_rows = df[(df["fov_id"] == fov_id) & (df["track_id"].isin(track_ids))]

        # Count the total number of timepoints
        total_timepoints = len(lineage_rows)

        # Only keep lineages with at least min_timepoints
        if total_timepoints >= min_timepoints:
            filtered_lineages.append((fov_id, track_ids))

    return filtered_lineages


def find_top_matching_tracks(cell_division_df, infection_df, n_top=10) -> pd.DataFrame:
    # Find common tracks between datasets
    intersection_df = pd.merge(
        cell_division_df,
        infection_df,
        on=["fov_name", "track_ids"],
        how="inner",
        suffixes=("_df1", "_df2"),
    )

    # Add column with sum of the values
    intersection_df["distance_sum"] = (
        intersection_df["distance_df1"] + intersection_df["distance_df2"]
    )

    # Find rows with the smallest sum
    intersection_df.sort_values(by="distance_sum", ascending=True, inplace=True)
    return intersection_df.head(n_top)
