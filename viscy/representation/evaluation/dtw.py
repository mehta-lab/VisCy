import numpy as np
import pandas as pd
import xarray as xr
import torch
from scipy.spatial.distance import cdist
from viscy.data.triplet import INDEX_COLUMNS
from warnings import warn
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from typing import Literal, cast
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def identify_lineages(
    embeddings: xr.Dataset,
    min_timepoints: int = 10,
    return_both_branches: bool = False,
    include_fovs: list[str] | None = None,
) -> list[tuple[str, list[int]]]:
    """
    Identifies all distinct lineages in cell tracking data from an xarray.Dataset.

    Parameters
    ----------
    embeddings : xr.Dataset
        Dataset containing tracking data with variables: 'fov_name', 'track_id', 'parent_track_id', and time.
    min_timepoints : int, optional
        Minimum number of timepoints to consider a lineage (default: 10).
    return_both_branches : bool, optional
        Whether to return both branches after each division event (default: False).
    include_fovs : list of str, str, or None, optional
        FOV(s) to process. If None, process all FOVs. If str or list, process only those.

    Returns
    -------
    list of tuple
        Each tuple contains (fov_id, [track_ids]) representing a single branch lineage within a single FOV.

    Notes
    -----
    - Uses tree traversal to extract lineages (see lineage tracing in cell tracking, e.g., Jaqaman et al., Nat Methods 2008).
    - Time filtering is performed using PyTorch for GPU acceleration.
    """

    all_lineages = []
    all_fov_names = np.unique(embeddings["fov_name"].values)
    if include_fovs is None:
        fov_names = all_fov_names
    else:
        # Validate that all requested FOVs exist in the dataset
        missing_fovs = set(include_fovs) - set(all_fov_names)
        if missing_fovs:
            raise ValueError(f"FOV(s) not found in dataset: {missing_fovs}")
        fov_names = include_fovs

    for fov_id in fov_names:
        fov_mask = embeddings["fov_name"].values == fov_id
        fov_ds = embeddings.isel(sample=fov_mask)

        track_ids = fov_ds["track_id"].values
        parent_track_ids = fov_ds["parent_track_id"].values
        times = fov_ds["t"].values

        # Map track_id to parent_track_id
        child_to_parent = {}
        for tid, ptid in zip(track_ids, parent_track_ids):
            if ptid != -1:
                child_to_parent[tid] = ptid

        all_tracks = set(track_ids)
        child_tracks = set(child_to_parent.keys())
        root_tracks = all_tracks - child_tracks

        # Validate root tracks
        validated_root_tracks = set()
        for track_id in root_tracks:
            idx = np.where(track_ids == track_id)[0][0]
            if parent_track_ids[idx] == -1 or parent_track_ids[idx] not in all_tracks:
                validated_root_tracks.add(track_id)
        root_tracks = validated_root_tracks

        # Build parent-to-children mapping
        parent_to_children = {}
        for child, parent in child_to_parent.items():
            parent_to_children.setdefault(parent, []).append(child)

        def get_all_branches(track_id):
            """Recursively get all branches from a parent (tree traversal)."""
            branches = []
            current_branch = [track_id]
            if track_id in parent_to_children:
                for child in parent_to_children[track_id]:
                    child_branches = get_all_branches(child)
                    for branch in child_branches:
                        branches.append(current_branch + branch)
            else:
                branches.append(current_branch)
            return branches

        def calculate_lineage_timepoints(branch_track_ids):
            """Calculate total number of timepoints for a lineage using PyTorch for speed."""
            track_ids_tensor = torch.tensor(track_ids)
            mask = torch.isin(track_ids_tensor, torch.tensor(branch_track_ids))
            return int(mask.sum().item())

        for root_track in root_tracks:
            lineage_tracks = get_all_branches(root_track)
            if return_both_branches:
                for branch in lineage_tracks:
                    total_timepoints = calculate_lineage_timepoints(branch)
                    if total_timepoints >= min_timepoints:
                        all_lineages.append((fov_id, branch))
            else:
                if lineage_tracks:
                    total_timepoints = calculate_lineage_timepoints(lineage_tracks[0])
                    if total_timepoints >= min_timepoints:
                        all_lineages.append((fov_id, lineage_tracks[0]))

    return all_lineages


def path_skew(warping_path, ref_len, query_len):
    """
    Calculate the skewness of a DTW warping path.

    Parameters
    ----------
    warping_path : List of tuples (ref_idx, query_idx)
        representing the warping path
    ref_len : int
        Length of the reference sequence
    query_len : int
        Length of the query sequence

    Returns
    -------
    float
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

    Parameters
    ----------
    distance_matrix : np.ndarray
        Pre-computed distance matrix between two sequences
    normalize : bool, optional
        Whether to normalize the distance by path length (default: True)

    Returns
    -------
    tuple
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


def find_best_match_dtw(
    lineage: np.ndarray,
    reference_pattern: np.ndarray,
    num_candidates: int = 5,
    window_step: int = 5,
    max_distance: float = float("inf"),
    max_skew: float = 0.8,
    normalize: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Find the best matches in a lineage using DTW with dtaidistance.

    Parameters
    ----------
    lineage : np.ndarray
        The lineage to search (t,embeddings).
    reference_pattern : np.ndarray
        The pattern to search for (t,embeddings).
    num_candidates : int, optional
        The number of candidates to return.
    window_step : int, optional
        The step size for the window.
    max_distance : float, optional
        Maximum distance threshold to consider a match.
        max_skew: Maximum allowed path skewness (0-1).
    show_progress : bool, optional
        Whether to show progress bar for window sliding (default: False).

    Returns
    -------
    pd.DataFrame
        DataFrame with position, warping_path, distance, and skewness for the best matches
    """
    from dtaidistance.dtw_ndim import warping_path

    dtw_results = []
    n_windows = len(lineage) - len(reference_pattern) + 1

    if n_windows <= 0:
        return pd.DataFrame(
            {"position": [], "path": [], "distance": [], "skewness": []}
        )

    # Create iterator with optional progress bar
    window_range = range(0, n_windows, window_step)
    if show_progress and n_windows > 100:  # Only show for large numbers of windows
        window_range = tqdm(window_range, desc="DTW window sliding", leave=False)

    for i in window_range:
        window = lineage[i : i + len(reference_pattern)]
        path, dist = warping_path(
            reference_pattern,
            window,
            include_distance=True,
        )
        if normalize:
            # Normalize by path length to match bernd_clifford method
            dist = dist / len(path)
        # Calculate skewness using the utils function
        skewness = path_skew(path, len(reference_pattern), len(window))

        if dist <= max_distance and skewness <= max_skew:
            dtw_results.append(
                {"position": i, "path": path, "distance": dist, "skewness": skewness}
            )

    # Convert to DataFrame
    results_df = pd.DataFrame(dtw_results)

    # Sort by distance first (primary) and then by skewness (secondary)
    if not results_df.empty:
        results_df = results_df.sort_values(by=["distance", "skewness"]).head(
            num_candidates
        )

    return results_df


def find_best_match_dtw_bernd_clifford(
    lineage: np.ndarray,
    reference_pattern: np.ndarray,
    num_candidates: int = 5,
    window_step: int = 5,
    normalize: bool = True,
    max_distance: float = float("inf"),
    max_skew: float = 0.8,
    metric: Literal["cosine", "euclidean"] = "cosine",
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Find the best matches in a lineage using DTW with the utils.py implementation.

    Parameters
    ----------
    lineage : np.ndarray
        The lineage to search (t,embeddings).
    reference_pattern : np.ndarray
        The pattern to search for (t,embeddings).
    num_candidates : int, optional
        The number of candidates to return.
    window_step : int, optional
        The step size for the window.
    max_distance : float, optional
        Maximum distance threshold to consider a match.
    max_skew: Maximum allowed path skewness (0-1).
    show_progress : bool, optional
        Whether to show progress bar for window sliding (default: False).

    Returns
    -------
    pd.DataFrame
        DataFrame with position, warping_path, distance, and skewness for the best matches
    """

    dtw_results = []
    n_windows = len(lineage) - len(reference_pattern) + 1

    if n_windows <= 0:
        return pd.DataFrame(
            {"position": [], "path": [], "distance": [], "skewness": []}
        )

    # Create iterator with optional progress bar
    window_range = range(0, n_windows, window_step)
    if show_progress and n_windows > 100:  # Only show for large numbers of windows
        window_range = tqdm(window_range, desc="DTW window sliding", leave=False)

    for i in window_range:
        window = lineage[i : i + len(reference_pattern)]

        # Create distance matrix
        distance_matrix = cdist(reference_pattern, window, metric=metric)

        # Apply DTW using our implementation
        distance, _, path = dtw_with_matrix(distance_matrix, normalize=normalize)

        # Calculate skewness
        skewness = path_skew(path, len(reference_pattern), len(window))

        # Only add if both distance and skewness pass thresholds
        if distance <= max_distance and skewness <= max_skew:
            logger.debug(
                f"Found match at {i} with distance {distance} and skewness {skewness}"
            )
            dtw_results.append(
                {
                    "position": i,
                    "path": path,
                    "distance": distance,
                    "skewness": skewness,
                }
            )

    # Convert to DataFrame
    results_df = pd.DataFrame(dtw_results)

    # Sort by distance first (primary) and then by skewness (secondary)
    if not results_df.empty:
        results_df = results_df.sort_values(by=["distance", "skewness"]).head(
            num_candidates
        )

    return results_df


def _process_lineage_dtw(
    lineage_embeddings: np.ndarray,
    reference_pattern: np.ndarray,
    method: str,
    num_candidates: int,
    window_step: int,
    max_distance: float,
    max_skew: float,
    normalize: bool,
    metric: str,
    show_inner_progress: bool,
) -> dict | None:
    """
    Shared function to process DTW matching for a single lineage.

    Returns the best match information or None if no matches found.
    """
    # Find best matches using the selected DTW method
    if method == "bernd_clifford":
        # Ensure metric is valid for bernd_clifford method
        valid_metric = cast(
            Literal["cosine", "euclidean"],
            metric if metric in ["cosine", "euclidean"] else "cosine",
        )
        matches_df = find_best_match_dtw_bernd_clifford(
            lineage_embeddings,
            reference_pattern=reference_pattern,
            num_candidates=num_candidates,
            window_step=window_step,
            max_distance=max_distance,
            max_skew=max_skew,
            normalize=normalize,
            metric=valid_metric,
            show_progress=show_inner_progress,
        )
    else:
        matches_df = find_best_match_dtw(
            lineage_embeddings,
            reference_pattern=reference_pattern,
            num_candidates=num_candidates,
            window_step=window_step,
            max_distance=max_distance,
            max_skew=max_skew,
            normalize=normalize,
            show_progress=show_inner_progress,
        )

    if not matches_df.empty:
        # Get the best match (first row of the sorted DataFrame)
        best_match = matches_df.iloc[0]
        return {
            "distance": best_match["distance"],
            "skewness": best_match["skewness"],
            "warp_path": best_match["path"],
            "start_timepoint": best_match["position"],
            "end_timepoint": best_match["position"] + len(reference_pattern),
        }
    else:
        return None


def _process_single_lineage_parallel(lineage_data: tuple, **kwargs) -> dict:
    """
    Worker function for parallel processing.

    Extracts lineage data and calls the shared processing function.
    """
    fov_name, track_ids, lineage_embeddings = lineage_data

    try:
        result = _process_lineage_dtw(lineage_embeddings, **kwargs)

        if result:
            result.update(
                {
                    "fov_name": fov_name,
                    "track_ids": track_ids,
                }
            )
        else:
            result = {
                "fov_name": fov_name,
                "track_ids": track_ids,
                "distance": None,
                "skewness": None,
                "warp_path": None,
                "start_timepoint": None,
                "end_timepoint": None,
            }
        return result

    except Exception as e:
        logger.error(f"Error processing lineage {fov_name}, {track_ids}: {e}")
        return {
            "fov_name": fov_name,
            "track_ids": track_ids,
            "distance": None,
            "skewness": None,
            "warp_path": None,
            "start_timepoint": None,
            "end_timepoint": None,
        }


def find_pattern_matches(
    reference_pattern: np.ndarray,
    filtered_lineages: list[tuple[str, list[int]]],
    embeddings_dataset: xr.Dataset,
    window_step: int = 1,
    num_candidates: int = 3,
    max_distance: float = float("inf"),
    max_skew: float = 0.8,
    save_path: str | None = None,
    method: str = "bernd_clifford",
    normalize: bool = True,
    metric: str = "cosine",
    n_jobs: int = 1,
    show_inner_progress: bool = False,
) -> pd.DataFrame:
    """
    Find the best matches of a reference pattern in multiple lineages using DTW.

    Parameters
    ----------
    reference_pattern : np.ndarray
        The reference pattern embeddings
    filtered_lineages : list[tuple[str, list[int]]]
        List of lineages to search in (fov_name, track_ids)
    embeddings_dataset : xr.Dataset
        Dataset containing embeddings
    window_step : int, optional
        Step size for the window (default: 1)
    num_candidates : int, optional
        Number of best candidates to consider per lineage (default: 3)
    max_distance : float, optional
        Maximum distance threshold to consider a match (default: inf)
    max_skew : float, optional
        Maximum allowed path skewness (0-1, where 0=perfect diagonal) (default: 0.8)
    save_path : str or None, optional
        Optional path to save the results CSV (default: None)
    method : str, optional
        DTW method to use - 'bernd_clifford' or 'dtaidistance' (default: 'bernd_clifford')
    normalize : bool, optional
        Whether to normalize DTW distance by path length (default: True)
    metric : str, optional
        Distance metric for embeddings comparison (default: 'cosine')
    n_jobs : int, optional
        Number of parallel jobs. If 1, runs sequentially. If >1, runs in parallel (default: 1)
    show_inner_progress : bool, optional
        Whether to show detailed progress bars for DTW window sliding (default: False)

    Returns
    -------
    pd.DataFrame
        DataFrame with match positions and distances

    Notes
    -----
    When n_jobs > 1, uses multiprocessing for faster computation on multi-core systems.
    The DTW algorithm (Sakoe & Chiba, 1978) is applied for time series alignment.
    """
    # FIXME: Check if we can pass other metrics to dtaidistance
    if metric != "euclidean" and method != "bernd_clifford":
        warn("dtaidistance only supports euclidean distance for now")

    # Prepare shared processing parameters
    processing_kwargs = {
        "reference_pattern": reference_pattern,
        "method": method,
        "num_candidates": num_candidates,
        "window_step": window_step,
        "max_distance": max_distance,
        "max_skew": max_skew,
        "normalize": normalize,
        "metric": metric,
        "show_inner_progress": show_inner_progress,
    }

    # Use parallel processing if n_jobs > 1
    if n_jobs > 1:
        return _find_pattern_matches_parallel(
            filtered_lineages=filtered_lineages,
            embeddings_dataset=embeddings_dataset,
            n_jobs=n_jobs,
            save_path=save_path,
            **processing_kwargs,
        )

    # Sequential implementation
    all_match_positions = []

    for fov_name, track_ids in tqdm(
        filtered_lineages,
        desc="Finding pattern matches",
    ):
        logger.info(
            f"Finding pattern matches for {fov_name} with track ids: {track_ids}"
        )

        # Reconstruct the concatenated lineage
        lineages = []
        for track_id in track_ids:
            track_embeddings = embeddings_dataset.sel(
                sample=(fov_name, track_id)
            ).features.values
            lineages.append(track_embeddings)

        lineage_embeddings = np.concatenate(lineages, axis=0)

        # Process this lineage using shared function
        result = _process_lineage_dtw(lineage_embeddings, **processing_kwargs)

        if result:
            result.update(
                {
                    "fov_name": fov_name,
                    "track_ids": track_ids,
                }
            )
        else:
            result = {
                "fov_name": fov_name,
                "track_ids": track_ids,
                "distance": None,
                "skewness": None,
                "warp_path": None,
                "start_timepoint": None,
                "end_timepoint": None,
            }

        all_match_positions.append(result)

    # Convert to DataFrame and drop rows with no matches
    all_match_positions = pd.DataFrame(all_match_positions)
    all_match_positions = all_match_positions.dropna()

    # Sort by distance (primary) and skewness (secondary)
    if not all_match_positions.empty:
        all_match_positions = all_match_positions.sort_values(
            by=["distance", "skewness"], ascending=[True, True]
        )

    # Save to CSV if path is provided
    if save_path:
        all_match_positions.to_csv(save_path, index=False)

    return all_match_positions


def _find_pattern_matches_parallel(
    filtered_lineages: list[tuple[str, list[int]]],
    embeddings_dataset: xr.Dataset,
    n_jobs: int,
    save_path: str | None = None,
    **processing_kwargs,
) -> pd.DataFrame:
    """
    Internal parallel implementation of find_pattern_matches.
    """
    logger.info(f"Using {n_jobs} parallel workers for pattern matching")

    # Prepare lineage data by extracting embeddings beforehand
    lineage_data_list = []

    logger.info("Preparing lineage data for parallel processing...")
    for fov_name, track_ids in tqdm(
        filtered_lineages, desc="Extracting lineage embeddings"
    ):
        # Reconstruct the concatenated lineage
        lineages = []
        for track_id in track_ids:
            track_embeddings = embeddings_dataset.sel(
                sample=(fov_name, track_id)
            ).features.values
            lineages.append(track_embeddings)

        lineage_embeddings = np.concatenate(lineages, axis=0)
        lineage_data_list.append((fov_name, track_ids, lineage_embeddings))

    # Create partial function with fixed parameters
    worker_func = partial(_process_single_lineage_parallel, **processing_kwargs)

    # Process lineages in parallel
    results = []
    logger.info(
        f"Processing {n_jobs} / {len(lineage_data_list)} lineages in parallel..."
    )

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        future_to_lineage = {
            executor.submit(worker_func, lineage_data): lineage_data[0:2]
            for lineage_data in lineage_data_list
        }

        # Collect results with progress bar
        for future in tqdm(
            as_completed(future_to_lineage),
            total=len(future_to_lineage),
            desc="Finding pattern matches",
        ):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                fov_name, track_ids = future_to_lineage[future]
                logger.error(f"Error processing lineage {fov_name}, {track_ids}: {e}")
                # Add empty result for failed lineages
                results.append(
                    {
                        "fov_name": fov_name,
                        "track_ids": track_ids,
                        "distance": None,
                        "skewness": None,
                        "warp_path": None,
                        "start_timepoint": None,
                        "end_timepoint": None,
                    }
                )

    # Convert results to DataFrame
    all_match_positions = pd.DataFrame(results)

    # Drop rows with no matches
    all_match_positions = all_match_positions.dropna()

    # Sort by distance (primary) and skewness (secondary)
    if not all_match_positions.empty:
        all_match_positions = all_match_positions.sort_values(
            by=["distance", "skewness"], ascending=[True, True]
        )

    # Save to CSV if path is provided
    if save_path:
        all_match_positions.to_csv(save_path, index=False)

    return all_match_positions
