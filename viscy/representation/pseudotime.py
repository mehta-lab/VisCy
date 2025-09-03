from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from tqdm import tqdm

from viscy.representation.embedding_writer import read_embedding_dataset


@dataclass
class DTWResult:
    """Results from DTW pattern matching."""
    matches: pd.DataFrame
    reference_pattern: np.ndarray
    reference_info: dict
    

class DTWAnalyzer:
    """High-level API for DTW-based pseudotime analysis."""
    
    def __init__(self, embedding_path: str | Path, tracks_path: Optional[str | Path] = None):
        """Initialize DTW analyzer.
        
        Parameters
        ----------
        embedding_path : str | Path
            Path to embedding dataset (zarr file)
        tracks_path : str | Path, optional
            Path to tracking data
        """
        self.embedding_path = Path(embedding_path)
        self.tracks_path = Path(tracks_path) if tracks_path else None
        self._embeddings = None
        self._tracking_df = None
        
    @property
    def embeddings(self) -> xr.Dataset:
        """Load embeddings dataset lazily."""
        if self._embeddings is None:
            self._embeddings = read_embedding_dataset(str(self.embedding_path))
        return self._embeddings
    
    @property  
    def tracking_df(self) -> pd.DataFrame:
        """Load tracking dataframe lazily."""
        if self._tracking_df is None and self.tracks_path:
            # This would need to be implemented based on your tracking data format
            raise NotImplementedError("Tracking data loading not yet implemented")
        return self._tracking_df
    
    def get_reference_pattern(self, fov_name: str, track_id: int | list[int], 
                            timepoints: tuple[int, int]) -> np.ndarray:
        """Extract reference pattern from embeddings.
        
        Parameters
        ----------
        fov_name : str
            FOV identifier
        track_id : int | list[int] 
            Track ID(s) to use as reference
        timepoints : tuple[int, int]
            Start and end timepoints (start, end)
            
        Returns
        -------
        np.ndarray
            Reference pattern embeddings
        """
        if isinstance(track_id, int):
            track_id = [track_id]
            
        # Extract embeddings for the reference track(s)
        reference_embeddings = []
        for tid in track_id:
            track_emb = self.embeddings.sel(sample=(fov_name, tid)).features.values
            reference_embeddings.append(track_emb)
            
        # Concatenate if multiple tracks
        reference_pattern = np.concatenate(reference_embeddings, axis=0)
        
        # Extract the specified timepoint range
        start_t, end_t = timepoints
        reference_pattern = reference_pattern[start_t:end_t]
        
        return reference_pattern
    
    def find_pattern_matches(self, reference_pattern: np.ndarray, 
                           filtered_lineages: list[tuple[str, list[int]]] = None,
                           window_step_fraction: float = 0.25,
                           num_candidates: int = 3, 
                           max_distance: float = float("inf"),
                           max_skew: float = 0.8,
                           method: str = "bernd_clifford",
                           normalize: bool = True,
                           metric: str = "euclidean",
                           save_path: str | Path = None) -> pd.DataFrame:
        """Find pattern matches across lineages using DTW.
        
        Parameters
        ----------
        reference_pattern : np.ndarray
            Reference pattern to search for
        filtered_lineages : list[tuple[str, list[int]]], optional
            List of (fov_name, track_ids) to search in. If None, searches all.
        window_step_fraction : float
            Fraction of pattern length to use as window step
        num_candidates : int
            Number of best candidates per lineage
        max_distance : float
            Maximum DTW distance threshold
        max_skew : float
            Maximum path skewness (0-1)
        method : str
            DTW method ('bernd_clifford' or 'dtai')
        normalize : bool
            Whether to normalize DTW distance by path length
        metric : str
            Distance metric for embeddings
        save_path : str | Path, optional
            Path to save results CSV
            
        Returns
        -------
        pd.DataFrame
            Match results with distances and warping paths
        """
        if filtered_lineages is None:
            # TODO: Auto-identify lineages from tracking data
            raise NotImplementedError("Auto-identification of lineages not yet implemented")
            
        return find_pattern_matches(
            reference_pattern=reference_pattern,
            filtered_lineages=filtered_lineages,
            embeddings_dataset=self.embeddings,
            window_step_fraction=window_step_fraction,
            num_candidates=num_candidates,
            max_distance=max_distance,
            max_skew=max_skew,
            method=method,
            normalize=normalize,
            metric=metric,
            save_path=save_path
        )
    
    def analyze_embeddings(self, fov_name: str, track_id: int | list[int],
                          timepoints: tuple[int, int], 
                          filtered_lineages: list[tuple[str, list[int]]] = None,
                          **kwargs) -> DTWResult:
        """Complete DTW analysis pipeline.
        
        Parameters
        ----------
        fov_name : str
            Reference FOV name
        track_id : int | list[int]
            Reference track ID(s) 
        timepoints : tuple[int, int]
            Reference timepoint range
        filtered_lineages : list[tuple[str, list[int]]], optional
            Lineages to search in
        **kwargs
            Additional parameters for find_pattern_matches
            
        Returns
        -------
        DTWResult
            Analysis results
        """
        # Extract reference pattern
        reference_pattern = self.get_reference_pattern(fov_name, track_id, timepoints)
        
        # Find matches
        matches = self.find_pattern_matches(
            reference_pattern=reference_pattern,
            filtered_lineages=filtered_lineages,
            **kwargs
        )
        
        # Package results
        reference_info = {
            'fov_name': fov_name,
            'track_id': track_id,
            'timepoints': timepoints
        }
        
        return DTWResult(
            matches=matches,
            reference_pattern=reference_pattern,
            reference_info=reference_info
        )


def identify_lineages(
    tracking_df: pd.DataFrame, return_both_branches: bool = False
) -> list[tuple[str, list[int]]]:
    """Identify distinct lineages in cell tracking data.
    
    Parameters
    ----------
    tracking_df : pd.DataFrame
        Tracking dataframe with columns: fov_name, track_id, parent_track_id
    return_both_branches : bool
        If True, return both branches after division. If False, return only first branch.
        
    Returns
    -------
    list[tuple[str, list[int]]]
        List of (fov_name, track_ids) representing lineages
    """
    all_lineages = []
    
    # Group by FOV to handle repeated track_ids across FOVs
    for fov_id, fov_df in tracking_df.groupby("fov_name"):
        # Create parent-child mapping
        child_to_parent = {}
        for track_id, track_group in fov_df.groupby("track_id"):
            first_row = track_group.iloc[0]
            parent_track_id = first_row["parent_track_id"]
            if parent_track_id != -1:
                child_to_parent[track_id] = parent_track_id

        # Find root tracks
        all_tracks = set(fov_df["track_id"].unique())
        root_tracks = set()
        for track_id in all_tracks:
            track_data = fov_df[fov_df["track_id"] == track_id]
            if (
                track_data.iloc[0]["parent_track_id"] == -1
                or track_data.iloc[0]["parent_track_id"] not in all_tracks
            ):
                root_tracks.add(track_id)

        # Build parent-to-children mapping
        parent_to_children = {}
        for child, parent in child_to_parent.items():
            if parent not in parent_to_children:
                parent_to_children[parent] = []
            parent_to_children[parent].append(child)

        def get_all_branches(track_id):
            """Get all branches from a parent track."""
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

        # Build lineages from root tracks
        for root_track in root_tracks:
            lineage_tracks = get_all_branches(root_track)
            if return_both_branches:
                for branch in lineage_tracks:
                    all_lineages.append((fov_id, branch))
            else:
                all_lineages.append((fov_id, lineage_tracks[0]))

    return all_lineages


def find_pattern_matches(
    reference_pattern: np.ndarray,
    filtered_lineages: list[tuple[str, list[int]]],
    embeddings_dataset: xr.Dataset,
    window_step_fraction: float = 0.25,
    num_candidates: int = 3,
    max_distance: float = float("inf"),
    max_skew: float = 0.8,
    save_path: str | None = None,
    method: str = "bernd_clifford",
    normalize: bool = True,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Find best matches of a reference pattern in multiple lineages using DTW.
    
    Parameters
    ----------
    reference_pattern : np.ndarray
        Reference pattern embeddings
    filtered_lineages : list[tuple[str, list[int]]]
        List of lineages to search in (fov_name, track_ids)
    embeddings_dataset : xr.Dataset
        Dataset containing embeddings
    window_step_fraction : float
        Fraction of reference pattern length to use as window step
    num_candidates : int
        Number of best candidates to consider per lineage
    max_distance : float
        Maximum distance threshold to consider a match
    max_skew : float
        Maximum allowed path skewness (0-1, where 0=perfect diagonal)
    save_path : str, optional
        Path to save the results CSV
    method : str
        DTW method to use - 'bernd_clifford' or 'dtai'
    normalize : bool
        Whether to normalize DTW distance by path length
    metric : str
        Distance metric for computing distance matrix
        
    Returns
    -------
    pd.DataFrame
        Match results with distances and warping paths
    """
    # Calculate window step based on reference pattern length
    window_step = max(1, int(len(reference_pattern) * window_step_fraction))

    all_match_positions = {
        "fov_name": [],
        "track_ids": [],
        "distance": [],
        "skewness": [],
        "warp_path": [],
        "start_timepoint": [],
        "end_timepoint": [],
    }

    for fov_name, track_ids in tqdm(
        filtered_lineages, desc="Finding pattern matches"
    ):
        # Reconstruct the concatenated lineage
        lineages = []
        for track_id in track_ids:
            track_embeddings = embeddings_dataset.sel(
                sample=(fov_name, track_id)
            ).features.values
            lineages.append(track_embeddings)

        lineage_embeddings = np.concatenate(lineages, axis=0)

        # Find best matches using the selected DTW method
        if method == "bernd_clifford":
            matches_df = find_best_match_dtw_bernd_clifford(
                lineage_embeddings,
                reference_pattern=reference_pattern,
                num_candidates=num_candidates,
                window_step=window_step,
                max_distance=max_distance,
                max_skew=max_skew,
                normalize=normalize,
                metric=metric,
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
            )

        if not matches_df.empty:
            # Get the best match (first row)
            best_match = matches_df.iloc[0]
            best_pos = best_match["position"]
            best_path = best_match["path"]
            best_dist = best_match["distance"]
            best_skew = best_match["skewness"]

            all_match_positions["fov_name"].append(fov_name)
            all_match_positions["track_ids"].append(track_ids)
            all_match_positions["distance"].append(best_dist)
            all_match_positions["skewness"].append(best_skew)
            all_match_positions["warp_path"].append(best_path)
            all_match_positions["start_timepoint"].append(best_pos)
            all_match_positions["end_timepoint"].append(
                best_pos + len(reference_pattern)
            )
        else:
            # No matches found
            all_match_positions["fov_name"].append(fov_name)
            all_match_positions["track_ids"].append(track_ids)
            all_match_positions["distance"].append(None)
            all_match_positions["skewness"].append(None)
            all_match_positions["warp_path"].append(None)
            all_match_positions["start_timepoint"].append(None)
            all_match_positions["end_timepoint"].append(None)

    # Convert to DataFrame and drop rows with no matches
    all_match_positions = pd.DataFrame(all_match_positions)
    all_match_positions = all_match_positions.dropna()

    # Sort by distance (primary) and skewness (secondary)
    all_match_positions = all_match_positions.sort_values(
        by=["distance", "skewness"], ascending=[True, True]
    )

    # Save to CSV if path is provided
    if save_path:
        all_match_positions.to_csv(save_path, index=False)

    return all_match_positions


def find_best_match_dtw(
    lineage: np.ndarray,
    reference_pattern: np.ndarray,
    num_candidates: int = 5,
    window_step: int = 5,
    max_distance: float = float("inf"),
    max_skew: float = 0.8,
    normalize: bool = True,
) -> pd.DataFrame:
    """Find best matches using DTW with dtaidistance library.
    
    Parameters
    ----------
    lineage : np.ndarray
        The lineage to search (t, embeddings)
    reference_pattern : np.ndarray
        The pattern to search for (t, embeddings)
    num_candidates : int
        Number of candidates to return
    window_step : int
        Step size for sliding window
    max_distance : float
        Maximum distance threshold
    max_skew : float
        Maximum allowed path skewness (0-1)
    normalize : bool
        Whether to normalize distance by path length
        
    Returns
    -------
    pd.DataFrame
        Results with position, path, distance, and skewness
    """
    from dtaidistance.dtw_ndim import warping_path

    dtw_results = []
    n_windows = len(lineage) - len(reference_pattern) + 1

    if n_windows <= 0:
        return pd.DataFrame(columns=["position", "path", "distance", "skewness"])

    for i in range(0, n_windows, window_step):
        window = lineage[i : i + len(reference_pattern)]
        path, dist = warping_path(
            reference_pattern,
            window,
            include_distance=True,
        )
        if normalize:
            # Normalize by path length
            dist = dist / len(path)
        
        # Calculate skewness
        skewness = path_skew(path, len(reference_pattern), len(window))

        if dist <= max_distance and skewness <= max_skew:
            dtw_results.append(
                {"position": i, "path": path, "distance": dist, "skewness": skewness}
            )

    # Convert to DataFrame and sort
    results_df = pd.DataFrame(dtw_results)
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
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Find best matches using custom DTW implementation.
    
    Parameters
    ----------
    lineage : np.ndarray
        The lineage to search (t, embeddings)
    reference_pattern : np.ndarray
        The pattern to search for (t, embeddings)
    num_candidates : int
        Number of candidates to return
    window_step : int
        Step size for sliding window
    normalize : bool
        Whether to normalize distance by path length
    max_distance : float
        Maximum distance threshold
    max_skew : float
        Maximum allowed path skewness (0-1)
    metric : str
        Distance metric for computing distance matrix
        
    Returns
    -------
    pd.DataFrame
        Results with position, path, distance, and skewness
    """
    dtw_results = []
    n_windows = len(lineage) - len(reference_pattern) + 1

    if n_windows <= 0:
        return pd.DataFrame(columns=["position", "path", "distance", "skewness"])

    for i in range(0, n_windows, window_step):
        window = lineage[i : i + len(reference_pattern)]

        # Create distance matrix
        distance_matrix = cdist(reference_pattern, window, metric=metric)

        # Apply DTW
        distance, _, path = dtw_with_matrix(distance_matrix, normalize=normalize)

        # Calculate skewness
        skewness = path_skew(path, len(reference_pattern), len(window))

        # Only add if both thresholds are met
        if distance <= max_distance and skewness <= max_skew:
            dtw_results.append(
                {
                    "position": i,
                    "path": path,
                    "distance": distance,
                    "skewness": skewness,
                }
            )

    # Convert to DataFrame and sort
    results_df = pd.DataFrame(dtw_results)
    if not results_df.empty:
        results_df = results_df.sort_values(by=["distance", "skewness"]).head(
            num_candidates
        )

    return results_df


def compute_dtw_distance(
    s1: ArrayLike, s2: ArrayLike, metric: Literal["cosine", "euclidean"] = "cosine"
) -> Tuple[float, float]:
    """Compute DTW distance between two embedding sequences.
    
    Parameters
    ----------
    s1 : ArrayLike
        First embedding sequence
    s2 : ArrayLike  
        Second embedding sequence
    metric : Literal["cosine", "euclidean"]
        Distance metric to use
        
    Returns
    -------
    Tuple[float, float]
        DTW distance and path skewness
    """
    # Create distance matrix
    distance_matrix = cdist(s1, s2, metric=metric)

    # Compute DTW
    warping_distance, _, dtw_path = dtw_with_matrix(distance_matrix, normalize=True)
    
    # Compute path skewness
    skewness = path_skew(dtw_path, len(s1), len(s2))

    return warping_distance, skewness


def dtw_with_matrix(distance_matrix: np.ndarray, normalize: bool = True) -> Tuple[float, np.ndarray, list]:
    """Compute DTW using a pre-computed distance matrix.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Pre-computed distance matrix between two sequences
    normalize : bool
        Whether to normalize the distance by path length
        
    Returns
    -------
    Tuple[float, np.ndarray, list]
        DTW distance, warping matrix, and optimal warping path
    """
    n, m = distance_matrix.shape

    # Initialize accumulated cost matrix
    warping_matrix = np.full((n, m), np.inf)
    warping_matrix[0, 0] = distance_matrix[0, 0]

    # Fill first column and row
    for i in range(1, n):
        warping_matrix[i, 0] = warping_matrix[i - 1, 0] + distance_matrix[i, 0]
    for j in range(1, m):
        warping_matrix[0, j] = warping_matrix[0, j - 1] + distance_matrix[0, j]

    # Fill the rest of the matrix
    for i in range(1, n):
        for j in range(1, m):
            warping_matrix[i, j] = distance_matrix[i, j] + min(
                warping_matrix[i - 1, j],      # insertion
                warping_matrix[i, j - 1],      # deletion
                warping_matrix[i - 1, j - 1],  # match
            )

    # Backtrack to find optimal path
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

    # Get DTW distance
    dtw_distance = warping_matrix[n - 1, m - 1]

    # Normalize by path length if requested
    if normalize:
        dtw_distance = dtw_distance / len(path)

    return dtw_distance, warping_matrix, path


def path_skew(warping_path: list, ref_len: int, query_len: int) -> float:
    """Calculate skewness of a DTW warping path.
    
    Parameters
    ----------
    warping_path : list
        List of (ref_idx, query_idx) tuples representing the warping path
    ref_len : int
        Length of the reference sequence
    query_len : int
        Length of the query sequence
        
    Returns
    -------
    float
        Skewness metric between 0 and 1, where 0 means perfectly diagonal path
        and 1 means completely skewed path
    """
    # Calculate "ideal" diagonal indices
    diagonal_x = np.linspace(0, ref_len - 1, len(warping_path))
    diagonal_y = np.linspace(0, query_len - 1, len(warping_path))
    diagonal_path = np.column_stack((diagonal_x, diagonal_y))

    # Calculate distances from points on the warping path to the diagonal
    max_distance = max(ref_len, query_len)

    distances = []
    for i, (x, y) in enumerate(warping_path):
        # Find the closest point on the diagonal
        dx, dy = diagonal_path[i]
        # Euclidean distance
        dist = np.sqrt((x - dx) ** 2 + (y - dy) ** 2)
        distances.append(dist)

    # Average normalized distance as skewness metric
    skew = np.mean(distances) / max_distance

    return skew