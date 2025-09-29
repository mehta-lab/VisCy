import logging
from pathlib import Path
from typing import Literal, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing_extensions import TypedDict

_logger = logging.getLogger("lightning.pytorch")


# Annotated Example TypeDict
class AnnotatedSample(TypedDict):
    fov_name: str
    track_id: int | list[int]
    timepoints: tuple[int, int]
    annotations: dict | list
    weight: float


class DtwSample(TypedDict, total=False):
    pattern: np.ndarray
    annotations: list[str, int, float] | None
    distance: float
    skewness: float
    warping_path: list[tuple[int, int]]
    metadata: dict


class CytoDtw:
    def __init__(self, adata: ad.AnnData):
        """
        DTW for Dynamic Cell Embeddings

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object containing:
            - X: features/embeddings
            - obs: tracking info (fov_name, track_id, t, x, y, etc.) and annotations
            - obsm: multi-dimensional embeddings (X_PCA, X_UMAP, etc.)
        """
        self.adata = adata
        self.lineages = None
        self.consensus_data = None
        self.reference_patterns = None

    def _validate_input(self):
        raise NotImplementedError("Validation of input not implemented")
    
    def save_consensus(self, path: str):
        """Save consensus pattern to a file."""
        import pickle
        if self.consensus_data is None:
            raise ValueError("Consensus pattern not found")
        with open(path, 'wb') as f:
            pickle.dump(self.consensus_data, f)

    def load_consensus(self, path: str):
        """Load consensus pattern from a file."""
        import pickle
        with open(path, 'rb') as f:
            self.consensus_data = pickle.load(f)


    def get_lineages(self, min_timepoints: int = 15) -> list[tuple[str, list[int]]]:
        """Get identified lineages with specified minimum timepoints."""
        return self._identify_lineages(min_timepoints)

    def _identify_lineages(
        self, min_timepoints: int = 15
    ) -> list[tuple[str, list[int]]]:
        """Auto-identify lineages from the data."""
        # Use parent_track_id if available for proper lineage identification
        if "parent_track_id" in self.adata.obs.columns:
            all_lineages = identify_lineages(
                self.adata.obs, return_both_branches=False
            )
        else:
            # Fallback: treat each track as individual lineage
            all_lineages = []
            for (fov, track_id), group in self.adata.obs.groupby(
                ["fov_name", "track_id"]
            ):
                all_lineages.append((fov, [track_id]))

        # Filter lineages by total timepoints across all tracks in lineage
        filtered_lineages = []
        for fov_id, track_ids in all_lineages:
            lineage_rows = self.adata.obs[
                (self.adata.obs["fov_name"] == fov_id)
                & (self.adata.obs["track_id"].isin(track_ids))
            ]
            total_timepoints = len(lineage_rows)
            if total_timepoints >= min_timepoints:
                filtered_lineages.append((fov_id, track_ids))
        self.lineages = filtered_lineages
        return self.lineages

    def get_reference_pattern(
        self,
        fov_name: str,
        track_id: int | list[int],
        timepoints: tuple[int, int],
        reference_type: str = "features",
    ) -> np.ndarray:
        """
        Extract reference pattern from embeddings.

        Parameters
        ----------
        fov_name : str
            FOV identifier
        track_id : int | list[int]
            Track ID(s) to use as reference
        timepoints : tuple[int, int]
            Start and end timepoints (start, end)
        reference_type : str
            Type of embedding to use ('features' for X, or obsm key like 'X_PCA')
        Returns
        -------
        np.ndarray
            Reference pattern embeddings
        """
        if isinstance(track_id, int):
            track_id = [track_id]

        reference_embeddings = []
        for tid in track_id:
            # Filter by fov_name and track_id
            mask = (self.adata.obs['fov_name'] == fov_name) & (self.adata.obs['track_id'] == tid)
            track_data = self.adata[mask]
            
            # Sort by timepoint to ensure correct order
            time_order = np.argsort(track_data.obs['t'].values)
            track_data = track_data[time_order]
            
            if reference_type == "features":
                track_emb = track_data.X
            else:
                # Assume it's an obsm key
                track_emb = track_data.obsm[reference_type]

            # Handle 1D arrays (PC components) by reshaping to (time, 1)
            if track_emb.ndim == 1:
                track_emb = track_emb.reshape(-1, 1)

            reference_embeddings.append(track_emb)

        reference_pattern = np.concatenate(reference_embeddings, axis=0)

        start_t, end_t = timepoints
        reference_pattern = reference_pattern[start_t:end_t]

        return reference_pattern

    def get_matches(
        self,
        reference_pattern: np.ndarray = None,
        lineages: list[tuple[str, list[int]]] = None,
        window_step: int = 5,
        num_candidates: int | None = None,
        max_distance: float = float("inf"),
        max_skew: float = 0.8,
        method: str = "bernd_clifford",
        normalize: bool = True,
        metric: str = "euclidean",
        reference_type: str = "features",
        constraint_type: str = "unconstrained",
        band_width_ratio: float = 0.0,
        save_path: str | Path = None,
    ) -> pd.DataFrame:
        """Find pattern matches across lineages using DTW.

        Parameters
        ----------
        reference_pattern : np.ndarray
            Reference pattern to search for
        lineages : list[tuple[str, list[int]]], optional
            List of (fov_name, track_ids) to search in. If None, searches all.
        window_step : int
            Step size for sliding window search
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
        if reference_pattern is None:
            reference_pattern = self.consensus_data["pattern"]
        if lineages is None:
            # FIXME: Auto-identify lineages from tracking data
            lineages = self.get_lineages()

        return find_pattern_matches(
            reference_pattern=reference_pattern,
            filtered_lineages=lineages,
            adata=self.adata,
            window_step=window_step,
            num_candidates=num_candidates,
            max_distance=max_distance,
            max_skew=max_skew,
            method=method,
            normalize=normalize,
            metric=metric,
            reference_type=reference_type,
            constraint_type=constraint_type,
            band_width_ratio=band_width_ratio,
            save_path=save_path,
        )

    def create_consensus_reference_pattern(
        self,
        annotated_samples: list[AnnotatedSample],
        reference_selection: str = "median_length",
        aggregation_method: str = "mean",
        annotations_name: str = "annotations",
        reference_type: str = "features",
        **kwargs,
    ) -> DtwSample:
        """
        Create consensus reference pattern from multiple annotated samples.

        This method takes multiple manually annotated cell examples and creates a
        consensus reference pattern by aligning them with DTW and aggregating.

        Parameters
        ----------
        annotated_samples : list[AnnotatedSample]
            List of annotated examples
        reference_selection : str
            mode of selection of reference: "median_length", "first", "longest", "shortest"
        aggregation_method : str
            mode of aggregation: "mean", "median", "weighted_mean"
        annotations_name : str
            name of the annotations column
        reference_type : str
            Type of embedding to use ("features", "projections", "PC1", etc.)
        Returns
        -------
        DtwSample
            DtwSample containing:
            - 'pattern': np.ndarray - The consensus embedding pattern
            - 'annotations': list - Consensus annotations (if available)
            - 'metadata': dict - Information about consensus creation including method used
            - 'distance': float - DTW distance
            - 'skewness': float - Path skewness
            - 'warping_path': list - DTW warping path

        Examples
        --------
        >>> analyzer = CytoDtw("embeddings.zarr")
        >>> examples = [
        ...     AnnotatedSample(
        ...         'fov_name': '/FOV1', 'track_id': 129,
        ...         'timepoints': (8, 70), 'annotations': ['G1', 'S', 'G2', ...]
        ...     ),
        ...     AnnotatedSample(
        ...         'fov_name': '/FOV2', 'track_id': 45,
        ...         'timepoints': (5, 55), 'weight': 1.2
        ...     )
        ... ]
        >>> consensus = analyzer.create_consensus_reference_pattern(examples)
        """
        if not annotated_samples:
            raise ValueError("No annotated examples provided")

        # Extract embedding patterns from each example
        extracted_patterns = {}
        for i, example in enumerate(annotated_samples):
            pattern = self.get_reference_pattern(
                fov_name=example["fov_name"],
                track_id=example["track_id"],
                timepoints=example["timepoints"],
                reference_type=reference_type,
            )

            extracted_patterns[f"example_{i}"] = {
                "pattern": pattern,
                "annotations": example.get(annotations_name, None),
                "weight": example.get("weight", 1.0),
                "source": example,
            }

        self.consensus_data = create_consensus_from_patterns(
            patterns=extracted_patterns,
            reference_selection=reference_selection,
            aggregation_method=aggregation_method,
            **kwargs,
        )
        return self.consensus_data

    def create_enhanced_alignment_dataframe(
        self,
        top_matches: pd.DataFrame,
        consensus_lineage: np.ndarray,
        alignment_name: str = "cell_division",
        reference_type: str = "features"
    ) -> pd.DataFrame:
        """
        Create enhanced DataFrame that:
        1. Preserves lineage relationships (lineage_id groups related tracks)
        2. Supports multiple alignment types (cell_division, apoptosis, migration, etc.)
        3. Stores computed features once, reused across alignments
        4. Maintains original timepoint relationships

        Parameters
        ----------
        top_matches : pd.DataFrame
            DTW match results
        consensus_lineage : np.ndarray
            Consensus pattern for this alignment
        alignment_name : str
            Name for this alignment type (e.g., "cell_division", "apoptosis")
        reference_type : str
            Feature type to use

        Returns
        -------
        pd.DataFrame
            Enhanced DataFrame with lineage preservation and extensible alignments
        """
        enhanced_data = []
        track_lineage_mapping = {}  # Map track_id to lineage_id

        # Check what features are available in obsm
        available_obsm = list(self.adata.obsm.keys())
        # Detect all PC components dynamically - look for X_PCA or individual PC columns in obs
        has_pca_obsm = 'X_PCA' in available_obsm
        pc_components = []
        
        if has_pca_obsm:
            # Use X_PCA from obsm
            n_pca_components = self.adata.obsm['X_PCA'].shape[1]
            pc_components = [f'PC{i+1}' for i in range(n_pca_components)]
        else:
            # Look for individual PC columns in obs
            pc_components = [col for col in self.adata.obs.columns if col.startswith('PC') and col[2:].isdigit()]
            pc_components.sort(key=lambda x: int(x[2:]))  # Sort PC1, PC2, PC3, etc.
        
        has_pc_components = len(pc_components) > 0
        
        # Build lineage mapping
        lineage_counter = 0
        for idx, match_row in top_matches.iterrows():
            fov_name = match_row['fov_name']
            track_ids = match_row['track_ids']

            # Assign lineage ID (tracks in same match belong to same lineage)
            lineage_id = lineage_counter
            lineage_counter += 1

            for track_id in track_ids:
                track_lineage_mapping[(fov_name, track_id)] = lineage_id

        # If no PC components available, compute PCA
        pca = None
        scaler = None
        consensus_pca = None
        
        if not has_pc_components:
            all_embeddings = []
            
            # Collect all embeddings for PCA
            for idx, match_row in top_matches.iterrows():
                fov_name = match_row['fov_name']
                track_ids = match_row['track_ids']
                
                for track_id in track_ids:
                    try:
                        # Filter by fov_name and track_id
                        mask = (self.adata.obs['fov_name'] == fov_name) & (self.adata.obs['track_id'] == track_id)
                        track_data = self.adata[mask]
                        
                        # Sort by timepoint to ensure correct order
                        time_order = np.argsort(track_data.obs['t'].values)
                        track_data = track_data[time_order]
                        
                        if reference_type == "features":
                            track_embeddings = track_data.X
                        else:
                            track_embeddings = track_data.obsm[reference_type]
                        
                        all_embeddings.append(track_embeddings)
                    except KeyError:
                        continue

            # Include consensus in PCA fit
            all_embeddings.append(consensus_lineage)
            all_concat = np.vstack(all_embeddings)

            # Fit PCA for consistent feature computation
            # Use min of 8 components or number of features available
            n_components = min(8, all_concat.shape[1])
            scaler = StandardScaler()
            scaled_all = scaler.fit_transform(all_concat)
            pca = PCA(n_components=n_components)
            pca_all = pca.fit_transform(scaled_all)
            
            # Get consensus PCA
            consensus_pca = pca_all[-len(consensus_lineage):]

        # Process each matched track to create DataFrame rows
        pca_start_idx = 0

        for idx, match_row in top_matches.iterrows():
            fov_name = match_row['fov_name']
            track_ids = match_row['track_ids']
            warp_path = match_row['warp_path']
            dtw_distance = match_row.get('distance', np.nan)

            # Create mapping from query timepoint to consensus timepoint
            query_to_consensus = {}
            for consensus_idx, query_timepoint in warp_path:
                query_to_consensus[query_timepoint] = consensus_idx

            # Process each track in this lineage
            for track_id in track_ids:
                try:
                    # Filter by fov_name and track_id
                    mask = (self.adata.obs['fov_name'] == fov_name) & (self.adata.obs['track_id'] == track_id)
                    track_data = self.adata[mask]
                    
                    # Sort by timepoint to ensure correct order
                    time_order = np.argsort(track_data.obs['t'].values)
                    track_data = track_data[time_order]
                    
                    track_timepoints = track_data.obs['t'].values

                    # Get PC features - either from obsm/obs or computed PCA
                    pc_values = {}
                    if has_pc_components:
                        if has_pca_obsm:
                            # Extract from X_PCA obsm
                            pca_data = track_data.obsm['X_PCA']
                            for i, pc_name in enumerate(pc_components):
                                pc_values[pc_name] = pca_data[:, i]
                        else:
                            # Extract from individual PC columns in obs
                            for pc_name in pc_components:
                                pc_values[pc_name] = track_data.obs[pc_name].values
                    else:
                        # Use computed PCA
                        if reference_type == "features":
                            track_embeddings = track_data.X
                        else:
                            track_embeddings = track_data.obsm[reference_type]
                        scaled_embeddings = scaler.transform(track_embeddings)
                        track_pca = pca.transform(scaled_embeddings)
                        # Create PC values for all computed components
                        for i in range(n_components):
                            pc_name = f'PC{i+1}'
                            pc_values[pc_name] = track_pca[:, i]

                    # Get lineage ID for this track
                    lineage_id = track_lineage_mapping.get((fov_name, track_id), -1)

                    # Create row for each timepoint
                    for i, t in enumerate(track_timepoints):
                        # Get spatial coordinates from track_data.obs (which is already filtered)
                        obs_row = track_data.obs.iloc[i]
                        x_coord = obs_row.get('x', np.nan)
                        y_coord = obs_row.get('y', np.nan)

                        # Determine alignment status for this specific alignment type
                        is_aligned = t in query_to_consensus
                        consensus_mapping = query_to_consensus.get(t, np.nan)

                        # Create dynamic column names based on alignment_name
                        row_data = {
                            # Core tracking info (preserves lineage relationships)
                            'fov_name': fov_name,
                            'lineage_id': lineage_id,
                            'track_id': track_id,
                            't': t,
                            'x': x_coord,
                            'y': y_coord,

                            # Alignment-specific columns (dynamic based on alignment_name)
                            f'dtw_{alignment_name}_consensus_mapping': consensus_mapping,
                            f'dtw_{alignment_name}_aligned': is_aligned,
                            f'dtw_{alignment_name}_distance': dtw_distance,
                            f'dtw_{alignment_name}_match_rank': idx,

                            # Placeholders for image-based features (to be computed later)
                            'mean_intensity': np.nan,
                            'cell_area': np.nan,
                            'cell_perimeter': np.nan,
                            'cell_eccentricity': np.nan,
                        }
                        
                        # Add all PC components dynamically
                        for pc_name, pc_vals in pc_values.items():
                            row_data[pc_name] = pc_vals[i]
                        enhanced_data.append(row_data)

                except KeyError as e:
                    _logger.warning(f"Could not find track {track_id} in FOV {fov_name}: {e}")
                    continue

        # Add consensus pattern as reference (special lineage_id = -1)
        consensus_pc_values = {}
        if has_pc_components:
            # If PC components exist, we'd need the consensus to have them too
            # For now, set to NaN - this might need adjustment based on how consensus is created
            for pc_name in pc_components:
                consensus_pc_values[pc_name] = [np.nan] * len(consensus_lineage)
        else:
            # Use computed PCA for consensus
            for i in range(n_components):
                pc_name = f'PC{i+1}'
                consensus_pc_values[pc_name] = consensus_pca[:, i]

        for i in range(len(consensus_lineage)):
            consensus_row = {
                'fov_name': 'consensus',
                'lineage_id': -1,  # Special lineage for consensus
                'track_id': -1,
                't': i,
                'x': np.nan,
                'y': np.nan,
                f'dtw_{alignment_name}_consensus_mapping': i,  # Maps to itself
                f'dtw_{alignment_name}_aligned': True,
                f'dtw_{alignment_name}_distance': 0.0,
                f'dtw_{alignment_name}_match_rank': -1,
                'mean_intensity': np.nan,
                'cell_area': np.nan,
                'cell_perimeter': np.nan,
                'cell_eccentricity': np.nan,
            }
            
            # Add all PC components dynamically for consensus
            for pc_name, pc_vals in consensus_pc_values.items():
                consensus_row[pc_name] = pc_vals[i]
                
            enhanced_data.append(consensus_row)

        return pd.DataFrame(enhanced_data)


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
    adata: ad.AnnData,
    window_step: int = 5,
    num_candidates: int = 3,
    max_distance: float = float("inf"),
    max_skew: float = 0.8,
    save_path: str | None = None,
    method: str = "bernd_clifford",
    normalize: bool = True,
    metric: str = "euclidean",
    reference_type: Literal[
        "features", "X_PCA", "X_UMAP", "X_PHATE"
    ] = "features",
    constraint_type: str = "unconstrained",
    band_width_ratio: float = 0.0,
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
    window_step : int
        Step size for sliding window search
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
    reference_type : str
        Type of embedding to use for reference pattern
    Returns
    -------
    pd.DataFrame
        Match results with distances and warping paths
    """
    # Use window_step directly as step size
    window_step = max(1, window_step)

    all_match_positions = {
        "fov_name": [],
        "track_ids": [],
        "distance": [],
        "skewness": [],
        "warp_path": [],
        "start_track_timepoint": [],
        "end_track_timepoint": [],
    }

    for fov_name, track_ids in tqdm(filtered_lineages, desc="Finding pattern matches"):
        lineages = []
        t_values = []
        for track_id in track_ids:
            # Filter by fov_name and track_id
            mask = (adata.obs['fov_name'] == fov_name) & (adata.obs['track_id'] == track_id)
            track_data = adata[mask]
            
            # Sort by timepoint to ensure correct order
            time_order = np.argsort(track_data.obs['t'].values)
            track_data = track_data[time_order]
            
            if reference_type == "features":
                track_embeddings = track_data.X
            else:
                # Assume it's an obsm key
                track_embeddings = track_data.obsm[reference_type]
            
            track_t = track_data.obs['t'].values

            # Handle 1D arrays (PC components) by reshaping to (time, 1)
            if track_embeddings.ndim == 1:
                track_embeddings = track_embeddings.reshape(-1, 1)

            lineages.append(track_embeddings)
            t_values.extend(track_t)  # Add t values to our mapping

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
                constraint_type=constraint_type,
                band_width_ratio=band_width_ratio,
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
                constraint_type=constraint_type,
                band_width_ratio=band_width_ratio,
            )

        if not matches_df.empty:
            # Get the best match (first row)
            best_match = matches_df.iloc[0]
            best_pos = best_match["position"]
            best_path = best_match["path"]
            best_dist = best_match["distance"]
            best_skew = best_match["skewness"]

            #warping path is relative to the reference pattern
            #query_idx is relative to the lineage
            converted_path = []
            for ref_idx, query_idx in best_path:
                query_t_idx = best_pos + query_idx
                if query_t_idx < len(t_values):
                    actual_t = t_values[query_t_idx]
                    converted_path.append((ref_idx, actual_t))
            
            start_t = t_values[best_pos] if best_pos < len(t_values) else None
            end_pos = best_pos + len(reference_pattern) - 1
            end_t = t_values[end_pos] if end_pos < len(t_values) else None

            all_match_positions["fov_name"].append(fov_name)
            all_match_positions["track_ids"].append(track_ids)
            all_match_positions["distance"].append(best_dist)
            all_match_positions["skewness"].append(best_skew)
            all_match_positions["warp_path"].append(converted_path)
            all_match_positions["start_track_timepoint"].append(start_t)
            all_match_positions["end_track_timepoint"].append(end_t)
        else:
            # No matches found
            all_match_positions["fov_name"].append(fov_name)
            all_match_positions["track_ids"].append(track_ids)
            all_match_positions["distance"].append(None)
            all_match_positions["skewness"].append(None)
            all_match_positions["warp_path"].append(None)
            all_match_positions["start_track_timepoint"].append(None)
            all_match_positions["end_track_timepoint"].append(None)

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
    constraint_type: str = "unconstrained",
    band_width_ratio: float = 0.0,
) -> pd.DataFrame:
    """Find best matches using DTW with dtaidistance library.

    Note: constraint_type and band_width_ratio are ignored for dtaidistance method.

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
    constraint_type : str
        Ignored for this method
    band_width_ratio : float
        Ignored for this method

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
    constraint_type: str = "unconstrained",
    band_width_ratio: float = 0.0,
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
        distance, _, path = dtw_with_matrix(
            distance_matrix,
            normalize=normalize,
            constraint_type=constraint_type,
            band_width_ratio=band_width_ratio,
        )

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
    s1: ArrayLike,
    s2: ArrayLike,
    metric: Literal["cosine", "euclidean"] = "cosine",
    constraint_type: str = "unconstrained",
    band_width_ratio: float = None,
) -> dict:
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
    dict
        - 'distance': float - DTW distance
        - 'skewness': float - Path skewness
        - 'warping_path': list - Warping path
    """
    # Create distance matrix
    distance_matrix = cdist(s1, s2, metric=metric)

    # Compute DTW
    dtw_distance, _, warping_path = dtw_with_matrix(
        distance_matrix,
        normalize=True,
        constraint_type=constraint_type,
        band_width_ratio=band_width_ratio,
    )

    # Compute path skewness
    skewness = path_skew(warping_path, len(s1), len(s2))

    return {
        "distance": dtw_distance,
        "skewness": skewness,
        "warping_path": warping_path,
    }


def dtw_with_matrix(
    distance_matrix: np.ndarray,
    normalize: bool = True,
    constraint_type: str = "unconstrained",
    band_width_ratio: float = 0.0,
) -> Tuple[float, np.ndarray, list]:
    """Compute DTW using a pre-computed distance matrix with constraints.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Pre-computed distance matrix between two sequences
    normalize : bool
        Whether to normalize the distance by path length
    constraint_type : str
        Type of constraint: "sakoe_chiba", "unconstrained"
    band_width_ratio : float
        Ratio of matrix size for Sakoe-Chiba band constraint

    Returns
    -------
    Tuple[float, np.ndarray, list]
        DTW distance, warping matrix, and optimal warping path
    """
    n, m = distance_matrix.shape
    warping_matrix = np.full((n, m), np.inf)

    if constraint_type == "sakoe_chiba":
        # Sakoe-Chiba band constraint
        band_width = int(max(n, m) * band_width_ratio)

        for i in range(n):
            for j in range(m):
                # Only allow alignment within the band
                diagonal_position = j * n / m
                if abs(i - diagonal_position) <= band_width:
                    if i == 0 and j == 0:
                        warping_matrix[i, j] = distance_matrix[i, j]
                    elif i == 0 and j > 0 and warping_matrix[i, j - 1] != np.inf:
                        warping_matrix[i, j] = (
                            warping_matrix[i, j - 1] + distance_matrix[i, j]
                        )
                    elif j == 0 and i > 0 and warping_matrix[i - 1, j] != np.inf:
                        warping_matrix[i, j] = (
                            warping_matrix[i - 1, j] + distance_matrix[i, j]
                        )
                    elif i > 0 and j > 0:
                        candidates = []
                        if warping_matrix[i - 1, j] != np.inf:
                            candidates.append(warping_matrix[i - 1, j])
                        if warping_matrix[i, j - 1] != np.inf:
                            candidates.append(warping_matrix[i, j - 1])
                        if warping_matrix[i - 1, j - 1] != np.inf:
                            candidates.append(warping_matrix[i - 1, j - 1])

                        if candidates:
                            warping_matrix[i, j] = distance_matrix[i, j] + min(
                                candidates
                            )
    else:
        # Unconstrained DTW
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
                    warping_matrix[i - 1, j],  # insertion
                    warping_matrix[i, j - 1],  # deletion
                    warping_matrix[i - 1, j - 1],  # match
                )

    # Backtrack to find optimal path
    i, j = n - 1, m - 1
    warping_path = [(i, j)]

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

        warping_path.append((i, j))

    warping_path.reverse()

    dtw_distance = warping_matrix[n - 1, m - 1]

    if normalize:
        dtw_distance = dtw_distance / len(warping_path)

    return dtw_distance, warping_matrix, warping_path


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

    max_distance = max(ref_len, query_len)

    distances = []
    for i, (x, y) in enumerate(warping_path):
        dx, dy = diagonal_path[i]
        dist = np.sqrt((x - dx) ** 2 + (y - dy) ** 2)
        distances.append(dist)

    skew = np.mean(distances) / max_distance

    return skew


def create_consensus_from_patterns(
    patterns: dict[str, dict],
    reference_selection: str = "median_length",
    aggregation_method: str = "mean",
    metric: Literal["cosine", "euclidean"] = "cosine",
    constraint_type: str = "unconstrained",
    band_width_ratio: float = 0.0,
) -> dict:
    """Create consensus pattern from multiple embedding patterns using DTW alignment.

    This function is compatible with CytoDtw workflow and creates consensus
    patterns from embedding vectors extracted from different cell examples.

    Parameters
    ----------
    patterns : dict[str, dict]
        Dictionary where keys are pattern identifiers and values contain:
        - 'pattern': np.ndarray - The embedding pattern (time, features)
        - 'annotations': list or dict - Optional annotations/labels
        - 'weight': float - Optional weight for this pattern (default 1.0)
        - Other metadata fields are preserved
    reference_selection : str
        How to select reference: "median_length", "first", "longest", "shortest"
    aggregation_method : str
        How to aggregate: "mean", "median", "weighted_mean"
    metric: Literal["cosine", "euclidean"]
        Distance metric for DTW alignment

    Returns
    -------
    dict
        Dictionary containing:
        - 'pattern': np.ndarray - The consensus embedding pattern
        - 'annotations': list - Consensus annotations (if available)
        - 'metadata': dict - Information about the consensus creation process
    """
    if not patterns:
        raise ValueError("No patterns provided")

    for pattern_id, pattern_data in patterns.items():
        if "pattern" not in pattern_data:
            raise ValueError(f"Pattern '{pattern_id}' missing 'pattern' key")
        if not isinstance(pattern_data["pattern"], np.ndarray):
            raise ValueError(f"Pattern '{pattern_id}' must be numpy array")

    reference_id = _select_reference_pattern(patterns, reference_selection)
    reference_pattern = patterns[reference_id]["pattern"]

    _logger.debug(f"Selected reference pattern: {reference_id}")
    _logger.debug(f"Reference shape: {reference_pattern.shape}")

    reference_pattern = patterns[reference_id]["pattern"]
    aligned_patterns = {reference_id: patterns[reference_id]}

    for pattern_id, pattern_data in patterns.items():
        if pattern_id == reference_id:
            continue  # Skip reference

        query_pattern = pattern_data["pattern"]
        alignment_result = align_embedding_patterns(
            query_pattern,
            reference_pattern,
            metric=metric,
            query_annotations=pattern_data.get("annotations"),
            constraint_type=constraint_type,
            band_width_ratio=band_width_ratio,
        )
        aligned_data = {
            "pattern": alignment_result["pattern"],
            "annotations": alignment_result["annotations"],
            "weight": pattern_data.get("weight", 1.0),
            "dtw_distance": alignment_result["distance"],
            "dtw_skewness": alignment_result["skewness"],
            "alignment_path": alignment_result["warping_path"],
        }
        # Copy other metadata
        for key, value in pattern_data.items():
            if key not in ["pattern", "annotations", "weight"]:
                aligned_data[key] = value

        aligned_patterns[pattern_id] = aligned_data
    consensus = _aggregate_aligned_patterns(aligned_patterns, aggregation_method)

    consensus = DtwSample(
        pattern=consensus["pattern"],
        annotations=consensus["annotations"],
        distance=alignment_result["distance"],
        skewness=alignment_result["skewness"],
        warping_path=alignment_result["warping_path"],
    )

    consensus["metadata"] = {
        "reference_pattern": reference_id,
        "source_patterns": list(patterns.keys()),
        "reference_selection": reference_selection,
        "aggregation_method": aggregation_method,
        "n_patterns": len(patterns),
    }

    return consensus


def _select_reference_pattern(patterns: dict, method: str) -> str:
    """Select which pattern to use as reference for DTW alignment."""
    if method == "first":
        return list(patterns.keys())[0]

    elif method == "median_length":
        lengths = {pid: len(pdata["pattern"]) for pid, pdata in patterns.items()}
        median_length = np.median(list(lengths.values()))
        closest_id = min(lengths.keys(), key=lambda x: abs(lengths[x] - median_length))
        return closest_id

    elif method == "longest":
        lengths = {pid: len(pdata["pattern"]) for pid, pdata in patterns.items()}
        return max(lengths.keys(), key=lambda x: lengths[x])

    elif method == "shortest":
        lengths = {pid: len(pdata["pattern"]) for pid, pdata in patterns.items()}
        return min(lengths.keys(), key=lambda x: lengths[x])

    else:
        raise ValueError(f"Unknown reference selection method: {method}")


def align_embedding_patterns(
    query_pattern: np.ndarray,
    reference_pattern: np.ndarray,
    metric: str = "cosine",
    query_annotations: list = None,
    constraint_type: str = "unconstrained",
    band_width_ratio: float = 0.0,
) -> DtwSample:
    """Align two embedding patterns using DTW.

    This is a modular function that aligns two embedding sequences (T, ndim)
    using DTW and returns comprehensive alignment information.

    Parameters
    ----------
    query_pattern : np.ndarray
        Query embedding pattern with shape (T1, ndim)
    reference_pattern : np.ndarray
        Reference embedding pattern with shape (T2, ndim)
    metric : str
        Distance metric for DTW alignment
    query_annotations : list, optional
        Optional annotations for query pattern to align alongside the embeddings

    Returns
    -------
    DtwSample
    """

    dtw_result = compute_dtw_distance(
        query_pattern,
        reference_pattern,
        metric=metric,
        constraint_type=constraint_type,
        band_width_ratio=band_width_ratio,
    )

    # Apply warping path once for both pattern and annotations
    aligned_query, aligned_annotations = _apply_warping_path(
        query_pattern=query_pattern,
        reference_pattern=reference_pattern,
        warping_path=dtw_result["warping_path"],
        query_annotations=query_annotations,
    )

    return DtwSample(
        pattern=aligned_query,
        annotations=aligned_annotations,
        distance=dtw_result["distance"],
        skewness=dtw_result["skewness"],
        warping_path=dtw_result["warping_path"],
    )


def _apply_warping_path(
    query_pattern: np.ndarray,
    reference_pattern: np.ndarray,
    warping_path: list[tuple[int, int]],
    query_annotations: list = None,
) -> tuple[np.ndarray, list]:
    """Apply DTW warping path to align query pattern to reference pattern.

    This is a modular helper function that applies a DTW warping path
    to align embedding patterns and their annotations.

    Parameters
    ----------
    query_pattern : np.ndarray
        Query embedding pattern to be aligned (time, features)
    reference_pattern : np.ndarray
        Reference pattern to align to (time, features)
    warping_path : list[tuple[int, int]]
        DTW warping path as list of (query_idx, ref_idx) tuples
    query_annotations : list, optional
        Optional annotations for query pattern

    Returns
    -------
    tuple[np.ndarray, list]
        Aligned pattern and aligned annotations (if provided)
    """
    ref_length, n_features = reference_pattern.shape
    aligned_pattern = np.zeros_like(reference_pattern)

    # Apply warping path to align the embedding vectors
    for query_idx, ref_idx in warping_path:
        if ref_idx < ref_length and query_idx < len(query_pattern):
            aligned_pattern[ref_idx] = query_pattern[query_idx]

    # Align annotations if present
    aligned_annotations = None
    if query_annotations is not None:
        aligned_annotations = ["Unknown"] * ref_length
        for query_idx, ref_idx in warping_path:
            if ref_idx < ref_length and query_idx < len(query_annotations):
                aligned_annotations[ref_idx] = query_annotations[query_idx]

    return aligned_pattern, aligned_annotations


def _aggregate_aligned_patterns(
    aligned_patterns: DtwSample, method: Literal["mean", "median", "weighted_mean"]
) -> DtwSample:
    """Aggregate aligned embedding patterns into consensus."""
    consensus = {}

    # Extract patterns and weights
    pattern_arrays = []
    weights = []

    for pattern_data in aligned_patterns.values():
        pattern_arrays.append(pattern_data["pattern"])
        weights.append(pattern_data.get("weight", 1.0))

    pattern_arrays = np.array(pattern_arrays)
    weights = np.array(weights)

    # Aggregate embedding patterns
    if method == "mean":
        pattern = np.mean(pattern_arrays, axis=0)
    elif method == "median":
        pattern = np.median(pattern_arrays, axis=0)
    elif method == "weighted_mean":
        weights = weights / np.sum(weights)
        pattern = np.average(pattern_arrays, axis=0, weights=weights)

    consensus["pattern"] = pattern

    # Aggregate annotations if present (use most common at each timepoint)
    annotation_lists = []
    for pattern_data in aligned_patterns.values():
        if pattern_data.get("annotations") is not None:
            annotation_lists.append(pattern_data["annotations"])

    if annotation_lists:
        annotations = []
        time_length = pattern.shape[0]

        for t in range(time_length):
            annotations_at_t = []
            for ann_list in annotation_lists:
                if t < len(ann_list) and ann_list[t] != "Unknown":
                    annotations_at_t.append(ann_list[t])

            if annotations_at_t:
                # Find most common annotation
                most_common = max(set(annotations_at_t), key=annotations_at_t.count)
                annotations.append(most_common)
            else:
                annotations.append("Unknown")

        consensus["annotations"] = annotations

    return consensus
