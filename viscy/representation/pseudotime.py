import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from tqdm import tqdm
from typing_extensions import TypedDict

_logger = logging.getLogger("lightning.pytorch")

#Annotated Example TypeDict
class AnnotatedSample(TypedDict):
    fov_name: str
    track_id: int | list[int]
    timepoints: tuple[int, int]
    annotations: dict | list
    weight: float

# DTW configuration
class DTWConfig(TypedDict, total=False):
    window_step: int
    num_candidates: int
    max_distance: float
    max_skew: float
    method: str
    normalize: bool
    metric: str

@dataclass
class DTWResult:
    """Results from DTW pattern matching."""
    matches: pd.DataFrame
    reference_pattern: np.ndarray
    reference_info: dict
    

class CytoDtw:    
    def __init__(self, embeddings: xr.Dataset, annotations_df: pd.DataFrame):
        """
        DTW for Dynamic Cell Embeddings 
               
        Parameters
        ----------
        embeddings  : xr.Dataset
            Embedding dataset (zarr file)
        """
        self.embeddings=embeddings
        self.annotations_df=annotations_df
        self.lineages = None


        self.consensus_data = None

    def _validate_input(self):
        raise NotImplementedError("Validation of input not implemented")
    
    def get_lineages(self, min_timepoints: int = 15) -> list[tuple[str, list[int]]]:
        """Get identified lineages with specified minimum timepoints."""
        return self._identify_lineages(min_timepoints)
        
    def _identify_lineages(self, min_timepoints: int = 15) -> list[tuple[str, list[int]]]:
        """Auto-identify lineages from the data."""
        # Use parent_track_id if available for proper lineage identification
        if 'parent_track_id' in self.annotations_df.columns:
            all_lineages = identify_lineages(self.annotations_df, return_both_branches=False)
        else:
            # Fallback: treat each track as individual lineage
            all_lineages = []
            for (fov, track_id), group in self.annotations_df.groupby(['fov_name', 'track_id']):
                all_lineages.append((fov, [track_id]))
        
        # Filter lineages by total timepoints across all tracks in lineage
        filtered_lineages = []
        for fov_id, track_ids in all_lineages:
            lineage_rows = self.annotations_df[
                (self.annotations_df["fov_name"] == fov_id) & (self.annotations_df["track_id"].isin(track_ids))
            ]
            total_timepoints = len(lineage_rows)
            if total_timepoints >= min_timepoints:
                filtered_lineages.append((fov_id, track_ids))
        self.lineages=filtered_lineages
        return self.lineages

    def get_reference_pattern(self, fov_name: str, track_id: int | list[int], 
                            timepoints: tuple[int, int]) -> np.ndarray:
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
            
        Returns
        -------
        np.ndarray
            Reference pattern embeddings
        """
        if isinstance(track_id, int):
            track_id = [track_id]
            
        reference_embeddings = []
        for tid in track_id:
            track_emb = self.embeddings.sel(sample=(fov_name, tid)).features.values
            reference_embeddings.append(track_emb)

        reference_pattern = np.concatenate(reference_embeddings, axis=0)
        
        start_t, end_t = timepoints
        reference_pattern = reference_pattern[start_t:end_t]
        
        return reference_pattern
    
    def get_matches(self, reference_pattern: np.ndarray=None, 
                           lineages: list[tuple[str, list[int]]] = None,
                           window_step: int = 5,
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
            reference_pattern = self.consensus_data['consensus_pattern']
        if lineages is None:
            # FIXME: Auto-identify lineages from tracking data
            lineages = pd.DataFrame(self.lineages, columns=["fov_name", "track_id"])
            
        return find_pattern_matches(
            reference_pattern=reference_pattern,
            filtered_lineages=lineages,
            embeddings_dataset=self.embeddings,
            window_step=window_step,
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
    
    def create_consensus_reference_pattern(
        self, 
        annotated_samples: list[AnnotatedSample],
        reference_selection: str = "median_length",
        aggregation_method: str = "mean",
        annotations_name: str = "annotations"
    ) -> dict:
        """
        Create consensus reference pattern from multiple annotated samples.
        
        This method takes multiple manually annotated cell examples and creates a
        consensus reference pattern by aligning them with DTW and aggregating.
        
        Parameters
        ----------
        annotated_samples : list[AnnotatedSample]
            List of annotated examples, each containing:
            - 'fov_name': str - FOV identifier
            - 'track_id': int or list[int] - Track ID(s)
            - 'timepoints': tuple[int, int] - (start, end) timepoints
            - 'annotations': dict or list - Optional annotations/labels
            - 'weight': float - Optional weight for this example (default 1.0)
        reference_selection : str
            mode of selection of reference: "median_length", "first", "longest", "shortest"
        aggregation_method : str
            mode of aggregation: "mean", "median", "weighted_mean"
        annotations_name : str
            name of the annotations column
        Returns
        -------
        dict
            Dictionary containing:
            - 'consensus_pattern': np.ndarray - The consensus embedding pattern
            - 'consensus_annotations': list - Consensus annotations (if available)
            - 'metadata': dict - Information about consensus creation including method used
            
        Examples
        --------
        >>> analyzer = CytoDtw("embeddings.zarr")
        >>> examples = [
        ...     {
        ...         'fov_name': '/FOV1', 'track_id': 129, 
        ...         'timepoints': (8, 70), 'annotations': ['G1', 'S', 'G2', ...]
        ...     },
        ...     {
        ...         'fov_name': '/FOV2', 'track_id': 45,
        ...         'timepoints': (5, 55), 'weight': 1.2
        ...     }
        ... ]
        >>> consensus = analyzer.create_consensus_reference_pattern(examples)
        """
        if not annotated_samples:
            raise ValueError("No annotated examples provided")
        
        # Extract embedding patterns from each example
        extracted_patterns = {}
        for i, example in enumerate(annotated_samples):
            pattern = self.get_reference_pattern(
                fov_name=example['fov_name'],
                track_id=example['track_id'],
                timepoints=example['timepoints']
            )
            
            extracted_patterns[f"example_{i}"] = {
                'pattern': pattern,
                'annotations': example.get(annotations_name, None),
                'weight': example.get('weight', 1.0),
                'source': example
            }
        
        # Use the standalone function to create consensus
        consensus_data = create_consensus_from_patterns(
            extracted_patterns,
            reference_selection=reference_selection,
            aggregation_method=aggregation_method
        )
        self.consensus_data=consensus_data
        return self.consensus_data
    
    def align_patterns(
        self,
        pattern1: np.ndarray,
        pattern2: np.ndarray,
        metric: str = "euclidean"
    ) -> dict:
        """Align two embedding patterns using DTW.
        
        This is a general-purpose alignment method that can be used for any
        two embedding patterns with shape (T, ndim). It provides the core DTW 
        functionality in a modular way that can be reused by other methods.
        
        Parameters
        ----------
        pattern1 : np.ndarray
            First embedding pattern (T1, ndim) - will be used as query
        pattern2 : np.ndarray
            Second embedding pattern (T2, ndim) - will be used as reference
        metric : str
            Distance metric for DTW alignment
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'distance': float - DTW distance
            - 'skewness': float - Path skewness
            - 'warping_path': list - DTW warping path
            - 'aligned_pattern1': np.ndarray - Pattern1 aligned to pattern2's timepoints
        """
        result = align_embedding_patterns(pattern1, pattern2, metric=metric)
        result['aligned_pattern1'] = result.pop('aligned_query')
        return result
    
    def visualize_alignment(
        self,
        pattern1: np.ndarray,
        pattern2: np.ndarray,
        plot_type: str = "trajectories_2d",
        feature_subset: list[int] = None,
        **kwargs
    ):
        """Visualize DTW alignment results.
        
        Parameters
        ----------
        pattern1 : np.ndarray
            First embedding pattern (T1, ndim)
        pattern2 : np.ndarray  
            Second embedding pattern (T2, ndim) - used as reference
        plot_type : str
            Type of visualization: "comparison", "trajectories_2d", "warping_path", "phase_portrait"
        feature_subset : list[int], optional
            Subset of features to visualize
        **kwargs
            Additional arguments for specific plot types
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        # Get alignment results
        alignment = self.align_patterns(pattern1, pattern2, **kwargs)
        aligned_pattern1 = alignment['aligned_pattern1']
        warping_path = alignment['warping_path']
        
        if plot_type == "comparison":
            return self._plot_aligned_comparison(
                pattern1, pattern2, aligned_pattern1, feature_subset
            )
        elif plot_type == "trajectories_2d":
            return self._plot_trajectories_2d(
                pattern1, pattern2, aligned_pattern1, **kwargs
            )
        elif plot_type == "warping_path":
            feature_idx = kwargs.get('feature_idx', 0)
            return self._plot_warping_path(
                pattern1, pattern2, warping_path, feature_idx
            )
        elif plot_type == "phase_portrait":
            feature_pairs = kwargs.get('feature_pairs', [(0, 1)])
            return self._plot_phase_portraits(
                pattern1, pattern2, aligned_pattern1, feature_pairs
            )
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
    
    def _plot_aligned_comparison(self, query, reference, aligned, feature_subset):
        """Plot before/after alignment comparison."""
        import matplotlib.pyplot as plt
        
        n_features = min(4, query.shape[1])
        if feature_subset:
            features_to_plot = feature_subset[:n_features]
        else:
            features_to_plot = list(range(n_features))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('DTW Alignment: Before vs After')
        
        # Original patterns
        axes[0,0].plot(query[:, features_to_plot])
        axes[0,0].set_title('Original Query Pattern')
        axes[0,0].set_ylabel('Embedding Value')
        
        axes[0,1].plot(reference[:, features_to_plot])
        axes[0,1].set_title('Reference Pattern') 
        axes[0,1].set_ylabel('Embedding Value')
        
        # Aligned patterns
        axes[1,0].plot(aligned[:, features_to_plot])
        axes[1,0].set_title('Aligned Query Pattern')
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Embedding Value')
        
        # Overlay
        axes[1,1].plot(reference[:, features_to_plot], alpha=0.7, label='Reference')
        axes[1,1].plot(aligned[:, features_to_plot], alpha=0.7, linestyle='--', label='Aligned Query')
        axes[1,1].set_title('Aligned Overlay')
        axes[1,1].set_xlabel('Time')
        axes[1,1].legend()
        
        plt.tight_layout()
        return fig
        
    def _plot_trajectories_2d(self, query, reference, aligned, reduction_method="pca", **kwargs):
        """Plot 2D embedding trajectories."""
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        
        # Use PCA for dimensionality reduction
        all_data = np.vstack([query, reference, aligned])
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(all_data)
        
        n_query = len(query)
        n_ref = len(reference)
        
        query_2d = reduced_data[:n_query]
        ref_2d = reduced_data[n_query:n_query+n_ref]
        aligned_2d = reduced_data[n_query+n_ref:]
        
        plt.figure(figsize=(10, 8))
        
        # Plot with colorblind-friendly colors (blue/orange)
        plt.plot(query_2d[:, 0], query_2d[:, 1], 'b-o', alpha=0.7, 
                label='Original Query', markersize=3)
        plt.plot(ref_2d[:, 0], ref_2d[:, 1], color='orange', marker='s', 
                linestyle='-', alpha=0.7, label='Reference', markersize=3)
        plt.plot(aligned_2d[:, 0], aligned_2d[:, 1], 'g--^', alpha=0.7, 
                label='Aligned Query', markersize=3)
        
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('Embedding Trajectories (PCA)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
        
    def _plot_warping_path(self, query, reference, warping_path, feature_idx):
        """Plot DTW warping path visualization."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Distance matrix with path
        distance_matrix = cdist(query, reference, metric='euclidean')
        ax1.imshow(distance_matrix, origin='lower', cmap='viridis', aspect='auto')
        
        path_array = np.array(warping_path)
        ax1.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2)
        ax1.set_xlabel('Reference Time')
        ax1.set_ylabel('Query Time')
        ax1.set_title('DTW Distance Matrix & Warping Path')
        
        # Signal alignment
        ax2.plot(query[:, feature_idx], 'b-', alpha=0.7, label='Original Query')
        ax2.plot(reference[:, feature_idx], color='orange', alpha=0.7, label='Reference')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel(f'Feature {feature_idx} Value')
        ax2.set_title(f'Signal Alignment (Feature {feature_idx})')
        ax2.legend()
        
        plt.tight_layout()
        return fig
        
    def _plot_phase_portraits(self, query, reference, aligned, feature_pairs):
        """Plot phase portraits in feature space."""
        import matplotlib.pyplot as plt
        
        n_pairs = len(feature_pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(5*n_pairs, 5))
        if n_pairs == 1:
            axes = [axes]
        
        for i, (f1, f2) in enumerate(feature_pairs):
            axes[i].plot(query[:, f1], query[:, f2], 'b-o', alpha=0.7, 
                        markersize=2, label='Original Query')
            axes[i].plot(reference[:, f1], reference[:, f2], color='orange', 
                        marker='s', linestyle='-', alpha=0.7, markersize=2, label='Reference')
            axes[i].plot(aligned[:, f1], aligned[:, f2], 'g--^', alpha=0.7, 
                        markersize=2, label='Aligned Query')
            
            # Mark start/end
            axes[i].scatter(reference[0, f1], reference[0, f2], 
                           c='red', s=50, marker='*', label='Start', zorder=5)
            axes[i].scatter(reference[-1, f1], reference[-1, f2], 
                           c='darkred', s=50, marker='X', label='End', zorder=5)
            
            axes[i].set_xlabel(f'Feature {f1}')
            axes[i].set_ylabel(f'Feature {f2}')
            axes[i].set_title(f'Phase Portrait (Features {f1} vs {f2})')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_consensus(
        self,
        consensus_result: dict,
        individual_patterns: dict,
        plot_types: list[str] = ["comparison", "trajectories_2d"],
        example_pattern_id: str = None
    ):
        """Visualize consensus reference pattern against individual examples.
        
        This method specifically visualizes the results of consensus creation,
        showing how individual cell patterns align to the consensus reference.
        
        Parameters
        ----------
        consensus_result : dict
            Result from create_consensus_from_patterns()
        individual_patterns : dict  
            Original patterns dictionary used to create consensus
        plot_types : list[str]
            Types of plots to generate
        example_pattern_id : str, optional
            Specific pattern to use for comparison. If None, uses first pattern.
            
        Returns
        -------
        dict
            Dictionary mapping plot_type to matplotlib.figure.Figure
        """
        consensus_pattern = consensus_result['consensus_pattern']
        metadata = consensus_result['metadata']
        
        # Select example pattern for comparison
        if example_pattern_id is None:
            example_pattern_id = list(individual_patterns.keys())[0]
        
        if example_pattern_id not in individual_patterns:
            raise ValueError(f"Pattern '{example_pattern_id}' not found in individual_patterns")
            
        example_pattern = individual_patterns[example_pattern_id]['pattern']
        
        # Align example to consensus
        alignment = self.align_patterns(example_pattern, consensus_pattern)
        aligned_example = alignment['aligned_pattern1']
        
        figures = {}
        
        for plot_type in plot_types:
            if plot_type == "comparison":
                fig = self._plot_aligned_comparison(
                    example_pattern, consensus_pattern, aligned_example, 
                    feature_subset=[0, 1, 2, 3]
                )
                fig.suptitle(f'Consensus Alignment: {example_pattern_id} → Consensus Reference\\n'
                           f'Distance: {alignment["distance"]:.3f}, Skewness: {alignment["skewness"]:.3f}',
                           fontsize=12)
                
            elif plot_type == "trajectories_2d":
                fig = self._plot_trajectories_2d(
                    example_pattern, consensus_pattern, aligned_example
                )
                fig.suptitle(f'Consensus Reference Pattern (built from {metadata["n_patterns"]} cells)\\n'
                           f'Method: {metadata["aggregation_method"]}, Reference: {metadata["reference_pattern"]}',
                           fontsize=12)
                
            elif plot_type == "phase_portrait":
                fig = self._plot_phase_portraits(
                    example_pattern, consensus_pattern, aligned_example, 
                    feature_pairs=[(0, 1), (2, 3)]
                )
                fig.suptitle(f'Cell State Transitions: Individual vs Consensus', fontsize=12)
                
            elif plot_type == "warping_path":
                fig = self._plot_warping_path(
                    example_pattern, consensus_pattern, alignment['warping_path'], 0
                )
                fig.suptitle(f'DTW Alignment to Consensus Reference', fontsize=12)
                
            else:
                raise ValueError(f"Unknown plot_type: {plot_type}")
                
            figures[plot_type] = fig
            
        return figures
    
    def get_annotated_patterns(
        self,
        annotation_specs: list[dict]
    ) -> dict[str, dict]:
        """Extract multiple patterns with their annotations for consensus building.
        
        This is a helper method to extract patterns from multiple examples
        that can then be used with the consensus creation functions.
        
        Parameters
        ----------
        annotation_specs : list[dict]
            List of pattern specifications, each containing:
            - 'fov_name': str
            - 'track_id': int or list[int]
            - 'timepoints': tuple[int, int]
            - 'label': str - identifier for this pattern
            - 'annotations': optional annotations
            
        Returns
        -------
        dict[str, dict]
            Dictionary mapping labels to pattern data compatible with
            create_consensus_from_patterns function
        """
        patterns = {}
        
        for spec in annotation_specs:
            pattern = self.get_reference_pattern(
                fov_name=spec['fov_name'],
                track_id=spec['track_id'],
                timepoints=spec['timepoints']
            )
            
            patterns[spec['label']] = {
                'pattern': pattern,
                'annotations': spec.get('annotations', None),
                'source_info': {
                    'fov_name': spec['fov_name'],
                    'track_id': spec['track_id'],
                    'timepoints': spec['timepoints']
                }
            }
        
        return patterns


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


def filter_lineages_by_timepoints(
    lineages: list[tuple[str, list[int]]],
    tracking_df: pd.DataFrame,
    min_timepoints: int = 20
) -> list[tuple[str, list[int]]]:
    """Filter lineages that have at least a minimum number of timepoints.
    
    This convenience function takes the output from identify_lineages() and filters
    out lineages that don't meet the minimum timepoint requirement.
    
    Parameters
    ----------
    lineages : list[tuple[str, list[int]]]
        List of (fov_name, track_ids) representing lineages from identify_lineages()
    tracking_df : pd.DataFrame
        Tracking dataframe with columns: fov_name, track_id, and other tracking data
    min_timepoints : int, default=20
        Minimum number of timepoints required to keep a lineage
        
    Returns
    -------
    list[tuple[str, list[int]]]
        Filtered list of lineages that meet the minimum timepoint requirement
    """
    filtered_lineages = []
    
    for fov_id, track_ids in lineages:
        # Count total timepoints for this lineage
        lineage_rows = tracking_df[
            (tracking_df["fov_name"] == fov_id) & 
            (tracking_df["track_id"].isin(track_ids))
        ]
        total_timepoints = len(lineage_rows)
        
        if total_timepoints >= min_timepoints:
            filtered_lineages.append((fov_id, track_ids))
    
    return filtered_lineages


def find_pattern_matches(
    reference_pattern: np.ndarray,
    filtered_lineages: list[tuple[str, list[int]]],
    embeddings_dataset: xr.Dataset,
    window_step: int = 5,
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


def create_consensus_from_patterns(
    patterns: dict[str, dict],
    reference_selection: str = "median_length",
    aggregation_method: str = "mean",
    distance_method: Literal["cosine", "euclidean"] = "cosine"
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
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'consensus_pattern': np.ndarray - The consensus embedding pattern
        - 'consensus_annotations': list - Consensus annotations (if available)
        - 'metadata': dict - Information about the consensus creation process
        
    Examples
    --------
    >>> patterns = {
    ...     'cell1': {
    ...         'pattern': np.random.rand(50, 128),  # 50 timepoints, 128 features
    ...         'annotations': ['G1'] * 20 + ['S'] * 15 + ['G2'] * 15,
    ...         'weight': 1.0
    ...     },
    ...     'cell2': {
    ...         'pattern': np.random.rand(45, 128),  # Different length
    ...         'annotations': ['G1'] * 18 + ['S'] * 12 + ['G2'] * 15
    ...     }
    ... }
    >>> consensus = create_consensus_from_patterns(patterns)
    """
    if not patterns:
        raise ValueError("No patterns provided")
    
    # Validate that all patterns have the 'pattern' key
    for pattern_id, pattern_data in patterns.items():
        if 'pattern' not in pattern_data:
            raise ValueError(f"Pattern '{pattern_id}' missing 'pattern' key")
        if not isinstance(pattern_data['pattern'], np.ndarray):
            raise ValueError(f"Pattern '{pattern_id}' must be numpy array")
    
    # Select reference pattern and align all patterns to it
    reference_id = _select_reference_pattern(patterns, reference_selection)
    reference_pattern = patterns[reference_id]['pattern']
    
    _logger.debug(f"Selected reference pattern: {reference_id}")
    _logger.debug(f"Reference shape: {reference_pattern.shape}")
    
    aligned_patterns = align_patterns_to_reference(patterns, reference_id, metric=distance_method)
    consensus = _aggregate_aligned_patterns(aligned_patterns, aggregation_method)
    
    consensus['metadata'] = {
        'reference_pattern': reference_id,
        'source_patterns': list(patterns.keys()),
        'reference_selection': reference_selection,
        'aggregation_method': aggregation_method,
        'n_patterns': len(patterns),
        'reference_shape': reference_pattern.shape,
        'consensus_shape': consensus['consensus_pattern'].shape
    }
    
    return consensus


def _select_reference_pattern(patterns: dict, method: str) -> str:
    """Select which pattern to use as reference for DTW alignment."""
    if method == "first":
        return list(patterns.keys())[0]
    
    elif method == "median_length":
        lengths = {pid: len(pdata['pattern']) for pid, pdata in patterns.items()}
        median_length = np.median(list(lengths.values()))
        closest_id = min(lengths.keys(), key=lambda x: abs(lengths[x] - median_length))
        return closest_id
    
    elif method == "longest":
        lengths = {pid: len(pdata['pattern']) for pid, pdata in patterns.items()}
        return max(lengths.keys(), key=lambda x: lengths[x])
    
    elif method == "shortest":
        lengths = {pid: len(pdata['pattern']) for pid, pdata in patterns.items()}
        return min(lengths.keys(), key=lambda x: lengths[x])
    
    else:
        raise ValueError(f"Unknown reference selection method: {method}")


def align_embedding_patterns(
    query_pattern: np.ndarray, 
    reference_pattern: np.ndarray,
    metric: str = "euclidean"
) -> dict:
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
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'distance': float - DTW distance
        - 'skewness': float - Path skewness  
        - 'warping_path': list - DTW warping path
        - 'aligned_query': np.ndarray - Query aligned to reference timepoints
    """

    distance, skewness = compute_dtw_distance(
        query_pattern, 
        reference_pattern, 
        metric=metric
    )
    
    distance_matrix = cdist(query_pattern, reference_pattern, metric=metric)
    _, _, dtw_path = dtw_with_matrix(distance_matrix, normalize=True)
    
    aligned_query, _ = _apply_warping_path(
        query_pattern, 
        reference_pattern,
        dtw_path
    )
    
    return {
        'distance': distance,
        'skewness': skewness,
        'warping_path': dtw_path,
        'aligned_query': aligned_query
    }


def align_patterns_to_reference(patterns: dict, reference_id: str, metric: str = "cosine") -> dict:
    """Align all patterns to the selected reference using DTW on embedding vectors.
    
    This function reuses the existing modular DTW functionality from the CytoDtw system
    instead of duplicating DTW alignment logic.
    """
    reference_pattern = patterns[reference_id]['pattern']
    aligned_patterns = {reference_id: patterns[reference_id]}  # Reference doesn't need alignment
    
    for pattern_id, pattern_data in patterns.items():
        if pattern_id == reference_id:
            continue  # Skip reference
        
        query_pattern = pattern_data['pattern']
        
        # Use our modular alignment function
        alignment_result = align_embedding_patterns(
            query_pattern, 
            reference_pattern, 
            metric=metric
        )
        
        # Apply the alignment to annotations if present
        _, aligned_annotations = _apply_warping_path(
            query_pattern, 
            reference_pattern,
            alignment_result['warping_path'],
            pattern_data.get('annotations')
        )
        
        # Create aligned pattern data
        aligned_data = {
            'pattern': alignment_result['aligned_query'],
            'annotations': aligned_annotations,
            'weight': pattern_data.get('weight', 1.0),
            'dtw_distance': alignment_result['distance'],
            'dtw_skewness': alignment_result['skewness'],
            'alignment_path': alignment_result['warping_path']
        }
        
        # Copy other metadata
        for key, value in pattern_data.items():
            if key not in ['pattern', 'annotations', 'weight']:
                aligned_data[key] = value
        
        aligned_patterns[pattern_id] = aligned_data
    
    return aligned_patterns


def _apply_warping_path(
    query_pattern: np.ndarray,
    reference_pattern: np.ndarray, 
    dtw_path: list[tuple[int, int]],
    query_annotations: list = None
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
    dtw_path : list[tuple[int, int]]
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
    for query_idx, ref_idx in dtw_path:
        if ref_idx < ref_length and query_idx < len(query_pattern):
            aligned_pattern[ref_idx] = query_pattern[query_idx]
    
    # Align annotations if present
    aligned_annotations = None
    if query_annotations is not None:
        aligned_annotations = ['Unknown'] * ref_length
        for query_idx, ref_idx in dtw_path:
            if (ref_idx < ref_length and 
                query_idx < len(query_annotations)):
                aligned_annotations[ref_idx] = query_annotations[query_idx]
    
    return aligned_pattern, aligned_annotations


def _aggregate_aligned_patterns(aligned_patterns: dict, method: str) -> dict:
    """Aggregate aligned embedding patterns into consensus."""
    consensus = {}
    
    # Extract patterns and weights
    pattern_arrays = []
    weights = []
    
    for pattern_data in aligned_patterns.values():
        pattern_arrays.append(pattern_data['pattern'])
        weights.append(pattern_data.get('weight', 1.0))
    
    pattern_arrays = np.array(pattern_arrays)  # Shape: (n_patterns, time, features)
    weights = np.array(weights)
    
    # Aggregate embedding patterns
    if method == "mean":
        consensus_pattern = np.mean(pattern_arrays, axis=0)
    elif method == "median":
        consensus_pattern = np.median(pattern_arrays, axis=0)
    elif method == "weighted_mean":
        weights = weights / np.sum(weights)
        consensus_pattern = np.average(pattern_arrays, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    consensus['consensus_pattern'] = consensus_pattern
    
    # Aggregate annotations if present (use most common at each timepoint)
    annotation_lists = []
    for pattern_data in aligned_patterns.values():
        if pattern_data.get('annotations') is not None:
            annotation_lists.append(pattern_data['annotations'])
    
    if annotation_lists:
        consensus_annotations = []
        time_length = consensus_pattern.shape[0]
        
        for t in range(time_length):
            annotations_at_t = []
            for ann_list in annotation_lists:
                if t < len(ann_list) and ann_list[t] != 'Unknown':
                    annotations_at_t.append(ann_list[t])
            
            if annotations_at_t:
                # Find most common annotation
                most_common = max(set(annotations_at_t), key=annotations_at_t.count)
                consensus_annotations.append(most_common)
            else:
                consensus_annotations.append('Unknown')
        
        consensus['consensus_annotations'] = consensus_annotations
    
    return consensus