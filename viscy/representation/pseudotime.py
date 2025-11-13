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


# Utility functions
def filter_tracks_by_fov_and_length(
    df: pd.DataFrame,
    fov_pattern: str | list[str],
    min_timepoints: int,
) -> pd.DataFrame:
    """
    Filter dataframe by FOV pattern(s) and minimum timepoints per track.

    Convenience function for filtering tracking data by field of view pattern
    and track length. Useful for separating experimental conditions (e.g.,
    infected vs uninfected) and ensuring sufficient temporal coverage.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'fov_name', 'track_id', and 't' columns
    fov_pattern : str or list[str]
        Pattern(s) to match FOV names (used with str.contains()).
        If list, matches any of the patterns (OR logic).
    min_timepoints : int
        Minimum number of timepoints required per track

    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing only tracks that:
        1. Match the FOV pattern(s)
        2. Have at least min_timepoints timepoints

    Examples
    --------
    >>> # Single pattern
    >>> infected_df = filter_tracks_by_fov_and_length(
    ...     master_df, fov_pattern="B/2", min_timepoints=20
    ... )
    >>>
    >>> # Multiple patterns (OR logic)
    >>> infected_df = filter_tracks_by_fov_and_length(
    ...     master_df, fov_pattern=["B/2", "C/2"], min_timepoints=20
    ... )
    """
    # Handle single pattern or multiple patterns
    if isinstance(fov_pattern, str):
        fov_patterns = [fov_pattern]
    else:
        fov_patterns = fov_pattern

    # Filter by FOV pattern(s) - OR logic
    fov_mask = pd.Series(False, index=df.index)
    for pattern in fov_patterns:
        fov_mask |= df["fov_name"].str.contains(pattern)

    fov_filtered = df[fov_mask].copy()

    if len(fov_filtered) == 0:
        _logger.warning(f"No FOVs matched pattern(s): {fov_patterns}")
        return pd.DataFrame()

    # Count timepoints per track
    track_lengths = fov_filtered.groupby(["fov_name", "track_id"]).size()
    valid_tracks = track_lengths[track_lengths >= min_timepoints].index

    # Filter to valid tracks only
    result = fov_filtered[
        fov_filtered.set_index(["fov_name", "track_id"]).index.isin(valid_tracks)
    ].reset_index(drop=True)

    pattern_str = fov_patterns if len(fov_patterns) > 1 else fov_patterns[0]
    _logger.info(
        f"Filtered by pattern '{pattern_str}' with min_timepoints={min_timepoints}: "
        f"{len(valid_tracks)} tracks from {fov_filtered['fov_name'].nunique()} FOVs"
    )

    return result


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
        with open(path, "wb") as f:
            pickle.dump(self.consensus_data, f)

    def load_consensus(self, path: str):
        """Load consensus pattern from a file."""
        import pickle

        with open(path, "rb") as f:
            self.consensus_data = pickle.load(f)

    def get_lineages(self, min_timepoints: int = 15) -> list[tuple[str, list[int]]]:
        """Get identified lineages with specified minimum timepoints."""
        return self._identify_lineages(min_timepoints)

    def get_track_statistics(
        self,
        lineages: list[tuple[str, list[int]]] = None,
        min_timepoints: int = 15,
        per_fov: bool = False,
    ) -> pd.DataFrame:
        """Get statistics for tracks/lineages.

        Parameters
        ----------
        lineages : list[tuple[str, list[int]]], optional
            List of (fov_name, track_ids) to analyze. If None, uses all lineages.
        min_timepoints : int
            Minimum timepoints required for lineage identification if lineages is None
        per_fov : bool
            If True, return aggregated statistics per FOV. If False, return per-lineage statistics.

        Returns
        -------
        pd.DataFrame
            If per_fov=False: DataFrame with per-lineage statistics:
            - fov_name: FOV identifier
            - track_ids: List of track IDs in lineage
            - n_tracks: Number of tracks in lineage
            - total_timepoints: Total timepoints across all tracks
            - mean_timepoints_per_track: Average timepoints per track
            - std_timepoints_per_track: Standard deviation of timepoints per track
            - min_t: Earliest timepoint
            - max_t: Latest timepoint

            If per_fov=True: DataFrame with aggregated per-FOV statistics:
            - fov_name: FOV identifier
            - n_lineages: Number of lineages in FOV
            - total_tracks: Total number of tracks across all lineages
            - mean_tracks_per_lineage: Average tracks per lineage
            - std_tracks_per_lineage: Standard deviation of tracks per lineage
            - mean_total_timepoints: Average total timepoints per lineage
            - std_total_timepoints: Standard deviation of total timepoints
            - mean_timepoints_per_track: Average timepoints per track across all tracks
            - std_timepoints_per_track: Standard deviation across all tracks
        """
        if lineages is None:
            lineages = self.get_lineages(min_timepoints)

        stats_list = []
        for fov_name, track_ids in lineages:
            lineage_rows = self.adata.obs[
                (self.adata.obs["fov_name"] == fov_name)
                & (self.adata.obs["track_id"].isin(track_ids))
            ]

            total_timepoints = len(lineage_rows)
            timepoints_per_track = []

            for track_id in track_ids:
                track_rows = lineage_rows[lineage_rows["track_id"] == track_id]
                timepoints_per_track.append(len(track_rows))

            stats_list.append(
                {
                    "fov_name": fov_name,
                    "track_ids": track_ids,
                    "n_tracks": len(track_ids),
                    "total_timepoints": total_timepoints,
                    "mean_timepoints_per_track": np.mean(timepoints_per_track),
                    "std_timepoints_per_track": np.std(timepoints_per_track),
                    "timepoints_per_track": timepoints_per_track,
                    "min_t": lineage_rows["t"].min(),
                    "max_t": lineage_rows["t"].max(),
                }
            )

        df = pd.DataFrame(stats_list)

        if not per_fov:
            return df.drop(columns=["timepoints_per_track"])

        # Aggregate by FOV
        fov_stats = []
        for fov_name in df["fov_name"].unique():
            fov_df = df[df["fov_name"] == fov_name]
            all_timepoints = [
                tp for tps in fov_df["timepoints_per_track"] for tp in tps
            ]

            fov_stats.append(
                {
                    "fov_name": fov_name,
                    "n_lineages": len(fov_df),
                    "total_tracks": fov_df["n_tracks"].sum(),
                    "mean_tracks_per_lineage": fov_df["n_tracks"].mean(),
                    "std_tracks_per_lineage": fov_df["n_tracks"].std(),
                    "mean_total_timepoints": fov_df["total_timepoints"].mean(),
                    "std_total_timepoints": fov_df["total_timepoints"].std(),
                    "mean_timepoints_per_track": np.mean(all_timepoints),
                    "std_timepoints_per_track": np.std(all_timepoints),
                }
            )

        return pd.DataFrame(fov_stats)

    def _identify_lineages(
        self, min_timepoints: int = 15
    ) -> list[tuple[str, list[int]]]:
        """Auto-identify lineages from the data."""
        # Use parent_track_id if available for proper lineage identification
        if "parent_track_id" in self.adata.obs.columns:
            all_lineages = identify_lineages(self.adata.obs, return_both_branches=False)
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
            mask = (self.adata.obs["fov_name"] == fov_name) & (
                self.adata.obs["track_id"] == tid
            )
            track_data = self.adata[mask]

            # Sort by timepoint to ensure correct order
            time_order = np.argsort(track_data.obs["t"].values)
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
        Create consensus reference pattern from annotated samples.

        This method takes one or more annotated cell examples and creates a
        consensus reference pattern. For single annotations, uses it directly. For
        multiple annotations, aligns them with DTW and aggregates.

        Parameters
        ----------
        annotated_samples : list[AnnotatedSample]
            List of annotated examples (minimum 1 required)
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
            raise ValueError("At least one annotated example is required")

        # Extract embedding patterns from each example
        extracted_patterns = {}
        for i, example in enumerate(annotated_samples):
            pattern = self.get_reference_pattern(
                fov_name=example["fov_name"],
                track_id=example["track_id"],
                timepoints=example["timepoints"],
                reference_type=reference_type,
            )

            extracted_patterns[
                f"example_{example['fov_name']}_{example['track_id']}"
            ] = {
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

    def create_alignment_dataframe(
        self,
        top_matches: pd.DataFrame,
        consensus_lineage: np.ndarray,
        alignment_name: str = "cell_division",
        reference_type: str = "features",
    ) -> pd.DataFrame:
        """
        Create alignment DataFrame that:
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
            DataFrame with lineage preservation and extensible alignments
        """
        alignment_data = []
        track_lineage_mapping = {}

        available_obsm = list(self.adata.obsm.keys())
        has_pca_obsm = "X_PCA" in available_obsm
        pc_components = []

        if has_pca_obsm:
            n_pca_components = self.adata.obsm["X_PCA"].shape[1]
            pc_components = [f"PC{i + 1}" for i in range(n_pca_components)]
        else:
            pc_components = [
                col
                for col in self.adata.obs.columns
                if col.startswith("PC") and col[2:].isdigit()
            ]
            pc_components.sort(key=lambda x: int(x[2:]))  # Sort PC1, PC2, PC3, etc.

        has_pc_components = len(pc_components) > 0

        lineage_counter = 0
        for idx, match_row in top_matches.iterrows():
            fov_name = match_row["fov_name"]
            track_ids = match_row["track_ids"]

            lineage_id = lineage_counter
            lineage_counter += 1

            for track_id in track_ids:
                track_lineage_mapping[(fov_name, track_id)] = lineage_id

        pca = None
        scaler = None
        consensus_pca = None

        if not has_pc_components:
            all_embeddings = []

            for idx, match_row in top_matches.iterrows():
                fov_name = match_row["fov_name"]
                track_ids = match_row["track_ids"]

                for track_id in track_ids:
                    try:
                        mask = (self.adata.obs["fov_name"] == fov_name) & (
                            self.adata.obs["track_id"] == track_id
                        )
                        track_data = self.adata[mask]

                        time_order = np.argsort(track_data.obs["t"].values)
                        track_data = track_data[time_order]

                        if reference_type == "features":
                            track_embeddings = track_data.X
                        else:
                            track_embeddings = track_data.obsm[reference_type]

                        all_embeddings.append(track_embeddings)
                    except KeyError:
                        continue
            all_embeddings.append(consensus_lineage)
            all_concat = np.vstack(all_embeddings)
            n_components = min(8, all_concat.shape[1])
            scaler = StandardScaler()
            scaled_all = scaler.fit_transform(all_concat)
            pca = PCA(n_components=n_components)
            pca_all = pca.fit_transform(scaled_all)

            consensus_pca = pca_all[-len(consensus_lineage) :]

        for idx, match_row in top_matches.iterrows():
            fov_name = match_row["fov_name"]
            track_ids = match_row["track_ids"]
            warp_path = match_row["warp_path"]
            dtw_distance = match_row.get("distance", np.nan)

            # Create mapping from query timepoint to consensus timepoint
            query_to_consensus = {}
            for consensus_idx, query_timepoint in warp_path:
                query_to_consensus[query_timepoint] = consensus_idx

            # Process each track in this lineage
            for track_id in track_ids:
                try:
                    # Filter by fov_name and track_id
                    mask = (self.adata.obs["fov_name"] == fov_name) & (
                        self.adata.obs["track_id"] == track_id
                    )
                    track_data = self.adata[mask]

                    # Sort by timepoint to ensure correct order
                    time_order = np.argsort(track_data.obs["t"].values)
                    track_data = track_data[time_order]

                    track_timepoints = track_data.obs["t"].values

                    # Get PC features - either from obsm/obs or computed PCA
                    pc_values = {}
                    if has_pc_components:
                        if has_pca_obsm:
                            # Extract from X_PCA obsm
                            pca_data = track_data.obsm["X_PCA"]
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
                            pc_name = f"PC{i + 1}"
                            pc_values[pc_name] = track_pca[:, i]

                    # Get lineage ID for this track
                    lineage_id = track_lineage_mapping.get((fov_name, track_id), -1)

                    # Create row for each timepoint
                    for i, t in enumerate(track_timepoints):
                        # Get spatial coordinates from track_data.obs (which is already filtered)
                        obs_row = track_data.obs.iloc[i]
                        x_coord = obs_row.get("x", np.nan)
                        y_coord = obs_row.get("y", np.nan)

                        # Determine alignment status for this specific alignment type
                        is_aligned = t in query_to_consensus
                        consensus_mapping = query_to_consensus.get(t, np.nan)

                        # Create dynamic column names based on alignment_name
                        row_data = {
                            # Core tracking info (preserves lineage relationships)
                            "fov_name": fov_name,
                            "lineage_id": lineage_id,
                            "track_id": track_id,
                            "t": t,
                            "x": x_coord,
                            "y": y_coord,
                            # Alignment-specific columns (dynamic based on alignment_name)
                            f"dtw_{alignment_name}_consensus_mapping": consensus_mapping,
                            f"dtw_{alignment_name}_aligned": is_aligned,
                            f"dtw_{alignment_name}_distance": dtw_distance,
                            f"dtw_{alignment_name}_match_rank": idx,
                        }

                        # Add all PC components dynamically
                        for pc_name, pc_vals in pc_values.items():
                            row_data[pc_name] = pc_vals[i]
                        alignment_data.append(row_data)

                except KeyError as e:
                    _logger.warning(
                        f"Could not find track {track_id} in FOV {fov_name}: {e}"
                    )
                    continue

        consensus_pc_values = {}
        if has_pc_components:
            for pc_name in pc_components:
                consensus_pc_values[pc_name] = [np.nan] * len(consensus_lineage)
        else:
            for i in range(n_components):
                pc_name = f"PC{i + 1}"
                consensus_pc_values[pc_name] = consensus_pca[:, i]

        for i in range(len(consensus_lineage)):
            consensus_row = {
                "fov_name": "consensus",
                "lineage_id": -1,
                "track_id": -1,
                "t": i,
                "x": np.nan,
                "y": np.nan,
                f"dtw_{alignment_name}_consensus_mapping": i,  # Maps to itself
                f"dtw_{alignment_name}_aligned": True,
                f"dtw_{alignment_name}_distance": np.nan,
                f"dtw_{alignment_name}_match_rank": -1,
            }

            for pc_name, pc_vals in consensus_pc_values.items():
                consensus_row[pc_name] = pc_vals[i]

            alignment_data.append(consensus_row)

        return pd.DataFrame(alignment_data)

    def get_concatenated_sequences(
        self,
        df: pd.DataFrame,
        alignment_name: str = "cell_division",
        feature_columns: list[str] = None,
        max_lineages: int = None,
    ) -> dict:
        """
        Extract concatenated [unaligned_before + aligned + unaligned_after] sequences from enhanced DataFrame.

        This is a shared method used by both plotting and image sequence functions.

        Parameters
        ----------
        df : pd.DataFrame
            Enhanced DataFrame with alignment information
        alignment_name : str
            Name of alignment type (e.g., "cell_division")
        feature_columns : list[str], optional
            Feature columns to extract. If None, extracts PC components only.
        max_lineages : int, optional
            Maximum number of lineages to process

        Returns
        -------
        dict
            Dictionary mapping lineage_id to:
            - 'unaligned_before_data': dict of arrays/dicts for timepoints before aligned region
            - 'aligned_data': dict of consensus-length aligned arrays/dicts
            - 'unaligned_after_data': dict of arrays/dicts for timepoints after aligned region
            - 'metadata': lineage metadata (fov_name, track_ids, etc.)
        """
        aligned_col = f"dtw_{alignment_name}_aligned"
        mapping_col = f"dtw_{alignment_name}_consensus_mapping"
        distance_col = f"dtw_{alignment_name}_distance"

        if aligned_col not in df.columns:
            _logger.error(f"Alignment '{alignment_name}' not found in DataFrame")
            return {}

        consensus_df = df[df["lineage_id"] == -1].sort_values("t").copy()
        lineages = df[df["lineage_id"] != -1]["lineage_id"].unique()

        if max_lineages is not None:
            lineages = lineages[:max_lineages]

        if consensus_df.empty:
            _logger.error("No consensus found in DataFrame")
            return {}

        consensus_length = len(consensus_df)
        concatenated_sequences = {}

        for lineage_id in lineages:
            lineage_df = df[df["lineage_id"] == lineage_id].copy().sort_values("t")
            if lineage_df.empty:
                continue

            aligned_rows = lineage_df[lineage_df[aligned_col]].copy()

            # Extract unaligned timepoints BEFORE and AFTER the aligned portion
            if not aligned_rows.empty:
                min_aligned_t = aligned_rows["t"].min()
                max_aligned_t = aligned_rows["t"].max()
                unaligned_before_rows = lineage_df[
                    (~lineage_df[aligned_col]) & (lineage_df["t"] < min_aligned_t)
                ].copy()
                unaligned_after_rows = lineage_df[
                    (~lineage_df[aligned_col]) & (lineage_df["t"] > max_aligned_t)
                ].copy()
            else:
                # No aligned portion - treat all as unaligned_before
                unaligned_before_rows = lineage_df[~lineage_df[aligned_col]].copy()
                unaligned_after_rows = pd.DataFrame()

            # Create consensus-length aligned portion using mapping
            # Make dict to map pseudotime to real time
            aligned_portion = {}
            for _, row in aligned_rows.iterrows():
                consensus_idx = row[mapping_col]
                if not pd.isna(consensus_idx):
                    consensus_idx = int(consensus_idx)
                    if 0 <= consensus_idx < consensus_length:
                        row_dict = {"t": row["t"], "row": row}
                        if feature_columns:
                            row_dict.update({col: row[col] for col in feature_columns})
                        aligned_portion[consensus_idx] = row_dict

            # Fill gaps in aligned portion
            filled_aligned = {}
            if aligned_portion:
                for i in range(consensus_length):
                    if i in aligned_portion:
                        filled_aligned[i] = aligned_portion[i]
                    else:
                        available_indices = list(aligned_portion.keys())
                        if available_indices:
                            closest_idx = min(
                                available_indices, key=lambda x: abs(x - i)
                            )
                            filled_aligned[i] = aligned_portion[closest_idx]
                        else:
                            consensus_row = consensus_df.iloc[i]
                            row_dict = {"row": consensus_row}
                            if feature_columns:
                                row_dict.update(
                                    {col: consensus_row[col] for col in feature_columns}
                                )
                            filled_aligned[i] = row_dict

            # Convert to arrays/lists for features
            aligned_data = {"length": consensus_length, "mapping": filled_aligned}
            if feature_columns:
                aligned_data["features"] = {}
                for col in feature_columns:
                    aligned_data["features"][col] = np.array(
                        [filled_aligned[i][col] for i in range(consensus_length)]
                    )

            # Process unaligned BEFORE portion
            unaligned_before_data = {
                "length": len(unaligned_before_rows),
                "rows": unaligned_before_rows,
            }
            if feature_columns and not unaligned_before_rows.empty:
                unaligned_before_rows = unaligned_before_rows.sort_values("t")
                unaligned_before_data["features"] = {}
                for col in feature_columns:
                    unaligned_before_data["features"][col] = unaligned_before_rows[
                        col
                    ].values

            # Process unaligned AFTER portion
            unaligned_after_data = {
                "length": len(unaligned_after_rows),
                "rows": unaligned_after_rows,
            }
            if feature_columns and not unaligned_after_rows.empty:
                unaligned_after_rows = unaligned_after_rows.sort_values("t")
                unaligned_after_data["features"] = {}
                for col in feature_columns:
                    unaligned_after_data["features"][col] = unaligned_after_rows[
                        col
                    ].values

            concatenated_sequences[lineage_id] = {
                "unaligned_before_data": unaligned_before_data,
                "aligned_data": aligned_data,
                "unaligned_after_data": unaligned_after_data,
                "metadata": {
                    "fov_name": lineage_df["fov_name"].iloc[0],
                    "track_ids": list(lineage_df["track_id"].unique()),
                    "dtw_distance": (
                        lineage_df[distance_col].iloc[0]
                        if not pd.isna(lineage_df[distance_col].iloc[0])
                        else np.nan
                    ),
                    "lineage_id": lineage_id,
                    "consensus_length": consensus_length,
                },
            }

        return concatenated_sequences

    def plot_global_trends(
        self,
        df: pd.DataFrame,
        alignment_name: str = "cell_division",
        feature_columns: list[str] = None,
        max_lineages: int = None,
        plot_type: str = "mean_bands",
        figsize: tuple = (15, 12),
        colors: tuple = ("#1f77b4", "#ff7f0e"),
        cmap: str = "RdBu_r",
        remove_outliers: bool = False,
        outlier_percentile: tuple = (1, 99),
    ):
        """
        Plot global trends across all aligned lineages.

        Parameters
        ----------
        df : pd.DataFrame
            Enhanced DataFrame with alignment information
        alignment_name : str
            Name of alignment type
        feature_columns : list[str], optional
            Feature columns to plot. If None, uses PC1, PC2, PC3
        max_lineages : int, optional
            Maximum number of lineages to include
        plot_type : str
            Type of plot: "mean_bands", "heatmap", "quantile_bands", or "individual_with_mean"
        figsize : tuple
            Figure size
        colors : tuple
            Colors for (aligned, unaligned) portions in line plots
        cmap : str
            Colormap for heatmap plot
        remove_outliers : bool
            Whether to clip outlier values for better visualization
        outlier_percentile : tuple
            (lower, upper) percentile bounds for clipping (default: 1st-99th percentile)
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        import matplotlib.pyplot as plt

        if feature_columns is None:
            feature_columns = ["PC1", "PC2", "PC3"]

        # Get concatenated sequences
        concatenated_seqs = self.get_concatenated_sequences(
            df=df,
            alignment_name=alignment_name,
            feature_columns=feature_columns,
            max_lineages=max_lineages,
        )

        if not concatenated_seqs:
            _logger.error("No concatenated sequences found")
            return None

        consensus_df = df[df["lineage_id"] == -1].sort_values("t").copy()
        consensus_length = len(consensus_df)

        n_features = len(feature_columns)
        fig, axes = plt.subplots(n_features, 1, figsize=figsize)
        if n_features == 1:
            axes = [axes]

        for feat_idx, feat_col in enumerate(feature_columns):
            ax = axes[feat_idx]

            all_unaligned_before = []
            all_aligned = []
            all_unaligned_after = []

            for lineage_id, seq_data in concatenated_seqs.items():
                unaligned_before_features = seq_data["unaligned_before_data"].get(
                    "features", {}
                )
                aligned_features = seq_data["aligned_data"]["features"]
                unaligned_after_features = seq_data["unaligned_after_data"].get(
                    "features", {}
                )

                if feat_col in unaligned_before_features:
                    all_unaligned_before.append(unaligned_before_features[feat_col])
                all_aligned.append(aligned_features[feat_col])
                if feat_col in unaligned_after_features:
                    all_unaligned_after.append(unaligned_after_features[feat_col])

            aligned_array = np.array(all_aligned)
            max_unaligned_before_len = (
                max([len(u) for u in all_unaligned_before])
                if all_unaligned_before
                else 0
            )
            max_unaligned_after_len = (
                max([len(u) for u in all_unaligned_after]) if all_unaligned_after else 0
            )

            if remove_outliers:
                all_values = []
                all_values.extend(consensus_df[feat_col].values)
                if all_unaligned_before:
                    for u in all_unaligned_before:
                        all_values.extend(u)
                all_values.extend(aligned_array.flatten())
                if all_unaligned_after:
                    for u in all_unaligned_after:
                        all_values.extend(u)

                all_values = np.array(all_values)
                all_values = all_values[~np.isnan(all_values)]

                if len(all_values) > 0:
                    lower_bound = np.percentile(all_values, outlier_percentile[0])
                    upper_bound = np.percentile(all_values, outlier_percentile[1])
                    _logger.info(
                        f"{feat_col}: setting y-limits to [{lower_bound:.3f}, {upper_bound:.3f}]"
                    )

            # Pad unaligned arrays to same length
            # Pad "before" arrays: prepend NaN to align shorter sequences to the right
            if max_unaligned_before_len > 0:
                unaligned_before_array = np.full(
                    (len(concatenated_seqs), max_unaligned_before_len), np.nan
                )
                for i, u in enumerate(all_unaligned_before):
                    padding_needed = max_unaligned_before_len - len(u)
                    # Right-align by prepending NaN padding
                    unaligned_before_array[i, padding_needed:] = u
            else:
                unaligned_before_array = np.array([]).reshape(0, 0)

            # Pad "after" arrays: append NaN (left-aligned)
            if max_unaligned_after_len > 0:
                unaligned_after_array = np.full(
                    (len(concatenated_seqs), max_unaligned_after_len), np.nan
                )
                for i, u in enumerate(all_unaligned_after):
                    unaligned_after_array[i, : len(u)] = u
            else:
                unaligned_after_array = np.array([]).reshape(0, 0)

            if plot_type == "mean_bands":
                # Plot mean ± SEM for all three segments
                time_offset = 0

                # 1. Plot unaligned BEFORE
                if unaligned_before_array.size > 0:
                    unaligned_before_mean = np.nanmean(unaligned_before_array, axis=0)
                    unaligned_before_sem = np.nanstd(
                        unaligned_before_array, axis=0
                    ) / np.sqrt(np.sum(~np.isnan(unaligned_before_array), axis=0))
                    unaligned_before_time = np.arange(
                        time_offset, time_offset + max_unaligned_before_len
                    )

                    ax.plot(
                        unaligned_before_time,
                        unaligned_before_mean,
                        "--",
                        color=colors[1],
                        linewidth=2,
                        label="Unaligned before",
                        alpha=0.7,
                        zorder=2,
                    )
                    ax.fill_between(
                        unaligned_before_time,
                        unaligned_before_mean - unaligned_before_sem,
                        unaligned_before_mean + unaligned_before_sem,
                        alpha=0.2,
                        color=colors[1],
                        zorder=1,
                    )
                    time_offset += max_unaligned_before_len

                # 2. Plot ALIGNED (highlight with thick lines)
                aligned_mean = np.nanmean(aligned_array, axis=0)
                aligned_sem = np.nanstd(aligned_array, axis=0) / np.sqrt(
                    np.sum(~np.isnan(aligned_array), axis=0)
                )
                aligned_time = np.arange(time_offset, time_offset + consensus_length)

                ax.plot(
                    aligned_time,
                    aligned_mean,
                    "-",
                    color=colors[0],
                    linewidth=3,
                    label="Aligned mean",
                    zorder=4,
                )
                ax.fill_between(
                    aligned_time,
                    aligned_mean - aligned_sem,
                    aligned_mean + aligned_sem,
                    alpha=0.3,
                    color=colors[0],
                    zorder=3,
                )
                time_offset += consensus_length

                # 3. Plot unaligned AFTER
                if unaligned_after_array.size > 0:
                    unaligned_after_mean = np.nanmean(unaligned_after_array, axis=0)
                    unaligned_after_sem = np.nanstd(
                        unaligned_after_array, axis=0
                    ) / np.sqrt(np.sum(~np.isnan(unaligned_after_array), axis=0))
                    unaligned_after_time = np.arange(
                        time_offset, time_offset + max_unaligned_after_len
                    )

                    ax.plot(
                        unaligned_after_time,
                        unaligned_after_mean,
                        "--",
                        color=colors[1],
                        linewidth=2,
                        label="Unaligned after",
                        alpha=0.7,
                        zorder=2,
                    )
                    ax.fill_between(
                        unaligned_after_time,
                        unaligned_after_mean - unaligned_after_sem,
                        unaligned_after_mean + unaligned_after_sem,
                        alpha=0.2,
                        color=colors[1],
                        zorder=1,
                    )

            elif plot_type == "quantile_bands":
                # Plot median + quartiles for all three segments
                time_offset = 0

                # 1. Plot unaligned BEFORE
                if unaligned_before_array.size > 0:
                    unaligned_before_median = np.nanmedian(
                        unaligned_before_array, axis=0
                    )
                    unaligned_before_q25 = np.nanpercentile(
                        unaligned_before_array, 25, axis=0
                    )
                    unaligned_before_q75 = np.nanpercentile(
                        unaligned_before_array, 75, axis=0
                    )
                    unaligned_before_time = np.arange(
                        time_offset, time_offset + max_unaligned_before_len
                    )

                    ax.plot(
                        unaligned_before_time,
                        unaligned_before_median,
                        "--",
                        color=colors[1],
                        linewidth=2,
                        label="Unaligned before",
                        alpha=0.7,
                        zorder=2,
                    )
                    ax.fill_between(
                        unaligned_before_time,
                        unaligned_before_q25,
                        unaligned_before_q75,
                        alpha=0.2,
                        color=colors[1],
                        zorder=1,
                    )
                    time_offset += max_unaligned_before_len

                # 2. Plot ALIGNED (highlight with thick lines)
                aligned_median = np.nanmedian(aligned_array, axis=0)
                aligned_q25 = np.nanpercentile(aligned_array, 25, axis=0)
                aligned_q75 = np.nanpercentile(aligned_array, 75, axis=0)
                aligned_time = np.arange(time_offset, time_offset + consensus_length)

                ax.plot(
                    aligned_time,
                    aligned_median,
                    "-",
                    color=colors[0],
                    linewidth=3,
                    label="Aligned median",
                    zorder=4,
                )
                ax.fill_between(
                    aligned_time,
                    aligned_q25,
                    aligned_q75,
                    alpha=0.3,
                    color=colors[0],
                    zorder=3,
                )
                time_offset += consensus_length

                # 3. Plot unaligned AFTER
                if unaligned_after_array.size > 0:
                    unaligned_after_median = np.nanmedian(unaligned_after_array, axis=0)
                    unaligned_after_q25 = np.nanpercentile(
                        unaligned_after_array, 25, axis=0
                    )
                    unaligned_after_q75 = np.nanpercentile(
                        unaligned_after_array, 75, axis=0
                    )
                    unaligned_after_time = np.arange(
                        time_offset, time_offset + max_unaligned_after_len
                    )

                    ax.plot(
                        unaligned_after_time,
                        unaligned_after_median,
                        "--",
                        color=colors[1],
                        linewidth=2,
                        label="Unaligned after",
                        alpha=0.7,
                        zorder=2,
                    )
                    ax.fill_between(
                        unaligned_after_time,
                        unaligned_after_q25,
                        unaligned_after_q75,
                        alpha=0.2,
                        color=colors[1],
                        zorder=1,
                    )

            elif plot_type == "heatmap":
                # Stack all data for heatmap [before, aligned, after]
                # Use padded arrays so boundaries align across all lineages
                full_data = []
                for i in range(len(all_aligned)):
                    parts = []
                    # Add padded before segment
                    if unaligned_before_array.size > 0:
                        parts.append(unaligned_before_array[i])
                    # Add aligned
                    parts.append(all_aligned[i])
                    # Add padded after segment
                    if unaligned_after_array.size > 0:
                        parts.append(unaligned_after_array[i])

                    full_seq = np.concatenate(parts) if len(parts) > 1 else parts[0]
                    full_data.append(full_seq)

                # Stack into 2D array (all sequences should have same length now due to padding)
                heatmap_data = np.array(full_data)

                im = ax.imshow(
                    heatmap_data, aspect="auto", cmap=cmap, interpolation="nearest"
                )
                # Mark aligned region boundaries with gray lines
                if max_unaligned_before_len > 0:
                    ax.axvline(
                        max_unaligned_before_len - 0.5,
                        color="gray",
                        linewidth=2,
                        linestyle=":",
                        alpha=0.7,
                        label="Aligned region start",
                    )
                if max_unaligned_after_len > 0:
                    ax.axvline(
                        max_unaligned_before_len + consensus_length - 0.5,
                        color="gray",
                        linewidth=2,
                        linestyle=":",
                        alpha=0.7,
                        label="Aligned region end",
                    )
                plt.colorbar(im, ax=ax, label=feat_col)
                ax.set_ylabel("Lineage")

            elif plot_type == "individual_with_mean":
                # Plot all individual traces + mean overlay
                time_offset = 0

                # 1. Plot unaligned BEFORE individual traces
                if max_unaligned_before_len > 0:
                    for i in range(len(all_unaligned_before)):
                        unaligned_before_time = np.arange(
                            time_offset, time_offset + len(all_unaligned_before[i])
                        )
                        ax.plot(
                            unaligned_before_time,
                            all_unaligned_before[i],
                            "--",
                            color=colors[1],
                            alpha=0.2,
                            linewidth=1,
                            zorder=1,
                        )
                    time_offset += max_unaligned_before_len

                # 2. Plot ALIGNED individual traces
                aligned_time = np.arange(time_offset, time_offset + consensus_length)
                for i in range(len(all_aligned)):
                    ax.plot(
                        aligned_time,
                        all_aligned[i],
                        "-",
                        color=colors[0],
                        alpha=0.2,
                        linewidth=1,
                        zorder=1,
                    )
                time_offset += consensus_length

                # 3. Plot unaligned AFTER individual traces
                if max_unaligned_after_len > 0:
                    for i in range(len(all_unaligned_after)):
                        unaligned_after_time = np.arange(
                            time_offset, time_offset + len(all_unaligned_after[i])
                        )
                        ax.plot(
                            unaligned_after_time,
                            all_unaligned_after[i],
                            "--",
                            color=colors[1],
                            alpha=0.2,
                            linewidth=1,
                            zorder=1,
                        )

                # Overlay mean traces
                time_offset = 0

                # 1. Plot unaligned BEFORE mean
                if unaligned_before_array.size > 0:
                    unaligned_before_mean = np.nanmean(unaligned_before_array, axis=0)
                    unaligned_before_time = np.arange(
                        time_offset, time_offset + max_unaligned_before_len
                    )
                    ax.plot(
                        unaligned_before_time,
                        unaligned_before_mean,
                        "--",
                        color="black",
                        linewidth=3,
                        label="Mean (before)",
                        zorder=3,
                    )
                    time_offset += max_unaligned_before_len

                # 2. Plot ALIGNED mean
                aligned_mean = np.nanmean(aligned_array, axis=0)
                aligned_time = np.arange(time_offset, time_offset + consensus_length)
                ax.plot(
                    aligned_time,
                    aligned_mean,
                    "-",
                    color="black",
                    linewidth=3,
                    label="Mean (aligned)",
                    zorder=3,
                )
                time_offset += consensus_length

                # 3. Plot unaligned AFTER mean
                if unaligned_after_array.size > 0:
                    unaligned_after_mean = np.nanmean(unaligned_after_array, axis=0)
                    unaligned_after_time = np.arange(
                        time_offset, time_offset + max_unaligned_after_len
                    )
                    ax.plot(
                        unaligned_after_time,
                        unaligned_after_mean,
                        "--",
                        color="black",
                        linewidth=3,
                        label="Mean (after)",
                        zorder=3,
                    )

            # Mark infection state if available in consensus
            if self.consensus_data is not None:
                consensus_annotations = self.consensus_data.get("annotations", None)
                if consensus_annotations and "infected" in consensus_annotations:
                    infection_idx = consensus_annotations.index("infected")
                    if plot_type == "heatmap":
                        # Offset by max_unaligned_before_len to align with padded structure
                        infection_t = max_unaligned_before_len + infection_idx
                        ax.axvline(
                            infection_t - 0.5,  # -0.5 for proper pixel alignment
                            color="orange",
                            linewidth=2.5,
                            linestyle="-",
                            alpha=0.9,
                            label="Infection",
                        )
                    else:
                        # For line plots, use time offset
                        infection_t = max_unaligned_before_len + infection_idx
                        ax.axvline(
                            infection_t,
                            color="orange",
                            alpha=0.7,
                            linestyle="--",
                            linewidth=2.5,
                            label="Infection",
                        )

            # Mark alignment boundary
            if plot_type != "heatmap":
                ax.axvline(
                    consensus_length,
                    color="gray",
                    alpha=0.5,
                    linestyle=":",
                    linewidth=2,
                )
                ax.text(
                    consensus_length,
                    ax.get_ylim()[1],
                    " Alignment end",
                    rotation=90,
                    verticalalignment="top",
                    fontsize=9,
                    alpha=0.7,
                )

            # Plot consensus reference
            if plot_type != "heatmap":
                consensus_values = consensus_df[feat_col].values.copy()
                consensus_time = np.arange(len(consensus_values))
                ax.plot(
                    consensus_time,
                    consensus_values,
                    "o-",
                    color="black",
                    linewidth=2,
                    markersize=6,
                    label="Consensus",
                    alpha=0.6,
                    zorder=4,
                )

            # Set y-axis limits based on outlier bounds if requested
            if remove_outliers and plot_type != "heatmap":
                ax.set_ylim(lower_bound, upper_bound)

            ax.set_ylabel(feat_col)
            if feat_idx == 0:
                ax.legend(loc="best")
            if feat_idx == n_features - 1:
                ax.set_xlabel("Time: [Aligned] | [Unaligned continuation]")
            ax.set_title(
                f"{feat_col} - {plot_type} (n={len(concatenated_seqs)} lineages)"
            )
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_sample_patterns(
        self,
        annotated_samples: list[AnnotatedSample],
        reference_type: str = "features",
        n_pca_components: int = 3,
        figsize: tuple = None,
    ):
        """
        Plot PCA projections of reference patterns with annotations before consensus creation.

        This method provides quality control visualization of the reference patterns
        before creating a consensus. It extracts patterns from annotated samples,
        computes PCA for dimensionality reduction, and plots each pattern's trajectory
        in PC space with annotation markers (e.g., mitosis, infection).

        Parameters
        ----------
        annotated_samples : list[AnnotatedSample]
            List of annotated examples to extract and plot patterns from. Each should contain:
            - fov_name: str - FOV identifier
            - track_id: int | list[int] - Track ID(s)
            - timepoints: tuple[int, int] - (start, end) timepoints
            - annotations: list - Optional annotations for each timepoint
        reference_type : str
            Type of embedding to use ("features" for X, or obsm key like "X_PCA")
        n_pca_components : int
            Number of PCA components to compute and plot (default 3)
        figsize : tuple, optional
            Figure size (width, height). If None, auto-computed based on number of patterns.

        Returns
        -------
        matplotlib.figure.Figure
            The created matplotlib figure

        Examples
        --------
        >>> cytodtw = CytoDtw(adata)
        >>> samples = [
        ...     {'fov_name': 'A/2/001', 'track_id': [10],
        ...      'timepoints': (0, 30), 'annotations': ['G1']*15 + ['mitosis'] + ['G1']*14}
        ... ]
        >>> fig = cytodtw.plot_sample_patterns(samples)
        """
        import matplotlib.pyplot as plt

        # Extract patterns from annotated samples
        patterns = []
        pattern_info = []

        for i, example in enumerate(annotated_samples):
            pattern = self.get_reference_pattern(
                fov_name=example["fov_name"],
                track_id=example["track_id"],
                timepoints=example["timepoints"],
                reference_type=reference_type,
            )
            patterns.append(pattern)
            pattern_info.append(
                {
                    "index": i,
                    "fov_name": example["fov_name"],
                    "track_id": example["track_id"],
                    "timepoints": example["timepoints"],
                    "annotations": example.get("annotations", None),
                }
            )

        # Concatenate all patterns and fit PCA
        all_patterns_concat = np.vstack(patterns)
        scaler = StandardScaler()
        scaled_patterns = scaler.fit_transform(all_patterns_concat)
        pca = PCA(n_components=n_pca_components)
        pca.fit(scaled_patterns)

        # Create figure
        n_patterns = len(patterns)
        if figsize is None:
            figsize = (12, 3 * n_patterns)

        fig, axes = plt.subplots(n_patterns, n_pca_components, figsize=figsize)
        if n_patterns == 1:
            axes = axes.reshape(1, -1)

        # Plot each pattern
        for i, (pattern, info) in enumerate(zip(patterns, pattern_info)):
            scaled_pattern = scaler.transform(pattern)
            pc_pattern = pca.transform(scaled_pattern)
            time_axis = np.arange(len(pattern))

            for pc_idx in range(n_pca_components):
                ax = axes[i, pc_idx]

                ax.plot(
                    time_axis,
                    pc_pattern[:, pc_idx],
                    "o-",
                    color="blue",
                    linewidth=2,
                    markersize=4,
                )

                # Add annotation markers
                annotations = info.get("annotations")
                if annotations:
                    for t, annotation in enumerate(annotations):
                        if annotation == "mitosis":
                            ax.axvline(
                                t,
                                color="orange",
                                alpha=0.7,
                                linestyle="--",
                                linewidth=2,
                            )
                            ax.scatter(
                                t, pc_pattern[t, pc_idx], c="orange", s=50, zorder=5
                            )
                        elif annotation == "infected":
                            ax.axvline(
                                t, color="red", alpha=0.5, linestyle="--", linewidth=1
                            )
                            ax.scatter(
                                t, pc_pattern[t, pc_idx], c="red", s=30, zorder=5
                            )
                            break  # Only mark the first infection timepoint

                ax.set_xlabel("Time")
                ax.set_ylabel(f"PC{pc_idx + 1}")
                ax.set_title(
                    f"Pattern {i + 1}: FOV {info['fov_name']}, Tracks {info['track_id']}\nPC{pc_idx + 1} over time"
                )
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_consensus_validation(
        self,
        annotated_samples: list[AnnotatedSample],
        reference_type: str = "features",
        metric: str = "cosine",
        constraint_type: str = "unconstrained",
        band_width_ratio: float = 0.0,
        n_pca_components: int = 3,
        figsize: tuple = (18, 5),
    ):
        """
        Plot DTW-aligned reference patterns overlaid with consensus for quality validation.

        This method validates consensus quality by aligning all reference patterns to the
        consensus using DTW, then visualizing them together in PCA space. This helps verify
        that the consensus captures the common structure across all reference patterns.

        Parameters
        ----------
        annotated_samples : list[AnnotatedSample]
            List of annotated examples that were used to create the consensus
        reference_type : str
            Type of embedding to use ("features" for X, or obsm key like "X_PCA")
        metric : str
            Distance metric for DTW alignment (default "cosine")
        constraint_type : str
            DTW constraint type ("unconstrained", "sakoe_chiba")
        band_width_ratio : float
            DTW band width ratio for Sakoe-Chiba constraint (default 0.0)
        n_pca_components : int
            Number of PCA components to compute and plot (default 3)
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        matplotlib.figure.Figure
            The created matplotlib figure

        Raises
        ------
        ValueError
            If consensus_data has not been set in the instance

        Examples
        --------
        >>> cytodtw = CytoDtw(adata)
        >>> samples = [...]
        >>> consensus = cytodtw.create_consensus_reference_pattern(samples)
        >>> fig = cytodtw.plot_consensus_validation(samples)
        """
        import matplotlib.pyplot as plt

        if self.consensus_data is None:
            raise ValueError(
                "Consensus data not found. Please create or load consensus first using "
                "create_consensus_reference_pattern() or load_consensus()."
            )

        consensus_lineage = self.consensus_data["pattern"]
        consensus_annotations = self.consensus_data.get("annotations", None)

        # Extract and align patterns to consensus
        aligned_patterns_list = []
        aligned_annotations_list = []
        all_patterns_for_pca = []

        for i, example in enumerate(annotated_samples):
            # Extract pattern
            pattern = self.get_reference_pattern(
                fov_name=example["fov_name"],
                track_id=example["track_id"],
                timepoints=example["timepoints"],
                reference_type=reference_type,
            )

            # Align to consensus
            if len(pattern) == len(consensus_lineage):
                # Already same length, likely the reference pattern
                aligned_patterns_list.append(pattern)
                aligned_annotations_list.append(example.get("annotations", None))
            else:
                # Align to consensus using DTW
                alignment_result = align_embedding_patterns(
                    query_pattern=pattern,
                    reference_pattern=consensus_lineage,
                    metric=metric,
                    query_annotations=example.get("annotations", None),
                    constraint_type=constraint_type,
                    band_width_ratio=band_width_ratio,
                )
                aligned_patterns_list.append(alignment_result["pattern"])
                aligned_annotations_list.append(
                    alignment_result.get("annotations", None)
                )

            all_patterns_for_pca.append(aligned_patterns_list[-1])

        # Add consensus to patterns for PCA fitting
        all_patterns_for_pca.append(consensus_lineage)
        all_patterns_concat = np.vstack(all_patterns_for_pca)

        # Fit PCA on all patterns (aligned references + consensus)
        scaler = StandardScaler()
        scaled_patterns = scaler.fit_transform(all_patterns_concat)
        pca = PCA(n_components=n_pca_components)
        pca.fit(scaled_patterns)

        # Create figure
        fig, axes = plt.subplots(1, n_pca_components, figsize=figsize)
        if n_pca_components == 1:
            axes = [axes]

        # Plot each aligned pattern
        for pc_idx in range(n_pca_components):
            ax = axes[pc_idx]

            # Transform each aligned pattern to PC space and plot
            for i, pattern in enumerate(aligned_patterns_list):
                scaled_ref = scaler.transform(pattern)
                pc_ref = pca.transform(scaled_ref)

                time_axis = np.arange(len(pc_ref))
                ax.plot(
                    time_axis,
                    pc_ref[:, pc_idx],
                    "o-",
                    label=f"Ref {i + 1}",
                    alpha=0.7,
                    linewidth=2,
                    markersize=4,
                )

            # Overlay consensus pattern
            scaled_consensus = scaler.transform(consensus_lineage)
            pc_consensus = pca.transform(scaled_consensus)
            time_axis = np.arange(len(pc_consensus))
            ax.plot(
                time_axis,
                pc_consensus[:, pc_idx],
                "s-",
                color="black",
                linewidth=3,
                markersize=6,
                label="Consensus",
                zorder=10,
            )

            # Mark consensus infection timepoint with a thicker, more prominent line
            if consensus_annotations and "infected" in consensus_annotations:
                consensus_infection_t = consensus_annotations.index("infected")
                ax.axvline(
                    consensus_infection_t,
                    color="orange",
                    alpha=0.9,
                    linestyle="--",
                    linewidth=2.5,
                    label="Infection",
                )

            ax.set_xlabel("Aligned Time")
            ax.set_ylabel(f"PC{pc_idx + 1}")
            ax.set_title(f"PC{pc_idx + 1}: All DTW-Aligned References + Consensus")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.suptitle(
            "Consensus Validation: DTW-Aligned References + Computed Consensus",
            fontsize=14,
        )
        plt.tight_layout()
        return fig

    def plot_individual_lineages(
        self,
        df: pd.DataFrame,
        alignment_name: str = "cell_division",
        feature_columns: list[str] = None,
        max_lineages: int = 5,
        y_offset_step: float = 0.0,
        figsize: tuple = (15, 12),
        aligned_linewidth: float = 2.5,
        unaligned_linewidth: float = 1.0,
        aligned_markersize: float = 4.0,
        unaligned_markersize: float = 2.0,
        remove_outliers: bool = False,
        outlier_percentile: tuple = (1, 99),
    ):
        """
        Plot individual lineages with y-offsets (waterfall plot).

        Each lineage is shown as a separate trace with vertical offset for clarity.
        The aligned portion is highlighted with thicker lines/markers.

        Parameters
        ----------
        df : pd.DataFrame
            Enhanced DataFrame with alignment information
        alignment_name : str
            Name of alignment type
        feature_columns : list[str], optional
            Feature columns to plot. If None, uses PC1, PC2, PC3
        max_lineages : int
            Maximum number of lineages to display
        y_offset_step : float
            Vertical separation between lineages
        figsize : tuple
            Figure size
        aligned_linewidth : float
            Line width for aligned portions
        unaligned_linewidth : float
            Line width for unaligned portions
        aligned_markersize : float
            Marker size for aligned portions
        unaligned_markersize : float
            Marker size for unaligned portions
        remove_outliers : bool
            Whether to clip outlier values for better visualization
        outlier_percentile : tuple
            (lower, upper) percentile bounds for clipping

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        import matplotlib.pyplot as plt

        if feature_columns is None:
            feature_columns = ["PC1", "PC2", "PC3"]

        # Get concatenated sequences
        concatenated_seqs = self.get_concatenated_sequences(
            df=df,
            alignment_name=alignment_name,
            feature_columns=feature_columns,
            max_lineages=max_lineages,
        )

        if not concatenated_seqs:
            _logger.error("No concatenated sequences found")
            return None

        # Get consensus for reference
        consensus_df = df[df["lineage_id"] == -1].sort_values("t").copy()
        if consensus_df.empty:
            _logger.error("No consensus found in DataFrame")
            return None

        # Find maximum "before" length across all lineages to align them
        max_unaligned_before_length = max(
            [
                seq_data["unaligned_before_data"]["length"]
                for seq_data in concatenated_seqs.values()
            ],
            default=0,
        )

        # Prepare concatenated lineages data with padding
        concatenated_lineages = {}
        for lineage_id, seq_data in concatenated_seqs.items():
            unaligned_before_features = seq_data["unaligned_before_data"].get(
                "features", {}
            )
            aligned_features = seq_data["aligned_data"]["features"]
            unaligned_after_features = seq_data["unaligned_after_data"].get(
                "features", {}
            )

            unaligned_before_length = seq_data["unaligned_before_data"]["length"]
            padding_needed = max_unaligned_before_length - unaligned_before_length

            concatenated_arrays = {}
            for col in feature_columns:
                # Pad "before" segment to max length for alignment
                parts = []
                if (
                    col in unaligned_before_features
                    and len(unaligned_before_features[col]) > 0
                ):
                    before_vals = unaligned_before_features[col]
                    if padding_needed > 0:
                        # Prepend NaN padding
                        before_padded = np.concatenate(
                            [np.full(padding_needed, np.nan), before_vals]
                        )
                    else:
                        before_padded = before_vals
                    parts.append(before_padded)
                else:
                    # No before data - pad with full max length
                    parts.append(np.full(max_unaligned_before_length, np.nan))

                # Add aligned and after portions
                parts.append(aligned_features[col])
                if (
                    col in unaligned_after_features
                    and len(unaligned_after_features[col]) > 0
                ):
                    parts.append(unaligned_after_features[col])

                concatenated_arrays[col] = np.concatenate(parts)

            concatenated_lineages[lineage_id] = {
                "concatenated": concatenated_arrays,
                "unaligned_before_length": seq_data["unaligned_before_data"]["length"],
                "aligned_length": seq_data["aligned_data"]["length"],
                "unaligned_after_length": seq_data["unaligned_after_data"]["length"],
                "dtw_distance": seq_data["metadata"]["dtw_distance"],
                "fov_name": seq_data["metadata"]["fov_name"],
                "track_ids": seq_data["metadata"]["track_ids"],
            }

        # Compute outlier bounds per feature if requested
        outlier_bounds = {}
        if remove_outliers:
            for feat_col in feature_columns:
                all_values = []
                all_values.extend(consensus_df[feat_col].values)
                for data in concatenated_lineages.values():
                    all_values.extend(data["concatenated"][feat_col])

                all_values = np.array(all_values)
                all_values = all_values[~np.isnan(all_values)]

                if len(all_values) > 0:
                    lower_bound = np.percentile(all_values, outlier_percentile[0])
                    upper_bound = np.percentile(all_values, outlier_percentile[1])
                    outlier_bounds[feat_col] = (lower_bound, upper_bound)
                    _logger.info(
                        f"{feat_col}: removing outliers outside [{lower_bound:.3f}, {upper_bound:.3f}]"
                    )

        n_features = len(feature_columns)
        fig, axes = plt.subplots(n_features, 1, figsize=figsize)
        if n_features == 1:
            axes = [axes]

        cmap = plt.cm.get_cmap(
            "tab10"
            if len(concatenated_lineages) <= 10
            else "tab20"
            if len(concatenated_lineages) <= 20
            else "hsv"
        )
        colors = [
            cmap(i / max(len(concatenated_lineages), 1))
            for i in range(len(concatenated_lineages))
        ]

        for feat_idx, feat_col in enumerate(feature_columns):
            ax = axes[feat_idx]

            # Plot consensus at offset where aligned regions are
            consensus_values = consensus_df[feat_col].values.copy()
            consensus_time = np.arange(
                max_unaligned_before_length,
                max_unaligned_before_length + len(consensus_values),
            )
            ax.plot(
                consensus_time,
                consensus_values,
                "o-",
                color="black",
                linewidth=4,
                markersize=8,
                label=f"Consensus ({alignment_name})",
                alpha=0.9,
                zorder=5,
            )

            for lineage_idx, (lineage_id, data) in enumerate(
                concatenated_lineages.items()
            ):
                y_offset = -(lineage_idx + 1) * y_offset_step
                color = colors[lineage_idx]

                concat_values = data["concatenated"][feat_col].copy() + y_offset
                time_axis = np.arange(len(concat_values))

                track_id_str = ",".join(map(str, data["track_ids"]))
                ax.plot(
                    time_axis,
                    concat_values,
                    ".-",
                    color=color,
                    linewidth=unaligned_linewidth,
                    markersize=unaligned_markersize,
                    alpha=0.8,
                    label=f"{data['fov_name']}, track={track_id_str} (d={data['dtw_distance']:.3f})",
                )

                # Overlay aligned portion with thick lines
                # All aligned portions now start at max_unaligned_before_length due to padding
                aligned_length = data["aligned_length"]
                aligned_start = max_unaligned_before_length
                aligned_end = aligned_start + aligned_length

                if aligned_length > 0:
                    aligned_time = time_axis[aligned_start:aligned_end]
                    aligned_vals = concat_values[aligned_start:aligned_end]
                    ax.plot(
                        aligned_time,
                        aligned_vals,
                        "o-",
                        color=color,
                        linewidth=aligned_linewidth,
                        markersize=aligned_markersize,
                        alpha=1.0,
                        zorder=3,
                    )

                # Add vertical lines at aligned region boundaries
                if max_unaligned_before_length > 0 and aligned_length > 0:
                    ax.axvline(
                        aligned_start - 0.5,
                        color=color,
                        linestyle=":",
                        alpha=0.3,
                        linewidth=1,
                    )
                if aligned_length > 0 and data["unaligned_after_length"] > 0:
                    ax.axvline(
                        aligned_end - 0.5,
                        color=color,
                        linestyle=":",
                        alpha=0.3,
                        linewidth=1,
                    )

            # Mark infection state if available in consensus
            if self.consensus_data is not None:
                consensus_annotations = self.consensus_data.get("annotations", None)
                if consensus_annotations and "infected" in consensus_annotations:
                    infection_idx = consensus_annotations.index("infected")
                    # Offset by max_unaligned_before_length to align with consensus position
                    infection_t = max_unaligned_before_length + infection_idx
                    ax.axvline(
                        infection_t,
                        color="orange",
                        alpha=0.7,
                        linestyle="--",
                        linewidth=2.5,
                        label="Infection",
                        zorder=6,
                    )

            if remove_outliers and feat_col in outlier_bounds:
                lower, upper = outlier_bounds[feat_col]
                max_offset = -(len(concatenated_lineages)) * y_offset_step
                ax.set_ylim(max_offset + lower - 1, upper + 1)

            ax.set_ylabel(feat_col)
            ax.set_xlabel("Time: [before alignment] + [aligned] + [after alignment]")
            ax.set_title(
                f"{feat_col} - Individual lineages (n={len(concatenated_lineages)})"
            )
            ax.legend(loc="best", fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def get_aligned_image_sequences(
    cytodtw_instance,
    df: pd.DataFrame,
    alignment_name: str,
    image_loader_fn,
    max_lineages: int = None,
) -> dict:
    """
    Create concatenated [unaligned_before + aligned + unaligned_after] image sequences from alignment.

    This is a generic function that works with any dataset by accepting a custom
    image loader function.

    Parameters
    ----------
    cytodtw_instance : CytoDtw
        CytoDtw instance with get_concatenated_sequences method
    df : pd.DataFrame
        Enhanced DataFrame with alignment information
    alignment_name : str
        Name of alignment to use (e.g., "cell_division", "infection_state")
    image_loader_fn : callable
        Function that takes (fov_name, track_ids) and returns a dict mapping
        timepoint -> image_data. Signature: fn(fov_name, track_ids) -> {t: img_data}
    max_lineages : int, optional
        Maximum number of lineages to process

    Returns
    -------
    dict
        Dictionary mapping lineage_id to:
        - 'concatenated_images': List of concatenated images (before + aligned + after)
        - 'unaligned_before_length': Number of unaligned images before aligned region
        - 'aligned_length': Number of DTW-aligned images
        - 'unaligned_after_length': Number of unaligned images after aligned region
        - 'unaligned_length': Total number of unaligned images (before + after, for backward compatibility)
        - 'metadata': Lineage metadata (fov_name, track_ids, dtw_distance, etc.)

    Examples
    --------
    >>> # Define custom image loader for your dataset
    >>> def load_images(fov_name, track_ids):
    ...     # Your dataset-specific loading logic
    ...     images = dataset.get_images_for_tracks(fov_name, track_ids)
    ...     return {img['t']: img for img in images}
    >>>
    >>> # Get image sequences
    >>> image_seqs = get_image_sequences_from_alignment(
    ...     cytodtw, alignment_df, "cell_division",
    ...     image_loader_fn=load_images, max_lineages=10
    ... )
    """
    aligned_col = f"dtw_{alignment_name}_aligned"
    if aligned_col not in df.columns:
        _logger.error(f"Alignment '{alignment_name}' not found in DataFrame")
        return {}

    consensus_df = df[df["lineage_id"] == -1].sort_values("t").copy()
    if consensus_df.empty:
        _logger.error("No consensus found in DataFrame")
        return {}

    concatenated_seqs = cytodtw_instance.get_concatenated_sequences(
        df=df,
        alignment_name=alignment_name,
        feature_columns=None,
        max_lineages=max_lineages,
    )

    if not concatenated_seqs:
        _logger.error("No concatenated sequences found")
        return {}

    concatenated_image_sequences = {}

    for lineage_id, seq_data in concatenated_seqs.items():
        fov_name = seq_data["metadata"]["fov_name"]
        track_ids = seq_data["metadata"]["track_ids"]

        try:
            time_to_image = image_loader_fn(fov_name, track_ids)
        except Exception as e:
            _logger.warning(f"Failed to load images for lineage {lineage_id}: {e}")
            continue

        if not time_to_image:
            _logger.warning(
                f"No images found for lineage {lineage_id}, FOV {fov_name}, tracks {track_ids}"
            )
            continue

        aligned_mapping = seq_data["aligned_data"]["mapping"]
        consensus_length = seq_data["metadata"]["consensus_length"]
        aligned_images = [None] * consensus_length

        # pseudotime-indexed image array for the aligned portion
        for i in range(consensus_length):
            if i in aligned_mapping:
                timepoint = aligned_mapping[i]["t"]
                if timepoint in time_to_image:
                    aligned_images[i] = time_to_image[timepoint]
                else:
                    available_times = list(time_to_image.keys())
                    if available_times:
                        closest_time = min(
                            available_times, key=lambda x: abs(x - timepoint)
                        )
                        aligned_images[i] = time_to_image[closest_time]

        # Gap filling for any missing aligned images
        for i in range(consensus_length):
            if aligned_images[i] is None:
                available_indices = [
                    j for j, img in enumerate(aligned_images) if img is not None
                ]
                if available_indices:
                    closest_idx = min(available_indices, key=lambda x: abs(x - i))
                    aligned_images[i] = aligned_images[closest_idx]
                elif time_to_image:
                    aligned_images[i] = next(iter(time_to_image.values()))

        # Map unaligned BEFORE portion
        unaligned_before_images = []
        unaligned_before_rows = seq_data["unaligned_before_data"]["rows"]
        if not unaligned_before_rows.empty:
            unaligned_before_rows = unaligned_before_rows.sort_values("t")
            for _, row in unaligned_before_rows.iterrows():
                timepoint = row["t"]
                if timepoint in time_to_image:
                    unaligned_before_images.append(time_to_image[timepoint])
                else:
                    # Find closest available time
                    available_times = list(time_to_image.keys())
                    if available_times:
                        closest_time = min(
                            available_times, key=lambda x: abs(x - timepoint)
                        )
                        unaligned_before_images.append(time_to_image[closest_time])

        # Map unaligned AFTER portion
        unaligned_after_images = []
        unaligned_after_rows = seq_data["unaligned_after_data"]["rows"]
        if not unaligned_after_rows.empty:
            unaligned_after_rows = unaligned_after_rows.sort_values("t")
            for _, row in unaligned_after_rows.iterrows():
                timepoint = row["t"]
                if timepoint in time_to_image:
                    unaligned_after_images.append(time_to_image[timepoint])
                else:
                    # Find closest available time
                    available_times = list(time_to_image.keys())
                    if available_times:
                        closest_time = min(
                            available_times, key=lambda x: abs(x - timepoint)
                        )
                        unaligned_after_images.append(time_to_image[closest_time])

        # Concatenate in order: [before, aligned, after]
        concatenated_images = (
            unaligned_before_images + aligned_images + unaligned_after_images
        )

        concatenated_image_sequences[lineage_id] = {
            "concatenated_images": concatenated_images,
            "unaligned_before_length": len(unaligned_before_images),
            "aligned_length": len(aligned_images),
            "unaligned_after_length": len(unaligned_after_images),
            "unaligned_length": len(unaligned_before_images)
            + len(unaligned_after_images),  # Keep for backward compatibility
            "metadata": seq_data["metadata"],
        }

    _logger.debug(
        f"Created image sequences for {len(concatenated_image_sequences)} lineages"
    )
    return concatenated_image_sequences


def create_synchronized_warped_sequences(
    concatenated_image_sequences: dict,
    alignment_df: pd.DataFrame,
    alignment_name: str,
    consensus_infection_idx: int,
    time_interval_minutes: float = 30,
) -> dict:
    """
    Create synchronized warped sequences with zero-padding and HPI metadata tracking.

    This function takes concatenated image sequences (from get_aligned_image_sequences)
    and creates synchronized warped sequences where all cells' aligned regions start
    at the same frame index. It also computes metadata for mapping to Hours Post Infection (HPI).

    Parameters
    ----------
    concatenated_image_sequences : dict
        Output from get_aligned_image_sequences(), mapping lineage_id to:
        - 'concatenated_images': List of [before + aligned + after] images
        - 'unaligned_before_length': Number of frames before aligned region
        - 'aligned_length': Number of aligned frames (consensus_length)
        - 'unaligned_after_length': Number of frames after aligned region
        - 'metadata': Lineage metadata (fov_name, track_ids, etc.)
    alignment_df : pd.DataFrame
        Enhanced DataFrame with DTW alignment columns including:
        - 'fov_name', 'track_id', 't' (absolute time)
        - f'dtw_{alignment_name}_aligned': Boolean marking aligned frames
        - f'dtw_{alignment_name}_consensus_mapping': Pseudotime index (0 to consensus_length-1)
    alignment_name : str
        Name of alignment (e.g., "cell_division", "infection_state")
    consensus_infection_idx : int
        Pseudotime index where biological event occurs (e.g., infection starts)
        This is used to map the event to absolute time for each cell.
    time_interval_minutes : float, optional
        Time between frames in minutes (default: 30)

    Returns
    -------
    dict
        Dictionary containing:
        - 'warped_sequences': dict mapping lineage_id to synchronized numpy arrays with shape
          (time, channels, z, y, x). All arrays have the same time dimension.
        - 'alignment_shifts': dict mapping lineage_id to:
            - 'fov_name': FOV name
            - 'track_ids': List of track IDs
            - 'first_aligned_t': Absolute time where pseudotime 0 starts
            - 'infection_t_abs': Absolute time where biological event occurs
            - 'shift': Value to add to absolute time for pseudotime-anchored coordinates
            - 'infection_offset_in_viz': Frame offset where event occurs in warped coordinates
        - 'warped_metadata': dict with:
            - 'max_unaligned_before': Maximum frames before aligned region across all cells
            - 'max_unaligned_after': Maximum frames after aligned region across all cells
            - 'consensus_aligned_length': Length of aligned region (same for all cells)
            - 'time_interval_minutes': Time interval for HPI conversion

    Notes
    -----
    Coordinate Systems:
    1. Absolute time (t_abs): Original experimental timepoints
    2. Pseudotime (consensus_idx): DTW-aligned indices (0 to consensus_length-1)
    3. Visualization time (t_viz): Pseudotime-anchored coordinates where t_viz=0 at pseudotime 0
    4. Warped frame index: Synchronized frame indices in output arrays
    5. Hours Post Infection (HPI): Biological time relative to event

    Conversions:
    - t_viz = t_abs + shift (where shift = -first_aligned_t)
    - HPI = (t_abs - infection_t_abs) * (time_interval_minutes / 60)
    - warped_frame_idx = max_unaligned_before + (varies by cell's history)

    Warped Time Structure:
    - Frames [0 to max_unaligned_before-1]: Before aligned region (padded with zeros where needed)
    - Frames [max_unaligned_before to max_unaligned_before+consensus_length-1]: ALIGNED region (synchronized!)
    - Frames [max_unaligned_before+consensus_length to end]: After aligned region (padded where needed)

    Examples
    --------
    >>> result = create_synchronized_warped_sequences(
    ...     concatenated_image_sequences,
    ...     alignment_df,
    ...     alignment_name="infection_state",
    ...     consensus_infection_idx=5,
    ...     time_interval_minutes=30
    ... )
    >>> warped_arrays = result['warped_sequences']
    >>> shifts = result['alignment_shifts']
    >>>
    >>> # Get Hours Post Infection for a specific cell at absolute time t=25
    >>> lineage_id = 1
    >>> t_abs = 25
    >>> infection_t_abs = shifts[lineage_id]['infection_t_abs']
    >>> hpi = (t_abs - infection_t_abs) * (30 / 60)  # Convert to hours
    """
    import numpy as np

    if len(concatenated_image_sequences) == 0:
        _logger.warning("No concatenated sequences provided")
        return {
            "warped_sequences": {},
            "alignment_shifts": {},
            "warped_metadata": {},
        }

    # Step 1: Find maximum unaligned lengths across all cells
    max_unaligned_before = 0
    max_unaligned_after = 0
    consensus_aligned_length = None

    for lineage_id, seq_data in concatenated_image_sequences.items():
        max_unaligned_before = max(
            max_unaligned_before, seq_data["unaligned_before_length"]
        )
        max_unaligned_after = max(
            max_unaligned_after, seq_data["unaligned_after_length"]
        )
        if consensus_aligned_length is None:
            consensus_aligned_length = seq_data["aligned_length"]

    _logger.info(f"Consensus aligned length: {consensus_aligned_length}")
    _logger.info(f"Max unaligned before across all cells: {max_unaligned_before}")
    _logger.info(f"Max unaligned after across all cells: {max_unaligned_after}")

    total_warped_length = (
        max_unaligned_before + consensus_aligned_length + max_unaligned_after
    )
    _logger.info(f"Total synchronized warped time length: {total_warped_length}")

    # Prepare output dictionaries
    warped_sequences = {}
    alignment_shifts = {}

    # Column names for alignment
    aligned_col = f"dtw_{alignment_name}_aligned"
    consensus_mapping_col = f"dtw_{alignment_name}_consensus_mapping"

    # Step 2: Process each lineage
    for lineage_id, seq_data in concatenated_image_sequences.items():
        meta = seq_data["metadata"]
        fov_name = meta["fov_name"]
        track_ids = meta["track_ids"]
        concatenated_images = seq_data["concatenated_images"]

        unaligned_before_length = seq_data["unaligned_before_length"]
        seq_data["aligned_length"]
        unaligned_after_length = seq_data["unaligned_after_length"]

        # Extract images from concatenated sequence
        image_stack = []
        for img_sample in concatenated_images:
            if img_sample is not None:
                img_tensor = img_sample["anchor"]
                img_np = img_tensor.cpu().numpy()
                image_stack.append(img_np)

        if len(image_stack) == 0:
            _logger.warning(f"No images found for lineage {lineage_id}")
            continue

        # Stack the raw concatenated sequence
        concatenated_stack = np.stack(image_stack, axis=0)

        # Compute padding needed for synchronization
        pad_before = max_unaligned_before - unaligned_before_length
        pad_after = max_unaligned_after - unaligned_after_length

        # Create zero frames for padding (same shape as images)
        dummy_frame = np.zeros_like(concatenated_stack[0])

        # Pad before and after
        padded_before = [dummy_frame] * pad_before
        padded_after = [dummy_frame] * pad_after

        # Create synchronized sequence
        synchronized_frames = padded_before + list(concatenated_stack) + padded_after
        warped_time_series = np.stack(synchronized_frames, axis=0)

        warped_sequences[lineage_id] = warped_time_series

        # Step 3: Compute alignment shifts and HPI metadata
        # Get alignment information for this lineage from alignment_df
        lineage_rows = alignment_df[
            (alignment_df["fov_name"] == fov_name)
            & (alignment_df["track_id"].isin(track_ids))
        ].copy()

        if len(lineage_rows) == 0:
            _logger.warning(
                f"No alignment data found for lineage {lineage_id} (FOV: {fov_name}, tracks: {track_ids})"
            )
            continue

        # Find aligned rows (where DTW matched)
        aligned_rows = lineage_rows[lineage_rows[aligned_col]].copy()

        if len(aligned_rows) == 0:
            _logger.warning(f"No aligned frames found for lineage {lineage_id}")
            continue

        # Find first_aligned_t: absolute time where pseudotime 0 starts
        # This is the minimum absolute time among frames mapped to consensus_idx == 0
        pseudotime_zero_rows = aligned_rows[
            aligned_rows[consensus_mapping_col] == 0
        ].copy()

        if len(pseudotime_zero_rows) > 0:
            first_aligned_t = pseudotime_zero_rows["t"].min()
        else:
            # Fallback: use minimum aligned time
            first_aligned_t = aligned_rows["t"].min()
            _logger.warning(
                f"No consensus_idx=0 found for lineage {lineage_id}, using min aligned time"
            )

        # Find infection_t_abs: absolute time where biological event occurs
        infection_rows = aligned_rows[
            aligned_rows[consensus_mapping_col] == consensus_infection_idx
        ].copy()

        if len(infection_rows) > 0:
            infection_t_abs = infection_rows["t"].iloc[0]
        else:
            # Fallback: estimate based on first_aligned_t
            infection_t_abs = first_aligned_t + consensus_infection_idx
            _logger.warning(
                f"No consensus_idx={consensus_infection_idx} found for lineage {lineage_id}, estimating infection time"
            )

        # Compute shift for pseudotime-anchored coordinates
        shift = -first_aligned_t

        # Compute infection offset in visualization coordinates
        infection_offset_in_viz = infection_t_abs - first_aligned_t

        # Store shift metadata
        alignment_shifts[lineage_id] = {
            "fov_name": fov_name,
            "track_ids": track_ids,
            "first_aligned_t": float(first_aligned_t),
            "infection_t_abs": float(infection_t_abs),
            "shift": float(shift),
            "infection_offset_in_viz": float(infection_offset_in_viz),
        }

        _logger.debug(
            f"Lineage {lineage_id}: first_aligned_t={first_aligned_t}, "
            f"infection_t_abs={infection_t_abs}, shift={shift}, "
            f"warped_shape={warped_time_series.shape}"
        )

    # Step 4: Create metadata summary
    warped_metadata = {
        "max_unaligned_before": int(max_unaligned_before),
        "max_unaligned_after": int(max_unaligned_after),
        "consensus_aligned_length": int(consensus_aligned_length),
        "time_interval_minutes": float(time_interval_minutes),
        "total_warped_length": int(total_warped_length),
        "aligned_region_start_idx": int(max_unaligned_before),
        "aligned_region_end_idx": int(max_unaligned_before + consensus_aligned_length),
    }

    _logger.info(
        f"Created synchronized warped sequences for {len(warped_sequences)} lineages"
    )
    _logger.info(
        f"Aligned region in warped coordinates: frames [{warped_metadata['aligned_region_start_idx']} to {warped_metadata['aligned_region_end_idx'] - 1}]"
    )

    return {
        "warped_sequences": warped_sequences,
        "alignment_shifts": alignment_shifts,
        "warped_metadata": warped_metadata,
    }


def compute_hpi_from_absolute_time(
    t_abs: float,
    alignment_shifts: dict,
    lineage_id: int,
    time_interval_minutes: float = 30,
) -> float:
    """
    Convert absolute time to Hours Post Infection (HPI) for a specific cell.

    Parameters
    ----------
    t_abs : float
        Absolute timepoint from experimental data
    alignment_shifts : dict
        Alignment shifts dictionary from create_synchronized_warped_sequences()
    lineage_id : int
        Lineage ID to compute HPI for
    time_interval_minutes : float, optional
        Time between frames in minutes (default: 30)

    Returns
    -------
    float
        Hours post infection (negative values = before infection)

    Examples
    --------
    >>> # Get HPI for cell 1 at absolute time t=25
    >>> hpi = compute_hpi_from_absolute_time(
    ...     t_abs=25,
    ...     alignment_shifts=shifts,
    ...     lineage_id=1,
    ...     time_interval_minutes=30
    ... )
    >>> print(f"t=25 corresponds to {hpi:.2f} hours post infection")
    """
    if lineage_id not in alignment_shifts:
        raise ValueError(f"Lineage {lineage_id} not found in alignment_shifts")

    infection_t_abs = alignment_shifts[lineage_id]["infection_t_abs"]
    hpi = (t_abs - infection_t_abs) * (time_interval_minutes / 60.0)
    return hpi


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
    reference_type: Literal["features", "X_PCA", "X_UMAP", "X_PHATE"] = "features",
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
            mask = (adata.obs["fov_name"] == fov_name) & (
                adata.obs["track_id"] == track_id
            )
            track_data = adata[mask]

            # Sort by timepoint to ensure correct order
            time_order = np.argsort(track_data.obs["t"].values)
            track_data = track_data[time_order]

            if reference_type == "features":
                track_embeddings = track_data.X
            else:
                # Assume it's an obsm key
                track_embeddings = track_data.obsm[reference_type]

            track_t = track_data.obs["t"].values

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

            # warping path is relative to the reference pattern
            # query_idx is relative to the lineage
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

    all_match_positions = pd.DataFrame(all_match_positions)
    all_match_positions = all_match_positions.dropna()

    all_match_positions = all_match_positions.sort_values(
        by=["distance", "skewness"], ascending=[True, True]
    )

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
        distance_matrix = cdist(reference_pattern, window, metric=metric)
        distance, _, path = dtw_with_matrix(
            distance_matrix,
            normalize=normalize,
            constraint_type=constraint_type,
            band_width_ratio=band_width_ratio,
        )
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
    """Create consensus pattern from one or more embedding patterns using DTW alignment.

    For single patterns, uses it directly. For multiple patterns, aligns them with DTW
    and aggregates.

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
        raise ValueError("At least one pattern is required")

    for pattern_id, pattern_data in patterns.items():
        if "pattern" not in pattern_data:
            raise ValueError(f"Pattern '{pattern_id}' missing 'pattern' key")
        if not isinstance(pattern_data["pattern"], np.ndarray):
            raise ValueError(f"Pattern '{pattern_id}' must be numpy array")

    # Handle single pattern case - use it directly as consensus
    if len(patterns) == 1:
        pattern_id = list(patterns.keys())[0]
        pattern_data = patterns[pattern_id]

        consensus = DtwSample(
            pattern=pattern_data["pattern"],
            annotations=pattern_data.get("annotations"),
            distance=np.nan,
            skewness=0.0,
            warping_path=[(i, i) for i in range(len(pattern_data["pattern"]))],
        )

        consensus["metadata"] = {
            "reference_pattern": pattern_id,
            "source_patterns": [pattern_id],
            "reference_selection": "single_pattern",
            "aggregation_method": "none",
            "n_patterns": 1,
        }

        return consensus

    # Multiple patterns - perform DTW alignment and aggregation
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
