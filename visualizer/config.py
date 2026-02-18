"""
Configuration classes and constants for PHATE Track Viewer.

This module provides configuration data structures for single and multi-dataset
PHATE visualization applications.

Classes
-------
DatasetConfig : Configuration for a single dataset source
MultiDatasetConfig : Configuration for multi-dataset PHATE viewer

Constants
---------
INFECTION_COLORS : Colorblind-friendly palette for infection status
"""

from pathlib import Path
from typing import NamedTuple


class DatasetConfig(NamedTuple):
    """
    Configuration for a single dataset source.

    Parameters
    ----------
    adata_path : Path
        Path to AnnData zarr store with features.
    data_path : Path
        Path to image zarr store.
    channels : tuple[str, ...]
        Channel names to load from images (e.g., ("Phase3D", "Nuclei")).
    z_range : tuple[int, int]
        Z slice range (start, end) (e.g., (0, 1)).
    fov_filter : list[str] or None, optional
        FOV names or patterns to filter. If None, include all FOVs.
    annotation_csv : Path or None, optional
        Path to CSV file with annotations.
    annotation_column : str or None, optional
        Column name in annotation CSV.
    categories : dict or None, optional
        Dictionary to remap annotation categories.
    dataset_id : str, optional
        Unique identifier for this dataset (default: "").
    """

    adata_path: Path
    data_path: Path
    channels: tuple[str, ...]
    z_range: tuple[int, int]
    fov_filter: list[str] | None = None
    annotation_csv: Path | None = None
    annotation_column: str | None = None
    categories: dict | None = None
    dataset_id: str = ""


class MultiDatasetConfig(NamedTuple):
    """
    Configuration for multi-dataset PHATE viewer.

    Parameters
    ----------
    datasets : list[DatasetConfig]
        List of dataset configurations to load and merge. Each dataset must specify
        its own channels and z_range.
    phate_kwargs : dict or None, optional
        PHATE parameters for computing embeddings. If None, uses existing embeddings
        from AnnData (single dataset only). For multi-dataset, always computes joint PHATE.
        Example: {"n_components": 2, "knn": 5, "decay": 40, "scale_embeddings": False}
        Set to empty dict {} to force recomputation with default parameters.
    yx_patch_size : tuple[int, int], optional
        Patch size in (Y, X) dimensions (default: (160, 160)).
    port : int, optional
        Server port (default: 8050).
    debug : bool, optional
        Enable debug mode (default: False).
    default_color_mode : str, optional
        Default coloring mode: "annotation", "time", "track_id", or "dataset" (default: "annotation").
    """

    datasets: list[DatasetConfig]
    phate_kwargs: dict | None = None
    yx_patch_size: tuple[int, int] = (160, 160)
    port: int = 8050
    debug: bool = False
    default_color_mode: str = "annotation"


# Colorblind-friendly palette (blue/orange)
INFECTION_COLORS = {
    "uninfected": "#3498db",  # Blue
    "infected": "#e67e22",  # Orange
    "unknown": "#95a5a6",  # Gray
}
