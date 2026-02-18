"""
PHATE Track Viewer - Interactive visualization for PHATE embeddings with track timelines.

This package provides a modular framework for visualizing PHATE embeddings of
cell tracking data, with support for multiple datasets, flexible coloring modes,
and interactive track timelines.

Modules
-------
config : Configuration classes and constants
data_loading : Data loading and PHATE computation
visualization : PHATE embedding visualization
image_cache : Image loading and caching
timeline : Track timeline display
app : Dash application setup
cli : Command-line interface

Examples
--------
Basic usage:

    from visualizer import create_app, run_app, MultiDatasetConfig, DatasetConfig
    from pathlib import Path

    config = MultiDatasetConfig(
        datasets=[
            DatasetConfig(
                adata_path=Path("path/to/adata.zarr"),
                data_path=Path("path/to/data.zarr"),
            )
        ]
    )

    app = create_app(config)
    run_app(app)
"""

from .app import create_app, run_app
from .config import INFECTION_COLORS, DatasetConfig, MultiDatasetConfig
from .config_loader import (
    load_config_from_json,
    load_config_from_yaml,
    save_config_to_json,
    save_config_to_yaml,
)
from .data_loading import load_multiple_datasets

__version__ = "0.1.0"
__all__ = [
    "DatasetConfig",
    "MultiDatasetConfig",
    "INFECTION_COLORS",
    "load_multiple_datasets",
    "create_app",
    "run_app",
    "load_config_from_yaml",
    "load_config_from_json",
    "save_config_to_yaml",
    "save_config_to_json",
    "__version__",
]
