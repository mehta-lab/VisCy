"""
Configuration file loading and saving for YAML and JSON formats.

This module provides functions for loading and saving visualizer configurations
from YAML and JSON files, enabling easy configuration management without
editing Python code.

Functions
---------
load_config_from_yaml : Load configuration from YAML file
load_config_from_json : Load configuration from JSON file
save_config_to_yaml : Save configuration to YAML file
save_config_to_json : Save configuration to JSON file
"""

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from .config import DatasetConfig, MultiDatasetConfig

logger = logging.getLogger(__name__)


def _parse_dataset_config(dataset_dict: dict[str, Any]) -> DatasetConfig:
    """
    Parse dataset configuration from dictionary.

    Parameters
    ----------
    dataset_dict : dict
        Dictionary containing dataset configuration.

    Returns
    -------
    DatasetConfig
        Parsed dataset configuration.
    """
    adata_path = Path(dataset_dict["adata_path"])
    data_path = Path(dataset_dict["data_path"])

    fov_filter = dataset_dict.get("fov_filter")
    annotation_csv = dataset_dict.get("annotation_csv")
    if annotation_csv is not None:
        annotation_csv = Path(annotation_csv)

    annotation_column = dataset_dict.get("annotation_column")
    categories = dataset_dict.get("categories")
    dataset_id = dataset_dict.get("dataset_id", "")

    if "channels" not in dataset_dict:
        raise ValueError("Each dataset must specify 'channels'")
    channels = tuple(dataset_dict["channels"])

    if "z_range" not in dataset_dict:
        raise ValueError("Each dataset must specify 'z_range'")
    z_range = tuple(dataset_dict["z_range"])

    return DatasetConfig(
        adata_path=adata_path,
        data_path=data_path,
        channels=channels,
        z_range=z_range,
        fov_filter=fov_filter,
        annotation_csv=annotation_csv,
        annotation_column=annotation_column,
        categories=categories,
        dataset_id=dataset_id,
    )


def _dataset_config_to_dict(config: DatasetConfig) -> dict[str, Any]:
    """
    Convert DatasetConfig to dictionary.

    Parameters
    ----------
    config : DatasetConfig
        Dataset configuration to convert.

    Returns
    -------
    dict
        Dictionary representation of configuration.
    """
    return {
        "adata_path": str(config.adata_path),
        "data_path": str(config.data_path),
        "channels": list(config.channels),
        "z_range": list(config.z_range),
        "fov_filter": config.fov_filter,
        "annotation_csv": str(config.annotation_csv) if config.annotation_csv else None,
        "annotation_column": config.annotation_column,
        "categories": config.categories,
        "dataset_id": config.dataset_id,
    }


def load_config_from_yaml(path: Path) -> MultiDatasetConfig:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    path : Path
        Path to YAML configuration file.

    Returns
    -------
    MultiDatasetConfig
        Loaded configuration.

    Raises
    ------
    FileNotFoundError
        If configuration file does not exist.
    ValueError
        If configuration file is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    logger.info(f"Loading configuration from {path}")

    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        raise ValueError(f"Configuration file is empty: {path}")

    if "datasets" not in config_dict:
        raise ValueError("Configuration must contain 'datasets' key")

    datasets = [_parse_dataset_config(ds) for ds in config_dict["datasets"]]

    phate_kwargs = config_dict.get("phate_kwargs")

    yx_patch_size = config_dict.get("yx_patch_size", [160, 160])
    if isinstance(yx_patch_size, list):
        yx_patch_size = tuple(yx_patch_size)

    port = config_dict.get("port", 8050)
    debug = config_dict.get("debug", False)
    default_color_mode = config_dict.get("default_color_mode", "annotation")

    config = MultiDatasetConfig(
        datasets=datasets,
        phate_kwargs=phate_kwargs,
        yx_patch_size=yx_patch_size,
        port=port,
        debug=debug,
        default_color_mode=default_color_mode,
    )

    logger.info(f"Successfully loaded configuration with {len(datasets)} dataset(s)")
    return config


def load_config_from_json(path: Path) -> MultiDatasetConfig:
    """
    Load configuration from JSON file.

    Parameters
    ----------
    path : Path
        Path to JSON configuration file.

    Returns
    -------
    MultiDatasetConfig
        Loaded configuration.

    Raises
    ------
    FileNotFoundError
        If configuration file does not exist.
    ValueError
        If configuration file is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    logger.info(f"Loading configuration from {path}")

    with open(path, "r") as f:
        config_dict = json.load(f)

    if "datasets" not in config_dict:
        raise ValueError("Configuration must contain 'datasets' key")

    datasets = [_parse_dataset_config(ds) for ds in config_dict["datasets"]]

    phate_kwargs = config_dict.get("phate_kwargs")

    yx_patch_size = config_dict.get("yx_patch_size", [160, 160])
    if isinstance(yx_patch_size, list):
        yx_patch_size = tuple(yx_patch_size)

    port = config_dict.get("port", 8050)
    debug = config_dict.get("debug", False)
    default_color_mode = config_dict.get("default_color_mode", "annotation")

    config = MultiDatasetConfig(
        datasets=datasets,
        phate_kwargs=phate_kwargs,
        yx_patch_size=yx_patch_size,
        port=port,
        debug=debug,
        default_color_mode=default_color_mode,
    )

    logger.info(f"Successfully loaded configuration with {len(datasets)} dataset(s)")
    return config


def save_config_to_yaml(config: MultiDatasetConfig, path: Path) -> None:
    """
    Save configuration to YAML file.

    Parameters
    ----------
    config : MultiDatasetConfig
        Configuration to save.
    path : Path
        Path to output YAML file.
    """
    logger.info(f"Saving configuration to {path}")

    config_dict = {
        "datasets": [_dataset_config_to_dict(ds) for ds in config.datasets],
        "phate_kwargs": config.phate_kwargs,
        "yx_patch_size": list(config.yx_patch_size),
        "port": config.port,
        "debug": config.debug,
        "default_color_mode": config.default_color_mode,
    }

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Successfully saved configuration to {path}")


def save_config_to_json(config: MultiDatasetConfig, path: Path) -> None:
    """
    Save configuration to JSON file.

    Parameters
    ----------
    config : MultiDatasetConfig
        Configuration to save.
    path : Path
        Path to output JSON file.
    """
    logger.info(f"Saving configuration to {path}")

    config_dict = {
        "datasets": [_dataset_config_to_dict(ds) for ds in config.datasets],
        "phate_kwargs": config.phate_kwargs,
        "yx_patch_size": list(config.yx_patch_size),
        "port": config.port,
        "debug": config.debug,
        "default_color_mode": config.default_color_mode,
    }

    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Successfully saved configuration to {path}")
