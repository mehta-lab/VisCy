"""Preprocessing config loading with OmegaConf."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_preprocess_config(config_path: Path) -> DictConfig:
    """Load a YAML config via OmegaConf.

    Parameters
    ----------
    config_path : Path
        Path to the YAML config file. Must exist.

    Returns
    -------
    DictConfig
        Loaded config.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)
