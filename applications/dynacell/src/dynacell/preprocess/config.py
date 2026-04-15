"""Preprocessing config loading with OmegaConf."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_preprocess_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config via OmegaConf.

    Parameters
    ----------
    config_path : Path
        Path to the YAML config file. Must exist.

    Returns
    -------
    dict[str, Any]
        Loaded config as an OmegaConf DictConfig.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)
