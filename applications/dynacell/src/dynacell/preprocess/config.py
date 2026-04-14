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
        Absolute path to the YAML config file.

    Returns
    -------
    dict[str, Any]
        Loaded config as an OmegaConf DictConfig.
    """
    if config_path.exists():
        return OmegaConf.load(config_path)
    return OmegaConf.create({})
