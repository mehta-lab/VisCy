"""Preprocessing config loading with OmegaConf fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_preprocess_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config via OmegaConf, falling back to an empty dict.

    Parameters
    ----------
    config_path : Path
        Absolute path to the YAML config file.

    Returns
    -------
    dict[str, Any]
        Loaded config as a dict-like object (OmegaConf DictConfig
        or plain dict if OmegaConf is not installed).
    """
    try:
        from omegaconf import OmegaConf

        if config_path.exists():
            return OmegaConf.load(config_path)
        return OmegaConf.create({})
    except ImportError:
        return {}
