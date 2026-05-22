"""Shared OmegaConf + Pydantic YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from omegaconf import OmegaConf
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_yaml(path: Path, model_class: type[T]) -> T:
    """Load a YAML file and validate it against a Pydantic model.

    Parameters
    ----------
    path : Path
        Path to a YAML file.
    model_class : type[T]
        Pydantic model class to validate against.

    Returns
    -------
    T
        Validated model instance.
    """
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return model_class.model_validate(raw)
