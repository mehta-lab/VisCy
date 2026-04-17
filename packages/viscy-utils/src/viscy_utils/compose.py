"""Config composition for YAML files with ``base:`` inheritance.

Provides a lightweight config composition mechanism using PyYAML.
Leaf configs declare dependencies via a ``base:`` list of relative paths;
the helper recursively merges them with deep dict merge (lists replace,
not append).  The final output is plain ``class_path`` / ``init_args``
YAML compatible with LightningCLI.
"""

from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    Dicts are merged key-by-key; all other types (including lists) are
    replaced entirely by the override value.
    """
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_composed_config(path: str | Path, _seen: frozenset[Path] | None = None) -> dict:
    """Load a YAML config, recursively resolving ``base:`` references.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file.  May contain a ``base:`` key with
        a list of relative paths to recipe fragments that are merged
        before the file's own keys.

    Returns
    -------
    dict
        Fully composed config dict with ``base:`` key removed.

    Raises
    ------
    ValueError
        If a circular ``base:`` reference is detected.
    """
    path = Path(path).resolve()
    if _seen is None:
        _seen = frozenset()
    if path in _seen:
        raise ValueError(f"Circular base: reference detected: {path}")
    _seen = _seen | {path}
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    bases = cfg.pop("base", [])
    if bases is None:
        bases = []
    elif isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for rel in bases:
        base_cfg = load_composed_config(path.parent / rel, _seen)
        merged = deep_merge(merged, base_cfg)
    return deep_merge(merged, cfg)
