"""Config composition for YAML files with ``base:`` inheritance.

Provides a lightweight config composition mechanism using PyYAML.
Leaf configs declare dependencies via a ``base:`` list of relative paths;
the helper recursively merges them with deep dict merge (lists replace,
not append).  The final output is plain ``class_path`` / ``init_args``
YAML compatible with LightningCLI.
"""

import copy
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(maxsize=256)
def _load_yaml_cached(resolved_path: Path) -> dict:
    """Parse a YAML file once per resolved path within the process.

    Keyed by the fully-resolved path so different symlinks to the same
    file share a cache entry. Callers must deep-copy the returned dict
    before mutating, since ``lru_cache`` hands out the same object on
    every hit.
    """
    with open(resolved_path) as f:
        return yaml.safe_load(f) or {}


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


def load_composed_config(
    path: str | Path,
    _seen: frozenset[Path] | None = None,
    *,
    resolver: Callable[[dict], dict] | None = None,
) -> dict:
    """Load a YAML config, recursively resolving ``base:`` references.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file.  May contain a ``base:`` key with
        a list of relative paths to recipe fragments that are merged
        before the file's own keys.
    resolver : callable, optional
        Post-composition hook ``dict -> dict`` invoked once on the final
        merged dict at the top-level call. Recursive calls that resolve
        ``base:`` fragments pass ``resolver=None``, so each fragment is
        merged raw and only the outermost composed dict is transformed.

    Returns
    -------
    dict
        Fully composed config dict with ``base:`` key removed and any
        top-level keys starting with ``_`` stripped (YAML anchor
        definitions and other private markers — see notes). If
        ``resolver`` is provided, the returned dict is the resolver's
        output, stripped.

    Notes
    -----
    Top-level keys whose name starts with ``_`` are treated as private
    to the YAML composition layer and are removed from the returned
    dict. This lets leaves define YAML merge anchors at top level (the
    only scope ``yaml.safe_load`` resolves) without those defining
    keys reaching downstream consumers like LightningCLI / jsonargparse,
    which reject unknown top-level keys::

        _hcs_init_args: &hcs_init_args
          source_channel: [Phase3D]
          ...
        data:
          init_args:
            data_modules:
              - init_args:
                  <<: *hcs_init_args
                  data_path: /path/to/zarr

    The merge expansion under ``data.init_args`` survives; the
    ``_hcs_init_args`` defining key is stripped. The strip applies at
    every recursion level so anchor defs in ``base:`` fragments do not
    leak through ``deep_merge``.

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
    cfg = copy.deepcopy(_load_yaml_cached(path))
    bases = cfg.pop("base", [])
    if bases is None:
        bases = []
    elif isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for rel in bases:
        base_cfg = load_composed_config(path.parent / rel, _seen)
        merged = deep_merge(merged, base_cfg)
    result = deep_merge(merged, cfg)
    if resolver is not None:
        result = resolver(result)
    return {k: v for k, v in result.items() if not k.startswith("_")}
