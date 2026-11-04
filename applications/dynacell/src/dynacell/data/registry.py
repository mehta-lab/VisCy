"""Name-based dataset registry for the DynaCell benchmark.

Resolves bare dataset names (e.g. ``"aics-hipsc"``,
``"a549-mantis-sec61b-mock"``) to validated manifest objects, backed by the
manifest-roots resolver (:mod:`dynacell.data.resolver`). This is the
canonical home for the name-lookup API that ``dynacell-paper`` previously
carried in ``dynacell_paper.data.registry``.

Distinct from :mod:`dynacell.data.manifests`, whose loaders take explicit
file paths, and from :func:`dynacell.data.manifests.get_target`, which takes
an already-loaded manifest object. Name-based target access is
``get_manifest(name).targets[target]`` (no shadowing symbol).
"""

from __future__ import annotations

from pathlib import Path

from dynacell.data.manifests import (
    DatasetManifest,
    SplitDefinition,
    load_manifest,
    load_splits,
)
from dynacell.data.resolver import (
    ManifestNotFoundError,
    _find_manifest,
    discover_manifest_roots,
)


def list_datasets() -> list[str]:
    """Return all registered dataset names across configured manifest roots.

    Enumerates immediate subdirectories that contain a ``manifest.yaml``
    under each root returned by :func:`discover_manifest_roots`, in
    precedence order (CLI → env var → entry points). A name discovered in a
    higher-precedence root shadows later duplicates, matching the
    first-hit-wins policy of the resolver.

    Returns
    -------
    list[str]
        Registered dataset names, e.g. ``["a549-mantis-caax-denv", ...]``.
    """
    names: list[str] = []
    seen: set[str] = set()
    for root in discover_manifest_roots():
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.name in seen or child.name.startswith(("_", ".")):
                continue
            if (child / "manifest.yaml").is_file():
                seen.add(child.name)
                names.append(child.name)
    return names


def _manifest_path(name: str) -> Path:
    """Locate a dataset's ``manifest.yaml``, raising ``KeyError`` if absent.

    Wraps the resolver's :func:`_find_manifest`, translating its
    :class:`ManifestNotFoundError` into ``KeyError`` so callers get the same
    "unknown dataset name" contract as dict-style lookup.
    """
    try:
        return _find_manifest(name, discover_manifest_roots())
    except ManifestNotFoundError:
        raise KeyError(f"Unknown dataset {name!r}. Available: {list_datasets()}") from None


def get_manifest(name: str) -> DatasetManifest:
    """Load and validate a dataset manifest by registered name.

    Parameters
    ----------
    name : str
        Registered dataset name, e.g. ``"aics-hipsc"``.

    Returns
    -------
    DatasetManifest
        Validated manifest object (name-based target access via
        ``get_manifest(name).targets[target]``).

    Raises
    ------
    KeyError
        If the dataset name is not registered under any manifest root.
    """
    return load_manifest(_manifest_path(name))


def get_splits(name: str, target: str) -> SplitDefinition:
    """Load the split definition for a dataset target by name.

    Resolves the target's ``splits`` path relative to the manifest
    directory, mirroring the path convention of the bundled registry.

    Parameters
    ----------
    name : str
        Registered dataset name.
    target : str
        Key in the manifest's ``targets`` dict, e.g. ``"sec61b"``.

    Returns
    -------
    SplitDefinition
        Validated split definition.

    Raises
    ------
    KeyError
        If the dataset name is unregistered or the target is absent.
    FileNotFoundError
        If the target's split file does not exist on disk.
    """
    manifest_path = _manifest_path(name)
    manifest = load_manifest(manifest_path)
    if target not in manifest.targets:
        available = ", ".join(sorted(manifest.targets)) or "(none)"
        raise KeyError(f"Target {target!r} not in {name!r}. Available: {available}")
    split_path = manifest_path.parent / manifest.targets[target].splits
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file for {name!r}/{target!r} not found: {split_path}")
    return load_splits(split_path)
