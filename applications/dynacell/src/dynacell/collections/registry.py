"""Collection registry: name -> YAML path mapping and loading.

Mirrors :mod:`dynacell.data.registry`. Uses a hardcoded dict
instead of a filesystem walk so a duplicate or malformed YAML does
not break every consumer at import time.

Import is side-effect-free for the same reason: the packaged YAMLs are
only touched by :func:`get_collection`, so an incomplete install cannot
make this module — or the sibling :mod:`dynacell.collections.freezer`,
re-exported from the same package — unimportable. The freezer is what
regenerates missing collection YAMLs, so it must stay reachable when
they are absent.
"""

from importlib.resources import files
from pathlib import Path

from dynacell.data import BenchmarkCollection, load_collection

_CONFIGS_ROOT = Path(str(files("dynacell") / "_configs" / "collections"))

_REGISTRY: dict[str, Path] = {
    "sec61b_ipsc_v1": _CONFIGS_ROOT / "virtual_staining" / "sec61b_ipsc_v1.yaml",
    "sec61b_joint_all_v1": _CONFIGS_ROOT / "virtual_staining" / "sec61b_joint_all_v1.yaml",
    "sec61b_joint_mock_v1": _CONFIGS_ROOT / "virtual_staining" / "sec61b_joint_mock_v1.yaml",
    "tomm20_joint_all_v1": _CONFIGS_ROOT / "virtual_staining" / "tomm20_joint_all_v1.yaml",
    "tomm20_joint_mock_v1": _CONFIGS_ROOT / "virtual_staining" / "tomm20_joint_mock_v1.yaml",
    "nucleus_ipsc_v1": _CONFIGS_ROOT / "virtual_staining" / "nucleus_ipsc_v1.yaml",
    "nucleus_joint_all_v1": _CONFIGS_ROOT / "virtual_staining" / "nucleus_joint_all_v1.yaml",
    "nucleus_joint_mock_v1": _CONFIGS_ROOT / "virtual_staining" / "nucleus_joint_mock_v1.yaml",
    "membrane_ipsc_v1": _CONFIGS_ROOT / "virtual_staining" / "membrane_ipsc_v1.yaml",
    "membrane_joint_all_v1": _CONFIGS_ROOT / "virtual_staining" / "membrane_joint_all_v1.yaml",
    "membrane_joint_mock_v1": _CONFIGS_ROOT / "virtual_staining" / "membrane_joint_mock_v1.yaml",
}


def list_collections() -> list[str]:
    """Return all registered collection names."""
    return list(_REGISTRY.keys())


def get_collection(name: str) -> BenchmarkCollection:
    """Load and validate a frozen benchmark collection by name.

    Parameters
    ----------
    name
        Key in the registry, e.g. ``"sec61b_ipsc_v1"``.

    Returns
    -------
    BenchmarkCollection
        Validated collection.

    Raises
    ------
    KeyError
        If the collection name is not registered.
    FileNotFoundError
        If the collection is registered but its packaged YAML is absent,
        which means the install is incomplete.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown collection {name!r}. Available: {list_collections()}")
    path = _REGISTRY[name]
    if not path.is_file():
        raise FileNotFoundError(
            f"Collection {name!r} is registered but its YAML is missing at {path}. "
            "The packaged configs are incomplete — reinstall dynacell: uv sync"
        )
    return load_collection(path)
