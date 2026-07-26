"""Collection registry: name -> YAML path mapping and loading.

Mirrors :mod:`dynacell.data.registry`. Uses a hardcoded dict
instead of a filesystem walk so a duplicate or malformed YAML does
not break every consumer at import time.
"""

from importlib.resources import files
from pathlib import Path

from dynacell.data import BenchmarkCollection, load_collection

_CONFIGS_ROOT = Path(str(files("dynacell") / "_configs" / "collections"))

if not _CONFIGS_ROOT.is_dir():
    raise RuntimeError(f"Collection configs not found at {_CONFIGS_ROOT}.\nReinstall dynacell: uv sync")

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
        If the YAML file does not exist.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown collection {name!r}. Available: {list_collections()}")
    return load_collection(_REGISTRY[name])
