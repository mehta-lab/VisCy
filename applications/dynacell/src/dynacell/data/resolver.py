"""Manifest-driven dataset reference resolution for the DynaCell benchmark.

Turns a :class:`DatasetRef` (``{dataset, target}``) into concrete paths and
channel names by reading a Pydantic :class:`DatasetManifest` YAML discovered
via manifest roots. Callers compose this with the config pipeline via
:mod:`dynacell._compose_hook`.

Manifest root precedence (highest wins):

1. ``cli_roots`` argument.
2. ``DYNACELL_MANIFEST_ROOTS`` env var (``os.pathsep``-separated paths).
3. Python entry points under group ``dynacell.manifest_roots``.

For each root (in order), the resolver looks for
``<root>/<dataset>/manifest.yaml``. First hit wins. No recursion, no
globbing.
"""

from __future__ import annotations

import os
from importlib import resources
from importlib.metadata import entry_points
from pathlib import Path

from pydantic import BaseModel

from dynacell.data.manifests import (
    DatasetRef,
    VoxelSpacing,
    load_manifest,
)


class NoManifestRootsError(RuntimeError):
    """No manifest roots could be discovered from CLI, env, or entry points."""


class ManifestNotFoundError(LookupError):
    """Dataset slug not found under any configured manifest root."""


class TargetNotFoundError(LookupError):
    """Target slug not present in the located dataset manifest."""


class ResolvedDataset(BaseModel):
    """Flat view of the manifest fields a composed config needs."""

    manifest_path: Path
    data_path_train: Path
    data_path_test: Path
    source_channel: str
    target_channel: str
    spacing: VoxelSpacing
    cell_segmentation_path: Path | None = None
    gt_cache_dir: Path | None = None


_ENV_VAR = "DYNACELL_MANIFEST_ROOTS"
_ENTRY_POINT_GROUP = "dynacell.manifest_roots"

REQUIRED_REF_KEYS: tuple[str, ...] = ("dataset", "target")


def dataset_ref_from_dict(ref_dict: object) -> DatasetRef | None:
    """Validate a ``benchmark.dataset_ref`` dict, returning ``None`` for partial refs.

    Shared between the Lightning-side compose hook and the Hydra-side
    eval hook so the "full ref vs partial ref vs no ref" policy stays
    identical across surfaces. A missing dict, non-dict value, or
    partial dict (either ``dataset`` or ``target`` missing) is treated
    as a no-op signal (returns ``None``). A dict with both keys present
    is validated via Pydantic — malformed values surface as the usual
    :class:`pydantic.ValidationError`.
    """
    if not isinstance(ref_dict, dict):
        return None
    if not all(k in ref_dict for k in REQUIRED_REF_KEYS):
        return None
    return DatasetRef.model_validate(ref_dict)


def _entry_point_roots() -> list[Path]:
    """Resolve entry-point-registered manifest roots to package resource dirs."""
    roots: list[Path] = []
    for ep in entry_points(group=_ENTRY_POINT_GROUP):
        module = ep.load()
        resource_dir = resources.files(module)
        roots.append(Path(str(resource_dir)))
    return roots


def discover_manifest_roots(cli_roots: list[Path] | None = None) -> list[Path]:
    """Return manifest roots in precedence order (CLI → env var → entry points).

    Parameters
    ----------
    cli_roots : list[Path] or None
        Explicit roots provided by the caller. If given, they take
        precedence over environment and entry points but do not replace
        them — lower-precedence roots still contribute.

    Returns
    -------
    list[Path]
        Non-empty list of roots to scan.

    Raises
    ------
    NoManifestRootsError
        If no roots are configured at any precedence level.
    """
    roots: list[Path] = []
    if cli_roots:
        roots.extend(Path(p) for p in cli_roots)
    env_value = os.environ.get(_ENV_VAR)
    if env_value:
        roots.extend(Path(p) for p in env_value.split(os.pathsep) if p)
    roots.extend(_entry_point_roots())
    if not roots:
        raise NoManifestRootsError(
            "No dynacell manifest roots configured.\n\n"
            "Configure via one of:\n"
            f"  - Env var:        export {_ENV_VAR}=/path/to/datasets\n"
            "  - Install a provider:  pip install dynacell-paper\n"
        )
    return roots


def _find_manifest(dataset: str, roots: list[Path]) -> Path:
    """Return the first ``<root>/<dataset>/manifest.yaml`` that exists."""
    searched: list[Path] = []
    for root in roots:
        candidate = root / dataset / "manifest.yaml"
        searched.append(candidate)
        if candidate.is_file():
            return candidate
    lines = "\n".join(f"  - {p}" for p in searched)
    raise ManifestNotFoundError(f"dataset {dataset!r} not found.\n\nSearched:\n{lines}\n")


def resolve_dataset_ref(
    ref: DatasetRef,
    roots: list[Path] | None = None,
) -> ResolvedDataset:
    """Resolve a :class:`DatasetRef` against the manifest registry.

    Parameters
    ----------
    ref : DatasetRef
        The reference to resolve.
    roots : list[Path] or None
        Optional explicit roots (CLI-provided). Falls back to env var and
        entry points per :func:`discover_manifest_roots`.

    Returns
    -------
    ResolvedDataset
        Flat view of the fields the composed config needs.

    Raises
    ------
    NoManifestRootsError
        If no manifest roots are configured.
    ManifestNotFoundError
        If the dataset slug is not found under any root.
    TargetNotFoundError
        If the target slug is not defined in the located manifest.
    """
    all_roots = discover_manifest_roots(roots)
    manifest_path = _find_manifest(ref.dataset, all_roots)
    manifest = load_manifest(manifest_path)
    if ref.target not in manifest.targets:
        available = ", ".join(sorted(manifest.targets)) or "(none)"
        raise TargetNotFoundError(
            f"target {ref.target!r} not found in dataset {ref.dataset!r}.\n\n"
            f"Manifest: {manifest_path}\n"
            f"Available targets: {available}\n"
        )
    target = manifest.targets[ref.target]
    return ResolvedDataset(
        manifest_path=manifest_path,
        data_path_train=target.stores.train,
        data_path_test=target.stores.test,
        source_channel=manifest.source_channel,
        target_channel=target.target_channel,
        spacing=manifest.spacing,
        cell_segmentation_path=target.stores.cell_segmentation,
        gt_cache_dir=target.stores.gt_cache_dir,
    )
