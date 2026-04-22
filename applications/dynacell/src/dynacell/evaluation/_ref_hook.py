"""Hydra-side ``dataset_ref`` resolution for eval and precompute_gt.

Parallel to :mod:`dynacell._compose_hook`, but operates on an OmegaConf
:class:`DictConfig` rather than a plain ``dict``. Called from the Hydra
entry points of the evaluation pipeline after config composition; reads
``benchmark.dataset_ref`` and splices manifest-derived ``io.*`` fields
and ``pixel_metrics.spacing`` into the composed config.

Partial references (only ``dataset`` or only ``target``) are treated as
a no-op, matching the Lightning-side policy so shared fragments can
carry half the ref without breaking leaves.
"""

from __future__ import annotations

from typing import Final

from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from dynacell.data.manifests import DatasetRef
from dynacell.data.resolver import ResolvedDataset, resolve_dataset_ref

_REQUIRED_REF_KEYS: Final = ("dataset", "target")

_RESOLVED_FIELDS: Final = (
    ("io.gt_path", "data_path_test"),
    ("io.cell_segmentation_path", "cell_segmentation_path"),
    ("io.gt_channel_name", "target_channel"),
    ("io.gt_cache_dir", "gt_cache_dir"),
)


def apply_dataset_ref(config: DictConfig) -> None:
    """Splice manifest-derived ``io.*`` and ``pixel_metrics.spacing`` into *config*.

    Mutates *config* in place. No-op when ``benchmark.dataset_ref`` is
    missing or carries fewer than both required keys (matches the
    Lightning-side partial-ref policy in
    :mod:`dynacell._compose_hook`).

    Parameters
    ----------
    config : DictConfig
        The composed Hydra config. Must be mutable (``struct`` mode is
        disabled as needed to write new keys).

    Raises
    ------
    ValueError
        If ``benchmark.dataset_ref`` is malformed, or if any resolved
        field is already explicitly set to a value that disagrees with
        the manifest.
    """
    ref_node = OmegaConf.select(config, "benchmark.dataset_ref", default=None)
    if ref_node is None:
        return
    ref_dict = OmegaConf.to_container(ref_node, resolve=True)
    if not isinstance(ref_dict, dict) or not all(k in ref_dict for k in _REQUIRED_REF_KEYS):
        return
    try:
        ref = DatasetRef.model_validate(ref_dict)
    except ValidationError as e:
        raise ValueError(f"Invalid benchmark.dataset_ref: {ref_dict}") from e
    resolved = resolve_dataset_ref(ref)
    _check_collisions(config, resolved)
    _splice(config, resolved)


def _check_collisions(config: DictConfig, resolved: ResolvedDataset) -> None:
    """Raise ``ValueError`` if any resolved field disagrees with an explicit value.

    Fields that are unset (missing, ``None``, or OmegaConf-missing
    ``???``) are not considered collisions.

    Parameters
    ----------
    config : DictConfig
        The composed config.
    resolved : ResolvedDataset
        Manifest-resolved dataset fields.

    Raises
    ------
    ValueError
        If one or more explicit config values disagree with the
        manifest-derived values.
    """
    conflicts: list[tuple[str, str, str]] = []
    for cfg_path, attr in _RESOLVED_FIELDS:
        current = OmegaConf.select(config, cfg_path, default=None)
        if current is None:
            continue
        resolved_val = getattr(resolved, attr)
        if resolved_val is None:
            continue
        if str(current) != str(resolved_val):
            conflicts.append((cfg_path, str(current), str(resolved_val)))
    pred_channel = f"{resolved.target_channel}_prediction"
    current_pred = OmegaConf.select(config, "io.pred_channel_name", default=None)
    if current_pred is not None and str(current_pred) != pred_channel:
        conflicts.append(("io.pred_channel_name", str(current_pred), pred_channel))
    resolved_spacing = resolved.spacing.as_list()
    current_spacing = OmegaConf.select(config, "pixel_metrics.spacing", default=None)
    if current_spacing is not None:
        current_list = OmegaConf.to_container(current_spacing, resolve=True)
        if current_list != resolved_spacing:
            conflicts.append(("pixel_metrics.spacing", str(current_list), str(resolved_spacing)))
    if conflicts:
        lines = "\n".join(f"  - {p}: explicit={a!r}, manifest={b!r}" for p, a, b in conflicts)
        raise ValueError(
            f"benchmark.dataset_ref conflicts with explicit fields:\n{lines}\n"
            f"Remove the explicit fields OR remove benchmark.dataset_ref."
        )


def _splice(config: DictConfig, resolved: ResolvedDataset) -> None:
    """Write manifest-derived fields into *config* in place.

    Disables struct mode on *config* so that new keys can be written to
    paths that may not pre-exist. Callers who need struct-mode checks
    after splicing should re-enable it.

    Parameters
    ----------
    config : DictConfig
        The composed config to mutate.
    resolved : ResolvedDataset
        Manifest-resolved dataset fields.
    """
    OmegaConf.set_struct(config, False)
    for cfg_path, attr in _RESOLVED_FIELDS:
        val = getattr(resolved, attr)
        if val is None:
            continue
        OmegaConf.update(config, cfg_path, str(val), merge=False)
    OmegaConf.update(config, "io.pred_channel_name", f"{resolved.target_channel}_prediction", merge=False)
    OmegaConf.update(config, "pixel_metrics.spacing", resolved.spacing.as_list(), merge=False)
