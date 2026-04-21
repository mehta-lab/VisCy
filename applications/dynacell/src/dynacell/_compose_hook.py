"""Composition-time resolver hook for DynaCell benchmark leaves.

Threaded into :func:`viscy_utils.compose.load_composed_config` via the
``resolver`` keyword argument; run once after the final deep-merge.
Reads ``benchmark.dataset_ref: {dataset, target}`` from the composed dict
and splices concrete ``data_path``, ``source_channel``, ``target_channel``
into ``data.init_args`` from the resolved :class:`DatasetManifest`.

Partial references (only ``dataset`` or only ``target``) are a strict
no-op, so shared train/predict-set fragments can declare one half of
``dataset_ref`` without breaking leaves whose target fragment has not
yet been migrated.
"""

from __future__ import annotations

import copy
import sys

from dynacell.data import DatasetRef, ResolvedDataset, resolve_dataset_ref

_REQUIRED_REF_KEYS = ("dataset", "target")
_DATA_FIELDS = ("data_path", "source_channel", "target_channel")


def _infer_mode(composed: dict) -> str:
    """Return the Lightning subcommand ("fit", "predict", or "validate")."""
    launcher_mode = composed.get("launcher", {}).get("mode")
    if launcher_mode in {"fit", "predict", "validate"}:
        return launcher_mode
    for arg in sys.argv[1:]:
        if arg in {"fit", "predict", "validate"}:
            return arg
    raise ValueError("Cannot infer Lightning mode for dataset_ref resolution; set launcher.mode in the leaf config.")


def _splice_resolved(composed: dict, resolved: ResolvedDataset, mode: str, ref: DatasetRef) -> dict:
    """Return a deep-copied composed dict with resolved fields spliced in.

    Raises ``ValueError`` if the composed dict already declares one of
    the data fields with a value different from the manifest's. Matching
    values are tolerated — shared fragments (e.g.
    ``train_sets/ipsc_confocal.yml``) may declare defaults that happen
    to agree with the manifest during the staged rollout.
    """
    out = copy.deepcopy(composed)
    data = out.setdefault("data", {})
    init_args = data.setdefault("init_args", {})
    resolved_values = {
        "data_path": str(resolved.data_path_test if mode == "predict" else resolved.data_path_train),
        "source_channel": resolved.source_channel,
        "target_channel": resolved.target_channel,
    }
    conflicts: dict[str, tuple[object, object]] = {}
    for field, value in resolved_values.items():
        existing = init_args.get(field)
        if existing is not None and existing != value:
            conflicts[field] = (existing, value)
    if conflicts:
        details = "; ".join(
            f"{k}: composed={existing!r} vs manifest={resolved!r}" for k, (existing, resolved) in conflicts.items()
        )
        raise ValueError(
            f"benchmark.dataset_ref={{dataset: {ref.dataset}, target: {ref.target}}} "
            f"conflicts with explicit data.init_args fields: {details}. "
            "Remove one side — either drop the conflicting explicit fields "
            "or remove dataset_ref."
        )
    init_args.update(resolved_values)
    out.setdefault("benchmark", {})["spacing"] = resolved.spacing.as_list()
    return out


def _dynacell_ref_resolver(composed: dict) -> dict:
    """Resolve ``benchmark.dataset_ref`` against the manifest registry.

    Strict partial-ref no-op: returns the input dict unchanged unless
    both ``dataset`` and ``target`` keys are present under
    ``benchmark.dataset_ref``.
    """
    ref_dict = composed.get("benchmark", {}).get("dataset_ref")
    if not isinstance(ref_dict, dict):
        return composed
    if not all(k in ref_dict for k in _REQUIRED_REF_KEYS):
        return composed
    ref = DatasetRef.model_validate(ref_dict)
    resolved = resolve_dataset_ref(ref)
    return _splice_resolved(composed, resolved, _infer_mode(composed), ref)
