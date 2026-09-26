"""Per-target CP (GLCM+) reference: one shared feature space for every eval of a target.

The CP track used to be scored in a model-dependent space. Each eval dir pooled
its own GT and prediction cells to choose a feature subset, then z-scored the
prediction and the GT each by its *own* mean/std. That gives every model a
different feature set. It also erases any per-feature offset or scale difference
between prediction and GT, which is the signature of over-smoothing, so that
error could not register in CP KID/FID/cosine at all.

A CP reference fixes both. It is fit once per target on **GT cells only**,
pooled across the target's benchmark test sets (iPSC + A549 mock/denv/zikv;
HEK and the movie sets are excluded from the fit). It holds:

* a feature keep-mask chosen by :func:`~dynacell.evaluation.feature_select.select_gt_features`
  with the pipeline's criteria constants, and
* one per-feature mean/std over the kept features of that same pooled GT.

Every eval of the target applies the same mask and the same scaler to BOTH
prediction and GT, at dataset level and per row. The artifact is JSON and is
built by ``applications/dynacell/tools/build_cp_reference.py`` from existing GT
CP caches.

CP regionprops are always computed on the full GT/prediction volume. The
2D-track evals score 3D prediction stores, and the deep-feature focus slab never
touches CP. So every current eval runs in the ``3d`` CP space, and
:data:`CP_REFERENCE_DIMENSION` is the only dimension the registry holds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from dynacell.evaluation.cache import StaleCacheError
from dynacell.evaluation.feature_select import (
    DEFAULT_CORR_THRESHOLD,
    DEFAULT_FREQ_CUT,
    DEFAULT_UNIQUE_CUT,
    select_gt_features,
)
from dynacell.evaluation.paths import cp_reference_path

#: Schema of the reference JSON. Bump on any change to its keys or their meaning.
CP_REFERENCE_SCHEMA = 1

#: The CP feature space every current eval runs in (CP regionprops are 3D).
CP_REFERENCE_DIMENSION = "3d"

#: Keys excluded from the content hash: the hash itself, and the build timestamp,
#: so rebuilding from identical GT caches yields an identical hash and does not
#: needlessly invalidate every final-metrics cache stamped with it.
_UNHASHED_KEYS = frozenset({"sha256", "created_at"})


def _build_command(target_name: str, dimension: str) -> str:
    """Return the command that builds the registry reference for one target."""
    return (
        "uv run --no-sync python applications/dynacell/tools/build_cp_reference.py "
        f"--target {target_name} --dimension {dimension}"
    )


def payload_sha256(payload: dict[str, Any]) -> str:
    """Return the content hash of a reference payload.

    Parameters
    ----------
    payload : dict
        Reference JSON payload. ``sha256`` and ``created_at`` are ignored.

    Returns
    -------
    str
        Hex sha256 over the canonical (sorted-key, compact) JSON encoding.
    """
    body = {k: v for k, v in payload.items() if k not in _UNHASHED_KEYS}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def gt_matrix_sha256(gt: np.ndarray) -> str:
    """Return a sha256 over a stacked GT cell matrix (shape + float64 C-order bytes).

    Recorded in the reference so a later GT re-cache that changes any fitted
    value, or the cell count, is detectable against the artifact.
    """
    arr = np.ascontiguousarray(gt, dtype=np.float64)
    digest = hashlib.sha256(f"{arr.shape[0]}x{arr.shape[1]}:".encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def fit_cp_reference(
    gt: np.ndarray,
    *,
    target_name: str,
    dimension: str,
    feature_names: tuple[str, ...],
    cp_identity: dict[str, Any],
    datasets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fit a CP reference on pooled GT cells and return its JSON payload.

    Parameters
    ----------
    gt : np.ndarray
        Shape ``(n_cells, n_features)``: every finite GT cell of every fit
        dataset, stacked in ``datasets`` order.
    target_name : str
        Eval ``target_name`` (``nucleus``/``membrane``/``er``/``mitochondria``).
    dimension : str
        CP feature-space dimension, :data:`CP_REFERENCE_DIMENSION`.
    feature_names : tuple of str
        Column names of ``gt``, from ``active_cp_feature_names``.
    cp_identity : dict
        Dataset-independent CP recipe identity
        (:func:`~dynacell.evaluation.pipeline_cache.cp_recipe_identity`).
    datasets : list of dict
        Per-dataset fit inputs (dataset ref, GT cache paths read, ``n_cells``).

    Returns
    -------
    dict
        JSON-serializable payload, including its own content ``sha256``.

    Raises
    ------
    ValueError
        If ``gt`` does not match ``feature_names``, holds a non-finite value, or
        selection keeps a zero-variance feature.
    """
    if gt.ndim != 2 or gt.shape[1] != len(feature_names):
        raise ValueError(f"GT matrix shape {gt.shape} does not match {len(feature_names)} CP feature names")
    if not np.isfinite(gt).all():
        raise ValueError("GT matrix holds non-finite values; drop them before fitting")
    n_cells = sum(int(d["n_cells"]) for d in datasets)
    if n_cells != gt.shape[0]:
        raise ValueError(f"datasets report {n_cells} cells but the GT matrix has {gt.shape[0]} rows")

    gt = np.asarray(gt, dtype=np.float64)
    keep_mask = select_gt_features(
        gt, freq_cut=DEFAULT_FREQ_CUT, unique_cut=DEFAULT_UNIQUE_CUT, corr_threshold=DEFAULT_CORR_THRESHOLD
    )
    kept = gt[:, keep_mask]
    mean = kept.mean(axis=0)
    std = kept.std(axis=0)
    if not (std > 0).all():
        raise ValueError(f"selection kept zero-variance features: {np.array(feature_names)[keep_mask][std == 0]}")

    payload: dict[str, Any] = {
        "schema": CP_REFERENCE_SCHEMA,
        "target_name": target_name,
        "dimension": dimension,
        "feature_names": list(feature_names),
        "keep_mask": [bool(b) for b in keep_mask],
        "kept_feature_names": [n for n, k in zip(feature_names, keep_mask, strict=True) if k],
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "criteria": {
            "freq_cut": DEFAULT_FREQ_CUT,
            "unique_cut": DEFAULT_UNIQUE_CUT,
            "corr_threshold": DEFAULT_CORR_THRESHOLD,
            "fit_on": "gt_only",
        },
        "cp_identity": cp_identity,
        "fit": {
            "datasets": datasets,
            "n_cells": int(gt.shape[0]),
            "gt_matrix_sha256": gt_matrix_sha256(gt),
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload["sha256"] = payload_sha256(payload)
    return payload


def write_cp_reference(payload: dict[str, Any], path: Path) -> None:
    """Write a reference payload to ``path`` (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n")


@dataclass(frozen=True)
class CPReference:
    """A loaded, hash-verified CP reference.

    Picklable (plain arrays + scalars) so it can be shipped to spawn workers.
    """

    path: str
    sha256: str
    target_name: str
    dimension: str
    feature_names: tuple[str, ...]
    keep_mask: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    cp_identity: dict[str, Any]

    def _check_width(self, x: np.ndarray) -> None:
        """Raise when a CP matrix does not have the reference's column count."""
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise StaleCacheError(
                f"CP feature dimension mismatch: matrix has shape {x.shape}, the CP reference "
                f"{self.path} has {len(self.feature_names)} features. A CP feature cache was built "
                "with a different recipe (e.g. a GLCM toggle or a CP_FEATURE_VERSION change without "
                "a cache rebuild). Rebuild with force_recompute.gt_cp=true and/or pred_cp=true "
                "(or force_recompute.all=true)."
            )

    def select(self, x: np.ndarray) -> np.ndarray:
        """Return the kept columns of a raw CP matrix, unscaled."""
        self._check_width(x)
        return x[:, self.keep_mask]

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Return the kept columns of a raw CP matrix, standardized by the shared GT scaler."""
        return (self.select(x) - self.mean) / self.std


def load_cp_reference(path: Path, *, target_name: str, dimension: str) -> CPReference:
    """Load and hash-verify a CP reference.

    Parameters
    ----------
    path : pathlib.Path
        Reference JSON.
    target_name : str
        The eval's ``target_name``; the reference must have been fit for it.
    dimension : str
        The eval's CP feature-space dimension.

    Returns
    -------
    CPReference
        The verified reference.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist; the message names the build command.
    ValueError
        If the stored hash disagrees with the content, or the reference was fit
        for another target, dimension, or schema.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"CP reference {path} not found. CP (GLCM+) feature metrics are scored in a "
            f"per-target GT feature space; build it with:\n  {_build_command(target_name, dimension)}\n"
            "(or point feature_metrics.cp.reference_path at an existing reference)."
        )
    payload = json.loads(path.read_text())
    if payload["sha256"] != payload_sha256(payload):
        raise ValueError(f"CP reference {path} content does not match its recorded sha256 (edited by hand?)")
    if payload["schema"] != CP_REFERENCE_SCHEMA:
        raise ValueError(f"CP reference {path} has schema {payload['schema']}, expected {CP_REFERENCE_SCHEMA}")
    if payload["target_name"] != target_name or payload["dimension"] != dimension:
        raise ValueError(
            f"CP reference {path} was fit for ({payload['target_name']!r}, {payload['dimension']!r}), "
            f"but this eval is ({target_name!r}, {dimension!r})"
        )
    return CPReference(
        path=str(path),
        sha256=payload["sha256"],
        target_name=payload["target_name"],
        dimension=payload["dimension"],
        feature_names=tuple(payload["feature_names"]),
        keep_mask=np.asarray(payload["keep_mask"], dtype=bool),
        mean=np.asarray(payload["mean"], dtype=np.float64),
        std=np.asarray(payload["std"], dtype=np.float64),
        cp_identity=payload["cp_identity"],
    )


def resolve_cp_reference_path(config: DictConfig) -> Path:
    """Return ``feature_metrics.cp.reference_path``, or the registry default when null.

    The registry default lives under ``DATA_ROOT`` for full and lite evals alike,
    so a lite eval scores in exactly the full benchmark's feature space.
    """
    override = OmegaConf.select(config, "feature_metrics.cp.reference_path", default=None)
    if override is not None:
        return Path(override)
    return cp_reference_path(config.target_name, CP_REFERENCE_DIMENSION)


def load_checked_cp_reference(
    config: DictConfig, *, cp_identity: dict[str, Any], feature_names: tuple[str, ...]
) -> CPReference:
    """Load the eval's CP reference and refuse one built for a different CP recipe.

    Parameters
    ----------
    config : DictConfig
        Eval config (``target_name``, ``feature_metrics.cp.reference_path``).
    cp_identity : dict
        The eval's own CP recipe identity.
    feature_names : tuple of str
        The eval's CP column names.

    Returns
    -------
    CPReference
        The verified reference.

    Raises
    ------
    FileNotFoundError
        If the reference does not exist.
    StaleCacheError
        If the reference's CP recipe identity or feature names differ from the eval's.
    """
    ref = load_cp_reference(
        resolve_cp_reference_path(config), target_name=config.target_name, dimension=CP_REFERENCE_DIMENSION
    )
    if ref.cp_identity != cp_identity or ref.feature_names != feature_names:
        raise StaleCacheError(
            f"CP reference {ref.path} was built for a different CP recipe.\n"
            f"  reference identity: {ref.cp_identity}\n  eval identity:      {cp_identity}\n"
            f"  reference features: {len(ref.feature_names)}; eval features: {len(feature_names)}\n"
            f"Rebuild it from caches of the current recipe:\n  {_build_command(ref.target_name, ref.dimension)}"
        )
    return ref
