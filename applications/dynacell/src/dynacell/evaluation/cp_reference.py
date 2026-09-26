"""Per-target CP (GLCM+) reference: one feature mask, one GT scaler per test set.

The CP track used to be scored in a model-dependent space. Each eval dir pooled
its own GT and prediction cells to choose a feature subset, then z-scored the
prediction and the GT each by its *own* mean/std. That gives every model a
different feature set. It also erases any per-feature offset or scale difference
between prediction and GT, which is the signature of over-smoothing, so that
error could not register in CP KID/FID/cosine at all.

A CP reference fixes both. Per target it holds:

* **One feature keep-mask**, chosen by
  :func:`~dynacell.evaluation.feature_select.select_gt_features` on **GT cells
  only**, pooled over the target's :data:`MASK_FIT_DATASETS` (iPSC + A549
  mock/denv/zikv). HEK and every lite dataset stay out of the mask fit.
* **One scaler per test set** (keyed by ``benchmark.dataset_ref.dataset``): the
  mean/std of that test set's GT cells on the kept features. The scaler is
  applied identically to prediction and GT. It is per test set, not pooled,
  because the between-dataset offset dominates the pooled variance (eta^2 up
  to 0.91). A pooled scaler would squeeze the within-dataset spread to 0.02-0.13
  sd on the kept features.
* **Lite datasets reuse their parent's scaler** (the parent is read from the lite
  split's ``selection_criteria.lite_of``); no scaler is ever fit on lite cells.

Each scaler is fit on exactly the GT cells an eval of that dataset scores: every
position and timepoint of the GT store, read from the GT CP cache, with the
non-finite rows dropped. The reference records every dataset's (lite included)
position list, cell count and raw per-feature GT moments (mean/std over ALL CP
columns). An eval refuses a different position set up front
(:meth:`DatasetCPSpace.check_positions`) and, once its GT CP cells are staged,
refuses to score unless the count matches exactly and every feature's mean/std
matches the recorded moments within :data:`GT_MOMENT_RTOL`
(:meth:`DatasetCPSpace.check_gt_cells`). The gate is a tolerance, not an exact
hash, because GPU ``cp_regionprops`` is not bit-reproducible (run-to-run jitter
~1e-15 on an A40), so a GPU recompute of identical GT must pass; any real value
change is many orders of magnitude larger. The exact sha256 of the canonical GT
matrix (:func:`canonical_gt_matrix`) is kept for audit and ``--verify`` only, and
the cache manifest's ``built_at`` for audit only.

A per-dataset std can collapse on a feature the pooled mask keeps (a feature
that is near-constant within one test set). Such a std is floored at
:data:`STD_FLOOR_FRACTION` of that feature's pooled GT std, and the floored
features are recorded per dataset.

CP regionprops are always computed on the full GT/prediction volume, so every
eval, the 2D-track ones included, scores in this one 3-D space.

The artifact is JSON and is built by ``applications/dynacell/tools/build_cp_reference.py``
from existing GT CP caches.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from iohub.ngff import open_ome_zarr
from omegaconf import DictConfig, OmegaConf

from dynacell.evaluation.cache import StaleCacheError, open_features_group, read_features_from_group
from dynacell.evaluation.feature_select import (
    DEFAULT_CORR_THRESHOLD,
    DEFAULT_FREQ_CUT,
    DEFAULT_UNIQUE_CUT,
    select_gt_features,
)
from dynacell.evaluation.metrics import CP_FEATURE_VERSION, active_cp_feature_names, round_device_dependent_cp_columns
from dynacell.evaluation.paths import cp_reference_path
from dynacell.evaluation.pipeline_cache import cached_cp_feature_names, cp_recipe_identity

#: Schema of the reference JSON. Bump on any change to its keys or their meaning.
CP_REFERENCE_SCHEMA = 6

#: Sidecar each eval writes beside its metrics, naming the CP reference it scored in.
CP_SIDECAR_FILENAME = "cp_selected_feature_mask.json"

#: Content-gate tolerance on each CP feature's GT moments (see
#: :meth:`DatasetCPSpace.check_gt_cells`): |mean - mean_fit| <= GT_MOMENT_RTOL * std_fit
#: and |std / std_fit - 1| <= GT_MOMENT_RTOL. Measured on real data: GPU regionprops
#: jitter is ~1e-15 relative, and a CPU-vs-GPU recompute differs by <= ~1e-14 now that
#: ``cp_regionprops`` rounds intensity_min/max to float32 on both devices (before that,
#: up to 4.3e-8 on intensity_min, only ~23x under this tolerance); a real GT change is
#: >= ~1e-3.
GT_MOMENT_RTOL = 1e-6
#: For a feature constant in the fit (std_fit == 0): |mean - mean_fit| and the staged std
#: must both be <= GT_MOMENT_ZERO_STD_ATOL * (1 + |mean_fit|).
GT_MOMENT_ZERO_STD_ATOL = 1e-12

#: Shared-space z is clipped to +-CP_Z_CLIP on both sides before KID/FID/cosine
#: (:meth:`DatasetCPSpace.transform_clipped`). The cubic KID kernel otherwise lets one
#: heavy-tailed predicted feature set the metric's magnitude (a pilot A549->iPSC nucleus
#: eval had pred glcm_ASM at z up to 239, KID 4e9). c = 20 was chosen by the paper owner
#: from an offline sweep over c in {5, 8, 10, 15, 20, 30, 50, none} on the nucleus-lite
#: pilot (see :data:`CP_Z_CLIP_RANK_STABILITY`). Surveying every registry GT cell of the
#: full test sets in its own test set's scaler, 2 of 23,970 distinct cells exceed it (max
#: |z| 26.55: er, a549-mantis-sec61b-denv, kurtosis); enforced max clip fraction 5.3e-4.
CP_Z_CLIP = 20.0

#: Per-dataset bound on the GT cells the clip may touch: the fraction of a dataset's GT
#: cells with any kept-feature |z| > CP_Z_CLIP in its own scaler. :func:`fit_cp_reference`
#: refuses to build a reference that violates it for any NON-lite dataset, so it is a
#: checked invariant. Lite datasets record their fraction with ``enforced: false``.
CP_GT_CLIP_FRAC_MAX = 1e-3

#: Why c = 20: recorded in the hashed criteria next to the clip.
CP_Z_CLIP_RANK_STABILITY = (
    "cross-domain model order on the nucleus-lite pilot identical for all c >= 20 (swept 5,8,10,15,20,30,50,none)"
)

#: A per-dataset std below this fraction of the feature's pooled GT std is floored to it.
STD_FLOOR_FRACTION = 1e-3

#: The mask-fit set per eval ``target_name``, as ``(dataset, manifest target)`` refs --
#: the refs the grouped leaves carry in ``benchmark.dataset_ref``. Its keys are the
#: targets that have a CP reference. HEK and the lite datasets are deliberately absent.
MASK_FIT_DATASETS: dict[str, tuple[tuple[str, str], ...]] = {
    "nucleus": (
        ("aics-hipsc", "nucleus"),
        ("a549-mantis-h2b-mock", "h2b"),
        ("a549-mantis-h2b-denv", "h2b"),
        ("a549-mantis-h2b-zikv", "h2b"),
    ),
    "membrane": (
        ("aics-hipsc", "membrane"),
        ("a549-mantis-caax-mock", "caax"),
        ("a549-mantis-caax-denv", "caax"),
        ("a549-mantis-caax-zikv", "caax"),
    ),
    "er": (
        ("aics-hipsc", "sec61b"),
        ("a549-mantis-sec61b-mock", "sec61b"),
        ("a549-mantis-sec61b-denv", "sec61b"),
        ("a549-mantis-sec61b-zikv", "sec61b"),
    ),
    "mitochondria": (
        ("aics-hipsc", "tomm20"),
        ("a549-mantis-tomm20-mock", "tomm20"),
        ("a549-mantis-tomm20-denv", "tomm20"),
        ("a549-mantis-tomm20-zikv", "tomm20"),
    ),
}

#: Top-level keys excluded from the content hash.
_UNHASHED_KEYS = frozenset({"sha256", "created_at"})

#: Audit-only fields of each ``fit`` provenance record, excluded from the content hash.
#: Everything else in ``fit`` -- positions, cell counts and GT-matrix hashes, which the
#: content gate enforces -- is hashed, so a hand-edited gate field fails the load-time
#: hash check. Paths and ``built_at`` are excluded because they move on a harmless GT
#: re-cache (identical cells) or a relocation, which must not change the hash; the exact
#: ``gt_matrix_sha256`` because GPU jitter moves it too (audit / ``--verify`` info only).
_AUDIT_FIT_KEYS = frozenset({"gt_path", "gt_cache_dir", "cp_cache_path", "cp_cache_built_at", "gt_matrix_sha256"})


def _hashed_fit(fit: dict[str, Any]) -> dict[str, Any]:
    """Return ``fit`` without its audit-only per-dataset fields."""
    return {
        "mask_fit": fit["mask_fit"],
        **{
            group: {
                name: {k: v for k, v in rec.items() if k not in _AUDIT_FIT_KEYS} for name, rec in fit[group].items()
            }
            for group in ("datasets", "lite")
        },
    }


def _build_command(target_name: str) -> str:
    """Return the command that builds the registry reference for one target."""
    return f"uv run --no-sync python applications/dynacell/tools/build_cp_reference.py --target {target_name}"


def cp_space(config: DictConfig) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return the CP recipe identity and column names an eval of ``config`` produces.

    The one derivation shared by the eval (to check a reference) and the builder
    (to stamp one). Reads ``feature_metrics.cp.norm`` / ``feature_metrics.cp.glcm``
    directly, so a config missing them raises instead of defaulting.

    Parameters
    ----------
    config : DictConfig
        Eval config.

    Returns
    -------
    tuple
        ``(cp_identity, feature_names)``.
    """
    cp = config.feature_metrics.cp
    norm = OmegaConf.to_container(cp.norm, resolve=True)
    glcm = OmegaConf.to_container(cp.glcm, resolve=True)
    return cp_recipe_identity(CP_FEATURE_VERSION, norm, glcm), tuple(active_cp_feature_names(bool(cp.glcm.enabled)))


def payload_sha256(payload: dict[str, Any]) -> str:
    """Return the content hash of a reference payload.

    Parameters
    ----------
    payload : dict
        Reference JSON payload. ``sha256``, ``created_at`` and the audit-only ``fit``
        fields (paths, ``cp_cache_built_at``) are ignored.

    Returns
    -------
    str
        Hex sha256 over the canonical (sorted-key, compact) JSON encoding.
    """
    body = {k: v for k, v in payload.items() if k not in _UNHASHED_KEYS}
    body["fit"] = _hashed_fit(payload["fit"])
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def gt_matrix_sha256(gt: np.ndarray) -> str:
    """Return a sha256 over a stacked GT cell matrix.

    The digest covers the shape and the float64 C-order bytes, so it changes with
    any cell value (GPU jitter of ~1e-15 included) or the cell count. It is recorded
    per dataset in the reference's ``fit`` section for audit, and
    ``build_cp_reference.py --verify`` reports an exact mismatch as information. It
    gates nothing: the eval's content gate compares tolerance-checked moments
    (:func:`gt_moments_mismatch`).

    Parameters
    ----------
    gt : np.ndarray
        ``(n_cells, n_features)`` GT CP matrix.

    Returns
    -------
    str
        Hex sha256 digest.
    """
    arr = np.ascontiguousarray(gt, dtype=np.float64)
    digest = hashlib.sha256(f"{arr.shape[0]}x{arr.shape[1]}:".encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def gt_moments(cells: np.ndarray) -> dict[str, list[float]]:
    """Return the raw per-feature GT moments the content gate compares.

    Parameters
    ----------
    cells : np.ndarray
        ``(n_cells, n_features)`` GT-finite CP cells, all columns.

    Returns
    -------
    dict
        ``{"mean": [...], "std": [...]}`` over axis 0, float64, one entry per column.
    """
    arr = np.asarray(cells, dtype=np.float64)
    return {"mean": [float(v) for v in arr.mean(axis=0)], "std": [float(v) for v in arr.std(axis=0)]}


def gt_moments_mismatch(
    cells: np.ndarray, mean_fit: np.ndarray, std_fit: np.ndarray, feature_names: tuple[str, ...]
) -> list[str]:
    """Return the features whose staged GT moments are outside the gate tolerance.

    Criterion per feature: ``|mean - mean_fit| <= GT_MOMENT_RTOL * std_fit`` and
    ``|std / std_fit - 1| <= GT_MOMENT_RTOL``; for ``std_fit == 0`` both
    ``|mean - mean_fit|`` and ``std`` must be ``<= GT_MOMENT_ZERO_STD_ATOL * (1 + |mean_fit|)``.

    Parameters
    ----------
    cells : np.ndarray
        Staged ``(n_cells, n_features)`` GT-finite CP cells.
    mean_fit, std_fit : np.ndarray
        The recorded raw moments of the fit cells.
    feature_names : tuple of str
        Column names, for the report.

    Returns
    -------
    list of str
        ``"name: mean a vs b, std c vs d"`` for every failing feature (empty = pass).
    """
    now = gt_moments(cells)
    mean, std = np.asarray(now["mean"]), np.asarray(now["std"])
    constant = std_fit == 0
    safe = np.where(constant, 1.0, std_fit)
    ok_var = (np.abs(mean - mean_fit) <= GT_MOMENT_RTOL * std_fit) & (np.abs(std / safe - 1) <= GT_MOMENT_RTOL)
    zero_tol = GT_MOMENT_ZERO_STD_ATOL * (1 + np.abs(mean_fit))
    ok_const = (np.abs(mean - mean_fit) <= zero_tol) & (std <= zero_tol)
    ok = np.where(constant, ok_const, ok_var)
    return [
        f"{name}: mean {mean[i]:.9g} vs {mean_fit[i]:.9g}, std {std[i]:.9g} vs {std_fit[i]:.9g}"
        for i, name in enumerate(feature_names)
        if not ok[i]
    ]


@dataclass
class DatasetFit:
    """One test set's GT cells and the record of where they came from.

    Parameters
    ----------
    dataset : str
        ``benchmark.dataset_ref.dataset`` of the test set.
    cells : np.ndarray
        ``(n_cells, n_features)`` finite raw GT CP cells in canonical order
        (:func:`canonical_gt_matrix`): every position and timepoint.
    record : dict
        JSON-serializable provenance: at least ``positions`` (sorted names).
    in_mask_fit : bool
        Whether these cells enter the pooled mask fit.
    parent : str or None, optional
        For a lite dataset, the dataset whose scaler it reuses; its cells are then
        recorded (hash, count, positions) but never fit.
    """

    dataset: str
    cells: np.ndarray
    record: dict[str, Any]
    in_mask_fit: bool
    parent: str | None = None


def canonical_gt_matrix(blocks: dict[tuple[str, int], np.ndarray], n_features: int) -> np.ndarray:
    """Stack per-``(position, timepoint)`` GT CP rows in the canonical order the fit is hashed in.

    Sorted position names, then ``t`` ascending, then cell (row) order within a
    block, as float64. The builder and the eval both hash this matrix, so it is the
    single definition of "the GT cells a scaler describes". Callers pass only
    GT-finite rows (non-finite GT rows dropped, BEFORE the pred/GT paired drop), so
    the matrix does not depend on the prediction.

    Parameters
    ----------
    blocks : dict
        ``{(position, t): (n_cells_t, n_features) array}``; empty blocks may be omitted.
    n_features : int
        Column count, for the empty result.

    Returns
    -------
    np.ndarray
        ``(n_cells, n_features)`` float64 matrix.
    """
    rows = [np.asarray(blocks[key], dtype=np.float64) for key in sorted(blocks) if blocks[key].shape[0]]
    return np.concatenate(rows, axis=0) if rows else np.empty((0, n_features))


def _read_gt_cp_blocks(
    ctx, gt_path: Path, n_features: int
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, Any], list[str]]:
    """Read the GT-finite CP rows of every cached ``(position, t)`` of a GT store.

    Returns ``(blocks, record, missing)``: ``missing`` lists the ``pos/t`` slots the
    cache does not hold. Raises on a malformed store or slot (non-3-D position,
    wrong column count, no CP cache at all). Each block's intensity_min/max are
    rounded to float32 by :func:`round_device_dependent_cp_columns`, locating the
    columns by the names the cache's manifest entry records, so a CPU-built cache
    reads as the GPU-equivalent one.
    """
    with open_ome_zarr(gt_path, mode="r") as plate:
        shapes = {name: (int(pos.data.shape[0]), int(pos.data.shape[2])) for name, pos in plate.positions()}
    blocks: dict[tuple[str, int], np.ndarray] = {}
    missing: list[str] = []
    n_timepoints = n_dropped = 0
    with open_features_group(ctx.paths, "cp", mode="r") as group:
        if group is None:
            raise StaleCacheError(f"no CP feature cache at {ctx.paths.cp_features()}")
        cached_names = cached_cp_feature_names(ctx.manifest["artifacts"]["cp_features"])
        for pos_name, (t_count, z_depth) in sorted(shapes.items()):
            if z_depth < 2:
                raise ValueError(f"{gt_path}/{pos_name} has Z={z_depth}; CP regionprops are 3-D")
            for t in range(t_count):
                feats = read_features_from_group(group, pos_name, t)
                if feats is None:
                    missing.append(f"{pos_name}/t{t}")
                    continue
                n_timepoints += 1
                if feats.shape[0] == 0:
                    continue
                if feats.shape[1] != n_features:
                    raise StaleCacheError(f"{pos_name}/t{t} has {feats.shape[1]} CP columns, expected {n_features}")
                finite = np.isfinite(feats).all(axis=1)
                n_dropped += int((~finite).sum())
                blocks[(pos_name, t)] = round_device_dependent_cp_columns(feats[finite], cached_names)
    record = {"positions": sorted(shapes), "n_timepoints": n_timepoints, "n_cells_dropped_nonfinite": n_dropped}
    return blocks, record, missing


def read_gt_cp_cells(ctx, gt_path: Path, n_features: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Read every finite GT CP cell of one GT store from its cache, read-only, in canonical order.

    Parameters
    ----------
    ctx : pipeline_cache._CacheContext
        Enabled GT-side cache context of the store.
    gt_path : pathlib.Path
        GT HCS store; its positions and per-position ``T`` define what must be cached.
    n_features : int
        Expected CP column count.

    Returns
    -------
    tuple
        ``(cells, record)``: the :func:`canonical_gt_matrix` of the GT-finite rows,
        and ``{"positions", "n_timepoints", "n_cells_dropped_nonfinite"}``.

    Raises
    ------
    StaleCacheError
        If the CP cache is missing, a ``(position, timepoint)`` of the GT store is
        missing from it, or an entry has the wrong column count.
    ValueError
        If a GT position is not a 3-D volume.
    """
    blocks, record, missing = _read_gt_cp_blocks(ctx, gt_path, n_features)
    if missing:
        raise StaleCacheError(f"CP cache miss at {missing[0]} in {ctx.paths.cp_features()} ({len(missing)} missing)")
    return canonical_gt_matrix(blocks, n_features), record


def complete_cached_gt_cp_blocks(ctx, gt_path: Path, n_features: int) -> dict[tuple[str, int], np.ndarray] | None:
    """Return the GT-finite CP blocks of a GT store when its CP cache holds every slot, else ``None``.

    Lets the eval run the content gate before its FOV loop when the GT CP cache is
    already complete (the common re-eval case), so a doomed run fails in seconds
    rather than after the loop. ``None`` for a disabled cache or any missing slot:
    the post-staging gate is then the only (and always the authoritative) check.
    """
    if not ctx.enabled:
        return None
    with open_features_group(ctx.paths, "cp", mode="r") as group:
        if group is None:
            return None
    blocks, _, missing = _read_gt_cp_blocks(ctx, gt_path, n_features)
    return None if missing else blocks


def fit_cp_reference(
    fits: list[DatasetFit],
    *,
    target_name: str,
    feature_names: tuple[str, ...],
    cp_identity: dict[str, Any],
) -> dict[str, Any]:
    """Fit a CP reference and return its JSON payload.

    Parameters
    ----------
    fits : list of DatasetFit
        Every test set of the target. A fit without ``parent`` gets a scaler; those
        with ``in_mask_fit`` are pooled for the feature mask and the floor's pooled
        std. A fit with ``parent`` (lite) is mapped to that parent's scaler, and
        only its cell hash, count and positions are recorded.
    target_name : str
        Eval ``target_name`` (``nucleus``/``membrane``/``er``/``mitochondria``).
    feature_names : tuple of str
        Column names of every ``cells`` matrix.
    cp_identity : dict
        Dataset-independent CP recipe identity (from :func:`cp_space`).

    Returns
    -------
    dict
        JSON-serializable payload, including its own content ``sha256``.

    Raises
    ------
    ValueError
        If a matrix does not match ``feature_names`` or holds a non-finite value,
        a dataset has no cells, a dataset appears twice, a lite parent has no
        scaler, a lite fit is in the mask fit, no dataset is in the mask fit, or
        the pooled fit keeps a zero-variance feature.
    """
    names = [f.dataset for f in fits]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate datasets: {names}")
    scaled = [f for f in fits if f.parent is None]
    for fit in fits:
        if fit.cells.ndim != 2 or fit.cells.shape[1] != len(feature_names):
            raise ValueError(f"{fit.dataset}: GT matrix shape {fit.cells.shape} does not match {len(feature_names)}")
        if fit.cells.shape[0] == 0:
            raise ValueError(f"{fit.dataset}: no GT cells")
        if not np.isfinite(fit.cells).all():
            raise ValueError(f"{fit.dataset}: GT matrix holds non-finite values; drop them before fitting")
        if fit.parent is not None and (fit.in_mask_fit or fit.parent not in {f.dataset for f in scaled}):
            raise ValueError(f"lite dataset {fit.dataset} must reuse a scaled parent, not {fit.parent!r}")

    pooled = np.concatenate([np.asarray(f.cells, dtype=np.float64) for f in scaled if f.in_mask_fit], axis=0)
    if pooled.shape[0] == 0:
        raise ValueError("no dataset is in the mask fit")
    keep_mask = select_gt_features(
        pooled, freq_cut=DEFAULT_FREQ_CUT, unique_cut=DEFAULT_UNIQUE_CUT, corr_threshold=DEFAULT_CORR_THRESHOLD
    )
    kept_names = [n for n, k in zip(feature_names, keep_mask, strict=True) if k]
    if not kept_names:
        raise ValueError(
            f"GT-only feature selection kept no CP features out of {len(feature_names)} "
            f"({pooled.shape[0]} pooled GT cells); a reference with an empty mask cannot score CP."
        )
    pooled_std = pooled[:, keep_mask].std(axis=0)
    if not (pooled_std > 0).all():
        raise ValueError(f"pooled fit kept zero-variance features: {np.array(kept_names)[pooled_std == 0]}")
    floor = STD_FLOOR_FRACTION * pooled_std

    def _provenance(fit: DatasetFit) -> dict[str, Any]:
        return {
            **fit.record,
            "in_mask_fit": fit.in_mask_fit,
            "n_cells": int(fit.cells.shape[0]),
            "gt_moments": gt_moments(fit.cells),
            "gt_matrix_sha256": gt_matrix_sha256(fit.cells),
        }

    scalers: dict[str, dict[str, Any]] = {}
    for fit in scaled:
        kept = np.asarray(fit.cells, dtype=np.float64)[:, keep_mask]
        std = kept.std(axis=0)
        floored = std < floor
        scalers[fit.dataset] = {
            "mean": [float(v) for v in kept.mean(axis=0)],
            "std": [float(v) for v in np.where(floored, floor, std)],
            "floored_features": [n for n, f in zip(kept_names, floored, strict=True) if f],
        }
    lite = [f for f in fits if f.parent is not None]
    survey = _gt_abs_z_survey(fits, scalers, keep_mask, kept_names)

    payload: dict[str, Any] = {
        "schema": CP_REFERENCE_SCHEMA,
        "target_name": target_name,
        "feature_names": list(feature_names),
        "keep_mask": [bool(b) for b in keep_mask],
        "kept_feature_names": kept_names,
        "criteria": {
            "freq_cut": DEFAULT_FREQ_CUT,
            "unique_cut": DEFAULT_UNIQUE_CUT,
            "corr_threshold": DEFAULT_CORR_THRESHOLD,
            "mask_fit_on": "gt_only_pooled",
            "scaler_fit_on": "gt_only_per_dataset",
            "std_floor_fraction_of_pooled": STD_FLOOR_FRACTION,
            "z_clip": CP_Z_CLIP,
            "gt_clip_frac_max": CP_GT_CLIP_FRAC_MAX,
            "rank_stability": CP_Z_CLIP_RANK_STABILITY,
            "gt_abs_z_survey": {k: v for k, v in survey.items() if k != "datasets"},
        },
        "cp_identity": cp_identity,
        "scalers": scalers,
        "lite": {f.dataset: {"parent": f.parent} for f in lite},
        "fit": {
            "mask_fit": {
                "datasets": [f.dataset for f in scaled if f.in_mask_fit],
                "n_cells": int(pooled.shape[0]),
                "gt_matrix_sha256": gt_matrix_sha256(pooled),
                "pooled_std": [float(v) for v in pooled_std],
            },
            "datasets": {f.dataset: {**_provenance(f), "gt_abs_z": survey["datasets"][f.dataset]} for f in scaled},
            "lite": {f.dataset: {**_provenance(f), "gt_abs_z": survey["datasets"][f.dataset]} for f in lite},
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload["sha256"] = payload_sha256(payload)
    return payload


def _gt_abs_z_survey(
    fits: list[DatasetFit], scalers: dict[str, dict[str, Any]], keep_mask: np.ndarray, kept_names: list[str]
) -> dict[str, Any]:
    """Survey |z| of every fit dataset's GT cells in its OWN test set's scaler (lite: the parent's).

    Returns, over the FULL (non-lite) test sets -- lite cells are subsets of their
    parent's, so pooling them would count cells twice -- the max |z|, the p99.99
    (pooled over every GT cell and kept feature), where the max sits, the largest
    enforced per-dataset GT clip fraction, the number of GT cells and of those with any
    |z| > :data:`CP_Z_CLIP`; plus, for every dataset (lite included, unenforced),
    ``{gt_clip_frac, enforced, max, p9999, max_feature}``.

    Raises
    ------
    ValueError
        If any non-lite dataset's ``gt_clip_frac`` (fraction of its GT cells with any
        |z| > :data:`CP_Z_CLIP`) exceeds :data:`CP_GT_CLIP_FRAC_MAX`: the clip would
        then discard more GT signal than the bound allows. Lite datasets are recorded
        with ``enforced: false``: their cells are a subset of the parent's, in the
        parent's scaler, so one extra cell on ~1000 must not brick every build.
    """
    per: dict[str, dict[str, Any]] = {}
    pooled: list[np.ndarray] = []  # full (non-lite) datasets only: lite cells duplicate their parent's
    n_full = n_full_beyond = 0
    for fit in fits:
        scaler = scalers[fit.parent or fit.dataset]
        z = np.abs(
            (np.asarray(fit.cells, dtype=np.float64)[:, keep_mask] - np.asarray(scaler["mean"]))
            / np.asarray(scaler["std"])
        )
        col = int(np.unravel_index(np.argmax(z), z.shape)[1])
        per[fit.dataset] = {
            "gt_clip_frac": float((z > CP_Z_CLIP).any(axis=1).mean()),
            # Lite GT is a subset of the parent's cells in the parent's scaler, so its fraction
            # is small-sample noise on the parent's: recorded, never enforced.
            "enforced": fit.parent is None,
            "max": float(z.max()),
            "p9999": float(np.quantile(z, 0.9999)),
            "max_feature": kept_names[col],
        }
        if fit.parent is None:
            pooled.append(z.ravel())
            n_full += z.shape[0]
            n_full_beyond += int((z > CP_Z_CLIP).any(axis=1).sum())
    over = {d: r["gt_clip_frac"] for d, r in per.items() if r["enforced"] and r["gt_clip_frac"] > CP_GT_CLIP_FRAC_MAX}
    if over:
        raise ValueError(
            f"GT clip fraction above the bound {CP_GT_CLIP_FRAC_MAX} at z_clip {CP_Z_CLIP} for {over}: the clip "
            "would discard too much GT signal. Revisit CP_Z_CLIP (and re-survey) or investigate the GT cache."
        )
    worst = max((d for d in per if per[d]["enforced"]), key=lambda d: per[d]["max"])
    return {
        # Over FULL test sets only, so no GT cell is counted twice through a lite subset.
        "max": per[worst]["max"],
        "p9999": float(np.quantile(np.concatenate(pooled), 0.9999)),
        "max_dataset": worst,
        "max_feature": per[worst]["max_feature"],
        "max_gt_clip_frac": max(r["gt_clip_frac"] for r in per.values() if r["enforced"]),
        "n_cells": n_full,
        "n_cells_beyond_clip": n_full_beyond,
        "datasets": per,
    }


def write_cp_reference(payload: dict[str, Any], path: Path, *, force: bool = False) -> bool:
    """Write a reference payload atomically (tmp file in the same dir, then ``os.replace``).

    Parameters
    ----------
    payload : dict
        Output of :func:`fit_cp_reference`.
    path : pathlib.Path
        Destination; parents are created.
    force : bool, optional
        Overwrite an existing reference whose content hash differs.

    Returns
    -------
    bool
        ``True`` if the file was written, ``False`` if an identical reference (same
        hash AND same fit provenance) was already there (no-op). Same hash with new
        audit-only provenance (a re-cache that moved ``built_at``, a relocated cache
        path) is written without ``force``: no scaler, mask or gate field changes.

    Raises
    ------
    FileExistsError
        If ``path`` holds a reference with a different hash and ``force`` is off:
        replacing it changes CP values and invalidates every eval stamped with it.
    """
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["sha256"] == payload["sha256"]:
            if existing["fit"] == payload["fit"]:
                return False
        elif not force:
            raise FileExistsError(
                f"{path} holds a different CP reference (sha256 {existing['sha256'][:12]} vs new "
                f"{payload['sha256'][:12]}); replacing it invalidates every eval scored with it. "
                "Pass --force to replace it."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    os.replace(tmp, path)
    return True


@dataclass(frozen=True)
class DatasetCPSpace:
    """The CP space one eval scores in: the reference mask plus one dataset's scaler.

    Picklable (plain arrays + scalars) so it can be shipped to spawn workers.
    """

    reference_path: str
    reference_sha256: str
    target_name: str
    dataset: str
    scaler_dataset: str
    is_lite: bool
    feature_names: tuple[str, ...]
    keep_mask: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    fit_positions: frozenset[str]
    fit_n_cells: int
    gt_mean: np.ndarray
    gt_std: np.ndarray
    cp_identity: dict[str, Any]
    criteria: dict[str, Any]
    floored_features: tuple[str, ...]

    @property
    def binding_sha256(self) -> str:
        """Return a sha256 over exactly what this eval's CP numbers depend on.

        Stamped as ``cp_space_sha256`` and compared on final-metrics cache reuse.
        It covers the CP recipe identity, the selection criteria, the feature
        names and keep-mask, THIS dataset's scaler (mean, std, floored features --
        the parent's for a lite set, plus the lite -> parent link), and this
        dataset's recorded GT position set, cell count and raw moments (a lite set's own) --
        not the exact GT-matrix sha256, so GPU jitter never moves it. It deliberately
        leaves out the whole-reference hash, so adding or refitting another dataset
        of the target does not invalidate this dataset's cached eval dirs (whose
        cache is all-or-nothing, pixel and mask metrics included), while another
        dataset's space, a changed scaler, or different GT cells all do.

        Returns
        -------
        str
            Hex sha256 over the canonical JSON of those fields.
        """
        body = {
            "cp_identity": self.cp_identity,
            # The GT |z| survey summarizes EVERY dataset of the target; it is hashed in the
            # reference but left out here, so adding a dataset keeps this binding.
            "criteria": {k: v for k, v in self.criteria.items() if k != "gt_abs_z_survey"},
            "feature_names": list(self.feature_names),
            "keep_mask": [bool(b) for b in self.keep_mask],
            "dataset": self.dataset,
            "scaler_dataset": self.scaler_dataset,
            "is_lite": self.is_lite,
            "scaler": {
                "mean": [float(v) for v in self.mean],
                "std": [float(v) for v in self.std],
                "floored_features": list(self.floored_features),
            },
            "gt_positions": sorted(self.fit_positions),
            "gt_n_cells": self.fit_n_cells,
            "gt_moments": {"mean": [float(v) for v in self.gt_mean], "std": [float(v) for v in self.gt_std]},
        }
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _check_width(self, x: np.ndarray) -> None:
        """Raise when a CP matrix does not have the reference's column count."""
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise StaleCacheError(
                f"CP feature dimension mismatch: matrix has shape {x.shape}, the CP reference "
                f"{self.reference_path} has {len(self.feature_names)} features. A CP feature cache was built "
                "with a different recipe (e.g. a GLCM toggle or a CP_FEATURE_VERSION change without "
                "a cache rebuild). Rebuild with force_recompute.gt_cp=true and/or pred_cp=true "
                "(or force_recompute.all=true)."
            )

    def select(self, x: np.ndarray) -> np.ndarray:
        """Return the kept columns of a raw CP matrix, unscaled."""
        self._check_width(x)
        return x[:, self.keep_mask]

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Return the kept columns of a raw CP matrix, standardized by this dataset's GT scaler."""
        return (self.select(x) - self.mean) / self.std

    @property
    def z_clip(self) -> float:
        """The reference's shared-space clip, :data:`CP_Z_CLIP` at build time."""
        return float(self.criteria["z_clip"])

    def transform_clipped(self, x: np.ndarray) -> np.ndarray:
        """Return :meth:`transform` clipped to ``+-z_clip``: the space KID/FID/cosine score in.

        Applied identically to pred and GT, at dataset level and per row. The build
        survey guarantees at most ``CP_GT_CLIP_FRAC_MAX`` of any dataset's GT cells are
        clipped, so this mainly bounds how far one heavy-tailed predicted feature can
        push the cubic KID kernel.
        """
        return np.clip(self.transform(x), -self.z_clip, self.z_clip)

    def clip_fraction(self, x: np.ndarray) -> dict[str, Any]:
        """Return what :meth:`transform_clipped` discards on ``x``: the fraction of cells clipped.

        Parameters
        ----------
        x : np.ndarray
            Raw ``(n_cells, n_features)`` CP matrix (the prediction's, in the eval).

        Returns
        -------
        dict
            ``{"any": fraction of cells with any kept-feature |z| > z_clip,
            "per_feature": {kept feature: fraction of cells clipped on it}}``; NaN
            fractions for zero cells.
        """
        kept_names = [n for n, k in zip(self.feature_names, self.keep_mask, strict=True) if k]
        if x.shape[0] == 0:
            return {"any": float("nan"), "per_feature": dict.fromkeys(kept_names, float("nan"))}
        over = np.abs(self.transform(x)) > self.z_clip
        return {
            "any": float(over.any(axis=1).mean()),
            "per_feature": {n: float(v) for n, v in zip(kept_names, over.mean(axis=0), strict=True)},
        }

    def check_positions(self, positions: list[str]) -> None:
        """Refuse an eval whose GT positions are not exactly the ones the fit recorded.

        Parameters
        ----------
        positions : list of str
            GT position names the eval scores (after any exclusion/limit).

        Raises
        ------
        ValueError
            If the positions differ from the fit's (e.g. ``limit_positions`` or
            ``io.exclude_fov_names``: a partial walk's CP cells are not the cells
            the scaler was fit on and hashed over).
        """
        if set(positions) != self.fit_positions:
            extra = sorted(set(positions) - self.fit_positions)
            missing = sorted(self.fit_positions - set(positions))
            raise ValueError(
                f"{self.dataset}: CP metrics need exactly the GT positions the CP reference recorded; this eval "
                f"adds {extra[:5]} and skips {len(missing)} (e.g. {missing[:5]}). limit_positions / "
                "io.exclude_fov_names cannot be combined with CP feature metrics: pass "
                "compute_feature_metrics=false for smoke runs."
            )

    def gt_cells_problems(self, cells: np.ndarray) -> list[str]:
        """Return why a canonical GT matrix fails the content gate (empty = it passes).

        Shared by the eval (:meth:`check_gt_cells`) and ``build_cp_reference.py
        --verify``: exact cell count, then per-feature moments within tolerance.
        """
        if cells.shape[0] != self.fit_n_cells:
            return [f"{cells.shape[0]} GT cells vs {self.fit_n_cells} in the fit"]
        return gt_moments_mismatch(cells, self.gt_mean, self.gt_std, self.feature_names)

    def check_gt_cells(self, blocks: dict[tuple[str, int], np.ndarray]) -> None:
        """Refuse to score unless the eval's GT CP cells are the reference's fit cells.

        The content gate, on the GT cells this run actually staged (after any GT CP
        recompute): the GT-finite cell count must equal the fit's exactly, and every
        CP feature's raw mean/std must match the recorded moments within
        :data:`GT_MOMENT_RTOL` (:func:`gt_moments_mismatch`). So a GT re-cache that
        reproduces the cells -- on CPU bit-exactly, on GPU up to ~1e-15 jitter
        (``--overwrite``, ``force_recompute.gt_cp``/``all``) -- passes, while a
        value or count change fails, however the cache manifest was merged.

        Parameters
        ----------
        blocks : dict
            ``{(position, t): GT-finite CP rows}`` over every scored FOV and
            timepoint, before the pred/GT paired non-finite drop.

        Raises
        ------
        StaleCacheError
            If the count differs or any feature's moments are out of tolerance.
        """
        problems = self.gt_cells_problems(canonical_gt_matrix(blocks, len(self.feature_names)))
        if problems:
            raise StaleCacheError(
                f"{self.dataset}: the GT CP cells this eval scores differ from the CP reference fit in "
                f"{self.reference_path}: {'; '.join(problems[:5])}. The GT CP cache changed since the reference "
                f"was built; rebuild it:\n  {_build_command(self.target_name)} --force"
            )


@dataclass(frozen=True)
class CPReference:
    """A loaded, hash-verified CP reference."""

    path: str
    sha256: str
    target_name: str
    feature_names: tuple[str, ...]
    keep_mask: np.ndarray
    cp_identity: dict[str, Any]
    criteria: dict[str, Any]
    scalers: dict[str, dict[str, Any]]
    lite: dict[str, dict[str, Any]]
    fit: dict[str, Any]

    def for_dataset(self, dataset: str) -> DatasetCPSpace:
        """Bind the reference to one eval dataset (a lite dataset gets its parent's scaler).

        Raises
        ------
        KeyError
            If the reference has neither a scaler nor a lite entry for ``dataset``.
        """
        is_lite = dataset in self.lite
        if not is_lite and dataset not in self.scalers:
            raise KeyError(
                f"CP reference {self.path} has no scaler for dataset {dataset!r} "
                f"(scalers: {sorted(self.scalers)}, lite: {sorted(self.lite)}); rebuild it:\n"
                f"  {_build_command(self.target_name)}"
            )
        scaler_dataset = self.lite[dataset]["parent"] if is_lite else dataset
        scaler = self.scalers[scaler_dataset]
        own = self.fit["lite"][dataset] if is_lite else self.fit["datasets"][dataset]
        return DatasetCPSpace(
            reference_path=self.path,
            reference_sha256=self.sha256,
            target_name=self.target_name,
            dataset=dataset,
            scaler_dataset=scaler_dataset,
            is_lite=is_lite,
            feature_names=self.feature_names,
            keep_mask=self.keep_mask,
            mean=np.asarray(scaler["mean"], dtype=np.float64),
            std=np.asarray(scaler["std"], dtype=np.float64),
            fit_positions=frozenset(own["positions"]),
            fit_n_cells=int(own["n_cells"]),
            gt_mean=np.asarray(own["gt_moments"]["mean"], dtype=np.float64),
            gt_std=np.asarray(own["gt_moments"]["std"], dtype=np.float64),
            cp_identity=self.cp_identity,
            criteria=self.criteria,
            floored_features=tuple(scaler["floored_features"]),
        )


def _read_cp_reference(path: Path, *, build_hint: str) -> CPReference:
    """Load a CP reference and verify its content hash and schema (no target check)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"CP reference {path} not found. CP (GLCM+) feature metrics are scored in a per-target "
            f"GT feature space; build it with:\n  {build_hint}\n"
            "(or point feature_metrics.cp.reference_path at an existing reference)."
        )
    payload = json.loads(path.read_text())
    if payload["sha256"] != payload_sha256(payload):
        raise ValueError(f"CP reference {path} content does not match its recorded sha256 (edited by hand?)")
    if payload["schema"] != CP_REFERENCE_SCHEMA:
        raise ValueError(f"CP reference {path} has schema {payload['schema']}, expected {CP_REFERENCE_SCHEMA}")
    return CPReference(
        path=str(path),
        sha256=payload["sha256"],
        target_name=payload["target_name"],
        feature_names=tuple(payload["feature_names"]),
        keep_mask=np.asarray(payload["keep_mask"], dtype=bool),
        cp_identity=payload["cp_identity"],
        criteria=payload["criteria"],
        scalers=payload["scalers"],
        lite=payload["lite"],
        fit=payload["fit"],
    )


def load_cp_reference(path: Path, *, target_name: str) -> CPReference:
    """Load and hash-verify a CP reference.

    Parameters
    ----------
    path : pathlib.Path
        Reference JSON.
    target_name : str
        The eval's ``target_name``; the reference must have been fit for it.

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
        for another target or schema.
    """
    ref = _read_cp_reference(path, build_hint=_build_command(target_name))
    if ref.target_name != target_name:
        raise ValueError(f"CP reference {path} was fit for {ref.target_name!r}, this eval is {target_name!r}")
    return ref


def cp_sidecar_payload(space: DatasetCPSpace) -> dict[str, Any]:
    """Return the :data:`CP_SIDECAR_FILENAME` payload an eval writes for the space it scored in.

    ``reference_path`` + ``dataset`` + ``cp_space_sha256`` are what
    :func:`sidecar_cp_space` reads back; the rest (whole-reference sha256, scaler
    dataset, mask) is audit.
    """
    return {
        "reference_path": space.reference_path,
        "dataset": space.dataset,
        # What post-hoc readers compare: this dataset's CP space, which survives rebuilds
        # that leave it unchanged. The whole-reference hash is audit only.
        "cp_space_sha256": space.binding_sha256,
        "reference_sha256": space.reference_sha256,
        "scaler_dataset": space.scaler_dataset,
        "feature_names": list(space.feature_names),
        "keep_mask": [bool(b) for b in space.keep_mask],
        "n_kept": int(space.keep_mask.sum()),
        "n_total": int(space.keep_mask.size),
    }


def sidecar_cp_space(eval_dir: Path) -> DatasetCPSpace:
    """Return the CP space a finished eval dir was scored in, from its :data:`CP_SIDECAR_FILENAME`.

    The sidecar records the reference path, the eval's dataset and that dataset's
    ``cp_space_sha256`` (:attr:`DatasetCPSpace.binding_sha256`). The current
    reference at that path is loaded and bound to the recorded dataset (a lite eval
    gets its parent's scaler exactly as the pipeline did), and its binding must equal
    the recorded one. A rebuild that leaves this dataset's CP space unchanged (another
    dataset added or refit, an audit field moved) therefore keeps the sidecar
    readable; one that changes it is refused. Used by post-hoc consumers of eval dirs
    (the cross-condition probe, the lite subset tools) so they score CP the way the
    eval did.

    Parameters
    ----------
    eval_dir : pathlib.Path
        Eval save dir holding the sidecar.

    Returns
    -------
    DatasetCPSpace
        The reference bound to the eval's dataset.

    Raises
    ------
    FileNotFoundError
        If the sidecar or the reference it names is missing.
    ValueError
        If the dataset's CP space in the current reference differs from the one the
        eval recorded.
    """
    sidecar = json.loads((eval_dir / CP_SIDECAR_FILENAME).read_text())
    path = Path(sidecar["reference_path"])
    ref = _read_cp_reference(path, build_hint="the reference this eval was scored in")
    space = ref.for_dataset(sidecar["dataset"])
    if space.binding_sha256 != sidecar["cp_space_sha256"]:
        raise ValueError(
            f"{eval_dir}: the CP space of {sidecar['dataset']} in {path} changed since the eval "
            f"(cp_space_sha256 {space.binding_sha256[:12]} now, {sidecar['cp_space_sha256'][:12]} at eval time); "
            "re-evaluate this dir"
        )
    return space


def resolve_cp_reference_path(config: DictConfig) -> Path:
    """Return ``feature_metrics.cp.reference_path``, or the registry default when null.

    The registry default lives under ``DATA_ROOT`` for full and lite evals alike.

    Raises
    ------
    ValueError
        If the default is requested for a target with no CP reference.
    """
    override = config.feature_metrics.cp.reference_path
    if override is not None:
        # Absolute, so the path stamped into an eval dir's sidecar still resolves when
        # a post-hoc tool (cross-condition probe, lite tools) runs from another cwd.
        return Path(override).resolve()
    if config.target_name not in MASK_FIT_DATASETS:
        raise ValueError(f"no CP reference for target {config.target_name!r}; expected {sorted(MASK_FIT_DATASETS)}")
    return cp_reference_path(config.target_name)


def eval_cp_space(config: DictConfig) -> DatasetCPSpace:
    """Load the eval's CP reference, refuse a foreign recipe, and bind it to the eval's dataset.

    Parameters
    ----------
    config : DictConfig
        Eval config (``target_name``, ``feature_metrics.cp.*``,
        ``benchmark.dataset_ref.dataset``).

    Returns
    -------
    DatasetCPSpace
        The verified reference bound to the eval's dataset.

    Raises
    ------
    FileNotFoundError
        If the reference does not exist.
    StaleCacheError
        If the reference's CP recipe identity or feature names differ from the eval's.
    KeyError
        If the reference has no scaler for the eval's dataset.
    ValueError
        If the config has no ``benchmark.dataset_ref.dataset``.
    """
    identity, names = cp_space(config)
    ref = load_cp_reference(resolve_cp_reference_path(config), target_name=config.target_name)
    if ref.cp_identity != identity or ref.feature_names != names:
        raise StaleCacheError(
            f"CP reference {ref.path} was built for a different CP recipe.\n"
            f"  reference identity: {ref.cp_identity}\n  eval identity:      {identity}\n"
            f"  reference features: {len(ref.feature_names)}; eval features: {len(names)}\n"
            f"Rebuild it from caches of the current recipe:\n  {_build_command(ref.target_name)}"
        )
    dataset = OmegaConf.select(config, "benchmark.dataset_ref.dataset", default=None)
    if dataset is None:
        raise ValueError(
            "CP feature metrics need benchmark.dataset_ref.dataset: the CP reference holds one GT scaler per "
            "test set, keyed by that name. Set benchmark.dataset_ref (dataset + target) in the eval config, or "
            "pass compute_feature_metrics=false."
        )
    return ref.for_dataset(dataset)
