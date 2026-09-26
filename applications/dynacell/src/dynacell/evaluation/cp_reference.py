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
non-finite rows dropped. The reference records each dataset's position list, cell
count, GT-matrix hash and GT-cache ``cp_features.built_at``, and an eval refuses
to score if its GT side differs (:meth:`DatasetCPSpace.check_gt_cache`,
:meth:`DatasetCPSpace.check_positions`, :meth:`DatasetCPSpace.check_n_cells`).

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
from omegaconf import DictConfig, OmegaConf

from dynacell.evaluation.cache import StaleCacheError, cache_paths, load_manifest
from dynacell.evaluation.feature_select import (
    DEFAULT_CORR_THRESHOLD,
    DEFAULT_FREQ_CUT,
    DEFAULT_UNIQUE_CUT,
    select_gt_features,
)
from dynacell.evaluation.metrics import CP_FEATURE_VERSION, active_cp_feature_names
from dynacell.evaluation.paths import cp_reference_path
from dynacell.evaluation.pipeline_cache import cp_recipe_identity

#: Schema of the reference JSON. Bump on any change to its keys or their meaning.
CP_REFERENCE_SCHEMA = 3

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

#: Keys excluded from the content hash. The hash covers only what changes CP numbers
#: (recipe identity, criteria, feature names, keep-mask, every scaler's mean/std and
#: floored features, the lite -> parent map). ``fit`` holds the fit provenance the eval
#: checks read (positions, cell counts, GT-cache ``built_at``, GT-matrix hashes), so a
#: rebuild after a harmless GT re-cache that leaves every scaler unchanged keeps the
#: hash and invalidates no final-metrics cache stamped with it.
_UNHASHED_KEYS = frozenset({"sha256", "created_at", "fit"})


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

    Recorded per dataset so a later GT re-cache that changes any fitted value, or
    the cell count, is detectable against the artifact.
    """
    arr = np.ascontiguousarray(gt, dtype=np.float64)
    digest = hashlib.sha256(f"{arr.shape[0]}x{arr.shape[1]}:".encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


@dataclass
class DatasetFit:
    """One test set's GT cells and the record of where they came from.

    Parameters
    ----------
    dataset : str
        ``benchmark.dataset_ref.dataset`` of the test set.
    cells : np.ndarray
        ``(n_cells, n_features)`` finite raw GT CP cells, every position and timepoint.
    record : dict
        JSON-serializable provenance: at least ``positions`` (sorted names),
        ``gt_cache_dir`` and ``cp_cache_built_at``.
    in_mask_fit : bool
        Whether these cells enter the pooled mask fit.
    """

    dataset: str
    cells: np.ndarray
    record: dict[str, Any]
    in_mask_fit: bool


def fit_cp_reference(
    fits: list[DatasetFit],
    *,
    target_name: str,
    feature_names: tuple[str, ...],
    cp_identity: dict[str, Any],
    lite: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Fit a CP reference and return its JSON payload.

    Parameters
    ----------
    fits : list of DatasetFit
        Every non-lite test set that needs a scaler. Those with ``in_mask_fit``
        are pooled for the feature mask and the floor's pooled std.
    target_name : str
        Eval ``target_name`` (``nucleus``/``membrane``/``er``/``mitochondria``).
    feature_names : tuple of str
        Column names of every ``cells`` matrix.
    cp_identity : dict
        Dataset-independent CP recipe identity (from :func:`cp_space`).
    lite : dict
        ``{lite dataset: {"parent": ..., "gt_cache_dir": ..., "cp_cache_built_at": ...}}``.
        Each parent must be one of ``fits``. Only ``parent`` enters the hash; the
        rest is fit provenance.

    Returns
    -------
    dict
        JSON-serializable payload, including its own content ``sha256``.

    Raises
    ------
    ValueError
        If a matrix does not match ``feature_names`` or holds a non-finite value,
        a dataset has no cells, a dataset appears twice, a lite parent has no
        scaler, no dataset is in the mask fit, or the pooled fit keeps a
        zero-variance feature.
    """
    names = [f.dataset for f in fits]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate datasets: {names}")
    for fit in fits:
        if fit.cells.ndim != 2 or fit.cells.shape[1] != len(feature_names):
            raise ValueError(f"{fit.dataset}: GT matrix shape {fit.cells.shape} does not match {len(feature_names)}")
        if fit.cells.shape[0] == 0:
            raise ValueError(f"{fit.dataset}: no GT cells")
        if not np.isfinite(fit.cells).all():
            raise ValueError(f"{fit.dataset}: GT matrix holds non-finite values; drop them before fitting")
    for lite_name, entry in lite.items():
        if entry["parent"] not in names:
            raise ValueError(f"lite dataset {lite_name} reuses {entry['parent']}, which has no scaler")

    pooled = np.concatenate([np.asarray(f.cells, dtype=np.float64) for f in fits if f.in_mask_fit], axis=0)
    if pooled.shape[0] == 0:
        raise ValueError("no dataset is in the mask fit")
    keep_mask = select_gt_features(
        pooled, freq_cut=DEFAULT_FREQ_CUT, unique_cut=DEFAULT_UNIQUE_CUT, corr_threshold=DEFAULT_CORR_THRESHOLD
    )
    kept_names = [n for n, k in zip(feature_names, keep_mask, strict=True) if k]
    pooled_std = pooled[:, keep_mask].std(axis=0)
    if not (pooled_std > 0).all():
        raise ValueError(f"pooled fit kept zero-variance features: {np.array(kept_names)[pooled_std == 0]}")
    floor = STD_FLOOR_FRACTION * pooled_std

    scalers: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for fit in fits:
        kept = np.asarray(fit.cells, dtype=np.float64)[:, keep_mask]
        std = kept.std(axis=0)
        floored = std < floor
        scalers[fit.dataset] = {
            "mean": [float(v) for v in kept.mean(axis=0)],
            "std": [float(v) for v in np.where(floored, floor, std)],
            "floored_features": [n for n, f in zip(kept_names, floored, strict=True) if f],
        }
        provenance[fit.dataset] = {
            **fit.record,
            "in_mask_fit": fit.in_mask_fit,
            "n_cells": int(kept.shape[0]),
            "gt_matrix_sha256": gt_matrix_sha256(fit.cells),
        }

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
        },
        "cp_identity": cp_identity,
        "scalers": scalers,
        "lite": {name: {"parent": entry["parent"]} for name, entry in lite.items()},
        "fit": {
            "mask_fit": {
                "datasets": [f.dataset for f in fits if f.in_mask_fit],
                "n_cells": int(pooled.shape[0]),
                "gt_matrix_sha256": gt_matrix_sha256(pooled),
                "pooled_std": [float(v) for v in pooled_std],
            },
            "datasets": provenance,
            "lite": {name: {k: v for k, v in entry.items() if k != "parent"} for name, entry in lite.items()},
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload["sha256"] = payload_sha256(payload)
    return payload


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
        provenance -- a harmless GT re-cache -- is written without ``force``: no CP
        number changes, and the eval checks need the new ``built_at``.

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
    gt_cache_dir: str | None
    cp_cache_built_at: str | None

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

    def check_gt_cache(self, gt_cache_dir: str | None) -> None:
        """Refuse an eval whose GT CP cache was re-cached after the reference was fit.

        Compares the eval's GT cache manifest ``cp_features.built_at`` with the one
        recorded at build time. A lite dataset is checked against its own cache
        only when the build recorded one (its scaler is the parent's, so a lite
        cache is otherwise not what the scaler describes); when nothing is
        recorded the check is skipped.

        Parameters
        ----------
        gt_cache_dir : str or None
            The eval's ``io.gt_cache_dir``.

        Raises
        ------
        ValueError
            If the eval reads a different GT cache dir than the recorded one.
        StaleCacheError
            If the recorded and current ``built_at`` differ.
        """
        if self.cp_cache_built_at is None:
            return
        if gt_cache_dir is None or Path(gt_cache_dir) != Path(self.gt_cache_dir):
            raise ValueError(
                f"{self.dataset}: the CP reference was fit on GT cache {self.gt_cache_dir}, "
                f"this eval reads {gt_cache_dir}"
            )
        entry = load_manifest(cache_paths(gt_cache_dir))["artifacts"].get("cp_features")
        built_at = entry["built_at"] if entry is not None else None
        if built_at != self.cp_cache_built_at:
            raise StaleCacheError(
                f"{self.dataset}: GT CP cache {gt_cache_dir} was built at {built_at}, but the CP reference "
                f"{self.reference_path} was fit on the cache built at {self.cp_cache_built_at}. "
                f"Rebuild the reference:\n  {_build_command(self.target_name)}"
            )

    def check_positions(self, positions: list[str]) -> None:
        """Refuse an eval whose GT positions are not the ones the scaler was fit on.

        Parameters
        ----------
        positions : list of str
            GT position names the eval scores (after any exclusion/limit).

        Raises
        ------
        ValueError
            If the positions are not a subset of the fit's, or, for a non-lite
            dataset, not exactly the fit's (e.g. ``limit_positions`` or
            ``io.exclude_fov_names``: a partial walk's CP values are not the
            cells the scaler describes).
        """
        extra = set(positions) - self.fit_positions
        if extra:
            raise ValueError(
                f"{self.dataset}: GT positions {sorted(extra)[:5]} are not in the CP reference fit of "
                f"{self.scaler_dataset} ({self.reference_path})"
            )
        if not self.is_lite and set(positions) != self.fit_positions:
            missing = sorted(self.fit_positions - set(positions))
            raise ValueError(
                f"{self.dataset}: CP metrics need every GT position the scaler was fit on; this eval "
                f"skips {len(missing)} (e.g. {missing[:5]}). limit_positions / io.exclude_fov_names "
                "cannot be combined with compute_feature_metrics=true."
            )

    def check_n_cells(self, n_gt_cells: int) -> None:
        """Refuse a dataset-level CP stage whose GT cell count differs from the fit's.

        Lite datasets are exempt: they score a subset of the parent's cells, and
        :meth:`CPReference.for_dataset` already required the parent to be recorded.

        Raises
        ------
        ValueError
            If ``n_gt_cells`` (finite GT CP rows scored) differs from the recorded count.
        """
        if not self.is_lite and n_gt_cells != self.fit_n_cells:
            raise ValueError(
                f"{self.dataset}: this eval scores {n_gt_cells} finite GT CP cells, the CP reference scaler was "
                f"fit on {self.fit_n_cells}. The GT CP cache changed since the reference was built."
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
        scaler_fit = self.fit["datasets"][scaler_dataset]
        own = self.fit["lite"][dataset] if is_lite else scaler_fit
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
            fit_positions=frozenset(scaler_fit["positions"]),
            fit_n_cells=int(scaler_fit["n_cells"]),
            gt_cache_dir=own["gt_cache_dir"],
            cp_cache_built_at=own["cp_cache_built_at"],
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
    if not path.is_file():
        raise FileNotFoundError(
            f"CP reference {path} not found. CP (GLCM+) feature metrics are scored in a per-target "
            f"GT feature space; build it with:\n  {_build_command(target_name)}\n"
            "(or point feature_metrics.cp.reference_path at an existing reference)."
        )
    payload = json.loads(path.read_text())
    if payload["sha256"] != payload_sha256(payload):
        raise ValueError(f"CP reference {path} content does not match its recorded sha256 (edited by hand?)")
    if payload["schema"] != CP_REFERENCE_SCHEMA:
        raise ValueError(f"CP reference {path} has schema {payload['schema']}, expected {CP_REFERENCE_SCHEMA}")
    if payload["target_name"] != target_name:
        raise ValueError(f"CP reference {path} was fit for {payload['target_name']!r}, this eval is {target_name!r}")
    return CPReference(
        path=str(path),
        sha256=payload["sha256"],
        target_name=payload["target_name"],
        feature_names=tuple(payload["feature_names"]),
        keep_mask=np.asarray(payload["keep_mask"], dtype=bool),
        cp_identity=payload["cp_identity"],
        scalers=payload["scalers"],
        lite=payload["lite"],
        fit=payload["fit"],
    )


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
        return Path(override)
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
    return ref.for_dataset(config.benchmark.dataset_ref.dataset)
