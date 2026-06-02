"""Feature-space similarity metrics backed by torch-fidelity.

Replaces the prior in-tree implementations
(:func:`dynacell.evaluation.utils._frechet_distance`,
:func:`dynacell.evaluation.utils._polynomial_mmd`,
:func:`dynacell.evaluation.utils._pairwise_feature_metrics`) with the
``*_features_to_metric`` helpers from ``torch_fidelity``, plus a
bootstrap loop around Kynkäänniemi precision / recall so the dataset
metrics ship mean *and* std for every kernel-based or manifold-based
quantity. Cosine similarity stays in-tree because torch-fidelity does
not expose a helper for it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
from torch_fidelity.metric_kid import kid_features_to_metric
from torch_fidelity.metric_mind import mind_features_to_metric
from torch_fidelity.metric_prc import prc_features_to_metric

from dynacell.evaluation.metrics import drop_paired_nonfinite_rows

_KID_MIN_SUBSET_SIZE = 16


def _median_cosine_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    """Per-row median cosine similarity between aligned ``(pred, target)``.

    Returns ``nan`` when no row pair has non-zero norms on both sides.
    """
    pred, target = drop_paired_nonfinite_rows(pred, target)
    if pred.shape[0] == 0:
        return float("nan")
    numerator = np.einsum("ij,ij->i", pred, target)
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1)
    nonzero = denominator > 0
    if not np.any(nonzero):
        return float("nan")
    cos = np.clip(numerator[nonzero] / denominator[nonzero], -1.0, 1.0)
    return float(np.median(cos))


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    """Wrap a ``(n, d)`` array as a CPU float32 ``torch.Tensor``."""
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def _fid(pred: np.ndarray, target: np.ndarray) -> float:
    """Frechet distance via torch-fidelity's eigvals-based composition.

    Mirrors the math at ``torch_fidelity.metric_fid.fid_statistics_to_metric``:
    for symmetric PSD ``Σ₁, Σ₂``, ``Σᵢ √λᵢ(Σ₁·Σ₂) == Tr(sqrt(Σ₁·Σ₂))``.
    Faster than ``scipy.linalg.sqrtm`` and avoids its convergence warnings.

    Returns ``nan`` for cohorts with fewer than 2 rows on either side —
    ``np.cov`` is undefined at N<2 and would produce a NaN covariance
    that crashes ``np.linalg.eigvals`` downstream.
    """
    if pred.shape[0] < 2 or target.shape[0] < 2:
        return float("nan")
    stats_pred = fid_features_to_statistics(_to_tensor(pred))
    stats_target = fid_features_to_statistics(_to_tensor(target))
    out = fid_statistics_to_metric(stats_pred, stats_target, verbose=False)
    return float(out["frechet_inception_distance"])


def _kid(
    pred: np.ndarray,
    target: np.ndarray,
    kid_subsets: int,
    kid_subset_size: int,
    rng_seed: int,
) -> tuple[float, float]:
    """KID mean and std with auto-shrunk subset size for small cohorts.

    Returns ``(nan, nan)`` when the effective subset size drops below
    :data:`_KID_MIN_SUBSET_SIZE` (KID std is uninformative when the
    library would otherwise resample with replacement-like overlap).
    """
    n_pred = pred.shape[0]
    n_target = target.shape[0]
    if n_pred < 2 or n_target < 2:
        return float("nan"), float("nan")
    effective_size = min(kid_subset_size, n_pred, n_target)
    if effective_size < _KID_MIN_SUBSET_SIZE:
        return float("nan"), float("nan")
    out = kid_features_to_metric(
        _to_tensor(pred),
        _to_tensor(target),
        kid_subsets=kid_subsets,
        kid_subset_size=effective_size,
        kid_kernel="poly",
        rng_seed=rng_seed,
        verbose=False,
    )
    return (
        float(out["kernel_inception_distance_mean"]),
        float(out["kernel_inception_distance_std"]),
    )


def _bootstrap_prc(
    pred: np.ndarray,
    target: np.ndarray,
    prc_neighborhood: int,
    prc_bootstrap_subsets: int,
    prc_bootstrap_size: int,
    rng_seed: int,
) -> tuple[float, float, float, float, float, float]:
    """Bootstrap Kynkäänniemi precision / recall / F1 means and stds.

    Each iteration draws ``prc_bootstrap_size`` rows with replacement
    from both ``pred`` and ``target``, rebuilds the k-NN manifolds on
    those resamples, and calls ``prc_features_to_metric`` (PRC
    convention: ``features_1=generated, features_2=real``).
    """
    rng = np.random.default_rng(rng_seed)
    precisions = np.empty(prc_bootstrap_subsets, dtype=np.float64)
    recalls = np.empty(prc_bootstrap_subsets, dtype=np.float64)
    f_scores = np.empty(prc_bootstrap_subsets, dtype=np.float64)
    n_pred = pred.shape[0]
    n_target = target.shape[0]
    pred_t = _to_tensor(pred)
    target_t = _to_tensor(target)
    for b in range(prc_bootstrap_subsets):
        idx_pred = torch.from_numpy(rng.integers(0, n_pred, size=prc_bootstrap_size))
        idx_target = torch.from_numpy(rng.integers(0, n_target, size=prc_bootstrap_size))
        out = prc_features_to_metric(
            pred_t[idx_pred],
            target_t[idx_target],
            prc_neighborhood=prc_neighborhood,
            verbose=False,
        )
        precisions[b] = out["precision"]
        recalls[b] = out["recall"]
        f_scores[b] = out["f_score"]
    return (
        float(precisions.mean()),
        float(precisions.std()),
        float(recalls.mean()),
        float(recalls.std()),
        float(f_scores.mean()),
        float(f_scores.std()),
    )


def _mind(pred: np.ndarray, target: np.ndarray, num_projections: int, rng_seed: int, use_gpu: bool = False) -> float:
    """Sliced 2-Wasserstein based MIND from torch-fidelity.

    ``use_gpu=True`` routes the projection + Wasserstein sort through CUDA
    via torch-fidelity's ``cuda`` kwarg. ``False`` (default) keeps the
    compute on CPU.

    The default is False because the only production caller — the
    dataset-level threadpool in ``evaluate_predictions`` — intentionally
    leaves it unset: 4 parallel threads on a shared CUDA context would
    serialize via the allocator and torch's CPU-vs-CUDA RNG produces
    different streams for the same ``rng_seed``, breaking cross-leaf
    comparability of the MIND column. The kwarg is plumbed through for
    ad-hoc single-threaded callers (notebook / debugging) that want GPU.
    """
    if pred.shape[0] == 0 or target.shape[0] == 0:
        return float("nan")
    out = mind_features_to_metric(
        _to_tensor(pred),
        _to_tensor(target),
        mind_num_projections=num_projections,
        rng_seed=rng_seed,
        cuda=bool(use_gpu and torch.cuda.is_available()),
        verbose=False,
    )
    return float(out["monge_inception_distance"])


def compute_feature_similarity(
    pred: np.ndarray,
    target: np.ndarray,
    prefix: str,
    kid_subsets: int = 100,
    kid_subset_size: int = 1000,
    prc_neighborhood: int = 5,
    prc_bootstrap_subsets: int = 100,
    prc_bootstrap_size: int | None = None,
    mind_num_projections: int = 1000,
    rng_seed: int = 2020,
    use_gpu: bool = False,
) -> dict[str, float]:
    """Compute dataset-level feature-similarity metrics for one prefix.

    Returns the FID / KID (mean + std) / Precision / Recall / F1 (each
    mean + bootstrap std) / MIND / median cosine similarity dict keyed
    by ``f"{prefix}_<METRIC>"``. Empty / single-row inputs return all-NaN.

    Parameters
    ----------
    pred : np.ndarray
        Predicted per-cell features, shape ``(n_pred, d)``. Treated as
        the "generated" side (``features_1`` in torch-fidelity's PRC
        convention).
    target : np.ndarray
        Ground-truth per-cell features, shape ``(n_target, d)``. Treated
        as the "real" side (``features_2``).
    prefix : str
        Column prefix, e.g. ``"CP"``, ``"DINOv3"``, ``"DynaCLR"``.
    kid_subsets, kid_subset_size : int
        Forwarded to ``kid_features_to_metric``; the subset size is
        auto-shrunk to ``min(kid_subset_size, n_pred, n_target)`` and
        the metric is NaN'd when the result is below 16.
    prc_neighborhood : int
        ``k`` for Kynkäänniemi manifold radii. ``5`` matches the
        in-repo paper script.
    prc_bootstrap_subsets : int
        Number of bootstrap resamples for Precision / Recall / F1.
    prc_bootstrap_size : int, optional
        Per-resample size; defaults to ``min(n_pred, n_target)``.

    Notes
    -----
    Precision / Recall / F1 are bootstrap *means* over resamples drawn
    with replacement at ``prc_bootstrap_size`` rows per side. At the
    default ``prc_bootstrap_size == min(n_pred, n_target)``, each draw
    omits ~37% of unique source rows, so the reported values are
    systematically lower than the single-shot all-cells PRC computed by
    the paper script. They are still directly comparable across models /
    plates / conditions evaluated with the same bootstrap scheme — but
    do not compare them to non-bootstrap PRC tables.
    mind_num_projections : int
        Number of random projections for MIND.
    rng_seed : int
        Seed shared across KID, PRC bootstrap, and MIND.
    """
    keys = (
        f"{prefix}_FID",
        f"{prefix}_KID",
        f"{prefix}_KID_std",
        f"{prefix}_Precision",
        f"{prefix}_Precision_std",
        f"{prefix}_Recall",
        f"{prefix}_Recall_std",
        f"{prefix}_F1",
        f"{prefix}_F1_std",
        f"{prefix}_MIND",
        f"{prefix}_Median_Cosine_Similarity",
    )
    if pred.size == 0 or target.size == 0:
        return dict.fromkeys(keys, float("nan"))
    if pred.shape[1] != target.shape[1]:
        raise ValueError(f"Feature dim mismatch: pred {pred.shape[1]} vs target {target.shape[1]}")

    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    fid = _fid(pred, target)
    kid_mean, kid_std = _kid(pred, target, kid_subsets, kid_subset_size, rng_seed)
    bootstrap_size = prc_bootstrap_size or min(pred.shape[0], target.shape[0])
    p_mean, p_std, r_mean, r_std, f_mean, f_std = _bootstrap_prc(
        pred, target, prc_neighborhood, prc_bootstrap_subsets, bootstrap_size, rng_seed
    )
    mind = _mind(pred, target, mind_num_projections, rng_seed, use_gpu=use_gpu)
    cos = _median_cosine_similarity(pred, target)

    return {
        f"{prefix}_FID": fid,
        f"{prefix}_KID": kid_mean,
        f"{prefix}_KID_std": kid_std,
        f"{prefix}_Precision": p_mean,
        f"{prefix}_Precision_std": p_std,
        f"{prefix}_Recall": r_mean,
        f"{prefix}_Recall_std": r_std,
        f"{prefix}_F1": f_mean,
        f"{prefix}_F1_std": f_std,
        f"{prefix}_MIND": mind,
        f"{prefix}_Median_Cosine_Similarity": cos,
    }


def compute_feature_similarity_pairwise(
    pred: np.ndarray,
    target: np.ndarray,
    prefix: str,
    kid_subsets: int = 100,
    kid_subset_size: int = 1000,
    rng_seed: int = 2020,
) -> dict[str, float]:
    """Per-(FOV, timepoint) variant: FID, KID mean + std, cosine only.

    PRC and MIND are dataset-level metrics; running them per-timepoint
    on ~50-cell cohorts is uninformative (the manifold is too sparse
    and the bootstrap variance dominates). Returns the four-column dict
    keyed by ``f"{prefix}_<METRIC>"``.
    """
    keys = (
        f"{prefix}_FID",
        f"{prefix}_KID",
        f"{prefix}_KID_std",
        f"{prefix}_Median_Cosine_Similarity",
    )
    if pred.size == 0 or target.size == 0:
        return dict.fromkeys(keys, float("nan"))
    if pred.shape[1] != target.shape[1]:
        raise ValueError(f"Feature dim mismatch: pred {pred.shape[1]} vs target {target.shape[1]}")

    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    fid = _fid(pred, target)
    kid_mean, kid_std = _kid(pred, target, kid_subsets, kid_subset_size, rng_seed)
    cos = _median_cosine_similarity(pred, target)

    return {
        f"{prefix}_FID": fid,
        f"{prefix}_KID": kid_mean,
        f"{prefix}_KID_std": kid_std,
        f"{prefix}_Median_Cosine_Similarity": cos,
    }
