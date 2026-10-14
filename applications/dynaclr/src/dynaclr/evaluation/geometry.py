"""Frozen-embedding geometry and lightweight probe diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FloatArray = NDArray[np.floating]


def _validate_features(features: FloatArray) -> NDArray[np.float64]:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError(f"features must have shape (samples, dimensions), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("features contain non-finite values")
    return array


def embedding_geometry(
    features: FloatArray,
    *,
    pairwise_sample: int = 2048,
    seed: int = 42,
) -> dict[str, float | int]:
    """Summarize covariance rank, scale, and cosine-distance geometry."""
    array = _validate_features(features)
    centered = array - array.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (len(centered) - 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance)[::-1], 0.0, None)
    variance = eigenvalues.sum()
    if variance <= 0:
        probabilities = np.zeros_like(eigenvalues)
        effective_rank = 0.0
        participation_ratio = 0.0
        rank_99 = 0
    else:
        probabilities = eigenvalues / variance
        nonzero = probabilities > 0
        effective_rank = float(np.exp(-(probabilities[nonzero] * np.log(probabilities[nonzero])).sum()))
        participation_ratio = float(variance**2 / np.square(eigenvalues).sum())
        rank_99 = int(np.searchsorted(np.cumsum(probabilities), 0.99) + 1)

    rng = np.random.default_rng(seed)
    sample_size = min(pairwise_sample, len(array))
    sample = array[rng.choice(len(array), size=sample_size, replace=False)]
    normalized = sample / np.clip(np.linalg.norm(sample, axis=1, keepdims=True), 1e-12, None)
    cosine_distance = 1.0 - normalized @ normalized.T
    upper = cosine_distance[np.triu_indices(sample_size, k=1)]
    feature_std = array.std(axis=0, ddof=1)
    feature_norm = np.linalg.norm(array, axis=1)

    return {
        "n_samples": int(array.shape[0]),
        "n_dimensions": int(array.shape[1]),
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "rank_99": rank_99,
        "top1_variance_fraction": float(probabilities[0]) if variance > 0 else 0.0,
        "top10_variance_fraction": float(probabilities[:10].sum()) if variance > 0 else 0.0,
        "mean_dimension_std": float(feature_std.mean()),
        "median_dimension_std": float(np.median(feature_std)),
        "mean_feature_norm": float(feature_norm.mean()),
        "cosine_distance_mean": float(upper.mean()),
        "cosine_distance_p05": float(np.quantile(upper, 0.05)),
        "cosine_distance_p50": float(np.quantile(upper, 0.50)),
        "cosine_distance_p95": float(np.quantile(upper, 0.95)),
    }


def knn_label_purity(
    features: FloatArray,
    labels: Sequence[object],
    *,
    n_neighbors: int = 10,
) -> dict[str, float]:
    """Return observed and prevalence-adjusted same-label neighbor purity."""
    array = _validate_features(features)
    label_array = np.asarray(labels)
    if len(label_array) != len(array):
        raise ValueError("labels and features must contain the same number of samples")
    if len(array) <= n_neighbors:
        raise ValueError("n_neighbors must be smaller than the sample count")

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine").fit(array)
    indices = neighbors.kneighbors(return_distance=False)[:, 1:]
    observed = float((label_array[indices] == label_array[:, None]).mean())
    _, counts = np.unique(label_array, return_counts=True)
    prevalence = counts / counts.sum()
    baseline = float(np.square(prevalence).sum())
    enrichment = float((observed - baseline) / max(1.0 - baseline, 1e-12))
    return {
        "observed": observed,
        "prevalence_baseline": baseline,
        "normalized_enrichment": enrichment,
    }


def grouped_binary_probe(
    features: FloatArray,
    labels: Sequence[object],
    groups: Sequence[object],
    *,
    n_splits: int = 3,
    seed: int = 42,
) -> dict[str, float | int]:
    """Cross-validate a linear binary probe while holding out whole groups."""
    array = _validate_features(features)
    label_array = np.asarray(labels)
    group_array = np.asarray(groups)
    if len(label_array) != len(array) or len(group_array) != len(array):
        raise ValueError("features, labels, and groups must contain the same number of samples")
    classes, encoded = np.unique(label_array, return_inverse=True)
    if len(classes) != 2:
        raise ValueError(f"binary probe requires two labels, got {classes.tolist()}")

    group_counts = [len(np.unique(group_array[encoded == class_id])) for class_id in range(2)]
    folds = min(n_splits, *group_counts)
    if folds < 2:
        raise ValueError("each class must span at least two groups")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    balanced_accuracies: list[float] = []
    aucs: list[float] = []
    for train, test in splitter.split(array, encoded, group_array):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
        classifier.fit(array[train], encoded[train])
        predicted = classifier.predict(array[test])
        probability = classifier.predict_proba(array[test])[:, 1]
        balanced_accuracies.append(balanced_accuracy_score(encoded[test], predicted))
        aucs.append(roc_auc_score(encoded[test], probability))

    return {
        "n_splits": folds,
        "balanced_accuracy_mean": float(np.mean(balanced_accuracies)),
        "balanced_accuracy_std": float(np.std(balanced_accuracies)),
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
    }
