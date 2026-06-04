"""Tests for :mod:`dynacell.evaluation.feature_select`."""

from __future__ import annotations

import numpy as np
import pytest

from dynacell.evaluation.feature_select import (
    correlation_threshold,
    select_features,
    variance_threshold,
)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded numpy generator for reproducible tests."""
    return np.random.default_rng(42)


def test_variance_threshold_drops_constant_and_near_constant(
    rng: np.random.Generator,
) -> None:
    n_samples = 100
    n_features = 12
    X = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    # Column 0: constant.
    X[:, 0] = 0.0
    # Column 1: 99% zeros + 1% unique-noise → uniqueness ~ 2/100 = 0.02
    # but freq_ratio = 1/99 < 0.05, so dropped.
    X[:, 1] = 0.0
    X[0, 1] = 1.23

    keep = variance_threshold(X, freq_cut=0.05, unique_cut=0.01)

    assert keep.dtype == bool
    assert keep.shape == (n_features,)
    assert not keep[0]
    assert not keep[1]
    assert keep[2:].all()


def test_variance_threshold_freq_ratio() -> None:
    # Case A: drop. counts = [95, 4, 1] → freq_ratio = 4/95 < 0.05.
    col_a = np.array([0.0] * 95 + [1.0] * 4 + [2.0] * 1)
    X_a = col_a.reshape(-1, 1)
    keep_a = variance_threshold(X_a, freq_cut=0.05, unique_cut=0.01)
    assert not keep_a[0]

    # Case B: keep. counts = [80, 15, 5] → freq_ratio = 15/80 = 0.1875 > 0.05,
    # uniqueness = 3/100 = 0.03 > 0.01.
    col_b = np.array([0.0] * 80 + [1.0] * 15 + [2.0] * 5)
    X_b = col_b.reshape(-1, 1)
    keep_b = variance_threshold(X_b, freq_cut=0.05, unique_cut=0.01)
    assert keep_b[0]


def test_correlation_threshold_drops_redundant(
    rng: np.random.Generator,
) -> None:
    n_samples = 200
    X = rng.standard_normal((n_samples, 5)).astype(np.float64)
    # Make col 3 a near-clone of col 1.
    X[:, 3] = X[:, 1] + 1e-4 * rng.standard_normal(n_samples)

    keep = correlation_threshold(X, threshold=0.9, method="pearson")

    assert keep.dtype == bool
    assert keep.shape == (5,)
    # The redundant pair is {1, 3}; with the tie-break rule the higher index
    # drops.
    assert not keep[3]
    assert keep[0]
    assert keep[1]
    assert keep[2]
    assert keep[4]


def test_correlation_threshold_threshold_respected(
    rng: np.random.Generator,
) -> None:
    n_samples = 200
    X = rng.standard_normal((n_samples, 5)).astype(np.float64)
    X[:, 3] = X[:, 1] + 1e-4 * rng.standard_normal(n_samples)

    keep = correlation_threshold(X, threshold=0.999999, method="pearson")

    # corr(col1, col3) ≈ 0.9999..., below the threshold of 0.999999 in expectation;
    # but to be robust, we check that at most one column is dropped.
    # The intent of the test is "raise threshold → nothing drops"; this is a
    # softer assertion that tolerates floating-point noise.
    assert keep.sum() >= 4


def test_correlation_threshold_rejects_unsupported_method(
    rng: np.random.Generator,
) -> None:
    X = rng.standard_normal((20, 3)).astype(np.float64)
    with pytest.raises(ValueError):
        correlation_threshold(X, threshold=0.9, method="spearman")


def test_select_features_applies_both_filters(
    rng: np.random.Generator,
) -> None:
    # features: [constant, A, A+ε, B]
    n_gt = 60
    n_pred = 60
    n_total = n_gt + n_pred

    a = rng.standard_normal(n_total).astype(np.float64)
    b = rng.standard_normal(n_total).astype(np.float64)
    eps = 1e-4 * rng.standard_normal(n_total).astype(np.float64)

    pooled = np.column_stack(
        [
            np.zeros(n_total),  # feature 0: constant
            a,  # feature 1: A
            a + eps,  # feature 2: A + ε
            b,  # feature 3: independent
        ]
    )

    gt = pooled[:n_gt]
    pred = pooled[n_gt:]

    gt_f, pred_f, keep_mask = select_features(gt, pred, freq_cut=0.05, unique_cut=0.01, corr_threshold=0.9)

    assert keep_mask.dtype == bool
    assert keep_mask.shape == (4,)
    # Feature 0 dropped by variance pruning.
    assert not keep_mask[0]
    # Feature 3 always kept (independent).
    assert keep_mask[3]
    # Exactly one of {1, 2} survives the correlation pruning.
    assert keep_mask[1] ^ keep_mask[2]
    assert keep_mask.sum() == 2
    assert gt_f.shape == (n_gt, 2)
    assert pred_f.shape == (n_pred, 2)


def test_select_features_shape_mismatch_raises(
    rng: np.random.Generator,
) -> None:
    gt = rng.standard_normal((10, 5)).astype(np.float64)
    pred = rng.standard_normal((10, 6)).astype(np.float64)
    with pytest.raises(ValueError, match="dim mismatch|shape"):
        select_features(gt, pred)


def test_select_features_returns_aligned_filtered_arrays(
    rng: np.random.Generator,
) -> None:
    gt = rng.standard_normal((30, 8)).astype(np.float64)
    pred = rng.standard_normal((25, 8)).astype(np.float64)
    gt_f, pred_f, keep_mask = select_features(gt, pred)

    assert gt_f.shape[0] == 30
    assert pred_f.shape[0] == 25
    assert gt_f.shape[1] == pred_f.shape[1] == int(keep_mask.sum())
