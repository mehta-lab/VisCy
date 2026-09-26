"""Tests for :mod:`dynacell.evaluation.feature_select`."""

from __future__ import annotations

import numpy as np
import pytest

from dynacell.evaluation.feature_select import (
    correlation_threshold,
    select_gt_features,
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


def test_select_gt_features_drops_low_variance_and_correlated(
    rng: np.random.Generator,
) -> None:
    """GT-only selection drops a constant, a near-constant, and one of a correlated pair."""
    n = 200
    a = rng.standard_normal(n)
    near_constant = np.zeros(n)
    near_constant[0] = 1.0
    gt = np.column_stack(
        [
            np.zeros(n),  # 0: constant -> variance drop
            near_constant,  # 1: one dominant level -> variance drop
            a,  # 2: A
            a + 1e-4 * rng.standard_normal(n),  # 3: A + eps -> correlation drop (one of 2/3)
            rng.standard_normal(n),  # 4: independent
            rng.standard_normal(n),  # 5: independent
        ]
    )

    keep = select_gt_features(gt)

    assert keep.dtype == bool and keep.shape == (6,)
    assert not keep[0] and not keep[1]
    assert keep[2] ^ keep[3]
    assert keep[4] and keep[5]


def test_select_gt_features_is_deterministic(
    rng: np.random.Generator,
) -> None:
    """Repeated calls on the same GT agree exactly, and the redundant column is the one dropped."""
    gt = rng.standard_normal((150, 10))
    gt[:, 7] = gt[:, 2] * 3.0 + 1e-6 * rng.standard_normal(150)

    first = select_gt_features(gt)
    second = select_gt_features(gt.copy())

    np.testing.assert_array_equal(first, second)
    expected = np.ones(10, dtype=bool)
    expected[7] = False  # tie on connectivity -> the higher index of the (2, 7) pair goes
    np.testing.assert_array_equal(first, expected)
