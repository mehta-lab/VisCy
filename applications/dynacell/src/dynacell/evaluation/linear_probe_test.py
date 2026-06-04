"""Tests for FOV-stratified linear-probe diagnostics."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from dynacell.evaluation.linear_probe import (
    MADScaler,
    fov_stratified_auroc,
    indistinguishability,
)


def test_indistinguishability_corner_values():
    assert indistinguishability(0.0) == pytest.approx(0.0)
    assert indistinguishability(0.25) == pytest.approx(0.5)
    assert indistinguishability(0.5) == pytest.approx(1.0)
    assert indistinguishability(0.75) == pytest.approx(0.5)
    assert indistinguishability(1.0) == pytest.approx(0.0)


def test_madscaler_fit_transform_shape():
    rng = np.random.default_rng(2020)
    X = rng.normal(size=(50, 4))
    scaler = MADScaler()
    Xt = scaler.fit(X).transform(X)
    assert Xt.shape == X.shape
    # After transform with statistics fit on the same data, per-column
    # median should be ~0.
    np.testing.assert_allclose(np.median(Xt, axis=0), np.zeros(4), atol=1e-10)


def test_madscaler_handles_zero_mad():
    # Constant column has MAD=0; eps guard prevents division-by-zero.
    X = np.column_stack([np.ones(20), np.arange(20).astype(float)])
    scaler = MADScaler()
    Xt = scaler.fit(X).transform(X)
    assert np.all(np.isfinite(Xt))


def test_madscaler_is_sklearn_compatible():
    rng = np.random.default_rng(2020)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    pipeline = Pipeline(
        [
            ("scaler", MADScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert preds.shape == y.shape
    proba = pipeline.predict_proba(X)
    assert proba.shape == (60, 2)


def test_fov_stratified_auroc_no_leak():
    rng = np.random.default_rng(2020)
    n_cells = 200
    n_fovs = 10
    # Assign cells to FOVs evenly.
    fov_id = np.repeat(np.arange(n_fovs), n_cells // n_fovs)
    # Column 0: small global signal correlated with y.
    col0 = rng.normal(size=n_cells)
    y = (col0 > 0).astype(int)
    # Add noise to col0 so the signal is weak but present.
    col0_noisy = col0 + rng.normal(scale=1.5, size=n_cells)
    # Column 1: per-FOV constant (different value per FOV); no signal across
    # FOVs. A leaky pipeline would memorize FOV identity from col 1.
    fov_constants = rng.normal(size=n_fovs) * 10.0
    col1 = fov_constants[fov_id]
    X = np.column_stack([col0_noisy, col1])

    result = fov_stratified_auroc(X, y, fov_id, n_splits=5, rng_seed=2020)
    assert result["n_folds"] == 5
    assert 0.55 < result["auroc_mean"] < 0.90
    assert result["auroc_std"] > 0


def test_fov_stratified_auroc_small_fov_fallback():
    rng = np.random.default_rng(2020)
    n_cells = 60
    # Only 3 unique FOV ids.
    fov_id = np.repeat(np.arange(3), n_cells // 3)
    X = rng.normal(size=(n_cells, 4))
    # Make y depend on X so AUROC is finite.
    y = (X[:, 0] > 0).astype(int)
    result = fov_stratified_auroc(X, y, fov_id, n_splits=5, rng_seed=2020)
    assert result["n_folds"] == 3
    assert np.isfinite(result["auroc_mean"])


def test_fov_stratified_auroc_single_fov_returns_nan():
    rng = np.random.default_rng(2020)
    n_cells = 40
    fov_id = np.zeros(n_cells, dtype=int)
    X = rng.normal(size=(n_cells, 3))
    y = (X[:, 0] > 0).astype(int)
    with pytest.warns(UserWarning):
        result = fov_stratified_auroc(X, y, fov_id, n_splits=5, rng_seed=2020)
    assert np.isnan(result["auroc_mean"])
    assert np.isnan(result["auroc_std"])
    assert result["n_folds"] == 1


def test_fov_stratified_auroc_degenerate_fold_skipped():
    rng = np.random.default_rng(2020)
    n_cells = 40
    # 2 unique FOVs; all cells in FOV 0 are class 0, all in FOV 1 are class 1.
    fov_id = np.repeat(np.arange(2), n_cells // 2)
    y = fov_id.copy()
    X = rng.normal(size=(n_cells, 3))
    with pytest.warns(UserWarning):
        result = fov_stratified_auroc(X, y, fov_id, n_splits=2, rng_seed=2020)
    assert np.isnan(result["auroc_mean"])


def test_fov_stratified_auroc_end_to_end_signal_present():
    rng = np.random.default_rng(2020)
    n_cells = 400
    n_fovs = 20
    fov_id = rng.integers(0, n_fovs, size=n_cells)
    # Strong feature-0 signal correlated with y.
    col0 = rng.normal(size=n_cells)
    y = (col0 > 0).astype(int)
    X = np.column_stack([col0, rng.normal(size=(n_cells, 3))])
    result = fov_stratified_auroc(X, y, fov_id, n_splits=5, rng_seed=2020)
    assert result["auroc_mean"] > 0.75
