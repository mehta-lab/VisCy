"""Unit tests for the enriched CP feature track and per-cell similarity.

The cubic-independent pieces (robust norm, distribution callables, active
feature names, per-cell PCC) run against any cubic. The end-to-end
``cp_regionprops`` path needs the ``extra_properties`` passthrough added in
cubic>=0.7.0a12 and is skipped otherwise.
"""

import inspect

import numpy as np
import pytest
import torch

from dynacell.evaluation.metrics import (
    _CP_BASE_FEATURE_NAMES,
    _CP_GLCM_FEATURE_NAMES,
    _DISTRIBUTION_PROPS,
    _kurtosis,
    _p10,
    _p50,
    _p90,
    _per_cell_glcm,
    _robust_norm,
    _skewness,
    active_cp_feature_names,
    cp_regionprops,
    per_cell_similarity,
)

# Skip the cubic-wrapper integration when the installed cubic predates the
# extra_properties passthrough (cubic<0.7.0a12).
try:
    from cubic.feature.voxel import regionprops_table as _cubic_rpt

    _HAS_EXTRA_PROPS = "extra_properties" in inspect.signature(_cubic_rpt).parameters
except ImportError:
    _HAS_EXTRA_PROPS = False


# --- robust normalization -----------------------------------------------------
def test_robust_norm_clips_and_scales() -> None:
    """Percentile-clip tames a hot pixel; the bulk maps into [0, 1]."""
    x = np.concatenate([np.linspace(0.0, 1.0, 100), np.array([1e6])])
    out = _robust_norm(x, p_lo=1.0, p_hi=99.0)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    # The outlier is clipped, so it sits at the top of the bulk, not at 1e6.
    assert out[-1] == pytest.approx(out[:-1].max(), abs=1e-6)


def test_robust_norm_constant_image_is_finite() -> None:
    """A constant image must not produce NaN/inf (the +eps guard)."""
    x = np.full((4, 4), 7.0)
    out = _robust_norm(x)
    assert np.isfinite(out).all()
    assert float(out.max()) == pytest.approx(0.0, abs=1e-6)


# --- distribution-shape extra_properties --------------------------------------
def test_distribution_callables_use_foreground_only() -> None:
    """Callables reduce over ``intensity[regionmask]``, ignoring bbox background."""
    intensity = np.array([[10.0, 20.0, 30.0], [0.0, 0.0, 0.0]])
    mask = np.array([[True, True, True], [False, False, False]])
    fg = intensity[mask]
    assert _p50(mask, intensity) == pytest.approx(np.percentile(fg, 50))
    assert _p10(mask, intensity) == pytest.approx(np.percentile(fg, 10))
    assert _p90(mask, intensity) == pytest.approx(np.percentile(fg, 90))


def test_skewness_kurtosis_match_manual_and_are_scale_free() -> None:
    """Skew/kurtosis equal the standardized moments and are scale-invariant."""
    rng = np.random.default_rng(0)
    vals = rng.gamma(2.0, 1.0, size=500)
    grid = vals.reshape(-1, 1)
    mask = np.ones_like(grid, dtype=bool)
    m, s = vals.mean(), vals.std()
    assert _skewness(mask, grid) == pytest.approx(((vals - m) ** 3).mean() / s**3, rel=1e-6)
    assert _kurtosis(mask, grid) == pytest.approx(((vals - m) ** 4).mean() / s**4 - 3.0, rel=1e-6)
    # affine rescale leaves standardized moments unchanged
    grid2 = 3.0 * grid + 5.0
    assert _skewness(mask, grid2) == pytest.approx(_skewness(mask, grid), rel=1e-6)
    assert _kurtosis(mask, grid2) == pytest.approx(_kurtosis(mask, grid), rel=1e-6)


def test_skewness_degenerate_region_is_nan() -> None:
    """Constant or single-voxel regions yield NaN, not a divide-by-zero."""
    const = np.full((3, 1), 4.0)
    mask = np.ones_like(const, dtype=bool)
    assert np.isnan(_skewness(mask, const))
    assert np.isnan(_kurtosis(mask, const))
    one = np.array([[5.0], [0.0]])
    m1 = np.array([[True], [False]])
    assert np.isnan(_skewness(m1, one))


def test_callables_integrate_via_skimage_regionprops() -> None:
    """The 2-arg callables drop into skimage regionprops_table cleanly."""
    from skimage.measure import regionprops_table as sk_rpt

    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[1:4, 4:6] = 2
    rng = np.random.default_rng(1)
    intensity = rng.random((6, 6))
    table = sk_rpt(labels, intensity, properties=["label"], extra_properties=_DISTRIBUTION_PROPS)
    for prop in _DISTRIBUTION_PROPS:
        assert prop.__name__ in table
        assert len(table[prop.__name__]) == 2
    # p50 of cell 1 matches manual median over its footprint.
    expected = np.percentile(intensity[labels == 1], 50)
    assert table["p50"][0] == pytest.approx(expected)


# --- column schema ------------------------------------------------------------
def test_active_cp_feature_names_schema() -> None:
    base = active_cp_feature_names(False)
    full = active_cp_feature_names(True)
    assert base == _CP_BASE_FEATURE_NAMES
    assert full == _CP_BASE_FEATURE_NAMES + _CP_GLCM_FEATURE_NAMES
    assert len(base) == 15
    assert len(full) == 22
    # no collisions; gradient/laplacian are distinct from base intensity keys
    assert len(set(full)) == len(full)
    assert {"gradient_mean", "gradient_std", "laplacian_var"} <= set(base)
    assert all(name.startswith("glcm_") for name in _CP_GLCM_FEATURE_NAMES)


# --- per-cell similarity ------------------------------------------------------
def test_per_cell_similarity_pcc_affine_invariant() -> None:
    """PCC is ~1 for an affine-rescaled prediction; reduced over cells."""
    labels = np.zeros((1, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[0, 4:7, 4:7] = 2
    target = np.zeros((1, 8, 8), dtype=np.float32)
    grad = np.arange(9.0).reshape(3, 3)
    target[0, 1:4, 1:4] = grad
    target[0, 4:7, 4:7] = grad.T
    predict = 2.5 * target + 3.0  # affine: PCC must stay ~1
    out = per_cell_similarity(target[None][0], predict[None][0], labels, metrics=("pcc",), use_gpu=False)
    assert out["PerCell_PCC_mean"] == pytest.approx(1.0, abs=1e-5)
    assert out["PerCell_PCC_median"] == pytest.approx(1.0, abs=1e-5)


def test_per_cell_similarity_nan_safe_and_empty() -> None:
    """Constant cells (NaN PCC) are skipped in the reduction; empty → NaN."""
    labels = np.zeros((1, 6, 6), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1  # constant cell → PCC NaN
    labels[0, 3:5, 3:5] = 2
    target = np.zeros((1, 6, 6), dtype=np.float32)
    target[0, 3:5, 3:5] = np.array([[1.0, 2.0], [3.0, 4.0]])
    predict = target.copy()
    out = per_cell_similarity(target, predict, labels, metrics=("pcc",), use_gpu=False)
    # cell 2 is a perfect match → ~1; cell 1 is NaN and dropped by the reduce
    assert out["PerCell_PCC_mean"] == pytest.approx(1.0, abs=1e-5)

    empty = np.zeros((1, 4, 4), dtype=np.int32)
    out_empty = per_cell_similarity(target, predict, empty, metrics=("pcc",), use_gpu=False)
    assert np.isnan(out_empty["PerCell_PCC_mean"])
    assert np.isnan(out_empty["PerCell_PCC_median"])


# --- GLCM gating --------------------------------------------------------------
def test_per_cell_glcm_requires_cubic_a12(monkeypatch) -> None:
    """The GLCM CP path raises a clear error when glcm_features is unavailable."""
    import dynacell.evaluation.metrics as m

    monkeypatch.setattr(m, "glcm_features", None)
    labels = np.ones((1, 4, 4), dtype=np.int32)
    img = np.random.default_rng(2).random((1, 4, 4))
    with pytest.raises(ImportError, match="glcm_features"):
        _per_cell_glcm(img, labels, {"enabled": True, "levels": 16, "distances": (1,)})


# --- end-to-end (needs cubic>=0.7.0a12) ---------------------------------------
@pytest.mark.skipif(not _HAS_EXTRA_PROPS, reason="needs cubic>=0.7.0a12 extra_properties passthrough")
def test_cp_regionprops_columns_glcm_off() -> None:
    """cp_regionprops emits exactly the base schema (GLCM off), no NaN moments."""
    labels = np.zeros((1, 12, 12), dtype=np.int32)
    labels[0, 1:6, 1:6] = 1
    labels[0, 6:11, 6:11] = 2
    rng = np.random.default_rng(3)
    image = rng.random((1, 12, 12)).astype(np.float32)
    feats = cp_regionprops(image, labels, spacing=[1.0, 1.0, 1.0], use_gpu=False)
    assert feats.shape == (2, len(active_cp_feature_names(False)))
    assert np.isfinite(feats).all()


@pytest.mark.skipif(not _HAS_EXTRA_PROPS, reason="needs cubic>=0.7.0a12 glcm_features + extra_properties")
def test_cp_regionprops_columns_glcm_on() -> None:
    """With GLCM enabled the matrix gains exactly the seven glcm_* columns."""
    labels = np.zeros((1, 16, 16), dtype=np.int32)
    labels[0, 1:8, 1:8] = 1
    labels[0, 8:15, 8:15] = 2
    rng = np.random.default_rng(4)
    image = rng.random((1, 16, 16)).astype(np.float32)
    glcm_cfg = {"enabled": True, "levels": 16, "distances": [1]}
    feats = cp_regionprops(image, labels, spacing=[1.0, 1.0, 1.0], glcm_cfg=glcm_cfg, use_gpu=False)
    assert feats.shape == (2, len(active_cp_feature_names(True)))
    assert np.isfinite(feats).all()


@pytest.mark.skipif(not _HAS_EXTRA_PROPS, reason="needs cubic>=0.7.0a12 glcm_features + extra_properties")
def test_cp_features_scale_invariant_to_intensity_range() -> None:
    """An affine intensity rescale (the GT/pred range mismatch) must leave the
    scale-free columns unchanged: skew/kurtosis and every per-image-quantized
    GLCM prop. This is the core scale-invariance contract for the CP track."""
    labels = np.zeros((1, 16, 16), dtype=np.int32)
    labels[0, 1:8, 1:8] = 1
    labels[0, 8:15, 8:15] = 2
    rng = np.random.default_rng(5)
    image = rng.random((1, 16, 16)).astype(np.float32)
    glcm_cfg = {"enabled": True, "levels": 16, "distances": [1]}
    names = active_cp_feature_names(True)
    idx = {n: i for i, n in enumerate(names)}
    a = cp_regionprops(image, labels, spacing=[1.0, 1.0, 1.0], glcm_cfg=glcm_cfg, use_gpu=False)
    b = cp_regionprops(2.5 * image + 3.0, labels, spacing=[1.0, 1.0, 1.0], glcm_cfg=glcm_cfg, use_gpu=False)
    for col in ("skewness", "kurtosis", "glcm_contrast", "glcm_correlation", "glcm_entropy"):
        np.testing.assert_allclose(a[:, idx[col]], b[:, idx[col]], rtol=0, atol=1e-5, err_msg=col)


@pytest.mark.skipif(
    not (_HAS_EXTRA_PROPS and torch.cuda.is_available()),
    reason="needs cubic>=0.7.0a12 + CUDA (cuCIM extra_properties callbacks + per-cell GLCM on cupy)",
)
def test_cp_regionprops_cpu_gpu_parity() -> None:
    """The GPU (cupy/cuCIM) path matches the CPU path within float32 tolerance.

    Exercises the cupy ``extra_properties`` callbacks and per-cell GLCM on
    device — the code paths CPU-only runs cannot reach."""
    labels = np.zeros((1, 24, 24), dtype=np.int32)
    labels[0, 2:11, 2:11] = 1
    labels[0, 12:22, 12:22] = 2
    rng = np.random.default_rng(6)
    image = rng.random((1, 24, 24)).astype(np.float32)
    glcm_cfg = {"enabled": True, "levels": 16, "distances": [1]}
    cpu = cp_regionprops(image, labels, spacing=[1.0, 1.0, 1.0], glcm_cfg=glcm_cfg, use_gpu=False)
    gpu = cp_regionprops(image, labels, spacing=[1.0, 1.0, 1.0], glcm_cfg=glcm_cfg, use_gpu=True)
    assert gpu.shape == cpu.shape
    assert np.isfinite(gpu).all()
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-4)
