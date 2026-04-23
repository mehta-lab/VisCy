"""Tests for all image-based phenotyping feature modules."""

import numpy as np
import pytest

from viscy_phenotyping.features_density import density_features
from viscy_phenotyping.features_gradient import gradient_features
from viscy_phenotyping.features_radial import concentric_uniformity_features, radial_distribution_features
from viscy_phenotyping.features_shape import shape_features
from viscy_phenotyping.features_structure import structure_features
from viscy_phenotyping.features_texture import texture_features
from viscy_phenotyping.profiler import compute_cell_features


# ---------- shared fixtures ----------


def _circle_mask(size=64, radius=20):
    """Binary circular mask centred in a square image."""
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    return (np.hypot(y - cy, x - cx) < radius).astype(np.uint8)


def _gaussian_image(size=64, sigma=10.0):
    """Gaussian blob image (float32)."""
    cy, cx = size // 2, size // 2
    y, x = np.mgrid[:size, :size]
    return np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2)).astype(np.float32)


def _ring_image(size=64, n_rings=3):
    """Concentric ring pattern image (float32)."""
    cy, cx = size // 2, size // 2
    y, x = np.mgrid[:size, :size]
    r = np.hypot(y - cy, x - cx)
    return (np.sin(r * n_rings * np.pi / (size / 2)) + 1).astype(np.float32)


def _spot_image(size=64, n_spots=10, rng_seed=0):
    """Random spots image (float32)."""
    rng = np.random.default_rng(rng_seed)
    img = np.zeros((size, size), dtype=np.float32)
    for _ in range(n_spots):
        y, x = rng.integers(5, size - 5, 2)
        img[y - 2 : y + 2, x - 2 : x + 2] = rng.uniform(0.5, 1.0)
    return img


# ---------- Problem 1: radial distribution ----------


def test_radial_distribution_keys():
    mask = _circle_mask()
    img = _gaussian_image()
    out = radial_distribution_features(img, mask)
    assert all(f"radial_frac_bin{i}" in out for i in range(8))
    assert "com_offset_norm" in out
    assert "angular_cv" in out
    assert "radial_frac_cv" in out
    assert "radial_slope" in out


def test_radial_slope_center_bright_is_positive():
    """Gaussian centred on nucleus → centre-bright → radial_slope > 0."""
    mask = _circle_mask()
    img = _gaussian_image()
    out = radial_distribution_features(img, mask)
    assert out["radial_slope"] > 0


def test_radial_slope_edge_bright_is_negative():
    """Ring at boundary → edge-bright → radial_slope < 0."""
    mask = _circle_mask()
    # Ring-like image: signal concentrated near the edge of the patch
    size = 64
    cy, cx = size // 2, size // 2
    y, x = np.mgrid[:size, :size]
    r = np.hypot(y - cy, x - cx)
    img = (r > 20).astype(np.float32)  # bright ring at outer boundary
    out = radial_distribution_features(img, mask)
    assert out["radial_slope"] < 0


def test_radial_distribution_gaussian_com_near_zero():
    """Gaussian centred on nucleus → centre-of-mass offset should be small."""
    mask = _circle_mask()
    img = _gaussian_image()
    out = radial_distribution_features(img, mask)
    assert out["com_offset_norm"] < 0.2


def test_radial_distribution_all_finite():
    mask = _circle_mask()
    img = _gaussian_image()
    out = radial_distribution_features(img, mask)
    assert all(np.isfinite(v) for v in out.values())


# ---------- Problem 3: concentric uniformity ----------


def test_concentric_uniformity_keys():
    mask = _circle_mask()
    img = _ring_image()
    out = concentric_uniformity_features(img, mask)
    assert "radial_profile_cv" in out
    assert "radial_dominant_freq" in out
    assert "radial_autocorr_lag1" in out


def test_uniform_image_low_profile_cv():
    """Uniform image → flat radial profile → low CV."""
    mask = _circle_mask()
    img = np.ones((64, 64), dtype=np.float32)
    out = concentric_uniformity_features(img, mask)
    assert out["radial_profile_cv"] < 0.05


# ---------- Problem 2: texture ----------


def test_texture_keys():
    img = _gaussian_image()
    out = texture_features(img)
    assert "intensity_mean" in out
    assert "intensity_median" in out
    assert "intensity_cv" in out
    assert "intensity_entropy" in out
    assert "glcm_homogeneity_mean" in out
    assert "lbp_entropy" in out


def test_texture_uniform_low_cv():
    img = np.ones((64, 64), dtype=np.float32)
    out = texture_features(img)
    assert out["intensity_cv"] < 1e-3


# ---------- Problem 4: density ----------


def test_density_keys():
    img = _spot_image()
    out = density_features(img)
    assert "binary_area_fraction" in out
    assert "spot_count" in out
    assert "granularity_1" in out
    assert "granularity_8" in out


def test_density_empty_image_zero_spots():
    img = np.zeros((64, 64), dtype=np.float32)
    out = density_features(img)
    assert out["spot_count"] == 0.0


# ---------- Problem 5: structure ----------


def test_structure_keys():
    img = _spot_image()
    out = structure_features(img)
    assert "edge_density" in out
    assert "skeleton_length" in out
    assert "skeleton_branch_points" in out
    assert "skeleton_endpoints" in out
    assert "n_connected_components" in out


def test_structure_all_finite():
    img = _gaussian_image()
    out = structure_features(img)
    assert all(np.isfinite(v) for v in out.values())


# ---------- Problem 6: shape ----------


def test_shape_keys():
    mask = _circle_mask()
    out = shape_features(mask)
    assert "circularity" in out
    assert "convexity" in out
    assert "radial_std_norm" in out
    assert all(f"fsd_{k}" in out for k in range(1, 7))


def test_circle_circularity_near_one():
    mask = _circle_mask(size=128, radius=40)
    out = shape_features(mask)
    assert out["circularity"] == pytest.approx(1.0, abs=0.1)


def test_circle_low_radial_std():
    mask = _circle_mask(size=128, radius=40)
    out = shape_features(mask)
    assert out["radial_std_norm"] < 0.05


# ---------- Problem 7: gradient ----------


def test_gradient_keys():
    mask = _circle_mask()
    img = _gaussian_image()
    out = gradient_features(img, mask)
    assert "gradient_mean" in out
    assert "gradient_p95" in out
    assert "laplacian_variance" in out
    assert "nucleus_mean_intensity" in out
    assert "cytoplasm_mean_intensity" in out
    assert "nucleus_to_cytoplasm_ratio" in out
    assert "gradient_entropy" in out


def test_nucleus_mean_intensity_higher_than_cytoplasm():
    """Gaussian centred on nucleus → nuclear pixels brighter than background."""
    mask = _circle_mask()
    img = _gaussian_image()
    out = gradient_features(img, mask)
    assert out["nucleus_mean_intensity"] > out["cytoplasm_mean_intensity"]


def test_gradient_uniform_near_zero():
    mask = _circle_mask()
    img = np.ones((64, 64), dtype=np.float32)
    out = gradient_features(img, mask)
    assert out["gradient_mean"] < 1e-5


# ---------- profiler orchestrator ----------


def test_compute_cell_features_returns_nonempty():
    label_patch = np.zeros((64, 64), dtype=np.int32)
    label_patch[20:45, 20:45] = 7
    img_patch = np.stack([_gaussian_image(), _ring_image()])  # (2, 64, 64)
    out = compute_cell_features(img_patch, label_patch, cell_id=7, channel_names=["ch0", "ch1"])
    assert len(out) > 0


def test_compute_cell_features_channel_prefix():
    label_patch = np.zeros((64, 64), dtype=np.int32)
    label_patch[20:45, 20:45] = 1
    img_patch = _gaussian_image()[np.newaxis]  # (1, 64, 64)
    out = compute_cell_features(img_patch, label_patch, cell_id=1, channel_names=["DAPI"])
    assert any(k.startswith("DAPI_") for k in out)


def test_compute_cell_features_shape_no_prefix():
    label_patch = np.zeros((64, 64), dtype=np.int32)
    label_patch[20:45, 20:45] = 1
    img_patch = _gaussian_image()[np.newaxis]
    out = compute_cell_features(img_patch, label_patch, cell_id=1, channel_names=["ch0"])
    # shape features have no channel prefix
    assert "circularity" in out


def test_compute_cell_features_missing_cell_returns_empty():
    label_patch = np.zeros((64, 64), dtype=np.int32)
    img_patch = _gaussian_image()[np.newaxis]
    out = compute_cell_features(img_patch, label_patch, cell_id=99, channel_names=["ch0"])
    assert out == {}
