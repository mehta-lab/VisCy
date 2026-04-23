"""Tests for nuclear morphology feature extraction."""

import numpy as np
import pytest

from viscy_phenotyping.features import extract_nuclear_morphology


def _make_label_2d():
    """2-D label image with two non-overlapping squares as nuclei."""
    img = np.zeros((80, 80), dtype=np.int32)
    img[5:25, 5:25] = 1   # nucleus 1: 20×20 = 400 pixels
    img[5:25, 50:70] = 2  # nucleus 2: 20×20 = 400 pixels
    return img


def _make_label_3d():
    """3-D label image with two non-overlapping cubes as nuclei."""
    img = np.zeros((10, 40, 40), dtype=np.int32)
    img[2:8, 2:12, 2:12] = 1   # nucleus 1: 6×10×10 = 600 voxels
    img[2:8, 25:35, 25:35] = 2  # nucleus 2: 6×10×10 = 600 voxels
    return img


# --- 2-D tests ---


def test_2d_returns_one_row_per_label():
    df = extract_nuclear_morphology(_make_label_2d(), np.array([1, 2]))
    assert len(df) == 2
    assert set(df["label"]) == {1, 2}


def test_2d_missing_label_is_dropped():
    df = extract_nuclear_morphology(_make_label_2d(), np.array([1, 99]))
    assert list(df["label"]) == [1]


def test_2d_feature_columns_present():
    df = extract_nuclear_morphology(_make_label_2d(), np.array([1]))
    expected = {
        "area", "eccentricity", "equivalent_diameter_area", "extent",
        "major_axis_length", "minor_axis_length", "orientation",
        "perimeter", "solidity", "euler_number", "aspect_ratio",
    }
    assert expected.issubset(set(df.columns))


def test_2d_area_matches_pixel_count():
    df = extract_nuclear_morphology(_make_label_2d(), np.array([1]))
    assert df.loc[df["label"] == 1, "area"].item() == pytest.approx(400)


def test_2d_aspect_ratio_finite():
    df = extract_nuclear_morphology(_make_label_2d(), np.array([1, 2]))
    assert np.all(np.isfinite(df["aspect_ratio"].to_numpy()))


# --- 3-D tests ---


def test_3d_returns_one_row_per_label():
    df = extract_nuclear_morphology(_make_label_3d(), np.array([1, 2]))
    assert len(df) == 2


def test_3d_inertia_eigval_columns_present():
    df = extract_nuclear_morphology(_make_label_3d(), np.array([1]))
    eigval_cols = [c for c in df.columns if c.startswith("inertia_eigval_")]
    assert len(eigval_cols) == 3  # 3-D has 3 eigenvalues


def test_3d_no_2d_only_columns():
    df = extract_nuclear_morphology(_make_label_3d(), np.array([1]))
    # eccentricity and perimeter are 2-D only
    assert "eccentricity" not in df.columns
    assert "perimeter" not in df.columns
