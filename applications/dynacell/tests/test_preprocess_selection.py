"""Tests for dynacell.preprocess.selection."""

from unittest.mock import patch

import numpy as np
import pytest

from dynacell.preprocess.selection import crop_depth, find_focus_depth, split_fovs


class TestCropDepth:
    """Tests for crop_depth."""

    def test_center_crop(self):
        """Crops are centered around focus_depth."""
        data = np.arange(20).reshape(1, 1, 20, 1, 1)
        result = crop_depth(data, focus_depth=10, target_depth=8)
        assert result.shape == (1, 1, 8, 1, 1)
        assert result[0, 0, 0, 0, 0] == 6  # d_start = 10 - 4 = 6

    def test_near_top_edge(self):
        """Focus near top clamps d_start to 0."""
        data = np.arange(20).reshape(1, 1, 20, 1, 1)
        result = crop_depth(data, focus_depth=2, target_depth=8)
        assert result.shape == (1, 1, 8, 1, 1)
        assert result[0, 0, 0, 0, 0] == 0  # d_start clamped to 0

    def test_near_bottom_edge(self):
        """Focus near bottom clamps to fit within array."""
        data = np.arange(20).reshape(1, 1, 20, 1, 1)
        result = crop_depth(data, focus_depth=18, target_depth=8)
        assert result.shape == (1, 1, 8, 1, 1)
        assert result[0, 0, 0, 0, 0] == 12  # d_start = 20 - 8 = 12

    def test_target_depth_exceeds_data_raises(self):
        """target_depth > data depth raises ValueError."""
        data = np.zeros((1, 1, 10, 1, 1))
        with pytest.raises(ValueError, match="exceeds data depth"):
            crop_depth(data, focus_depth=5, target_depth=20)


class TestFindFocusDepth:
    """Tests for find_focus_depth."""

    def test_returns_median(self):
        """Returns the median of per-timepoint best depths."""
        volume = np.random.default_rng(0).random((3, 10, 32, 32)).astype(np.float32)

        with patch(
            "waveorder.focus.focus_from_transverse_band",
            side_effect=[5, 7, 6],
        ):
            result = find_focus_depth(volume, na_det=1.25, lambda_ill=0.405, pixel_size=0.108)

        assert result == 6  # median of [5, 7, 6]


class TestSplitFovs:
    """Tests for split_fovs."""

    def test_correct_sizes(self):
        """Returns correct train/test sizes."""
        fovs = list(range(50))
        train, test = split_fovs(fovs, num_test=10, num_train=30)
        assert len(test) == 10
        assert len(train) == 30

    def test_deterministic(self):
        """Same seed produces same split."""
        fovs = list(range(100))
        train1, test1 = split_fovs(fovs, num_test=10, num_train=30, seed=42)
        train2, test2 = split_fovs(fovs, num_test=10, num_train=30, seed=42)
        assert train1 == train2
        assert test1 == test2

    def test_different_seed(self):
        """Different seeds produce different splits."""
        fovs = list(range(100))
        train1, _ = split_fovs(fovs, num_test=10, num_train=30, seed=42)
        train2, _ = split_fovs(fovs, num_test=10, num_train=30, seed=99)
        assert train1 != train2

    def test_no_overlap(self):
        """Train and test sets do not overlap."""
        fovs = list(range(50))
        train, test = split_fovs(fovs, num_test=10, num_train=30)
        assert set(train).isdisjoint(set(test))
