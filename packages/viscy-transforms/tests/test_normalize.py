"""Tests for normalization transforms."""

import pytest
import torch

from viscy_transforms import MinMaxSampled, NormalizeSampled


@pytest.fixture
def sample_with_norm_meta():
    """Sample dict with norm_meta for testing."""
    return {
        "Phase3D": torch.rand(1, 1, 8, 64, 64) * 100,
        "Structure": torch.rand(1, 1, 8, 64, 64) * 500 + 400,
        "norm_meta": {
            "Phase3D": {
                "fov_statistics": {
                    "mean": torch.tensor(50.0),
                    "std": torch.tensor(10.0),
                    "median": torch.tensor(49.0),
                    "iqr": torch.tensor(12.0),
                    "p1": torch.tensor(5.0),
                    "p99": torch.tensor(95.0),
                    "p5": torch.tensor(15.0),
                    "p95": torch.tensor(85.0),
                    "min": torch.tensor(0.0),
                    "max": torch.tensor(100.0),
                },
            },
            "Structure": {
                "fov_statistics": {
                    "mean": torch.tensor(540.0),
                    "std": torch.tensor(90.0),
                    "median": torch.tensor(524.0),
                    "iqr": torch.tensor(106.0),
                    "p1": torch.tensor(399.0),
                    "p99": torch.tensor(837.0),
                    "p5": torch.tensor(424.0),
                    "p95": torch.tensor(716.0),
                    "min": torch.tensor(356.0),
                    "max": torch.tensor(2103.0),
                },
            },
        },
    }


class TestNormalizeSampled:
    def test_mean_std(self, sample_with_norm_meta):
        transform = NormalizeSampled(keys=["Phase3D"], level="fov_statistics")
        result = transform(sample_with_norm_meta)
        # (x - mean) / std
        assert result["Phase3D"].shape == sample_with_norm_meta["Phase3D"].shape
        assert "norm_meta" in result

    def test_remove_meta(self, sample_with_norm_meta):
        transform = NormalizeSampled(keys=["Phase3D"], level="fov_statistics", remove_meta=True)
        result = transform(sample_with_norm_meta)
        assert "norm_meta" not in result


class TestMinMaxSampled:
    def test_p1_p99_output_range(self, sample_with_norm_meta):
        transform = MinMaxSampled(keys=["Phase3D"], level="fov_statistics", data_range="p1_p99")
        result = transform(sample_with_norm_meta)
        # After clipping to [p1, p99] and rescaling, output is in [-1, 1]
        assert result["Phase3D"].min() >= -1.0 - 1e-6
        assert result["Phase3D"].max() <= 1.0 + 1e-6
        assert result["Phase3D"].shape == sample_with_norm_meta["Phase3D"].shape

    def test_min_max_range(self, sample_with_norm_meta):
        transform = MinMaxSampled(keys=["Phase3D"], level="fov_statistics", data_range="min_max")
        result = transform(sample_with_norm_meta)
        assert result["Phase3D"].min() >= -1.0 - 1e-6
        assert result["Phase3D"].max() <= 1.0 + 1e-6

    def test_p5_p95_range(self, sample_with_norm_meta):
        transform = MinMaxSampled(keys=["Phase3D"], level="fov_statistics", data_range="p5_p95")
        result = transform(sample_with_norm_meta)
        assert result["Phase3D"].min() >= -1.0 - 1e-6
        assert result["Phase3D"].max() <= 1.0 + 1e-6

    def test_clipping(self, sample_with_norm_meta):
        # Set data well outside the p1-p99 range to verify clipping
        sample_with_norm_meta["Phase3D"] = torch.full((1, 1, 8, 64, 64), 200.0)
        transform = MinMaxSampled(keys=["Phase3D"], level="fov_statistics", data_range="p1_p99")
        result = transform(sample_with_norm_meta)
        # All values above p99=95 get clipped, then rescaled to 1.0
        assert torch.allclose(result["Phase3D"], torch.ones_like(result["Phase3D"]))

    def test_multiple_keys(self, sample_with_norm_meta):
        transform = MinMaxSampled(keys=["Phase3D", "Structure"], level="fov_statistics", data_range="p1_p99")
        result = transform(sample_with_norm_meta)
        assert result["Phase3D"].min() >= -1.0 - 1e-6
        assert result["Structure"].min() >= -1.0 - 1e-6

    def test_remove_meta(self, sample_with_norm_meta):
        transform = MinMaxSampled(keys=["Phase3D"], level="fov_statistics", remove_meta=True)
        result = transform(sample_with_norm_meta)
        assert "norm_meta" not in result

    def test_invalid_data_range(self, sample_with_norm_meta):
        transform = MinMaxSampled(keys=["Phase3D"], level="fov_statistics", data_range="invalid")
        with pytest.raises(ValueError, match="Invalid data_range"):
            transform(sample_with_norm_meta)

    def test_preserves_shape(self, sample_with_norm_meta):
        original_shape = sample_with_norm_meta["Phase3D"].shape
        transform = MinMaxSampled(keys=["Phase3D"], level="fov_statistics", data_range="p1_p99")
        result = transform(sample_with_norm_meta)
        assert result["Phase3D"].shape == original_shape
