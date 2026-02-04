import numpy as np
import pytest
from numpy.typing import NDArray

from viscy.representation.evaluation.clustering import pairwise_distance_matrix


@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    np.random.seed(42)
    return np.random.randn(50, 128).astype(np.float64)


@pytest.fixture
def small_features():
    """Create small sample with known values for numerical testing."""
    return np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])


class TestPairwiseDistanceMatrix:
    """Tests for pairwise_distance_matrix function."""

    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_scipy_baseline(self, sample_features: NDArray, metric: str):
        """Test that scipy backend produces valid distance matrices."""
        dist_matrix = pairwise_distance_matrix(
            sample_features, metric=metric, device="scipy"
        )

        # Check shape
        n = len(sample_features)
        assert dist_matrix.shape == (n, n)

        # Check symmetry
        assert np.allclose(dist_matrix, dist_matrix.T)

        # Check diagonal is zero (or near zero for numerical precision)
        assert np.allclose(np.diag(dist_matrix), 0, atol=1e-10)

        # Check all distances are non-negative
        assert np.all(dist_matrix >= 0)

    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    @pytest.mark.parametrize("device", ["cpu", "auto"])
    def test_torch_vs_scipy(self, sample_features: NDArray, metric: str, device: str):
        """Test that PyTorch implementation matches scipy results."""
        pytest.importorskip("torch")

        dist_scipy = pairwise_distance_matrix(
            sample_features, metric=metric, device="scipy"
        )
        dist_torch = pairwise_distance_matrix(
            sample_features, metric=metric, device=device
        )

        # Check numerical agreement
        assert np.allclose(dist_scipy, dist_torch, rtol=1e-5, atol=1e-6)

    @pytest.mark.skipif(
        not pytest.importorskip("torch").cuda.is_available(),
        reason="CUDA not available",
    )
    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_gpu_vs_scipy(self, sample_features: NDArray, metric: str):
        """Test that GPU implementation matches scipy results."""
        dist_scipy = pairwise_distance_matrix(
            sample_features, metric=metric, device="scipy"
        )
        dist_gpu = pairwise_distance_matrix(
            sample_features, metric=metric, device="cuda"
        )

        # Check numerical agreement
        assert np.allclose(dist_scipy, dist_gpu, rtol=1e-5, atol=1e-6)

    def test_cosine_distance_known_values(self, small_features: NDArray):
        """Test cosine distance with known values."""
        dist_matrix = pairwise_distance_matrix(
            small_features, metric="cosine", device="scipy"
        )

        # [1,0] and [0,1] are orthogonal: cosine distance = 1
        assert np.isclose(dist_matrix[0, 1], 1.0, atol=1e-10)

        # [1,1] and [0.5, 0.5] are parallel: cosine distance = 0
        assert np.isclose(dist_matrix[2, 3], 0.0, atol=1e-10)

        # [1,0] and [1,1]: cosine similarity = 1/sqrt(2), distance = 1 - 1/sqrt(2)
        expected = 1 - 1 / np.sqrt(2)
        assert np.isclose(dist_matrix[0, 2], expected, atol=1e-10)

    def test_euclidean_distance_known_values(self, small_features: NDArray):
        """Test euclidean distance with known values."""
        dist_matrix = pairwise_distance_matrix(
            small_features, metric="euclidean", device="scipy"
        )

        # Distance between [1,0] and [0,1] is sqrt(2)
        assert np.isclose(dist_matrix[0, 1], np.sqrt(2), atol=1e-10)

        # Distance between [1,1] and [0.5, 0.5] is sqrt(0.5)
        assert np.isclose(dist_matrix[2, 3], np.sqrt(0.5), atol=1e-10)

    def test_unsupported_metric_falls_back_to_scipy(self, sample_features: NDArray):
        """Test that unsupported metrics fall back to scipy."""
        # These metrics are only supported by scipy, not PyTorch
        dist_matrix = pairwise_distance_matrix(
            sample_features, metric="cityblock", device="auto"
        )

        # Should still produce valid results via scipy fallback
        n = len(sample_features)
        assert dist_matrix.shape == (n, n)
        assert np.allclose(dist_matrix, dist_matrix.T)

    def test_device_options(self, sample_features: NDArray):
        """Test various device options."""
        # Test scipy explicitly
        dist_scipy = pairwise_distance_matrix(
            sample_features, metric="cosine", device="scipy"
        )
        assert dist_scipy is not None

        # Test None as scipy
        dist_none = pairwise_distance_matrix(
            sample_features, metric="cosine", device=None
        )
        assert np.allclose(dist_scipy, dist_none)

    @pytest.mark.skipif(
        not pytest.importorskip("torch").cuda.is_available(),
        reason="CUDA not available",
    )
    def test_cuda_aliases(self, sample_features: NDArray):
        """Test that cuda and gpu device names work."""
        dist_cuda = pairwise_distance_matrix(
            sample_features, metric="cosine", device="cuda"
        )
        dist_gpu = pairwise_distance_matrix(
            sample_features, metric="cosine", device="gpu"
        )

        assert np.allclose(dist_cuda, dist_gpu)

    def test_invalid_device_raises_error(self, sample_features: NDArray):
        """Test that invalid device names raise appropriate errors."""
        pytest.importorskip("torch")

        with pytest.raises(ValueError, match="Invalid device"):
            pairwise_distance_matrix(
                sample_features, metric="cosine", device="invalid_device"
            )

    def test_float32_input_preserves_precision(self):
        """Test that float32 input is converted to float64 for precision."""
        pytest.importorskip("torch")

        features_f32 = np.random.randn(10, 32).astype(np.float32)

        dist_scipy = pairwise_distance_matrix(
            features_f32, metric="cosine", device="scipy"
        )
        dist_torch = pairwise_distance_matrix(
            features_f32, metric="cosine", device="cpu"
        )

        # Should still have good agreement despite float32 input
        assert np.allclose(dist_scipy, dist_torch, rtol=1e-5, atol=1e-6)

    def test_large_matrix_shape(self):
        """Test with larger matrix to ensure it works at scale."""
        large_features = np.random.randn(500, 64).astype(np.float64)

        dist_matrix = pairwise_distance_matrix(
            large_features, metric="cosine", device="auto"
        )

        assert dist_matrix.shape == (500, 500)
        assert np.allclose(dist_matrix, dist_matrix.T)
        assert np.allclose(np.diag(dist_matrix), 0, atol=1e-6)
