"""Tests for OnlineEvalCallback metrics."""

import numpy as np

from viscy_utils.callbacks.online_eval import effective_rank, temporal_smoothness


class TestEffectiveRank:
    """Tests for effective_rank()."""

    def test_full_rank_identity(self):
        """Identity-like matrix should have high effective rank."""
        rng = np.random.default_rng(42)
        features = rng.standard_normal((100, 10))
        erank = effective_rank(features)
        assert erank > 8, f"Expected near-full rank for random matrix, got {erank}"

    def test_collapsed_features(self):
        """All-identical rows should have effective rank ~1."""
        features = np.ones((50, 10))
        features += np.random.default_rng(0).normal(0, 1e-8, features.shape)
        erank = effective_rank(features)
        assert erank < 2, f"Expected ~1 for collapsed features, got {erank}"

    def test_rank_deficient(self):
        """Matrix living in a 3D subspace of 10D should have erank ~3."""
        rng = np.random.default_rng(42)
        basis = rng.standard_normal((3, 10))
        coeffs = rng.standard_normal((100, 3))
        features = coeffs @ basis
        erank = effective_rank(features)
        assert 2 < erank < 5, f"Expected ~3 for rank-3 subspace, got {erank}"


class TestTemporalSmoothness:
    """Tests for temporal_smoothness()."""

    def test_perfectly_smooth(self):
        """Embeddings that linearly evolve with time should give high rho."""
        n_tracks = 5
        n_timepoints = 20
        rng = np.random.default_rng(42)

        features_list = []
        track_ids = []
        timepoints = []

        for tid in range(n_tracks):
            direction = rng.standard_normal(32)
            direction /= np.linalg.norm(direction)
            for t in range(n_timepoints):
                features_list.append(direction * t + rng.normal(0, 0.01, 32))
                track_ids.append(tid)
                timepoints.append(t)

        features = np.array(features_list)
        rho = temporal_smoothness(features, np.array(track_ids), np.array(timepoints))
        assert rho > 0.3, f"Expected positive correlation, got {rho}"

    def test_random_embeddings(self):
        """Random embeddings should have near-zero temporal smoothness."""
        rng = np.random.default_rng(42)
        n = 100
        features = rng.standard_normal((n, 32))
        track_ids = np.repeat(np.arange(5), 20)
        timepoints = np.tile(np.arange(20), 5)

        rho = temporal_smoothness(features, track_ids, timepoints)
        assert abs(rho) < 0.3, f"Expected near-zero for random embeddings, got {rho}"

    def test_single_sample_tracks_return_nan(self):
        """Tracks with only 1 sample each should return NaN."""
        features = np.random.default_rng(0).standard_normal((5, 16))
        track_ids = np.arange(5)
        timepoints = np.zeros(5)

        rho = temporal_smoothness(features, track_ids, timepoints)
        assert np.isnan(rho)

    def test_two_samples_insufficient_pairs(self):
        """Single track with 2 samples gives only 1 pair — need >=3."""
        features = np.random.default_rng(0).standard_normal((2, 16))
        track_ids = np.array([0, 0])
        timepoints = np.array([0, 1])

        rho = temporal_smoothness(features, track_ids, timepoints)
        assert np.isnan(rho)
