"""Tests for OnlineEvalCallback metrics."""

from types import SimpleNamespace

import numpy as np
import torch

from viscy_utils.callbacks.online_eval import (
    OnlineEvalCallback,
    effective_rank,
    temporal_smoothness,
)


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


class TestGatherAcrossRanks:
    """_gather_across_ranks must produce full-set arrays under DDP.

    The world_size=1 path is exercised by the integration test; this
    class covers the multi-rank gather + missing-key passthrough that
    are otherwise unreachable without a real distributed backend.
    """

    def test_world_size_two_concatenates_features_and_handles_missing(self):
        """world_size=2 with identical per-rank inputs stacks features and labels;
        a globally-missing optional array returns None instead of stalling."""

        class _FakeModule:
            def __init__(self, world_size: int):
                self.trainer = SimpleNamespace(world_size=world_size)
                self.device = torch.device("cpu")
                self._w = world_size

            def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
                return torch.stack([tensor] * self._w, dim=0)

        callback = OnlineEvalCallback()
        module = _FakeModule(world_size=2)
        features_local = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        labels_local = np.array([7, 8, 9])

        # labels present, track_ids/timepoints missing (None on every rank).
        features_np, labels, track_ids, timepoints = callback._gather_across_ranks(
            module, features_local, labels_local, None, None
        )

        assert features_np.shape == (6, 4)
        np.testing.assert_array_equal(features_np[:3], features_local.numpy())
        np.testing.assert_array_equal(features_np[3:], features_local.numpy())
        np.testing.assert_array_equal(labels, np.array([7, 8, 9, 7, 8, 9]))
        assert track_ids is None
        assert timepoints is None
