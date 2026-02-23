"""TDD tests for variable tau sampling with exponential decay."""

import numpy as np
import pytest

from dynaclr.tau_sampling import sample_tau


class TestSampleTauRange:
    """Test that sampled tau values stay within bounds."""

    def test_sample_tau_within_range(self):
        """All samples must be in [tau_min, tau_max]."""
        rng = np.random.default_rng(42)
        tau_min, tau_max = 1, 10
        for _ in range(1000):
            tau = sample_tau(tau_min, tau_max, rng)
            assert tau_min <= tau <= tau_max, f"tau={tau} outside [{tau_min}, {tau_max}]"


class TestSampleTauDistribution:
    """Test exponential decay distribution properties."""

    def test_sample_tau_exponential_favors_small(self):
        """decay_rate=2.0, N=10000: median should be less than midpoint (5.5)."""
        rng = np.random.default_rng(42)
        tau_min, tau_max = 1, 10
        midpoint = (tau_min + tau_max) / 2
        samples = [sample_tau(tau_min, tau_max, rng, decay_rate=2.0) for _ in range(10000)]
        median = np.median(samples)
        assert median < midpoint, f"Median {median} should be < midpoint {midpoint}"

    def test_sample_tau_uniform_when_zero_decay(self):
        """decay_rate=0.0, N=10000: mean should be approximately midpoint (tolerance 0.5)."""
        rng = np.random.default_rng(42)
        tau_min, tau_max = 1, 10
        midpoint = (tau_min + tau_max) / 2
        samples = [sample_tau(tau_min, tau_max, rng, decay_rate=0.0) for _ in range(10000)]
        mean = np.mean(samples)
        assert abs(mean - midpoint) < 0.5, f"Mean {mean:.2f} should be ~{midpoint} (tolerance 0.5)"

    def test_sample_tau_strong_decay(self):
        """decay_rate=5.0: >50% of 10000 samples should be tau_min or tau_min+1."""
        rng = np.random.default_rng(42)
        tau_min, tau_max = 1, 10
        samples = [sample_tau(tau_min, tau_max, rng, decay_rate=5.0) for _ in range(10000)]
        near_min = sum(1 for s in samples if s <= tau_min + 1)
        fraction = near_min / len(samples)
        assert fraction > 0.50, f"Only {fraction:.2%} near tau_min, expected >50%"


class TestSampleTauEdgeCases:
    """Test edge cases and special values."""

    def test_sample_tau_single_value(self):
        """tau_min == tau_max: always returns that value."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            tau = sample_tau(5, 5, rng)
            assert tau == 5, f"Expected 5, got {tau}"


class TestSampleTauDeterminism:
    """Test reproducibility with same seed."""

    def test_sample_tau_deterministic(self):
        """Same seed produces same sequence of tau values."""
        seq1 = []
        rng1 = np.random.default_rng(123)
        for _ in range(50):
            seq1.append(sample_tau(1, 10, rng1))

        seq2 = []
        rng2 = np.random.default_rng(123)
        for _ in range(50):
            seq2.append(sample_tau(1, 10, rng2))

        assert seq1 == seq2, "Same seed should produce same sequence"


class TestSampleTauReturnType:
    """Test return type is Python int."""

    def test_sample_tau_returns_int(self):
        """Return type must be int (not numpy int64)."""
        rng = np.random.default_rng(42)
        tau = sample_tau(1, 10, rng)
        assert type(tau) is int, f"Expected int, got {type(tau).__name__}"
