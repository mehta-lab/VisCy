"""TDD test suite for FlexibleBatchSampler core sampling axes.

Tests cover:
- Experiment-aware batching (SAMP-01)
- Condition balancing (SAMP-02)
- Leaky experiment mixing (SAMP-05)
- Deterministic reproducibility
- Replacement sampling fallback for small groups
- DDP rank partitioning
- set_epoch behavior
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from viscy_data.sampler import FlexibleBatchSampler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_experiment_anchors() -> pd.DataFrame:
    """DataFrame with 2 experiments, 2 conditions each, 200 rows total."""
    rng = np.random.default_rng(42)
    rows = []
    for exp_name in ["exp_A", "exp_B"]:
        for cond in ["infected", "uninfected"]:
            for i in range(50):
                rows.append(
                    {
                        "experiment": exp_name,
                        "condition": cond,
                        "hours_post_infection": rng.uniform(0, 48),
                        "global_track_id": f"{exp_name}_{cond}_{i}",
                        "t": rng.integers(0, 20),
                    }
                )
    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)


@pytest.fixture()
def three_experiment_anchors() -> pd.DataFrame:
    """DataFrame with 3 experiments, 2 conditions, 300 rows total."""
    rng = np.random.default_rng(99)
    rows = []
    for exp_name in ["exp_X", "exp_Y", "exp_Z"]:
        for cond in ["ctrl", "treated"]:
            for i in range(50):
                rows.append(
                    {
                        "experiment": exp_name,
                        "condition": cond,
                        "hours_post_infection": rng.uniform(0, 24),
                        "global_track_id": f"{exp_name}_{cond}_{i}",
                        "t": rng.integers(0, 10),
                    }
                )
    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)


@pytest.fixture()
def three_condition_anchors() -> pd.DataFrame:
    """DataFrame with 1 experiment, 3 conditions, 150 rows total."""
    rows = []
    for cond in ["ctrl", "low_moi", "high_moi"]:
        for i in range(50):
            rows.append(
                {
                    "experiment": "exp_single",
                    "condition": cond,
                    "hours_post_infection": float(i),
                    "global_track_id": f"exp_single_{cond}_{i}",
                    "t": i,
                }
            )
    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)


@pytest.fixture()
def small_group_anchors() -> pd.DataFrame:
    """DataFrame where one group has fewer samples than batch_size."""
    rows = []
    # Tiny experiment with only 5 rows
    for i in range(5):
        rows.append(
            {
                "experiment": "tiny_exp",
                "condition": "ctrl",
                "hours_post_infection": float(i),
                "global_track_id": f"tiny_{i}",
                "t": i,
            }
        )
    # Larger experiment with 100 rows
    for i in range(100):
        rows.append(
            {
                "experiment": "big_exp",
                "condition": "ctrl",
                "hours_post_infection": float(i),
                "global_track_id": f"big_{i}",
                "t": i,
            }
        )
    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Experiment-aware batching (SAMP-01)
# ---------------------------------------------------------------------------


class TestExperimentAware:
    """experiment_aware=True restricts every batch to one experiment."""

    def test_batch_indices_from_single_experiment(
        self, two_experiment_anchors: pd.DataFrame
    ):
        """Every batch should contain indices from exactly one experiment."""
        sampler = FlexibleBatchSampler(
            valid_anchors=two_experiment_anchors,
            batch_size=8,
            experiment_aware=True,
            condition_balanced=False,
            leaky=0.0,
            seed=42,
        )
        batches = list(sampler)
        assert len(batches) > 0, "Sampler should yield batches"
        for batch in batches:
            experiments = two_experiment_anchors.iloc[batch]["experiment"].unique()
            assert len(experiments) == 1, (
                f"Experiment-aware batch has indices from {len(experiments)} experiments"
            )

    def test_all_experiments_appear(
        self, three_experiment_anchors: pd.DataFrame
    ):
        """Over many batches, all experiments should appear at least once."""
        sampler = FlexibleBatchSampler(
            valid_anchors=three_experiment_anchors,
            batch_size=8,
            experiment_aware=True,
            condition_balanced=False,
            leaky=0.0,
            seed=42,
        )
        batches = list(sampler)
        seen_experiments: set[str] = set()
        for batch in batches:
            exps = three_experiment_anchors.iloc[batch]["experiment"].unique()
            seen_experiments.update(exps)
        expected = {"exp_X", "exp_Y", "exp_Z"}
        assert seen_experiments == expected, (
            f"Not all experiments seen: {seen_experiments} vs {expected}"
        )

    def test_experiment_aware_false_allows_mixing(
        self, two_experiment_anchors: pd.DataFrame
    ):
        """experiment_aware=False should allow multiple experiments per batch."""
        sampler = FlexibleBatchSampler(
            valid_anchors=two_experiment_anchors,
            batch_size=8,
            experiment_aware=False,
            condition_balanced=False,
            leaky=0.0,
            seed=42,
        )
        batches = list(sampler)
        any_mixed = False
        for batch in batches:
            experiments = two_experiment_anchors.iloc[batch]["experiment"].unique()
            if len(experiments) > 1:
                any_mixed = True
                break
        assert any_mixed, (
            "With experiment_aware=False, at least one batch should mix experiments"
        )


# ---------------------------------------------------------------------------
# Condition balancing (SAMP-02)
# ---------------------------------------------------------------------------


class TestConditionBalanced:
    """condition_balanced=True produces ~equal condition representation."""

    def test_two_conditions_balanced(
        self, two_experiment_anchors: pd.DataFrame
    ):
        """Each batch should have ~50% of each condition (within tolerance)."""
        sampler = FlexibleBatchSampler(
            valid_anchors=two_experiment_anchors,
            batch_size=16,
            experiment_aware=True,
            condition_balanced=True,
            leaky=0.0,
            seed=42,
        )
        batches = list(sampler)
        assert len(batches) > 0
        for batch in batches:
            conditions = two_experiment_anchors.iloc[batch]["condition"]
            counts = conditions.value_counts()
            for cond_count in counts.to_numpy():
                fraction = cond_count / len(batch)
                # Within +/-20% of 50% = between 30% and 70%
                assert 0.3 <= fraction <= 0.7, (
                    f"Condition fraction {fraction:.2f} outside tolerance for "
                    f"2-condition balance (expected ~0.5)"
                )

    def test_three_conditions_balanced(
        self, three_condition_anchors: pd.DataFrame
    ):
        """Each batch should have ~33% of each condition."""
        sampler = FlexibleBatchSampler(
            valid_anchors=three_condition_anchors,
            batch_size=18,
            experiment_aware=True,
            condition_balanced=True,
            leaky=0.0,
            seed=42,
        )
        batches = list(sampler)
        assert len(batches) > 0
        for batch in batches:
            conditions = three_condition_anchors.iloc[batch]["condition"]
            counts = conditions.value_counts()
            for cond, cnt in counts.items():
                fraction = cnt / len(batch)
                # Within +/-20% of 33% = between 13% and 53%
                assert 0.13 <= fraction <= 0.54, (
                    f"Condition '{cond}' fraction {fraction:.2f} outside tolerance "
                    f"for 3-condition balance (expected ~0.33)"
                )

    def test_condition_balanced_false_no_constraint(
        self, two_experiment_anchors: pd.DataFrame
    ):
        """condition_balanced=False should not enforce any condition ratio."""
        sampler = FlexibleBatchSampler(
            valid_anchors=two_experiment_anchors,
            batch_size=8,
            experiment_aware=True,
            condition_balanced=False,
            leaky=0.0,
            seed=42,
        )
        # Just verify it runs without error and yields batches
        batches = list(sampler)
        assert len(batches) > 0


# ---------------------------------------------------------------------------
# Leaky experiment mixing (SAMP-05)
# ---------------------------------------------------------------------------


class TestLeakyMixing:
    """leaky > 0.0 injects cross-experiment samples into experiment-aware batches."""

    def test_leaky_zero_no_cross_experiment(
        self, two_experiment_anchors: pd.DataFrame
    ):
        """leaky=0.0 with experiment_aware should have 0 cross-experiment indices."""
        sampler = FlexibleBatchSampler(
            valid_anchors=two_experiment_anchors,
            batch_size=10,
            experiment_aware=True,
            condition_balanced=False,
            leaky=0.0,
            seed=42,
        )
        for batch in sampler:
            experiments = two_experiment_anchors.iloc[batch]["experiment"].unique()
            assert len(experiments) == 1

    def test_leaky_injects_cross_experiment(
        self, two_experiment_anchors: pd.DataFrame
    ):
        """leaky=0.2, batch_size=10 -> ~2 cross-experiment indices per batch."""
        sampler = FlexibleBatchSampler(
            valid_anchors=two_experiment_anchors,
            batch_size=10,
            experiment_aware=True,
            condition_balanced=False,
            leaky=0.2,
            seed=42,
        )
        batches = list(sampler)
        assert len(batches) > 0
        any_leaked = False
        for batch in batches:
            experiments = two_experiment_anchors.iloc[batch]["experiment"]
            if len(experiments.unique()) > 1:
                any_leaked = True
                # Check approximate count: expect ~2 from other experiment
                counts = experiments.value_counts()
                minority_count = counts.min()
                # Should be approximately int(10 * 0.2) = 2
                assert minority_count <= 4, (
                    f"Too many leaked samples: {minority_count} (expected ~2)"
                )
        assert any_leaked, "leaky=0.2 should inject cross-experiment samples"

    def test_leaky_ignored_when_not_experiment_aware(
        self, two_experiment_anchors: pd.DataFrame
    ):
        """leaky has no effect when experiment_aware=False."""
        sampler = FlexibleBatchSampler(
            valid_anchors=two_experiment_anchors,
            batch_size=10,
            experiment_aware=False,
            condition_balanced=False,
            leaky=0.5,
            seed=42,
        )
        # Should run without error and yield batches
        batches = list(sampler)
        assert len(batches) > 0


# ---------------------------------------------------------------------------
# Small group fallback
# ---------------------------------------------------------------------------


class TestSmallGroupFallback:
    """Small groups fall back to replacement sampling with a logged warning."""

    def test_small_group_does_not_crash(
        self, small_group_anchors: pd.DataFrame
    ):
        """batch_size > smallest group should not raise."""
        sampler = FlexibleBatchSampler(
            valid_anchors=small_group_anchors,
            batch_size=32,
            experiment_aware=True,
            condition_balanced=False,
            leaky=0.0,
            seed=42,
        )
        batches = list(sampler)
        assert len(batches) > 0

    def test_small_group_emits_warning(
        self, small_group_anchors: pd.DataFrame, caplog
    ):
        """Sampler should warn when a group < batch_size."""
        import logging

        with caplog.at_level(logging.WARNING):
            FlexibleBatchSampler(
                valid_anchors=small_group_anchors,
                batch_size=32,
                experiment_aware=True,
                condition_balanced=False,
                leaky=0.0,
                seed=42,
            )
        assert any(
            "replacement" in record.message.lower()
            or "small" in record.message.lower()
            or "fewer" in record.message.lower()
            for record in caplog.records
        ), f"Expected warning about small group, got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Determinism and set_epoch
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Deterministic: same seed + same epoch -> same batch sequence."""

    def test_same_seed_same_result(
        self, two_experiment_anchors: pd.DataFrame
    ):
        """Two samplers with same config should produce identical batches."""
        kwargs = dict(
            valid_anchors=two_experiment_anchors,
            batch_size=8,
            experiment_aware=True,
            condition_balanced=True,
            leaky=0.0,
            seed=123,
        )
        sampler1 = FlexibleBatchSampler(**kwargs)
        sampler2 = FlexibleBatchSampler(**kwargs)
        batches1 = list(sampler1)
        batches2 = list(sampler2)
        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2):
            assert b1 == b2, "Same seed should produce identical batches"

    def test_set_epoch_changes_sequence(
        self, two_experiment_anchors: pd.DataFrame
    ):
        """set_epoch(n) should change the batch sequence."""
        sampler = FlexibleBatchSampler(
            valid_anchors=two_experiment_anchors,
            batch_size=8,
            experiment_aware=True,
            condition_balanced=False,
            leaky=0.0,
            seed=42,
        )
        sampler.set_epoch(0)
        batches_epoch0 = list(sampler)
        sampler.set_epoch(1)
        batches_epoch1 = list(sampler)
        # At least one batch should differ
        assert batches_epoch0 != batches_epoch1, (
            "Different epochs should produce different batch sequences"
        )

    def test_set_epoch_same_epoch_same_result(
        self, two_experiment_anchors: pd.DataFrame
    ):
        """Calling set_epoch(5) twice should produce the same sequence."""
        sampler = FlexibleBatchSampler(
            valid_anchors=two_experiment_anchors,
            batch_size=8,
            experiment_aware=True,
            condition_balanced=False,
            leaky=0.0,
            seed=42,
        )
        sampler.set_epoch(5)
        batches_a = list(sampler)
        sampler.set_epoch(5)
        batches_b = list(sampler)
        assert batches_a == batches_b


# ---------------------------------------------------------------------------
# __len__ and __iter__ protocol
# ---------------------------------------------------------------------------


class TestSamplerProtocol:
    """Verify Sampler[list[int]] protocol."""

    def test_yields_list_of_int(
        self, two_experiment_anchors: pd.DataFrame
    ):
        """__iter__ should yield list[int], not individual ints."""
        sampler = FlexibleBatchSampler(
            valid_anchors=two_experiment_anchors,
            batch_size=8,
            experiment_aware=True,
            condition_balanced=False,
            leaky=0.0,
            seed=42,
        )
        for batch in sampler:
            assert isinstance(batch, list), f"Expected list, got {type(batch)}"
            assert len(batch) == 8, f"Batch size should be 8, got {len(batch)}"
            for idx in batch:
                assert isinstance(idx, (int, np.integer)), (
                    f"Expected int, got {type(idx)}"
                )

    def test_len_returns_expected_value(
        self, two_experiment_anchors: pd.DataFrame
    ):
        """__len__ should return total_batches // num_replicas."""
        sampler = FlexibleBatchSampler(
            valid_anchors=two_experiment_anchors,
            batch_size=8,
            experiment_aware=True,
            condition_balanced=False,
            leaky=0.0,
            seed=42,
            num_replicas=1,
        )
        expected = len(two_experiment_anchors) // 8  # 200 // 8 = 25
        assert len(sampler) == expected, (
            f"Expected __len__={expected}, got {len(sampler)}"
        )

    def test_len_with_replicas(
        self, two_experiment_anchors: pd.DataFrame
    ):
        """__len__ with num_replicas=2 should halve the count."""
        import math

        sampler = FlexibleBatchSampler(
            valid_anchors=two_experiment_anchors,
            batch_size=8,
            experiment_aware=True,
            condition_balanced=False,
            leaky=0.0,
            seed=42,
            num_replicas=2,
            rank=0,
        )
        total_batches = len(two_experiment_anchors) // 8  # 25
        expected = math.ceil(total_batches / 2)  # 13
        assert len(sampler) == expected


# ---------------------------------------------------------------------------
# DDP rank partitioning
# ---------------------------------------------------------------------------


class TestDDPPartitioning:
    """DDP: ranks get disjoint interleaved batch slices."""

    def test_two_ranks_disjoint_batches(
        self, two_experiment_anchors: pd.DataFrame
    ):
        """Rank 0 and rank 1 should get different (interleaved) batches."""
        common = dict(
            valid_anchors=two_experiment_anchors,
            batch_size=8,
            experiment_aware=True,
            condition_balanced=False,
            leaky=0.0,
            seed=42,
            num_replicas=2,
        )
        sampler_r0 = FlexibleBatchSampler(**common, rank=0)
        sampler_r1 = FlexibleBatchSampler(**common, rank=1)
        sampler_r0.set_epoch(0)
        sampler_r1.set_epoch(0)

        batches_r0 = list(sampler_r0)
        batches_r1 = list(sampler_r1)

        # Combined should cover all total batches
        total = len(two_experiment_anchors) // 8
        assert len(batches_r0) + len(batches_r1) >= total - 1

        # Batches should not be identical (different ranks get different slices)
        assert batches_r0 != batches_r1

    def test_ddp_same_seed_deterministic(
        self, two_experiment_anchors: pd.DataFrame
    ):
        """Both ranks with same seed+epoch should yield deterministic batches."""
        common = dict(
            valid_anchors=two_experiment_anchors,
            batch_size=8,
            experiment_aware=True,
            condition_balanced=False,
            leaky=0.0,
            seed=42,
            num_replicas=2,
        )
        s0a = FlexibleBatchSampler(**common, rank=0)
        s0b = FlexibleBatchSampler(**common, rank=0)
        s0a.set_epoch(3)
        s0b.set_epoch(3)
        assert list(s0a) == list(s0b)
