"""Tests for ShardedDistributedSampler."""

import torch
from torch.utils.data import TensorDataset

from viscy_data import ShardedDistributedSampler


def _make_dataset(size: int) -> TensorDataset:
    return TensorDataset(torch.arange(size))


def test_sampler_length():
    """Sampler length equals num_samples (dataset size / num_replicas)."""
    dataset = _make_dataset(100)
    sampler = ShardedDistributedSampler(dataset, num_replicas=4, rank=0, shuffle=False)
    assert sampler.num_samples == 25
    indices = list(sampler)
    assert len(indices) == 25


def test_sampler_no_shuffle_deterministic():
    """Without shuffle, indices are sequential and deterministic."""
    dataset = _make_dataset(8)
    sampler = ShardedDistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False)
    indices_r0 = list(sampler)

    sampler_r1 = ShardedDistributedSampler(dataset, num_replicas=2, rank=1, shuffle=False)
    indices_r1 = list(sampler_r1)

    # All indices should be covered
    assert sorted(indices_r0 + indices_r1) == list(range(8))
    # No overlap
    assert set(indices_r0).isdisjoint(set(indices_r1))


def test_sampler_shuffle_deterministic_across_epochs():
    """Shuffled sampler is deterministic given the same epoch/seed."""
    dataset = _make_dataset(20)
    sampler = ShardedDistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True, seed=42)
    sampler.set_epoch(0)
    indices_epoch0 = list(sampler)

    sampler.set_epoch(0)
    indices_epoch0_again = list(sampler)
    assert indices_epoch0 == indices_epoch0_again

    sampler.set_epoch(1)
    indices_epoch1 = list(sampler)
    assert indices_epoch0 != indices_epoch1


def test_sampler_all_indices_in_range():
    """All sampled indices are valid dataset indices."""
    dataset = _make_dataset(50)
    sampler = ShardedDistributedSampler(dataset, num_replicas=3, rank=1, shuffle=True, seed=0)
    for idx in sampler:
        assert 0 <= idx < len(dataset)


def test_sampler_drop_last():
    """With drop_last=True, no padding is added."""
    dataset = _make_dataset(10)
    sampler = ShardedDistributedSampler(dataset, num_replicas=3, rank=0, shuffle=False, drop_last=True)
    indices = list(sampler)
    # 10 // 3 = 3 per rank (drops remainder)
    assert len(indices) == 3
