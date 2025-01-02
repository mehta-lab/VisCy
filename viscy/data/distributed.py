"""Utilities for DDP training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.distributed
from torch.utils.data.distributed import DistributedSampler

if TYPE_CHECKING:
    from torch import Generator


class ShardedDistributedSampler(DistributedSampler):
    def _sharded_randperm(self, max_size: int, generator: Generator) -> list[int]:
        """Generate a sharded random permutation of indices.
        Overlap may occur in between the last two shards to maintain divisibility."""
        sharded_randperm = [
            torch.randperm(self.num_samples, generator=generator)
            + min(i * self.num_samples, max_size - self.num_samples)
            for i in range(self.num_replicas)
        ]
        indices = torch.stack(sharded_randperm, dim=1).reshape(-1)
        return indices.tolist()

    def __iter__(self):
        """Modified __iter__ method to shard data across distributed ranks."""
        max_size = len(self.dataset)  # type: ignore[arg-type]
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = self._sharded_randperm(max_size, g)
        else:
            indices = list(range(max_size))
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
