"""Utilities for DDP training."""

import math

import torch
from torch.utils.data.distributed import DistributedSampler


class ShardedDistributedSampler(DistributedSampler):
    def _sharded_randperm(self, generator):
        """Generate a sharded random permutation of indices."""
        indices = torch.tensor(range(len(self.dataset)))
        permuted = torch.stack(
            [
                torch.randperm(self.num_samples, generator=generator)
                + i * self.num_samples
                for i in range(self.num_replicas)
            ],
            dim=1,
        ).reshape(-1)
        return indices[permuted].tolist()

    def __iter__(self):
        """Modified __iter__ method to shard data across distributed ranks."""
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = self._sharded_randperm(g)
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

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
