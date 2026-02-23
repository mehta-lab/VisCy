"""Composable batch sampler with experiment-aware, condition-balanced,
and leaky mixing axes.

Yields lists of integer indices into a ``valid_anchors`` DataFrame
produced by :class:`~dynaclr.index.MultiExperimentIndex`.
Implements the :class:`torch.utils.data.Sampler` ``[list[int]]`` protocol
for use as a ``batch_sampler`` in :class:`torch.utils.data.DataLoader`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator

import numpy as np
import pandas as pd
from torch.utils.data import Sampler

__all__ = ["FlexibleBatchSampler"]

_logger = logging.getLogger(__name__)


class FlexibleBatchSampler(Sampler[list[int]]):
    """Composable batch sampler with experiment-aware, condition-balanced,
    and leaky experiment mixing axes.

    Each batch is constructed by a cascade:

    1. **Experiment selection** (``experiment_aware``): pick a single
       experiment to draw from, or draw from all experiments.
    2. **Leaky mixing** (``leaky``): optionally inject a fraction of
       cross-experiment samples into experiment-restricted batches.
    3. **Condition balancing** (``condition_balanced``): within the
       selected pool, balance condition representation.

    Parameters
    ----------
    valid_anchors : pd.DataFrame
        DataFrame with at least ``"experiment"`` and ``"condition"``
        columns.  Must have a clean integer index (0..N-1).
    batch_size : int
        Number of indices per batch.
    experiment_aware : bool
        If ``True``, every batch draws from a single experiment.
    leaky : float
        Fraction of the batch drawn from *other* experiments when
        ``experiment_aware`` is ``True``.  Ignored otherwise.
    experiment_weights : dict[str, float] | None
        Per-experiment sampling weight.  Defaults to proportional to
        group size.
    condition_balanced : bool
        If ``True``, balance condition representation within each batch.
    condition_ratio : dict[str, float] | None
        Per-condition target ratio.  Defaults to equal across conditions.
    num_replicas : int
        Number of DDP processes (1 for single-process).
    rank : int
        Rank of the current process (0 for single-process).
    seed : int
        Base RNG seed for deterministic sampling.
    drop_last : bool
        If ``True``, drop the last incomplete batch.
    """

    def __init__(
        self,
        valid_anchors: pd.DataFrame,
        batch_size: int = 128,
        experiment_aware: bool = True,
        leaky: float = 0.0,
        experiment_weights: dict[str, float] | None = None,
        condition_balanced: bool = True,
        condition_ratio: dict[str, float] | None = None,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        self.valid_anchors = valid_anchors
        self.batch_size = batch_size
        self.experiment_aware = experiment_aware
        self.leaky = leaky
        self.experiment_weights = experiment_weights
        self.condition_balanced = condition_balanced
        self.condition_ratio = condition_ratio
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self._precompute_groups()

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _precompute_groups(self) -> None:
        """Build index lookup tables from valid_anchors columns."""
        # Per-experiment indices
        self._experiment_indices: dict[str, np.ndarray] = {
            str(name): group.index.to_numpy()
            for name, group in self.valid_anchors.groupby("experiment")
        }
        self._experiment_names: list[str] = list(self._experiment_indices.keys())

        # Per-experiment-condition indices
        self._exp_cond_indices: dict[tuple[str, str], np.ndarray] = {}
        for (exp, cond), group in self.valid_anchors.groupby(
            ["experiment", "condition"]
        ):
            self._exp_cond_indices[(str(exp), str(cond))] = group.index.to_numpy()

        # Per-condition indices (global, for experiment_aware=False with condition balancing)
        self._condition_indices: dict[str, np.ndarray] = {
            str(name): group.index.to_numpy()
            for name, group in self.valid_anchors.groupby("condition")
        }
        self._condition_names: list[str] = list(self._condition_indices.keys())

        # All indices
        self._all_indices = np.arange(len(self.valid_anchors))

        # Compute experiment selection weights
        total = len(self.valid_anchors)
        if self.experiment_weights is not None:
            raw = np.array(
                [self.experiment_weights.get(n, 0.0) for n in self._experiment_names]
            )
            self._exp_probs = raw / raw.sum()
        else:
            # Default: proportional to group size
            self._exp_probs = np.array(
                [len(self._experiment_indices[n]) / total for n in self._experiment_names]
            )

        # Warn about small groups
        for name, indices in self._experiment_indices.items():
            if len(indices) < self.batch_size:
                _logger.warning(
                    "Experiment '%s' has %d samples, fewer than batch_size=%d. "
                    "Will use replacement sampling for this group.",
                    name,
                    len(indices),
                    self.batch_size,
                )

    # ------------------------------------------------------------------
    # Epoch management
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling across DDP ranks."""
        self.epoch = epoch

    # ------------------------------------------------------------------
    # Length and iteration
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return number of batches this rank will yield."""
        total_batches = len(self.valid_anchors) // self.batch_size
        return math.ceil(total_batches / self.num_replicas)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batch-sized lists of integer indices."""
        rng = np.random.default_rng(self.seed + self.epoch)
        total_batches = len(self.valid_anchors) // self.batch_size
        all_batches = [self._build_one_batch(rng) for _ in range(total_batches)]
        # DDP: each rank takes its interleaved slice
        my_batches = all_batches[self.rank :: self.num_replicas]
        yield from my_batches

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------

    def _build_one_batch(self, rng: np.random.Generator) -> list[int]:
        """Construct a single batch by cascading sampling axes."""
        chosen_exp: str | None = None

        # Step 1: Experiment selection
        if self.experiment_aware:
            chosen_exp = rng.choice(self._experiment_names, p=self._exp_probs)
            pool = self._experiment_indices[chosen_exp]
        else:
            pool = self._all_indices

        # Step 2: Leaky mixing
        leak_samples: np.ndarray | None = None
        if self.experiment_aware and self.leaky > 0.0 and chosen_exp is not None:
            n_leak = int(self.batch_size * self.leaky)
            n_primary = self.batch_size - n_leak
            if n_leak > 0:
                other_indices = np.concatenate(
                    [
                        v
                        for k, v in self._experiment_indices.items()
                        if k != chosen_exp
                    ]
                )
                if len(other_indices) > 0:
                    leak_samples = rng.choice(
                        other_indices,
                        size=min(n_leak, len(other_indices)),
                        replace=len(other_indices) < n_leak,
                    )
        else:
            n_primary = self.batch_size

        # Step 3: Condition balancing or plain sampling
        if self.condition_balanced:
            primary = self._sample_condition_balanced(
                pool, n_primary, chosen_exp, rng
            )
        else:
            replace = len(pool) < n_primary
            primary = rng.choice(pool, size=n_primary, replace=replace)

        # Combine primary + leak
        if leak_samples is not None and len(leak_samples) > 0:
            combined = np.concatenate([primary, leak_samples])
        else:
            combined = primary

        return [int(x) for x in combined]

    def _sample_condition_balanced(
        self,
        pool: np.ndarray,
        n_samples: int,
        chosen_exp: str | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample indices with balanced conditions.

        If ``chosen_exp`` is not None, balances conditions within that
        experiment.  Otherwise, balances conditions globally.
        """
        if chosen_exp is not None:
            # Conditions available in this experiment
            conditions = [
                cond
                for (exp, cond) in self._exp_cond_indices
                if exp == chosen_exp
            ]
            if not conditions:
                replace = len(pool) < n_samples
                return rng.choice(pool, size=n_samples, replace=replace)

            # Determine per-condition ratios
            if self.condition_ratio is not None:
                ratios = {
                    c: self.condition_ratio.get(c, 1.0 / len(conditions))
                    for c in conditions
                }
            else:
                ratios = {c: 1.0 / len(conditions) for c in conditions}

            # Normalize ratios
            total_ratio = sum(ratios.values())
            ratios = {c: r / total_ratio for c, r in ratios.items()}

            indices_parts: list[np.ndarray] = []
            remaining = n_samples
            for i, cond in enumerate(conditions):
                cond_pool = self._exp_cond_indices.get(
                    (chosen_exp, cond), np.array([], dtype=int)
                )
                if len(cond_pool) == 0:
                    continue
                if i == len(conditions) - 1:
                    # Last condition gets the remainder to avoid rounding issues
                    n_cond = remaining
                else:
                    n_cond = int(n_samples * ratios[cond])
                    remaining -= n_cond

                replace = len(cond_pool) < n_cond
                chosen = rng.choice(cond_pool, size=n_cond, replace=replace)
                indices_parts.append(chosen)

            if indices_parts:
                return np.concatenate(indices_parts)
            replace = len(pool) < n_samples
            return rng.choice(pool, size=n_samples, replace=replace)

        else:
            # experiment_aware=False: balance conditions globally
            conditions = self._condition_names
            if not conditions:
                replace = len(pool) < n_samples
                return rng.choice(pool, size=n_samples, replace=replace)

            if self.condition_ratio is not None:
                ratios = {
                    c: self.condition_ratio.get(c, 1.0 / len(conditions))
                    for c in conditions
                }
            else:
                ratios = {c: 1.0 / len(conditions) for c in conditions}

            total_ratio = sum(ratios.values())
            ratios = {c: r / total_ratio for c, r in ratios.items()}

            indices_parts: list[np.ndarray] = []
            remaining = n_samples
            for i, cond in enumerate(conditions):
                cond_pool = self._condition_indices.get(
                    cond, np.array([], dtype=int)
                )
                if len(cond_pool) == 0:
                    continue
                if i == len(conditions) - 1:
                    n_cond = remaining
                else:
                    n_cond = int(n_samples * ratios[cond])
                    remaining -= n_cond

                replace = len(cond_pool) < n_cond
                chosen = rng.choice(cond_pool, size=n_cond, replace=replace)
                indices_parts.append(chosen)

            if indices_parts:
                return np.concatenate(indices_parts)
            replace = len(pool) < n_samples
            return rng.choice(pool, size=n_samples, replace=replace)
