"""Composable batch sampler with experiment-aware, condition-balanced,
temporal enrichment, and leaky mixing axes.

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
    temporal enrichment, and leaky experiment mixing axes.

    Each batch is constructed by a cascade:

    1. **Experiment selection** (``experiment_aware``): pick a single
       experiment to draw from, or draw from all experiments.
    2. **Leaky mixing** (``leaky``): optionally inject a fraction of
       cross-experiment samples into experiment-restricted batches.
    3. **Condition balancing** (``condition_balanced``): within the
       selected pool, balance condition representation.
    4. **Temporal enrichment** (``temporal_enrichment``): concentrate
       batch indices around a randomly chosen focal HPI, with a
       configurable global fraction drawn from all timepoints.

    Parameters
    ----------
    valid_anchors : pd.DataFrame
        DataFrame with at least ``"experiment"`` and ``"condition"``
        columns.  Must have a clean integer index (0..N-1).
        When ``temporal_enrichment=True``, must also have
        ``"hours_post_infection"`` column.
    batch_size : int
        Number of indices per batch.
    experiment_aware : bool
        If ``True``, every batch draws from a single experiment.
        Requires ``"experiment"`` column in *valid_anchors*.
    leaky : float
        Fraction of the batch drawn from *other* experiments when
        ``experiment_aware`` is ``True``.  Ignored otherwise.
    experiment_weights : dict[str, float] | None
        Per-experiment sampling weight.  Defaults to proportional to
        group size.
    condition_balanced : bool
        If ``True``, balance condition representation within each batch.
        Requires ``"condition"`` column in *valid_anchors*.
    condition_ratio : dict[str, float] | None
        Per-condition target ratio.  Defaults to equal across conditions.
    temporal_enrichment : bool
        If ``True``, concentrate batch indices around a randomly chosen
        focal hours-post-infection (HPI) value.
        Requires ``"hours_post_infection"`` column in *valid_anchors*.
    temporal_window_hours : float
        Half-width of the focal window around the chosen HPI.
        Indices with ``|hpi - focal| <= temporal_window_hours`` are
        considered focal.
    temporal_global_fraction : float
        Fraction of the batch drawn from all timepoints (global).
        The remaining ``1 - temporal_global_fraction`` fraction is drawn
        from the focal window.
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
        temporal_enrichment: bool = False,
        temporal_window_hours: float = 2.0,
        temporal_global_fraction: float = 0.3,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        # ------------------------------------------------------------------
        # Validate required columns for enabled features
        # ------------------------------------------------------------------
        if experiment_aware and "experiment" not in valid_anchors.columns:
            raise ValueError(
                "experiment_aware=True requires 'experiment' column in "
                "valid_anchors, but columns are: "
                f"{list(valid_anchors.columns)}"
            )
        if condition_balanced and "condition" not in valid_anchors.columns:
            raise ValueError(
                "condition_balanced=True requires 'condition' column in "
                "valid_anchors, but columns are: "
                f"{list(valid_anchors.columns)}"
            )
        if temporal_enrichment and "hours_post_infection" not in valid_anchors.columns:
            raise ValueError(
                "temporal_enrichment=True requires 'hours_post_infection' "
                "column in valid_anchors, but columns are: "
                f"{list(valid_anchors.columns)}"
            )

        self.valid_anchors = valid_anchors
        self.batch_size = batch_size
        self.experiment_aware = experiment_aware
        self.leaky = leaky
        self.experiment_weights = experiment_weights
        self.condition_balanced = condition_balanced
        self.condition_ratio = condition_ratio
        self.temporal_enrichment = temporal_enrichment
        self.temporal_window_hours = temporal_window_hours
        self.temporal_global_fraction = temporal_global_fraction
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Pre-compute HPI values for temporal enrichment
        if self.temporal_enrichment:
            self._hpi_values: np.ndarray = (
                valid_anchors["hours_post_infection"].to_numpy()
            )

        self._precompute_groups()

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _precompute_groups(self) -> None:
        """Build index lookup tables from valid_anchors columns."""
        # Per-experiment indices
        if self.experiment_aware:
            self._experiment_indices: dict[str, np.ndarray] = {
                str(name): group.index.to_numpy()
                for name, group in self.valid_anchors.groupby("experiment")
            }
            self._experiment_names: list[str] = list(
                self._experiment_indices.keys()
            )
        else:
            self._experiment_indices = {}
            self._experiment_names = []

        # Per-experiment-condition indices
        self._exp_cond_indices: dict[tuple[str, str], np.ndarray] = {}
        if self.experiment_aware and self.condition_balanced:
            for (exp, cond), group in self.valid_anchors.groupby(
                ["experiment", "condition"]
            ):
                self._exp_cond_indices[(str(exp), str(cond))] = (
                    group.index.to_numpy()
                )

        # Per-condition indices (global, for experiment_aware=False with
        # condition balancing)
        if self.condition_balanced:
            self._condition_indices: dict[str, np.ndarray] = {
                str(name): group.index.to_numpy()
                for name, group in self.valid_anchors.groupby("condition")
            }
            self._condition_names: list[str] = list(
                self._condition_indices.keys()
            )
        else:
            self._condition_indices = {}
            self._condition_names = []

        # All indices
        self._all_indices = np.arange(len(self.valid_anchors))

        # Compute experiment selection weights
        if self.experiment_aware:
            total = len(self.valid_anchors)
            if self.experiment_weights is not None:
                raw = np.array(
                    [
                        self.experiment_weights.get(n, 0.0)
                        for n in self._experiment_names
                    ]
                )
                self._exp_probs = raw / raw.sum()
            else:
                # Default: proportional to group size
                self._exp_probs = np.array(
                    [
                        len(self._experiment_indices[n]) / total
                        for n in self._experiment_names
                    ]
                )

            # Warn about small groups
            for name, indices in self._experiment_indices.items():
                if len(indices) < self.batch_size:
                    _logger.warning(
                        "Experiment '%s' has %d samples, fewer than "
                        "batch_size=%d. Will use replacement sampling "
                        "for this group.",
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
        """Construct a single batch by cascading sampling axes.

        Cascade order:
        1. Experiment selection (if experiment_aware)
        2. Leaky mixing (if leaky > 0)
        3. Temporal enrichment OR condition balancing OR plain sampling
        4. Combine primary + leak
        """
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

        # Step 3: Sample primary indices
        if self.temporal_enrichment:
            # Temporal enrichment draws focal + global from the pool directly
            primary = self._enrich_temporal(pool, n_primary, rng, chosen_exp)
        elif self.condition_balanced:
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

    # ------------------------------------------------------------------
    # Temporal enrichment
    # ------------------------------------------------------------------

    def _enrich_temporal(
        self,
        pool: np.ndarray,
        n_target: int,
        rng: np.random.Generator,
        chosen_exp: str | None,
    ) -> np.ndarray:
        """Sample *n_target* indices from *pool* with focal HPI concentration.

        Picks a random focal HPI from the unique HPIs available in *pool*.
        Then splits *pool* into focal (within window) and non-focal indices,
        and assembles the batch with the specified focal/global mix.

        Parameters
        ----------
        pool : np.ndarray
            Experiment-filtered (or global) index array to sample from.
        n_target : int
            Number of indices to produce.
        rng : np.random.Generator
            Shared RNG for deterministic sampling.
        chosen_exp : str | None
            If experiment-aware, the chosen experiment name.

        Returns
        -------
        np.ndarray
            Sampled indices of length *n_target*.
        """
        hpi = self._hpi_values

        # Pick focal HPI from unique values in the pool
        unique_hpi = np.unique(hpi[pool])
        focal_hpi = rng.choice(unique_hpi)

        # Split pool into focal and non-focal
        pool_hpi = hpi[pool]
        focal_mask = np.abs(pool_hpi - focal_hpi) <= self.temporal_window_hours
        focal_pool = pool[focal_mask]
        global_pool = pool[~focal_mask]

        # Compute how many global vs focal samples
        n_global = int(n_target * self.temporal_global_fraction)
        n_focal = n_target - n_global

        # Sample focal indices
        if n_focal > 0 and len(focal_pool) > 0:
            focal_replace = len(focal_pool) < n_focal
            focal_samples = rng.choice(
                focal_pool, size=n_focal, replace=focal_replace
            )
        elif n_focal > 0:
            # No focal indices available -- fall back to pool
            focal_replace = len(pool) < n_focal
            focal_samples = rng.choice(
                pool, size=n_focal, replace=focal_replace
            )
        else:
            focal_samples = np.array([], dtype=int)

        # Sample global indices (from non-focal to avoid duplicating focal)
        if n_global > 0 and len(global_pool) > 0:
            global_replace = len(global_pool) < n_global
            global_samples = rng.choice(
                global_pool, size=n_global, replace=global_replace
            )
        elif n_global > 0:
            # No non-focal indices -- draw from full pool
            global_replace = len(pool) < n_global
            global_samples = rng.choice(
                pool, size=n_global, replace=global_replace
            )
        else:
            global_samples = np.array([], dtype=int)

        return np.concatenate([focal_samples, global_samples])

    # ------------------------------------------------------------------
    # Condition balancing
    # ------------------------------------------------------------------

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
                    # Last condition gets the remainder to avoid rounding
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
