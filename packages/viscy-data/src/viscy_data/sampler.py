"""Composable batch sampler with batch grouping, stratification, and temporal enrichment.

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
    """Composable batch sampler with batch grouping and stratification.

    Each batch is constructed by a cascade:

    1. **Group selection** (``batch_group_by``): pick a single group
       to draw from, or draw from all samples.
    2. **Leaky mixing** (``leaky``): optionally inject a fraction of
       cross-group samples into group-restricted batches.
    3. **Stratified sampling** (``stratify_by``): within the selected
       pool, balance representation across groups defined by one or
       more DataFrame columns.
    4. **Temporal enrichment** (``temporal_enrichment``): concentrate
       batch indices around a randomly chosen focal HPI, with a
       configurable global fraction drawn from all timepoints.

    Parameters
    ----------
    valid_anchors : pd.DataFrame
        DataFrame with a clean integer index (0..N-1).
        Must contain any columns referenced by ``batch_group_by``,
        ``stratify_by``, or ``temporal_enrichment``.
    batch_size : int
        Number of indices per batch.
    batch_group_by : str | list[str] | None
        Column(s) in *valid_anchors* that define batch-level groups.
        Each batch draws from a single group.
        ``"experiment"`` — one experiment per batch (old ``experiment_aware=True``).
        ``"marker"`` — one marker per batch.
        ``["marker", "source_channel"]`` — one (marker, channel) per batch.
        ``None`` — no grouping, draw from all samples.
    leaky : float
        Fraction of the batch drawn from *other* groups when
        ``batch_group_by`` is not ``None``. Ignored otherwise.
    group_weights : dict[str, float] | None
        Per-group sampling weight (keyed by group string key).
        Defaults to proportional to group size.
    stratify_by : str | list[str] | None
        Column name(s) in *valid_anchors* to stratify batches by.
        Groups are balanced equally within each batch.
        ``None`` disables stratification.
    temporal_enrichment : bool
        If ``True``, concentrate batch indices around a randomly chosen
        focal hours-post-infection (HPI) value.
        Requires ``"hours_post_perturbation"`` column in *valid_anchors*.
    temporal_window_hours : float
        Half-width of the focal window around the chosen HPI.
    temporal_global_fraction : float
        Fraction of the batch drawn from all timepoints (global).
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
        batch_group_by: str | list[str] | None = None,
        leaky: float = 0.0,
        group_weights: dict[str, float] | None = None,
        stratify_by: str | list[str] | None = "condition",
        temporal_enrichment: bool = False,
        temporal_window_hours: float = 2.0,
        temporal_global_fraction: float = 0.3,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        # Normalize to list or None
        if isinstance(batch_group_by, str):
            batch_group_by = [batch_group_by]
        if isinstance(stratify_by, str):
            stratify_by = [stratify_by]

        # ------------------------------------------------------------------
        # Validate required columns for enabled features
        # ------------------------------------------------------------------
        if batch_group_by is not None:
            missing = [c for c in batch_group_by if c not in valid_anchors.columns]
            if missing:
                raise ValueError(
                    f"batch_group_by={batch_group_by} requires columns {missing} "
                    f"in valid_anchors, but columns are: "
                    f"{list(valid_anchors.columns)}"
                )
        if stratify_by is not None:
            missing = [c for c in stratify_by if c not in valid_anchors.columns]
            if missing:
                raise ValueError(
                    f"stratify_by={stratify_by} requires columns {missing} "
                    f"in valid_anchors, but columns are: "
                    f"{list(valid_anchors.columns)}"
                )
        if temporal_enrichment and "hours_post_perturbation" not in valid_anchors.columns:
            raise ValueError(
                "temporal_enrichment=True requires 'hours_post_perturbation' "
                "column in valid_anchors, but columns are: "
                f"{list(valid_anchors.columns)}"
            )

        self.valid_anchors = valid_anchors
        self.batch_size = batch_size
        self.batch_group_by = batch_group_by
        self.leaky = leaky
        self.group_weights = group_weights
        self.stratify_by = stratify_by
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
            self._hpi_values: np.ndarray = valid_anchors["hours_post_perturbation"].to_numpy()

        self._precompute_groups()

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _precompute_groups(self) -> None:
        """Build index lookup tables from valid_anchors columns."""
        # Per-group indices
        if self.batch_group_by is not None:
            group_keys = self._compute_strat_keys(self.valid_anchors, self.batch_group_by)
            self._group_indices: dict[str, np.ndarray] = {
                str(name): group.index.to_numpy() for name, group in self.valid_anchors.groupby(group_keys)
            }
            self._group_names: list[str] = list(self._group_indices.keys())
        else:
            self._group_indices = {}
            self._group_names = []

        # Stratification indices
        self._strat_indices: dict[str, np.ndarray] = {}
        self._group_strat_indices: dict[tuple[str, str], np.ndarray] = {}
        self._strat_names: list[str] = []

        if self.stratify_by is not None:
            strat_keys = self._compute_strat_keys(self.valid_anchors, self.stratify_by)

            # Global stratification indices
            for key in strat_keys.unique():
                self._strat_indices[key] = self.valid_anchors.index[strat_keys == key].to_numpy()
            self._strat_names = list(self._strat_indices.keys())

            # Per-group stratification indices
            if self.batch_group_by is not None:
                group_keys = self._compute_strat_keys(self.valid_anchors, self.batch_group_by)
                for (grp, strat_key), group in self.valid_anchors.groupby([group_keys, strat_keys]):
                    self._group_strat_indices[(str(grp), str(strat_key))] = group.index.to_numpy()

        # All indices
        self._all_indices = np.arange(len(self.valid_anchors))

        # Compute group selection weights
        if self.batch_group_by is not None:
            total = len(self.valid_anchors)
            if self.group_weights is not None:
                raw = np.array([self.group_weights.get(n, 0.0) for n in self._group_names])
                self._group_probs = raw / raw.sum()
            else:
                # Default: proportional to group size
                self._group_probs = np.array([len(self._group_indices[n]) / total for n in self._group_names])

            # Warn about small groups
            for name, indices in self._group_indices.items():
                if len(indices) < self.batch_size:
                    _logger.warning(
                        "Group '%s' has %d samples, fewer than "
                        "batch_size=%d. Will use replacement sampling "
                        "for this group.",
                        name,
                        len(indices),
                        self.batch_size,
                    )

    @staticmethod
    def _compute_strat_keys(df: pd.DataFrame, columns: list[str]) -> pd.Series:
        """Compute a single string key per row for grouping.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to compute keys for.
        columns : list[str]
            Column names to combine into group keys.

        Returns
        -------
        pd.Series
            String keys, one per row. Single-column uses values directly;
            multi-column joins with ``"|"``.
        """
        if len(columns) == 1:
            return df[columns[0]].astype(str)
        return df[columns].astype(str).agg("|".join, axis=1)

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
        1. Group selection (if batch_group_by is set)
        2. Leaky mixing (if leaky > 0)
        3. Temporal enrichment OR stratified sampling OR plain sampling
        4. Combine primary + leak
        """
        chosen_group: str | None = None

        # Step 1: Group selection
        if self.batch_group_by is not None:
            chosen_group = rng.choice(self._group_names, p=self._group_probs)
            pool = self._group_indices[chosen_group]
        else:
            pool = self._all_indices

        # Step 2: Leaky mixing
        leak_samples: np.ndarray | None = None
        if self.batch_group_by is not None and self.leaky > 0.0 and chosen_group is not None:
            n_leak = int(self.batch_size * self.leaky)
            n_primary = self.batch_size - n_leak
            if n_leak > 0:
                other_indices = np.concatenate([v for k, v in self._group_indices.items() if k != chosen_group])
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
            primary = self._enrich_temporal(pool, n_primary, rng, chosen_group)
        elif self.stratify_by is not None:
            primary = self._sample_stratified(pool, n_primary, chosen_group, rng)
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
        chosen_group: str | None,
    ) -> np.ndarray:
        """Sample *n_target* indices from *pool* with focal HPI concentration.

        Picks a random focal HPI from the unique HPIs available in *pool*.
        Then splits *pool* into focal (within window) and non-focal indices,
        and assembles the batch with the specified focal/global mix.

        Parameters
        ----------
        pool : np.ndarray
            Group-filtered (or global) index array to sample from.
        n_target : int
            Number of indices to produce.
        rng : np.random.Generator
            Shared RNG for deterministic sampling.
        chosen_group : str | None
            If batch_group_by is set, the chosen group name.

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
            focal_samples = rng.choice(focal_pool, size=n_focal, replace=focal_replace)
        elif n_focal > 0:
            focal_replace = len(pool) < n_focal
            focal_samples = rng.choice(pool, size=n_focal, replace=focal_replace)
        else:
            focal_samples = np.array([], dtype=int)

        # Sample global indices (from non-focal to avoid duplicating focal)
        if n_global > 0 and len(global_pool) > 0:
            global_replace = len(global_pool) < n_global
            global_samples = rng.choice(global_pool, size=n_global, replace=global_replace)
        elif n_global > 0:
            global_replace = len(pool) < n_global
            global_samples = rng.choice(pool, size=n_global, replace=global_replace)
        else:
            global_samples = np.array([], dtype=int)

        return np.concatenate([focal_samples, global_samples])

    # ------------------------------------------------------------------
    # Stratified sampling
    # ------------------------------------------------------------------

    def _sample_stratified(
        self,
        pool: np.ndarray,
        n_samples: int,
        chosen_group: str | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample indices with balanced representation across strata.

        If ``chosen_group`` is not None, balances strata within that
        group. Otherwise, balances strata globally.

        Parameters
        ----------
        pool : np.ndarray
            Candidate index pool (group-filtered or global).
        n_samples : int
            Number of indices to produce.
        chosen_group : str | None
            If batch_group_by is set, the chosen group name.
        rng : np.random.Generator
            Shared RNG.

        Returns
        -------
        np.ndarray
            Sampled indices of length *n_samples*.
        """
        if chosen_group is not None:
            # Strata available in this group
            strata = [key for (grp, key) in self._group_strat_indices if grp == chosen_group]
            if not strata:
                replace = len(pool) < n_samples
                return rng.choice(pool, size=n_samples, replace=replace)

            ratios = self._compute_ratios(strata)

            indices_parts: list[np.ndarray] = []
            remaining = n_samples
            for i, key in enumerate(strata):
                strat_pool = self._group_strat_indices.get((chosen_group, key), np.array([], dtype=int))
                if len(strat_pool) == 0:
                    continue
                if i == len(strata) - 1:
                    n_stratum = remaining
                else:
                    n_stratum = int(n_samples * ratios[key])
                    remaining -= n_stratum

                replace = len(strat_pool) < n_stratum
                chosen = rng.choice(strat_pool, size=n_stratum, replace=replace)
                indices_parts.append(chosen)

            if indices_parts:
                return np.concatenate(indices_parts)
            replace = len(pool) < n_samples
            return rng.choice(pool, size=n_samples, replace=replace)

        else:
            # No batch grouping: balance strata globally
            strata = self._strat_names
            if not strata:
                replace = len(pool) < n_samples
                return rng.choice(pool, size=n_samples, replace=replace)

            ratios = self._compute_ratios(strata)

            indices_parts: list[np.ndarray] = []
            remaining = n_samples
            for i, key in enumerate(strata):
                strat_pool = self._strat_indices.get(key, np.array([], dtype=int))
                if len(strat_pool) == 0:
                    continue
                if i == len(strata) - 1:
                    n_stratum = remaining
                else:
                    n_stratum = int(n_samples * ratios[key])
                    remaining -= n_stratum

                replace = len(strat_pool) < n_stratum
                chosen = rng.choice(strat_pool, size=n_stratum, replace=replace)
                indices_parts.append(chosen)

            if indices_parts:
                return np.concatenate(indices_parts)
            replace = len(pool) < n_samples
            return rng.choice(pool, size=n_samples, replace=replace)

    @staticmethod
    def _compute_ratios(strata: list[str]) -> dict[str, float]:
        """Compute equal sampling ratios for strata.

        Parameters
        ----------
        strata : list[str]
            Group keys to compute ratios for.

        Returns
        -------
        dict[str, float]
            Equal ratios summing to 1.0.
        """
        n = len(strata)
        return {s: 1.0 / n for s in strata}
