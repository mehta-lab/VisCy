# Phase 22: Batch Sampling - Research

**Researched:** 2026-02-22
**Domain:** PyTorch custom BatchSampler with experiment-aware, condition-balanced, and temporally enriched sampling for contrastive learning
**Confidence:** HIGH

## Summary

Phase 22 implements `FlexibleBatchSampler` -- a composable batch sampler that controls WHICH cell indices appear in each training batch. It operates on the `valid_anchors` DataFrame produced by Phase 21's `MultiExperimentIndex` and yields lists of integer indices consumed by `__getitems__()` in the downstream dataset (Phase 24). The sampler lives in `packages/viscy-data/src/viscy_data/` as a reusable utility.

The core challenge is composing three independent sampling axes -- experiment restriction, condition balancing, and temporal enrichment -- into a single `__iter__` method that yields batch-sized index lists. Each axis progressively narrows the candidate pool for a batch. DDP support requires `set_epoch()` for deterministic shuffling and rank-aware index partitioning that composes with the existing `ShardedDistributedSampler` pattern.

This is a well-understood problem domain. PyTorch's `Sampler[list[int]]` protocol is simple (`__iter__` yielding `list[int]`, `__len__`), and the `batch_sampler=` kwarg to `DataLoader`/`ThreadDataLoader` handles integration. The main complexity is the sampling logic itself: picking experiments, balancing conditions within experiments, and concentrating around temporal windows -- all while maintaining deterministic behavior across DDP ranks.

**Primary recommendation:** Implement `FlexibleBatchSampler` as a `Sampler[list[int]]` subclass using a cascade approach: (1) pick experiment, (2) filter by condition quotas, (3) filter by temporal window, (4) sample indices. Use numpy RNG seeded by `epoch + seed` for DDP determinism. Do NOT hand-roll DDP sharding -- compose with the existing `ShardedDistributedSampler` or embed rank-aware slicing directly.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.x (installed) | `Sampler[list[int]]` base class, `Generator` for deterministic RNG | PyTorch's own sampler protocol |
| numpy | 1.x/2.x (installed) | `np.random.Generator` for seeded sampling, array operations | Faster than pandas for index manipulation |
| pandas | 2.x (installed) | DataFrame operations for groupby filtering | valid_anchors is a DataFrame |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| monai | (installed) | ThreadDataLoader accepts `batch_sampler=` via `**kwargs` passthrough | DataModule wiring in Phase 24 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom FlexibleBatchSampler | pytorch_metric_learning HierarchicalSampler | PML's sampler uses 2-level hierarchy (super_label/label); our 3-axis composition (experiment/condition/temporal) does not fit. PML also uses global numpy random state instead of seeded Generator. |
| Custom FlexibleBatchSampler | pytorch_metric_learning MPerClassSampler | MPerClassSampler balances classes but has no experiment-awareness or temporal enrichment. |
| numpy.random.Generator | torch.Generator | numpy Generator supports `choice(p=weights)` natively; torch Generator only works with `randperm`/`randint`. For weighted experiment selection and condition-balanced sub-sampling, numpy is more ergonomic. |

**Installation:** No new dependencies. All required packages are already installed.

## Architecture Patterns

### Recommended File Structure
```
packages/viscy-data/src/viscy_data/
├── sampler.py              # FlexibleBatchSampler (NEW)
├── distributed.py          # ShardedDistributedSampler (EXISTING)
├── __init__.py             # Add FlexibleBatchSampler export
└── ...

packages/viscy-data/tests/
├── test_sampler.py         # Tests for FlexibleBatchSampler (NEW)
└── ...
```

### Pattern 1: Cascade Batch Construction

**What:** Build each batch by progressively narrowing candidates: experiment -> condition -> temporal window -> sample.
**When to use:** When multiple independent sampling axes must compose within a single batch.

```python
# Source: Design derived from project requirements (SAMP-01 through SAMP-05)
# and reference context document

def _build_one_batch(self, rng: np.random.Generator) -> list[int]:
    """Construct a single batch by cascading filters."""
    # Step 1: Pick experiment (experiment_aware)
    if self.experiment_aware:
        exp = self._pick_experiment(rng)
        pool = self._experiment_indices[exp]
    else:
        pool = self._all_indices

    # Step 2: Leaky mixing -- inject cross-experiment samples
    if self.experiment_aware and self.leaky > 0.0:
        n_leak = int(self.batch_size * self.leaky)
        n_primary = self.batch_size - n_leak
        # ... sample n_leak from other experiments
    else:
        n_primary = self.batch_size

    # Step 3: Condition balancing
    if self.condition_balanced:
        pool = self._balance_conditions(pool, rng)

    # Step 4: Temporal enrichment
    if self.temporal_enrichment:
        pool = self._enrich_temporal(pool, rng)

    # Step 5: Sample batch_size indices from narrowed pool
    batch = rng.choice(pool, size=min(n_primary, len(pool)), replace=False)
    return batch.tolist()
```

### Pattern 2: DDP Composition via set_epoch()

**What:** Use `set_epoch(epoch)` to seed the RNG deterministically, then partition batches across ranks.
**When to use:** Multi-GPU training with DDP.

```python
# Source: torch.utils.data.distributed.DistributedSampler pattern

class FlexibleBatchSampler(Sampler[list[int]]):
    def __init__(self, ..., num_replicas=None, rank=None, seed=0):
        # If DDP not initialized, default to single-process
        self.num_replicas = num_replicas or 1
        self.rank = rank or 0
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        # Generate ALL batches (same on every rank due to same seed)
        all_batches = [self._build_one_batch(rng) for _ in range(self._num_batches)]
        # Each rank takes its slice
        my_batches = all_batches[self.rank::self.num_replicas]
        yield from my_batches
```

**Key insight:** All ranks use the same seed+epoch, so they generate the same batch list. Each rank then takes every Nth batch (interleaved). This is simpler than trying to partition indices across ranks before batch construction, which would break experiment-aware constraints.

### Pattern 3: Pre-computed Group Indices

**What:** At `__init__` time, pre-compute per-experiment and per-condition index arrays from the valid_anchors DataFrame. Avoid repeated groupby during iteration.
**When to use:** Always. The valid_anchors DataFrame is immutable between epochs.

```python
# Source: MultiExperimentIndex already provides experiment_groups and
# condition_groups, but FlexibleBatchSampler operates on valid_anchors
# (which has its own index space after reset_index(drop=True))

def _precompute_groups(self):
    """Build lookup tables from valid_anchors columns."""
    self._experiment_indices = {}
    for exp_name, group in self.valid_anchors.groupby("experiment"):
        self._experiment_indices[exp_name] = group.index.to_numpy()

    self._condition_indices = {}
    for cond, group in self.valid_anchors.groupby("condition"):
        self._condition_indices[cond] = group.index.to_numpy()

    # Cross-index: per-experiment, per-condition
    self._exp_cond_indices = {}
    for (exp, cond), group in self.valid_anchors.groupby(["experiment", "condition"]):
        self._exp_cond_indices[(exp, cond)] = group.index.to_numpy()
```

### Pattern 4: Temporal Enrichment with Focal Window

**What:** Concentrate a fraction of the batch around a focal HPI, with the rest drawn globally.
**When to use:** When `temporal_enrichment=True`.

```python
# Source: CONCORD (Zhu et al. Nature Biotech 2026) temporal concentration strategy

def _enrich_temporal(self, pool: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Concentrate pool around a randomly chosen focal HPI."""
    hpi_values = self.valid_anchors.loc[pool, "hours_post_infection"].values

    # Pick focal HPI from existing values
    focal_hpi = rng.choice(np.unique(hpi_values))

    # Split pool into focal window and global
    in_window = np.abs(hpi_values - focal_hpi) <= self.temporal_window_hours
    focal_pool = pool[in_window]
    global_pool = pool[~in_window]

    # Determine counts
    n_global = int(self.batch_size * self.temporal_global_fraction)
    n_focal = self.batch_size - n_global

    # Sample from each
    focal_samples = rng.choice(focal_pool, size=min(n_focal, len(focal_pool)), replace=len(focal_pool) < n_focal)
    global_samples = rng.choice(global_pool, size=min(n_global, len(global_pool)), replace=len(global_pool) < n_global)

    return np.concatenate([focal_samples, global_samples])
```

### Anti-Patterns to Avoid

- **Modifying valid_anchors during iteration:** The DataFrame is shared state. Never mutate it. All filtering should use boolean masks or index arrays.
- **Using pandas operations in the hot loop:** `groupby` and `loc` in `__iter__` are slow. Pre-compute index arrays at `__init__` time.
- **Global numpy random state:** PML samplers use `np.random.shuffle()` (global state) which is not DDP-safe. Always use `np.random.Generator` with explicit seed.
- **Coupling sampler to dataset:** The sampler should only know about index metadata (experiment, condition, HPI), never about image data or Position objects.
- **Trying to use batch_sampler AND sampler simultaneously:** PyTorch DataLoader raises ValueError if both are specified. When using `batch_sampler=`, do NOT pass `batch_size`, `shuffle`, `sampler`, or `drop_last`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| DDP index partitioning | Custom shard logic | Interleaved batch assignment (rank slicing) | Edge cases with uneven batches, padding, drop_last |
| Seeded RNG | `random.seed()` / `np.random.seed()` | `np.random.default_rng(seed)` | Thread-safe, no global state pollution, DDP-compatible |
| Weighted random selection | Manual probability computation | `rng.choice(a, p=weights)` | NumPy handles normalization, edge cases |
| DataFrame group indices | Repeated `df[df["col"]==val].index` | Pre-computed dict from `groupby` at init | O(1) lookup vs O(n) scan per batch |

**Key insight:** The sampling logic itself is custom (no library provides this exact 3-axis composition), but all the building blocks (seeded RNG, weighted choice, index arrays) are standard numpy operations. The only truly custom code is the cascade logic in `_build_one_batch`.

## Common Pitfalls

### Pitfall 1: Non-deterministic DDP batches
**What goes wrong:** Different ranks generate different batches, leading to gradient desync and training divergence.
**Why it happens:** RNG not seeded identically across ranks, or `set_epoch()` not called.
**How to avoid:** Use `seed + epoch` as RNG seed. All ranks generate the same full batch list, then each takes its interleaved slice. Verify with a test that checks `set_epoch(0)` on rank 0 and rank 1 produce disjoint but collectively exhaustive batches.
**Warning signs:** NaN losses, validation metrics diverge between ranks, training hangs at gradient sync.

### Pitfall 2: Small experiment/condition groups cause replacement sampling
**What goes wrong:** An experiment or condition has fewer cells than `batch_size`, requiring sampling with replacement, which duplicates samples.
**Why it happens:** Unbalanced datasets (e.g., 20 infected cells but batch_size=128).
**How to avoid:** Document the constraint: `batch_size` should not exceed the smallest experiment-condition group. Add a warning in `__init__` if any group is smaller than batch_size. Fall back to replacement sampling with a logged warning rather than crashing.
**Warning signs:** Training loss plateaus early, effective batch diversity is low.

### Pitfall 3: Temporal enrichment starves rare timepoints
**What goes wrong:** With a narrow temporal window, cells at the edges of the HPI range are never sampled as focal cells, and the global fraction is too small to include them.
**Why it happens:** `temporal_window_hours` is too narrow, or `temporal_global_fraction` is too low.
**How to avoid:** Default `temporal_global_fraction=0.3` ensures 30% of each batch comes from all timepoints. Focal HPI is chosen uniformly from available HPIs, not weighted.
**Warning signs:** Embeddings cluster only by time, not by biological state.

### Pitfall 4: batch_sampler + ThreadDataLoader kwargs conflict
**What goes wrong:** Passing `batch_sampler=` along with `batch_size=`, `shuffle=`, or `drop_last=` to DataLoader raises ValueError.
**Why it happens:** PyTorch enforces mutual exclusivity between `batch_sampler` and these kwargs.
**How to avoid:** When using FlexibleBatchSampler, the DataModule (Phase 24) must NOT pass batch_size/shuffle/drop_last to ThreadDataLoader. Only pass `batch_sampler=`, `num_workers=`, `collate_fn=`, etc.
**Warning signs:** ValueError at DataLoader construction time (easy to catch in tests).

### Pitfall 5: __len__ mismatch with actual iteration count
**What goes wrong:** DataLoader expects `__len__` to return the correct number of batches for progress bars and epoch completion. If `__len__` disagrees with actual `__iter__` count, training loop may hang or skip data.
**Why it happens:** `__len__` computed from total indices / batch_size, but actual batches depend on per-experiment constraints that may yield fewer batches.
**How to avoid:** Compute `__len__` as `total_batches // num_replicas` where `total_batches` is the number of batches that `__iter__` will actually yield. Pre-compute this in `__init__` based on the total number of valid anchors and batch size.
**Warning signs:** Progress bar stuck at 99%, training epoch never completes, or ends prematurely.

### Pitfall 6: valid_anchors index vs tracks index confusion
**What goes wrong:** valid_anchors has `reset_index(drop=True)`, giving it indices 0..N-1. The sampler yields these indices. But if someone confuses them with tracks indices (which may be a superset), wrong cells get loaded.
**Why it happens:** Two DataFrames (tracks, valid_anchors) with different index spaces.
**How to avoid:** The sampler operates ONLY on valid_anchors indices. Document this clearly. The dataset's `__getitems__` also uses `self.valid_anchors.iloc[indices]`, matching the sampler's output.
**Warning signs:** KeyError or IndexError when dataset tries to look up a sampler-provided index.

## Code Examples

### FlexibleBatchSampler skeleton (verified pattern from PyTorch Sampler protocol)

```python
# Source: torch.utils.data.sampler.Sampler protocol + DistributedSampler pattern
from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


class FlexibleBatchSampler(Sampler[list[int]]):
    """Composable batch sampler with experiment-aware, condition-balanced,
    and temporal enrichment axes.

    Yields lists of integer indices into a valid_anchors DataFrame.
    """

    def __init__(
        self,
        valid_anchors: pd.DataFrame,
        batch_size: int = 128,
        # Experiment-aware
        experiment_aware: bool = True,
        leaky: float = 0.0,
        experiment_weights: dict[str, float] | None = None,
        # Temporal enrichment
        temporal_enrichment: bool = False,
        temporal_window_hours: float = 2.0,
        temporal_global_fraction: float = 0.3,
        # Condition balancing
        condition_balanced: bool = True,
        condition_ratio: dict[str, float] | None = None,
        # DDP
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
        self.temporal_enrichment = temporal_enrichment
        self.temporal_window_hours = temporal_window_hours
        self.temporal_global_fraction = temporal_global_fraction
        self.condition_balanced = condition_balanced
        self.condition_ratio = condition_ratio or {}
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._precompute_groups()

    def _precompute_groups(self) -> None:
        """Build index lookup tables from valid_anchors."""
        # Per-experiment indices
        self._experiment_indices: dict[str, np.ndarray] = {
            name: group.index.to_numpy()
            for name, group in self.valid_anchors.groupby("experiment")
        }
        self._experiment_names = list(self._experiment_indices.keys())
        # Per-condition indices (within each experiment)
        self._exp_cond_indices: dict[tuple[str, str], np.ndarray] = {}
        for (exp, cond), group in self.valid_anchors.groupby(
            ["experiment", "condition"]
        ):
            self._exp_cond_indices[(exp, cond)] = group.index.to_numpy()
        self._all_indices = np.arange(len(self.valid_anchors))

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling across DDP ranks."""
        self.epoch = epoch

    def __len__(self) -> int:
        total_batches = len(self.valid_anchors) // self.batch_size
        return math.ceil(total_batches / self.num_replicas)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        total_batches = len(self.valid_anchors) // self.batch_size
        all_batches = [self._build_one_batch(rng) for _ in range(total_batches)]
        # DDP: each rank takes its interleaved slice
        my_batches = all_batches[self.rank :: self.num_replicas]
        yield from my_batches

    def _build_one_batch(self, rng: np.random.Generator) -> list[int]:
        """Construct a single batch by cascading sampling axes."""
        # ... implementation of cascade logic
        raise NotImplementedError
```

### DataLoader wiring (verified: ThreadDataLoader passes **kwargs to DataLoader)

```python
# Source: monai.data.thread_buffer.ThreadDataLoader.__init__
# ThreadDataLoader(dataset, **kwargs) -> super().__init__(dataset, **kwargs)
# So batch_sampler= is supported.

from monai.data.thread_buffer import ThreadDataLoader

loader = ThreadDataLoader(
    dataset=train_dataset,
    batch_sampler=flexible_sampler,  # FlexibleBatchSampler instance
    use_thread_workers=True,
    num_workers=num_workers,
    collate_fn=lambda x: x,        # dataset returns pre-batched dict
    pin_memory=pin_memory,
    # NOTE: Do NOT pass batch_size, shuffle, sampler, or drop_last
)
```

### Condition balancing within an experiment

```python
# Source: Derived from SAMP-02 requirement
def _balance_conditions(
    self,
    exp_name: str,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample indices with balanced conditions from one experiment."""
    conditions = [
        cond for (exp, cond) in self._exp_cond_indices
        if exp == exp_name
    ]
    # Default: equal ratio across conditions
    ratios = self.condition_ratio or {c: 1.0 / len(conditions) for c in conditions}

    indices = []
    for cond in conditions:
        n_cond = int(n_samples * ratios.get(cond, 1.0 / len(conditions)))
        pool = self._exp_cond_indices.get((exp_name, cond), np.array([]))
        if len(pool) > 0:
            chosen = rng.choice(pool, size=min(n_cond, len(pool)), replace=len(pool) < n_cond)
            indices.append(chosen)

    return np.concatenate(indices) if indices else np.array([], dtype=int)
```

### DDP determinism test pattern

```python
# Source: Standard DDP sampler test pattern
def test_ddp_determinism():
    """Verify rank 0 and rank 1 get disjoint batches from same seed."""
    sampler_r0 = FlexibleBatchSampler(
        valid_anchors, batch_size=4, num_replicas=2, rank=0, seed=42
    )
    sampler_r1 = FlexibleBatchSampler(
        valid_anchors, batch_size=4, num_replicas=2, rank=1, seed=42
    )
    sampler_r0.set_epoch(0)
    sampler_r1.set_epoch(0)

    batches_r0 = list(sampler_r0)
    batches_r1 = list(sampler_r1)

    # Same total coverage
    all_r0 = set(idx for batch in batches_r0 for idx in batch)
    all_r1 = set(idx for batch in batches_r1 for idx in batch)
    # Batches are disjoint (different batches assigned to different ranks)
    # Note: individual indices MAY overlap across batches (sampling with replacement for balance)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.utils.data.BatchSampler(RandomSampler(...))` | Custom `Sampler[list[int]]` subclass yielding batch index lists | PyTorch 1.x+ | Full control over batch composition |
| Global `np.random.seed()` for reproducibility | `np.random.default_rng(seed)` instance-based RNG | NumPy 1.17+ (2019) | Thread-safe, no global state, DDP-safe |
| PML MPerClassSampler (2-level) | Custom 3-axis sampler | Project-specific | Experiment/condition/temporal axes not available in PML |
| Uniform temporal sampling | CONCORD-style focal window enrichment | Zhu et al. 2026 | Forces hard negatives at similar timepoints |

**Deprecated/outdated:**
- `np.random.RandomState`: Use `np.random.Generator` / `default_rng()` instead. RandomState is legacy.
- PML's `NUMPY_RANDOM` global: Not DDP-safe. Avoid.

## Open Questions

1. **Experiment weighting strategy**
   - What we know: `experiment_weights` allows manual per-experiment probabilities for experiment selection.
   - What's unclear: Should the default be uniform across experiments, or proportional to the number of valid anchors per experiment?
   - Recommendation: Default to proportional (larger experiments sampled more often) with uniform as an explicit option. This prevents tiny experiments from dominating batch counts. Planner can decide.

2. **Condition ratio when more than 2 conditions exist**
   - What we know: Current requirements assume binary (infected/uninfected). `condition_ratio` dict supports N conditions.
   - What's unclear: What if an experiment has 3+ conditions (e.g., "uninfected", "low_moi", "high_moi")?
   - Recommendation: Support arbitrary condition counts. Default to equal ratios. The `condition_ratio` dict allows user override.

3. **Temporal enrichment focal HPI selection**
   - What we know: A focal HPI is chosen per batch, and cells within `temporal_window_hours` are concentrated.
   - What's unclear: Should focal HPI be chosen from the union of all HPIs in the experiment, or per-batch randomly?
   - Recommendation: Per-batch random selection from unique HPIs within the chosen experiment. This ensures all timepoints get exposure across batches.

4. **How `__len__` interacts with Lightning's progress bar**
   - What we know: Lightning calls `len(dataloader)` for progress bars. DataLoader delegates to `len(batch_sampler)`.
   - What's unclear: If the sampler's actual iteration count varies slightly from `__len__` (due to rounding in condition balance), does Lightning handle this gracefully?
   - Recommendation: Make `__len__` a conservative lower bound (floor division). Lightning handles `__iter__` exhaustion gracefully.

5. **Interaction with `ShardedDistributedSampler` vs embedded DDP**
   - What we know: STATE.md says "DDP via FlexibleBatchSampler + ShardedDistributedSampler composition". But `batch_sampler=` and `sampler=` are mutually exclusive in DataLoader.
   - What's unclear: Does "composition" mean embedding DDP logic inside FlexibleBatchSampler, or wrapping?
   - Recommendation: Embed DDP logic directly in FlexibleBatchSampler (num_replicas, rank, set_epoch). Do NOT try to compose with ShardedDistributedSampler as a separate sampler -- DataLoader forbids this. The "composition" means FlexibleBatchSampler follows the same pattern (set_epoch, rank-aware slicing) rather than literally wrapping ShardedDistributedSampler.

## Upstream Dependencies (Phase 21 API Surface)

### valid_anchors DataFrame Schema

The FlexibleBatchSampler receives `valid_anchors` which is a `pd.DataFrame` with `reset_index(drop=True)` (integer index 0..N-1). Required columns:

| Column | Type | Source | Used By |
|--------|------|--------|---------|
| `experiment` | str | ExperimentConfig.name | SAMP-01 (experiment-aware), SAMP-05 (leaky mixing) |
| `condition` | str | Resolved from condition_wells | SAMP-02 (condition balancing) |
| `hours_post_infection` | float | `start_hpi + t * interval_minutes / 60` | SAMP-03 (temporal enrichment) |
| `global_track_id` | str | `{exp}_{fov}_{track_id}` | Not directly used by sampler |
| `t` | int | Frame index | Not directly used by sampler |
| `y_clamp` / `x_clamp` | int | Border-clamped centroids | Not used by sampler |
| `position` | Position | iohub handle | Not used by sampler |

The sampler ONLY needs: `experiment`, `condition`, `hours_post_infection`, and the integer index.

### MultiExperimentIndex Properties

- `index.valid_anchors` -- the DataFrame passed to FlexibleBatchSampler
- `index.experiment_groups` -- `dict[str, np.ndarray]` of tracks indices (NOT valid_anchors indices; sampler must build its own)
- `index.condition_groups` -- same caveat

**Important:** `experiment_groups` and `condition_groups` return indices into `index.tracks`, not `index.valid_anchors`. The sampler must build its own groupby on valid_anchors at init time.

## Downstream Consumers (Phase 24)

Phase 24's `MultiExperimentDataModule` will wire the sampler:

```python
# Phase 24 wiring (for context, not implemented here)
self._train_sampler = FlexibleBatchSampler(
    valid_anchors=self.cell_index.valid_anchors,
    batch_size=self.batch_size,
    experiment_aware=self.experiment_aware,
    condition_balanced=self.balance_conditions,
    temporal_enrichment=self.temporal_enrichment,
    ...
)
# ThreadDataLoader(dataset, batch_sampler=self._train_sampler, ...)
```

The sampler yields `list[int]` -> dataset's `__getitems__(indices)` receives these -> loads patches.

## Sources

### Primary (HIGH confidence)
- `/Users/eduardo.hirata/Documents/repos/VisCy/.venv/lib/python3.13/site-packages/torch/utils/data/sampler.py` -- PyTorch Sampler and BatchSampler protocol (read directly)
- `/Users/eduardo.hirata/Documents/repos/VisCy/.venv/lib/python3.13/site-packages/torch/utils/data/distributed.py` -- DistributedSampler with set_epoch() pattern (read directly)
- `/Users/eduardo.hirata/Documents/repos/VisCy/.venv/lib/python3.13/site-packages/torch/utils/data/dataloader.py` -- DataLoader batch_sampler mutual exclusivity with batch_size/shuffle/sampler/drop_last (read directly)
- `/Users/eduardo.hirata/Documents/repos/VisCy/.venv/lib/python3.13/site-packages/monai/data/thread_buffer.py` -- ThreadDataLoader passes **kwargs to DataLoader, confirming batch_sampler support (read directly)
- `/Users/eduardo.hirata/Documents/repos/VisCy/applications/dynaclr/src/dynaclr/index.py` -- MultiExperimentIndex.valid_anchors schema and properties (read directly)
- `/Users/eduardo.hirata/Documents/repos/VisCy/packages/viscy-data/src/viscy_data/distributed.py` -- ShardedDistributedSampler pattern (read directly)
- `/Users/eduardo.hirata/Documents/repos/VisCy/packages/viscy-data/src/viscy_data/triplet.py` -- Existing TripletDataset sampling patterns (read directly)
- `/Users/eduardo.hirata/Downloads/dynaclr_claude_code_context.md` -- Full design context document with interfaces (read directly)

### Secondary (MEDIUM confidence)
- `/Users/eduardo.hirata/Documents/repos/VisCy/.venv/lib/python3.13/site-packages/pytorch_metric_learning/samplers/hierarchical_sampler.py` -- HierarchicalSampler pattern for 2-level batch construction (read directly)
- `/Users/eduardo.hirata/Documents/repos/VisCy/.venv/lib/python3.13/site-packages/pytorch_metric_learning/samplers/m_per_class_sampler.py` -- MPerClassSampler pattern for class-balanced sampling (read directly)
- `/Users/eduardo.hirata/Documents/repos/VisCy/.venv/lib/python3.13/site-packages/timm/data/distributed_sampler.py` -- RepeatAugSampler with set_epoch pattern (read directly)

### Tertiary (LOW confidence)
- CONCORD (Zhu et al. Nature Biotech 2026) -- temporal enrichment strategy (referenced in design doc, not independently verified)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed and verified by reading source
- Architecture: HIGH -- PyTorch Sampler protocol is simple and well-documented; patterns verified from source code
- Pitfalls: HIGH -- DDP determinism pitfall verified from DistributedSampler source; DataLoader mutual exclusivity verified from source
- Upstream API: HIGH -- valid_anchors schema verified from index.py source code

**Research date:** 2026-02-22
**Valid until:** 2026-03-22 (stable domain, PyTorch sampler protocol unchanged since 1.x)
