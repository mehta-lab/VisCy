# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** Milestone v2.2 -- Composable Sampling Framework -- COMPLETE

## Current Position

Phase: 25 of 25 (Integration) -- COMPLETE
Plan: 01 of 01 complete
Status: Phase 25 complete. Milestone v2.2 Composable Sampling Framework complete.
Last activity: 2026-02-24 -- Completed 25-01 Integration (end-to-end tests + YAML config)

Progress: [##############################] 25/25 phases complete (100%)

## Performance Metrics

**Combined velocity (all branches):**
- Total plans completed: 40 (v1.0: 7, v1.1: 9, v1.2: 9, v2.1: 2, v2.2: 13) + v2.0 manual phases

**By Milestone:**

| Milestone | Phases | Plans | Branch |
|-----------|--------|-------|--------|
| v1.0 Transforms | 1-5 | 7 | shared |
| v1.1 Data | 6-9 | 9 | modular-data |
| v1.2 Models | 10-14 | 9 | modular-models |
| v2.0 DynaCLR | 15-17 | manual | app-dynaclr |
| v2.1 Validation | 18-19 | 2 | app-dynaclr |
| v2.2 Sampling | 20-25 | 13 | dynav2 |

## Accumulated Context

### Decisions

Key decisions carrying forward:

- Clean break on imports: `from viscy_{pkg} import X` (no backward compatibility)
- Applications compose packages: dynaclr depends on viscy-data, viscy-models, viscy-transforms, viscy-utils
- triplet.py is NOT modified -- new composable sampling code in new files only
- FlexibleBatchSampler + ChannelDropout in packages/viscy-data/ (reusable)
- ExperimentConfig, Registry, Index, Dataset, DataModule in applications/dynaclr/ (domain-specific)
- NTXentHCL as nn.Module drop-in for ContrastiveModule(loss_function=...)
- NTXentHCL overrides _compute_loss (pair-based) to integrate with pytorch_metric_learning's reducer/distance pipeline
- beta=0.0 fast-path delegates to super()._compute_loss for exact numerical identity with NTXentLoss
- HCL weight normalization: per-anchor sum equals neg_count, preserving loss magnitude across beta values
- 2-channel input (Phase + Fluorescence) with channel dropout on channel 1
- HCL in loss only, no kNN sampler -- FlexibleBatchSampler handles experiment/condition/temporal axes
- Train/val split by whole experiments, not FOVs
- DDP via FlexibleBatchSampler + ShardedDistributedSampler composition
- ExperimentConfig is pure data container (dataclass, no validation); ExperimentRegistry validates the ensemble at __post_init__
- Positional alignment for source channels across experiments (names can differ, count must match)
- Excluded stale applications/dynacrl (typo) from uv workspace
- Explicit iohub/pyyaml deps in dynaclr even though transitive (direct imports require explicit declaration)
- Border clamping retains all cells within image bounds; only cells with centroid completely outside image are excluded
- Lineage reconstruction chases parent_track_id to root ancestor; missing parents fall back to self
- Position objects stored directly in DataFrame column for downstream data loading
- Global track ID format: {exp_name}_{fov_name}_{track_id} for cross-experiment uniqueness
- Anchor validity uses lineage_id for same-track and daughter-track positive matching -- simple set lookup
- tau=0 skipped to prevent anchor from being its own positive
- valid_anchors is reset_index(drop=True) for clean downstream indexing
- Properties (experiment_groups, condition_groups) use groupby on tracks rather than caching
- FlexibleBatchSampler uses numpy RNG (default_rng) over torch Generator for weighted choice ergonomics
- Default experiment weights proportional to group size, not uniform
- DDP interleaved batch slicing: all ranks generate same full batch list, each takes rank::num_replicas
- Condition balancing: last condition gets remainder to prevent rounding-induced batch size mismatch
- Temporal enrichment draws focal+global directly from experiment pool (not post-filter on pre-sampled primary)
- Conditional precomputation: groupby only runs for enabled features (avoids KeyError on missing columns)
- temporal_global_fraction=0.0 yields all-focal batches; 1.0 yields no enrichment effect
- ChannelDropout clones input tensor (non-destructive) for pipeline safety
- Per-sample independent dropout via torch.rand mask on batch dimension
- Exponential decay tau sampling uses normalized offset for consistent behavior across tau ranges
- Lineage-timepoint pre-built lookup indexed by (experiment, lineage_id) -> {t: [row_indices]} for O(1) positive candidate retrieval
- Fallback tau strategy: sample_tau first, then linear scan of full tau range if no candidate at sampled offset
- Dataset uses numpy.random.default_rng() without fixed seed; determinism delegated to external sampler
- Generic channel names (ch_0, ch_1) for DataModule transform pipeline -- experiments have different names but same count
- Norm_meta all-None coalescing in on_after_batch_transfer to prevent collate_meta_tensor crash
- Separate ExperimentRegistry instances for train/val splits, each building own MultiExperimentIndex
- ChannelDropout applied AFTER normalizations+augmentations+final_crop in on_after_batch_transfer
- Integration test uses SimpleEncoder (fc+proj) for fast CPU testing
- YAML config uses generic ch_0/ch_1 keys for normalizations/augmentations
- use_distributed_sampler: false in config since FlexibleBatchSampler handles DDP

### Blockers/Concerns

- None. Milestone v2.2 Composable Sampling Framework complete.

## Session Continuity

Last session: 2026-02-24
Stopped at: Completed 25-01-PLAN.md (Integration). Phase 25 complete. Milestone v2.2 complete.
Resume file: None

---
*State initialized: 2025-01-27*
*Updated for v2.2 Composable Sampling Framework roadmap: 2026-02-21*
*Updated for 20-01 completion: 2026-02-21*
*Updated for 20-02 completion: 2026-02-22*
*Updated for 21-01 completion: 2026-02-22*
*Updated for 21-02 completion: 2026-02-22*
*Updated for 22-01 completion: 2026-02-22*
*Updated for 22-02 completion: 2026-02-22*
*Updated for 23-02 completion: 2026-02-23*
*Updated for 23-01 completion: 2026-02-23*
*Updated for 24-01 completion: 2026-02-23*
*Updated for 24-02 completion: 2026-02-23*
*Updated for 25-01 completion: 2026-02-24*
