# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** Milestone v2.2 -- Composable Sampling Framework, Phase 22

## Current Position

Phase: 22 of 25 (Flexible Batch Sampler)
Plan: 00 of ?? complete
Status: Phase 21 complete (MultiExperimentIndex with valid anchors), ready for Phase 22
Last activity: 2026-02-22 -- Completed 21-02 valid anchors, properties, summary

Progress: [######################........] 21/25 phases complete (84%)

## Performance Metrics

**Combined velocity (all branches):**
- Total plans completed: 32 (v1.0: 7, v1.1: 9, v1.2: 9, v2.1: 2, v2.2: 5) + v2.0 manual phases

**By Milestone:**

| Milestone | Phases | Plans | Branch |
|-----------|--------|-------|--------|
| v1.0 Transforms | 1-5 | 7 | shared |
| v1.1 Data | 6-9 | 9 | modular-data |
| v1.2 Models | 10-14 | 9 | modular-models |
| v2.0 DynaCLR | 15-17 | manual | app-dynaclr |
| v2.1 Validation | 18-19 | 2 | app-dynaclr |
| v2.2 Sampling | 20-25 | 5 | dynav2 |

## Accumulated Context

### Decisions

Key decisions carrying forward:

- Clean break on imports: `from viscy_{pkg} import X` (no backward compatibility)
- Applications compose packages: dynaclr depends on viscy-data, viscy-models, viscy-transforms, viscy-utils
- triplet.py is NOT modified -- new composable sampling code in new files only
- FlexibleBatchSampler + ChannelDropout in packages/viscy-data/ (reusable)
- ExperimentConfig, Registry, Index, Dataset, DataModule in applications/dynaclr/ (domain-specific)
- NTXentHCL as nn.Module drop-in for ContrastiveModule(loss_function=...)
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

### Blockers/Concerns

- None. Phase 21 complete, ready for Phase 22.

## Session Continuity

Last session: 2026-02-22
Stopped at: Completed 21-02-PLAN.md (valid anchors, properties, summary). Phase 21 complete. Ready for Phase 22.
Resume file: None

---
*State initialized: 2025-01-27*
*Updated for v2.2 Composable Sampling Framework roadmap: 2026-02-21*
*Updated for 20-01 completion: 2026-02-21*
*Updated for 20-02 completion: 2026-02-22*
*Updated for 21-01 completion: 2026-02-22*
*Updated for 21-02 completion: 2026-02-22*
