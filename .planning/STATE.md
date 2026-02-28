# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** Phase 26 -- Refactor Translation Application

## Current Position

Phase: 26 (Refactor Translation Application)
Plan: 02 of 2 complete
Status: Phase Complete (all plans executed)
Last activity: 2026-02-28 -- Plan 26-02 executed (engine migration + test suite)

Progress: [######################........] 20/25 phases complete (80%)

## Performance Metrics

**Combined velocity (all branches):**
- Total plans completed: 29 (v1.0: 7, v1.1: 9, v1.2: 9, v2.1: 2, v2.3: 2) + v2.0 manual phases

**By Milestone:**

| Milestone | Phases | Plans | Branch |
|-----------|--------|-------|--------|
| v1.0 Transforms | 1-5 | 7 | shared |
| v1.1 Data | 6-9 | 9 | modular-data |
| v1.2 Models | 10-14 | 9 | modular-models |
| v2.0 DynaCLR | 15-17 | manual | app-dynaclr |
| v2.1 Validation | 18-19 | 2 | app-dynaclr |
| v2.2 Sampling | 20-25 | TBD | app-dynaclr |
| v2.3 Translation | 26 | 2/2 | app-cytoland |

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
- TYPE_CHECKING guard for cross-package type-only imports (HCSPredictionWriter pattern)
- viscy_utils.losses as shared location for reconstruction losses (MixedLoss)
- Translation app delegates to viscy_utils.cli.main for LightningCLI entry point
- MixedLoss removed from translation engine.py (imported from viscy_utils.losses)
- Top-level viscy_data imports for Sample/SegmentationSample (both exported at top level)

### Roadmap Evolution

- Phase 26 added: Refactor translation application
- Phase 26 planned: 2 plans (26-01 shared infra, 26-02 engine migration)
- Phase 26 Plan 01 executed: HCSPredictionWriter + MixedLoss to viscy-utils, translation scaffold created
- Phase 26 Plan 02 executed: VSUNet, FcmaeUNet, AugmentedPredictionVSUNet migrated with test suite

### Blockers/Concerns

- None. Phase 26 is complete.

## Session Continuity

Last session: 2026-02-28
Stopped at: Completed 26-02-PLAN.md. Phase 26 complete.
Resume file: None

---
*State initialized: 2025-01-27*
*Updated for v2.2 Composable Sampling Framework roadmap: 2026-02-21*
