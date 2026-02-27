# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** Phase 26 -- Refactor Translation Application

## Current Position

Phase: 26 (Refactor Translation Application)
Plan: 01 of 2 complete
Status: In Progress (Plan 01 complete, Plan 02 pending)
Last activity: 2026-02-27 -- Plan 26-01 executed (shared infra extraction + app scaffold)

Progress: [####################..........] 19/25 phases complete (76%)

## Performance Metrics

**Combined velocity (all branches):**
- Total plans completed: 28 (v1.0: 7, v1.1: 9, v1.2: 9, v2.1: 2, v2.3: 1) + v2.0 manual phases

**By Milestone:**

| Milestone | Phases | Plans | Branch |
|-----------|--------|-------|--------|
| v1.0 Transforms | 1-5 | 7 | shared |
| v1.1 Data | 6-9 | 9 | modular-data |
| v1.2 Models | 10-14 | 9 | modular-models |
| v2.0 DynaCLR | 15-17 | manual | app-dynaclr |
| v2.1 Validation | 18-19 | 2 | app-dynaclr |
| v2.2 Sampling | 20-25 | TBD | app-dynaclr |
| v2.3 Translation | 26 | 1/2 | app-cytoland |

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

### Roadmap Evolution

- Phase 26 added: Refactor translation application
- Phase 26 planned: 2 plans (26-01 shared infra, 26-02 engine migration)
- Phase 26 Plan 01 executed: HCSPredictionWriter + MixedLoss to viscy-utils, translation scaffold created

### Blockers/Concerns

- None. Phase 26 is planned and ready for execution.

## Session Continuity

Last session: 2026-02-27
Stopped at: Completed 26-01-PLAN.md. Ready to execute Plan 26-02.
Resume file: None

---
*State initialized: 2025-01-27*
*Updated for v2.2 Composable Sampling Framework roadmap: 2026-02-21*
