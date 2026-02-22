# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** Milestone v2.2 -- Composable Sampling Framework, Phase 20

## Current Position

Phase: 20 of 25 (Experiment Configuration)
Plan: --
Status: Ready to plan
Last activity: 2026-02-21 -- Roadmap created for v2.2 (6 phases, 26 requirements)

Progress: [####################..........] 19/25 phases complete (76%)

## Performance Metrics

**Combined velocity (all branches):**
- Total plans completed: 27 (v1.0: 7, v1.1: 9, v1.2: 9, v2.1: 2) + v2.0 manual phases

**By Milestone:**

| Milestone | Phases | Plans | Branch |
|-----------|--------|-------|--------|
| v1.0 Transforms | 1-5 | 7 | shared |
| v1.1 Data | 6-9 | 9 | modular-data |
| v1.2 Models | 10-14 | 9 | modular-models |
| v2.0 DynaCLR | 15-17 | manual | app-dynaclr |
| v2.1 Validation | 18-19 | 2 | app-dynaclr |
| v2.2 Sampling | 20-25 | TBD | app-dynaclr |

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

### Blockers/Concerns

- None. Ready to start Phase 20.

## Session Continuity

Last session: 2026-02-21
Stopped at: Roadmap created for v2.2. Ready to plan Phase 20.
Resume file: None

---
*State initialized: 2025-01-27*
*Updated for v2.2 Composable Sampling Framework roadmap: 2026-02-21*
