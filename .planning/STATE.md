# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-19)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** Phase 18 — Training Validation (v2.1) COMPLETE

## Current Position

Phase: 18 of 19 (Training Validation)
Plan: 1 of 1 in current phase (COMPLETE)
Status: Phase 18 complete, ready for Phase 19
Last activity: 2026-02-20 — Completed 18-01 training integration tests

Progress: [===================░] 95% (18/19 phases complete)

## Performance Metrics

**Combined velocity (all branches):**
- Total plans completed: 26 (v1.0: 7, v1.1: 9, v1.2: 9, v2.1: 1) + v2.0 manual phases

**By Milestone:**

| Milestone | Phases | Plans | Branch |
|-----------|--------|-------|--------|
| v1.0 Transforms | 1-5 | 7 | shared |
| v1.1 Data | 6-9 | 9 | modular-data |
| v1.2 Models | 10-14 | 9 | modular-models |
| v2.0 DynaCLR | 15-17 | manual | app-dynaclr |
| v2.1 Validation | 18-19 | 1 done | app-dynaclr |

## Accumulated Context

### Decisions

Key decisions carrying forward:

- Clean break on imports: `from viscy_{pkg} import X` (no backward compatibility)
- Applications compose packages: dynacrl depends on viscy-data, viscy-models, viscy-transforms, viscy-utils
- State dict key compatibility non-negotiable for checkpoint loading
- YAML config class_path references: dynacrl.engine, viscy_models, viscy_data, viscy_transforms
- Tests inside packages: `applications/dynacrl/tests/`, runnable via `uv run --package dynacrl pytest`
- TensorBoardLogger with tmp_path for integration tests instead of logger=False (exercises full logging pipeline)
- Workspace exclude needed for non-package application directories (benchmarking, contrastive_phenotyping, qc)
- Synthetic data shape (1,1,4,4) required for render_images compatibility in tests

### Blockers/Concerns

- Checkpoint + reference output paths needed from user during Phase 19 implementation
- fast_dev_run synthetic data pattern established and working (blocker resolved)

## Session Continuity

Last session: 2026-02-20
Stopped at: Completed 18-01-PLAN.md (training integration tests)
Resume file: None

---
*State initialized: 2025-01-27*
*Updated for v2.1 DynaCLR Integration Validation: 2026-02-20*
