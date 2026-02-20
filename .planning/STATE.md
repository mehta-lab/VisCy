# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-19)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** Phase 18 — Training Validation (v2.1)

## Current Position

Phase: 18 of 19 (Training Validation)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-02-19 — Roadmap created for v2.1 DynaCLR Integration Validation

Progress: [==================░░] 90% (17/19 phases complete)

## Performance Metrics

**Combined velocity (all branches):**
- Total plans completed: 25 (v1.0: 7, v1.1: 9, v1.2: 9) + v2.0 manual phases

**By Milestone:**

| Milestone | Phases | Plans | Branch |
|-----------|--------|-------|--------|
| v1.0 Transforms | 1-5 | 7 | shared |
| v1.1 Data | 6-9 | 9 | modular-data |
| v1.2 Models | 10-14 | 9 | modular-models |
| v2.0 DynaCLR | 15-17 | manual | app-dynaclr |
| v2.1 Validation | 18-19 | TBD | app-dynaclr |

## Accumulated Context

### Decisions

Key decisions carrying forward:

- Clean break on imports: `from viscy_{pkg} import X` (no backward compatibility)
- Applications compose packages: dynacrl depends on viscy-data, viscy-models, viscy-transforms, viscy-utils
- State dict key compatibility non-negotiable for checkpoint loading
- YAML config class_path references: dynacrl.engine, viscy_models, viscy_data, viscy_transforms
- Tests inside packages: `applications/dynacrl/tests/`, runnable via `uv run --package dynacrl pytest`

### Blockers/Concerns

- Checkpoint + reference output paths needed from user during Phase 19 implementation
- fast_dev_run requires synthetic or small real data accessible in test environment

## Session Continuity

Last session: 2026-02-19
Stopped at: Roadmap created for v2.1, ready to plan Phase 18
Resume file: None

---
*State initialized: 2025-01-27*
*Updated for v2.1 DynaCLR Integration Validation: 2026-02-19*
