# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** Milestone v2.2 — Composable Sampling Framework

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-02-21 — Milestone v2.2 started

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
| v2.1 Validation | 18-19 | 2 done | app-dynaclr |

## Accumulated Context

### Decisions

Key decisions carrying forward:

- Clean break on imports: `from viscy_{pkg} import X` (no backward compatibility)
- Applications compose packages: dynaclr depends on viscy-data, viscy-models, viscy-transforms, viscy-utils
- State dict key compatibility non-negotiable for checkpoint loading
- YAML config class_path references: dynaclr.engine, viscy_models, viscy_data, viscy_transforms
- Tests inside packages: `applications/dynaclr/tests/`, runnable via `uv run --package dynaclr pytest`
- TensorBoardLogger with tmp_path for integration tests instead of logger=False (exercises full logging pipeline)
- Workspace exclude needed for non-package application directories (benchmarking, contrastive_phenotyping, qc)
- Synthetic data shape (1,1,4,4) required for render_images compatibility in tests
- GPU tolerance: atol=0.02, rtol=1e-2 with Pearson r>0.999 for cross-environment reproducibility
- Lazy imports in EmbeddingWriter to avoid hard umap/phate/sklearn dependency for basic prediction

### Blockers/Concerns

- All blockers resolved. v2.1 milestone complete.

## Session Continuity

Last session: 2026-02-21
Stopped at: Defining requirements for v2.2 Composable Sampling Framework
Resume file: None

---
*State initialized: 2025-01-27*
*Updated for v2.2 Composable Sampling Framework: 2026-02-21*
