# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-13)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** Phase 8 -- Test Migration and Validation

## Current Position

Phase: 8 of 9 (Test Migration and Validation)
Plan: 2 of 2 in current phase (08-01, 08-02 complete)
Status: Phase Complete
Last activity: 2026-02-14 -- Completed 08-02 (Smoke Tests for Import and Public API)

Progress: [==========] 100% (phase 8 fully done)

## Performance Metrics

**Velocity (from v1.0):**
- Total plans completed: 14
- Average duration: 4.6 min
- Total execution time: 65 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 2 | ~8 min | ~4 min |
| 2 | 1 | ~4 min | ~4 min |
| 3 | 3 | ~13 min | ~4.3 min |
| 5 | 1 | ~4 min | ~4 min |
| 6 | 2 | ~7 min | ~3.5 min |
| 7 | 4 | ~22 min | ~5.5 min |
| 8 | 2 | ~21 min | ~10.5 min |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Key decisions carrying forward:

- Clean break on imports: `from viscy_data import X` (no backward compatibility)
- hatchling + uv-dynamic-versioning for build system
- No viscy-transforms dependency: assert batch shape instead of BatchedCenterSpatialCropd
- Optional dependency groups: tensorstore, tensordict, pycocotools as extras
- Extract shared utilities from hcs.py into _utils.py before migration
- Updated typing_extensions.NotRequired to typing.NotRequired (Python >=3.11 stdlib)
- Type definitions in _typing.py (private), re-exported from __init__.py (public API pattern)
- Internal utility functions accessed via `from viscy_data._utils import X` (not re-exported from __init__.py)
- Utility functions use `viscy_data._typing` for type imports (not `viscy.data.typing`)
- gpu_aug.py imports utilities from viscy_data._utils (not from hcs.py) for clean decoupling
- Removed unused imports (re, collate_meta_tensor) from hcs.py after utility extraction
- Lazy import pattern for optional deps: try/except at module level with None sentinel, guard in __init__ with pip extras hint
- combined.py preserved as-is (no split per REF-02 deferral)
- DATA-PKG-03 revised: BatchedCenterSpatialCropd added to _utils.py (CenterSpatialCropd cannot handle batch dim in triplet on_after_batch_transfer)
- String-literal type annotations for optional dep types (e.g., "pd.DataFrame") to avoid import-time failures
- Eager top-level imports in __init__.py: each module handles its own optional dep guards, so package import always succeeds
- Flat public API: all 45 names (DataModules, Datasets, types, utilities, enums) re-exported from package root
- Source inspection pattern for testing optional dep error messages: inspect.getsource() works regardless of dep installation state
- Parametrized __all__ tests: each of 45 exports as separate test case for clear reporting
- tensorstore added to test dependency group (needed for triplet tests to pass)

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-14
Stopped at: Re-executed 08-01-PLAN.md (Data Test Migration) with BatchedCenterSpatialCropd fix -- Phase 8 COMPLETE
Resume file: None

---
*State initialized: 2025-01-27*
*Last updated: 2026-02-14 (08-01 re-executed with bug fix, Phase 8 fully done)*
