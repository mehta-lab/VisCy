# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-13)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** Phase 6 -- Package Scaffolding and Foundation

## Current Position

Phase: 6 of 9 (Package Scaffolding and Foundation) -- COMPLETE
Plan: 2 of 2 in current phase (all plans complete)
Status: Phase Complete
Last activity: 2026-02-13 -- Completed 06-02 (Utility Module Extraction)

Progress: [========..] 80% (v1.0 complete, v1.1 phase 6 complete)

## Performance Metrics

**Velocity (from v1.0):**
- Total plans completed: 8
- Average duration: 4.1 min
- Total execution time: 33 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 2 | ~8 min | ~4 min |
| 2 | 1 | ~4 min | ~4 min |
| 3 | 3 | ~13 min | ~4.3 min |
| 5 | 1 | ~4 min | ~4 min |
| 6 | 2 | ~7 min | ~3.5 min |

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

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-13
Stopped at: Completed 06-02-PLAN.md (Utility Module Extraction) -- Phase 6 complete
Resume file: None

---
*State initialized: 2025-01-27*
*Last updated: 2026-02-13 (06-02 complete, Phase 6 complete)*
