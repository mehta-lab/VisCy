# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-13)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** Phase 7 -- Code Migration

## Current Position

Phase: 7 of 9 (Code Migration)
Plan: 1 of 4 in current phase (07-01 complete)
Status: In Progress
Last activity: 2026-02-14 -- Completed 07-01 (Core Data Module Migration)

Progress: [========..] 82% (v1.0 complete, phase 7 plan 1 of 4)

## Performance Metrics

**Velocity (from v1.0):**
- Total plans completed: 9
- Average duration: 4.7 min
- Total execution time: 42 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 2 | ~8 min | ~4 min |
| 2 | 1 | ~4 min | ~4 min |
| 3 | 3 | ~13 min | ~4.3 min |
| 5 | 1 | ~4 min | ~4 min |
| 6 | 2 | ~7 min | ~3.5 min |
| 7 | 1 | ~9 min | ~9 min |

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

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-14
Stopped at: Completed 07-01-PLAN.md (Core Data Module Migration)
Resume file: None

---
*State initialized: 2025-01-27*
*Last updated: 2026-02-14 (07-01 complete, Phase 7 in progress)*
