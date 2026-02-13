# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-13)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** Milestone v1.1 — Extract viscy-data

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-02-13 — Milestone v1.1 started

## Performance Metrics

**Velocity (from v1.0):**
- Total plans completed: 7
- Average duration: 4.2 min
- Total execution time: 29 min

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Key decisions from v1.0 carrying forward:

- Clean break on imports: `from viscy_data import X` (no backward compatibility)
- hatchling over setuptools: Modern build system with plugin support
- Root package is `viscy` umbrella with `package=true` (installable)
- `viscy` re-exports from subpackages, has dynamic versioning from git tags
- Use prek instead of pre-commit for faster hook execution
- ty type checker removed (too many false positives with MONAI)
- ruff per-file-ignores updated for monorepo pattern (**/tests/**)
- alls-green pattern for CI branch protection

New decisions for v1.1:
- **No viscy-transforms dependency**: Remove BatchedCenterSpatialCropd from triplet.py, assert batch shape instead
- **Optional dependency groups**: tensorstore, tensordict, pycocotools as extras

### Blockers/Concerns

(None yet)

## Session Continuity

Last session: 2026-02-13
Stopped at: Starting milestone v1.1 — defining requirements
Resume file: None

---
*State initialized: 2025-01-27*
*Last updated: 2026-02-13*
