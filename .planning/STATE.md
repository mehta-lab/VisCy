# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2025-01-27)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** Phase 5 - CI/CD (COMPLETE)

## Current Position

Phase: 5 of 5 (CI/CD) - COMPLETE
Plan: 1 of 1 complete
Status: Phase complete
Last activity: 2026-01-29 - Completed 05-01-PLAN.md (CI workflows)

Progress: [==========] 100% (All phases complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 4.2 min
- Total execution time: 29 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Workspace Foundation | 2/2 | 5 min | 2.5 min |
| 2. Package Structure | 1/1 | 4 min | 4 min |
| 3. Code Migration | 3/3 | 18 min | 6 min |
| 4. Documentation | 0/0 | - | - |
| 5. CI/CD | 1/1 | 2 min | 2 min |

**Recent Trend:**
- Last 5 plans: 4 min, 4 min, 8 min, 6 min, 2 min
- Trend: CI/CD was fast due to clear research findings

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Clean break on imports: `from viscy_transforms import X` (no backward compatibility)
- Clean slate approach: Wipe repo, keep only LICENSE, CITATION.cff, .gitignore
- hatchling over setuptools: Modern build system with plugin support
- Root package is `viscy` umbrella with `package=true` (installable)
- `viscy` re-exports from subpackages, has dynamic versioning from git tags
- Use prek instead of pre-commit for faster hook execution
- ty type checker removed (too many false positives with MONAI)
- Removed dependency-groups from package (root has `dev` not `test`, avoids cycle)
- uv-dynamic-versioning verified working with pattern-prefix for monorepo
- Extract only transform-relevant types (not dataset-specific types like SegmentationSample)
- Fixed _redef.py nested class bug (RandFlipd was nested inside CenterSpatialCropd)
- ruff per-file-ignores updated for monorepo pattern (**/tests/**)
- **NEW (05-01):** Matrix with fail-fast: true for quick feedback on failures
- **NEW (05-01):** alls-green pattern for single status check in branch protection
- **NEW (05-01):** Conditional cancel-in-progress: only for PRs, not main

### Blockers/Concerns

- **RESOLVED (Phase 2):** hatch-vcs tag pattern verified working via uv-dynamic-versioning pattern-prefix
- **RESOLVED:** ty type checker removed due to false positives with MONAI

## Phase 5 Completion Summary

CI/CD workflows are complete:
- `.github/workflows/test.yml` - 9-job matrix (3 OS x 3 Python) with alls-green
- `.github/workflows/lint.yml` - prek hooks + ruff format check
- Concurrency control with conditional cancel-in-progress

## Session Continuity

Last session: 2026-01-29
Stopped at: Completed 05-01-PLAN.md (CI workflows) - Phase 5 complete
Resume file: None

---
*State initialized: 2025-01-27*
*Last updated: 2026-01-29*
