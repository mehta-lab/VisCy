# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-12)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** Not started (defining requirements)

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-02-12 — Milestone v1.1 started

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Clean break on imports: `from viscy_models import X` (no backward compatibility)
- hatchling over setuptools: Modern build system with plugin support
- Root package is `viscy` umbrella with `package=true` (installable)
- `viscy` re-exports from subpackages, has dynamic versioning from git tags
- Use prek instead of pre-commit for faster hook execution
- ty type checker removed (too many false positives with MONAI)
- uv-dynamic-versioning verified working with pattern-prefix for monorepo
- ruff per-file-ignores updated for monorepo pattern (**/tests/**)
- alls-green pattern for single status check in branch protection
- Registry metaclass for Hydra: models self-register by name
- Architectures only in viscy-models: training logic goes to applications
- viscy-models independent of viscy-transforms

### Blockers/Concerns

None currently.

## v1.0 Completion Summary

All 5 phases complete (Phase 4 Documentation deferred). See MILESTONES.md.

## Session Continuity

Last session: 2026-02-12
Stopped at: Defining requirements for milestone v1.1
Resume file: None

---
*State initialized: 2025-01-27*
*Last updated: 2026-02-12*
