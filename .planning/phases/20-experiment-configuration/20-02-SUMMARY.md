---
phase: 20-experiment-configuration
plan: 02
subsystem: data
tags: [pyproject, dependencies, iohub, pyyaml, yaml-config, public-api, experiment-config]

# Dependency graph
requires:
  - phase: 20-01
    provides: ExperimentConfig and ExperimentRegistry dataclasses in dynaclr.experiment
provides:
  - Top-level imports of ExperimentConfig and ExperimentRegistry from dynaclr package
  - Explicit iohub and pyyaml dependencies in dynaclr pyproject.toml
  - Example experiments.yml demonstrating multi-experiment YAML config structure
affects: [21-cell-index-builder, 22-flexible-batch-sampler, 23-dataset-construction, 24-datamodule-assembly]

# Tech tracking
tech-stack:
  added: []
  patterns: [top-level re-exports for public API, example configs as documentation]

key-files:
  created:
    - applications/dynaclr/examples/configs/experiments.yml
  modified:
    - applications/dynaclr/pyproject.toml
    - applications/dynaclr/src/dynaclr/__init__.py

key-decisions:
  - "Explicit iohub/pyyaml deps even though transitive via viscy-utils (dynaclr.experiment imports both directly)"
  - "Alphabetical ordering in dependencies list and __all__ for consistency"

patterns-established:
  - "Top-level re-exports: public API classes exported via __init__.py for `from dynaclr import X`"
  - "Example configs: YAML reference files in examples/configs/ with inline comments"

# Metrics
duration: 2min
completed: 2026-02-22
---

# Phase 20 Plan 02: Package Wiring and Example Config Summary

**Top-level dynaclr imports for ExperimentConfig/Registry, explicit iohub+pyyaml deps, and example multi-experiment YAML config with positional channel alignment**

## Performance

- **Duration:** 2 min 16s
- **Started:** 2026-02-22T05:04:19Z
- **Completed:** 2026-02-22T05:06:35Z
- **Tasks:** 2
- **Files created:** 1
- **Files modified:** 2

## Accomplishments
- ExperimentConfig and ExperimentRegistry now importable from top-level `dynaclr` package (`from dynaclr import ExperimentConfig, ExperimentRegistry`)
- Explicit iohub>=0.3a2 and pyyaml dependencies in dynaclr pyproject.toml (were previously transitive only)
- Example experiments.yml with 2 experiments demonstrating positional channel alignment, different interval_minutes (30 vs 15), multiple conditions (infected/uninfected/mock), and detailed inline comments

## Task Commits

Each task was committed atomically:

1. **Task 1: Add explicit dependencies and update public API** - `3ca1ebb` (feat) - pyproject.toml deps + __init__.py re-exports
2. **Task 2: Create example experiments YAML configuration** - `3e68cc1` (feat) - experiments.yml with 2 experiments

## Files Created/Modified
- `applications/dynaclr/pyproject.toml` - Added iohub>=0.3a2 and pyyaml to dependencies list
- `applications/dynaclr/src/dynaclr/__init__.py` - Re-exports ExperimentConfig and ExperimentRegistry, added to __all__
- `applications/dynaclr/examples/configs/experiments.yml` - Example multi-experiment YAML config (64 lines) with SEC61 (ER, 30min) and TOMM20 (mito, 15min) experiments

## Decisions Made
- Added iohub and pyyaml as explicit dependencies even though they are transitive via viscy-utils, because dynaclr.experiment imports from both directly -- explicit is better than implicit
- Maintained alphabetical ordering in both the dependencies list and __all__ for consistency with project conventions

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 20 (Experiment Configuration) is now fully complete
- `from dynaclr import ExperimentConfig, ExperimentRegistry` works for all downstream phases
- Example experiments.yml provides reference for Phase 24 (DataModule assembly) and Phase 25 (integration tests)
- Phase 21 (Cell Index Builder) can proceed with ExperimentRegistry as input

## Self-Check: PASSED

- All files exist (pyproject.toml, __init__.py, experiments.yml, SUMMARY.md)
- All 2 commits verified (3ca1ebb, 3e68cc1)
- Key content verified: iohub in pyproject.toml, pyyaml in pyproject.toml, ExperimentConfig in __init__.py
- Key link verified: `from dynaclr.experiment import ExperimentConfig, ExperimentRegistry` in __init__.py
- Min lines met: experiments.yml=64 (>=20)

---
*Phase: 20-experiment-configuration*
*Completed: 2026-02-22*
