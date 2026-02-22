---
phase: 20-experiment-configuration
plan: 01
subsystem: data
tags: [dataclass, yaml, iohub, ome-zarr, channel-mapping, experiment-config]

# Dependency graph
requires: []
provides:
  - ExperimentConfig dataclass for per-experiment metadata
  - ExperimentRegistry with fail-fast validation and channel_maps
  - from_yaml classmethod for YAML config loading
  - tau_range_frames for hours-to-frames conversion
  - get_experiment lookup by name
affects: [21-cell-index-builder, 22-flexible-batch-sampler, 23-dataset-construction, 24-datamodule-assembly, 25-ntxent-hcl-loss]

# Tech tracking
tech-stack:
  added: []
  patterns: [dataclass-based config, fail-fast validation at __post_init__, iohub zarr metadata reading]

key-files:
  created:
    - applications/dynaclr/src/dynaclr/experiment.py
    - applications/dynaclr/tests/test_experiment.py
  modified:
    - pyproject.toml

key-decisions:
  - "Used plain dataclass (not pydantic) per project convention"
  - "Validation concentrated in ExperimentRegistry.__post_init__, not ExperimentConfig"
  - "Positional alignment for source channels across experiments (names can differ, count must match)"
  - "Excluded stale applications/dynacrl (typo) from uv workspace"

patterns-established:
  - "ExperimentConfig: pure data container with no validation logic"
  - "ExperimentRegistry: fail-fast validation at creation, channel_maps computed post-validation"
  - "iohub open_ome_zarr pattern for zarr channel metadata reading"

# Metrics
duration: 4min
completed: 2026-02-21
---

# Phase 20 Plan 01: ExperimentConfig and ExperimentRegistry Summary

**Dataclass-based ExperimentConfig and ExperimentRegistry with fail-fast validation, iohub zarr channel verification, YAML loading, and tau-range conversion via TDD (19 tests)**

## Performance

- **Duration:** 4 min 21s
- **Started:** 2026-02-22T04:57:16Z
- **Completed:** 2026-02-22T05:01:37Z
- **Tasks:** 3 (TDD: RED, GREEN, REFACTOR)
- **Files created:** 2
- **Files modified:** 1

## Accomplishments
- ExperimentConfig dataclass with 11 fields (6 required, 5 optional with defaults) for per-experiment metadata
- ExperimentRegistry with 8 validation rules in __post_init__: empty check, duplicate names, source_channel membership, channel count consistency, positive interval_minutes, non-empty condition_wells, data_path existence, zarr channel match
- channel_maps computation: per-experiment mapping of source position index to zarr channel index
- from_yaml classmethod for YAML config loading
- tau_range_frames conversion from hours to frames per-experiment using interval_minutes, with warning on degenerate ranges
- 19 passing tests with full coverage of all validation rules and public API

## Task Commits

Each task was committed atomically (TDD):

1. **RED: Failing tests** - `142b1a4` (test) - 19 test cases, all fail with ModuleNotFoundError
2. **GREEN: Implementation** - `8bda967` (feat) - experiment.py with ExperimentConfig + ExperimentRegistry, all 19 tests pass
3. **REFACTOR: Cleanup** - `4f2d772` (refactor) - Fix ruff lint issues (import sorting, unused import), exclude dynacrl from workspace

## Files Created/Modified
- `applications/dynaclr/src/dynaclr/experiment.py` - ExperimentConfig and ExperimentRegistry dataclasses (291 lines)
- `applications/dynaclr/tests/test_experiment.py` - Comprehensive test suite with 19 tests (304 lines)
- `pyproject.toml` - Exclude stale `applications/dynacrl` (typo) from uv workspace

## Decisions Made
- Used plain dataclass (not pydantic) per project convention established in prior phases
- Validation concentrated in ExperimentRegistry.__post_init__ rather than ExperimentConfig -- config is a pure data container, registry validates the ensemble
- Positional alignment for source channels across experiments: names can differ between experiments (GFP in exp A = Mito in exp B) as long as count matches
- Excluded stale applications/dynacrl (typo directory) from uv workspace to unblock builds

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Excluded stale dynacrl workspace member**
- **Found during:** Task 1 (RED phase, running tests)
- **Issue:** `applications/dynacrl` directory (typo) exists without pyproject.toml, breaking uv workspace resolution for all packages
- **Fix:** Added "applications/dynacrl" to workspace exclude list in root pyproject.toml
- **Files modified:** pyproject.toml
- **Verification:** `uv run --package dynaclr python -c "from iohub.ngff import open_ome_zarr; print('OK')"` succeeds
- **Committed in:** 4f2d772 (refactor commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential for any uv-based operation. No scope creep.

## Issues Encountered
None beyond the workspace blocking issue documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- ExperimentConfig and ExperimentRegistry are ready for downstream consumption
- Phase 20 Plan 02 can build on this for any additional experiment configuration needs
- Phase 21 (Cell Index Builder) can import ExperimentRegistry for cell indexing
- All exports available via `from dynaclr.experiment import ExperimentConfig, ExperimentRegistry`

## Self-Check: PASSED

- All files exist (experiment.py, test_experiment.py, SUMMARY.md)
- All 3 commits verified (142b1a4, 8bda967, 4f2d772)
- Module importable: `from dynaclr.experiment import ExperimentConfig, ExperimentRegistry`
- Key links verified: test imports, iohub import, yaml import
- Min lines met: experiment.py=291 (>=120), test_experiment.py=304 (>=150)

---
*Phase: 20-experiment-configuration*
*Completed: 2026-02-21*
