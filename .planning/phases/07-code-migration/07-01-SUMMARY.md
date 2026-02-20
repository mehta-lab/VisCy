---
phase: 07-code-migration
plan: 01
subsystem: data
tags: [pytorch, lightning, datamodule, hcs, monai, ome-zarr]

# Dependency graph
requires:
  - phase: 06-package-scaffolding-and-foundation
    provides: "viscy-data package skeleton with _typing.py and _utils.py"
provides:
  - "select.py: SelectWell mixin, _filter_wells, _filter_fovs"
  - "distributed.py: ShardedDistributedSampler for DDP training"
  - "segmentation.py: SegmentationDataset, SegmentationDataModule"
  - "hcs.py: HCSDataModule, SlidingWindowDataset, MaskTestDataset"
  - "gpu_aug.py: GPUTransformDataModule, CachedOmeZarrDataset, CachedOmeZarrDataModule"
affects: [07-02, 07-03, 07-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Import rewiring: viscy.data.X -> viscy_data.X"
    - "Utility import from _utils: functions shared across modules imported from viscy_data._utils"
    - "Type import from _typing: type definitions imported from viscy_data._typing"

key-files:
  created:
    - packages/viscy-data/src/viscy_data/select.py
    - packages/viscy-data/src/viscy_data/distributed.py
    - packages/viscy-data/src/viscy_data/segmentation.py
    - packages/viscy-data/src/viscy_data/hcs.py
    - packages/viscy-data/src/viscy_data/gpu_aug.py
  modified: []

key-decisions:
  - "Removed unused re and collate_meta_tensor imports from hcs.py (no longer needed after utility extraction)"
  - "Added minimal docstrings to satisfy ruff D rules enforced by pre-commit hooks"
  - "gpu_aug.py imports _ensure_channel_list and _read_norm_meta from viscy_data._utils (not from hcs.py)"

patterns-established:
  - "Import rewiring pattern: viscy.data.X -> viscy_data.X for all internal references"
  - "Utility deduplication: shared functions accessed via viscy_data._utils, not from original module"

# Metrics
duration: 9min
completed: 2026-02-14
---

# Phase 7 Plan 1: Core Data Module Migration Summary

**5 core data modules (select, distributed, segmentation, hcs, gpu_aug) migrated to viscy-data with all internal imports rewired from viscy.data to viscy_data prefix**

## Performance

- **Duration:** 9 min
- **Started:** 2026-02-14T00:52:39Z
- **Completed:** 2026-02-14T01:01:41Z
- **Tasks:** 3
- **Files created:** 5

## Accomplishments
- Migrated 5 core/standalone data modules into packages/viscy-data/src/viscy_data/
- Rewired all internal imports from viscy.data.X to viscy_data.X absolute prefix
- Removed duplicate utility function definitions from hcs.py (imported from _utils.py instead)
- All modules pass ruff check with full D-series docstring enforcement
- Zero viscy.data or viscy.transforms references across all 5 files

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate select.py, distributed.py, segmentation.py** - `d66e17b` (feat)
2. **Task 2: Migrate hcs.py with utility import rewiring** - `378d5e2` (feat)
3. **Task 3: Migrate gpu_aug.py with dependency rewiring** - `bd08483` (feat)

## Files Created
- `packages/viscy-data/src/viscy_data/select.py` - Well/FOV selection utilities: SelectWell mixin, _filter_wells, _filter_fovs
- `packages/viscy-data/src/viscy_data/distributed.py` - ShardedDistributedSampler for DDP training
- `packages/viscy-data/src/viscy_data/segmentation.py` - SegmentationDataset and SegmentationDataModule for test-stage evaluation
- `packages/viscy-data/src/viscy_data/hcs.py` - HCSDataModule, SlidingWindowDataset, MaskTestDataset (663 lines, utility functions imported from _utils)
- `packages/viscy-data/src/viscy_data/gpu_aug.py` - GPUTransformDataModule ABC, CachedOmeZarrDataset, CachedOmeZarrDataModule

## Decisions Made
- Removed unused `re` and `collate_meta_tensor` imports from hcs.py since those were only used by the 4 utility functions now in _utils.py
- Added minimal docstrings to all public classes/methods to satisfy ruff D rules enforced by pre-commit hooks (the original source lacked some)
- gpu_aug.py imports `_ensure_channel_list` and `_read_norm_meta` directly from `viscy_data._utils` rather than from `viscy_data.hcs`, matching the plan's intent to decouple utility access

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed viscy-data package in editable mode**
- **Found during:** Task 1 (import verification)
- **Issue:** viscy-data package was not installed in the Python environment, so import verification failed
- **Fix:** Ran `pip install -e packages/viscy-data` to install in editable mode
- **Files modified:** None (pip metadata only)
- **Verification:** Package installed successfully

**2. [Rule 1 - Bug] Removed unused imports from hcs.py**
- **Found during:** Task 2 (ruff check)
- **Issue:** `re` and `collate_meta_tensor` were imported but no longer used after utility function extraction to _utils.py
- **Fix:** Removed both unused imports
- **Files modified:** packages/viscy-data/src/viscy_data/hcs.py
- **Verification:** ruff check passes

**3. [Rule 2 - Missing Critical] Added docstrings for ruff D compliance**
- **Found during:** Tasks 1-3 (pre-commit hook enforcement)
- **Issue:** Original source code lacked docstrings on several public classes/methods; ruff D rules enforced by pre-commit hooks blocked commits
- **Fix:** Added minimal NumPy-style docstrings to all public classes and methods
- **Files modified:** All 5 migrated files
- **Verification:** ruff check passes, pre-commit hooks pass

---

**Total deviations:** 3 auto-fixed (1 blocking, 1 bug, 1 missing critical)
**Impact on plan:** All auto-fixes necessary for correctness and CI compliance. No scope creep.

## Issues Encountered
- NumPy version incompatibility in the HPC environment (NumPy 2.4.2 vs packages compiled for NumPy 1.x) prevented runtime import verification. Used AST-based parsing as alternative verification method. All modules parse correctly with expected class/function definitions.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 5 core modules are in place and ready for Wave 2 migration (07-02, 07-03, 07-04)
- Wave 2 modules (triplet.py, mmap.py, livecell.py) can now import from these core modules
- The import rewiring pattern is established and consistent across all files

## Self-Check: PASSED

- All 5 created files verified on disk
- All 3 task commits verified in git log (d66e17b, 378d5e2, bd08483)

---
*Phase: 07-code-migration*
*Completed: 2026-02-14*
