---
phase: 07-code-migration
plan: 02
subsystem: data
tags: [pytorch, lightning, triplet, classification, contrastive-learning, monai, pandas, tensorstore]

# Dependency graph
requires:
  - phase: 07-code-migration
    plan: 01
    provides: "Core data modules (hcs.py, select.py) and utility modules (_typing.py, _utils.py)"
provides:
  - "triplet.py: TripletDataset, TripletDataModule with CenterSpatialCropd (DATA-PKG-03)"
  - "cell_classification.py: ClassificationDataset, ClassificationDataModule"
  - "cell_division_triplet.py: CellDivisionTripletDataset, CellDivisionTripletDataModule"
affects: [07-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "BatchedCenterSpatialCropd replaced with MONAI CenterSpatialCropd (DATA-PKG-03)"
    - "Lazy optional dependency imports with try/except and ImportError guards in __init__"
    - "Type annotations for optional deps use string literals to avoid import-time failures"

key-files:
  created:
    - packages/viscy-data/src/viscy_data/triplet.py
    - packages/viscy-data/src/viscy_data/cell_classification.py
    - packages/viscy-data/src/viscy_data/cell_division_triplet.py
  modified: []

key-decisions:
  - "Removed DictTransform import from triplet.py (unused after utility extraction to _utils.py)"
  - "Added noqa PD013 for ts.stack() call (tensorstore method, not pandas)"
  - "Used string-literal type annotations for pandas/tensorstore types to avoid import-time errors"

patterns-established:
  - "Lazy import pattern: try/import/except at module level, guard in __init__ with pip install hint"
  - "DATA-PKG-03: CenterSpatialCropd from MONAI replaces BatchedCenterSpatialCropd from viscy.transforms"

# Metrics
duration: 6min
completed: 2026-02-14
---

# Phase 7 Plan 2: Specialized Module Migration Summary

**Triplet, classification, and cell division modules migrated with BatchedCenterSpatialCropd replaced by MONAI CenterSpatialCropd and lazy pandas/tensorstore imports**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-14T01:03:54Z
- **Completed:** 2026-02-14T01:10:05Z
- **Tasks:** 2
- **Files created:** 3

## Accomplishments
- Migrated triplet.py with the critical DATA-PKG-03 change: BatchedCenterSpatialCropd fully removed and replaced with MONAI CenterSpatialCropd
- Added lazy imports for pandas and tensorstore in triplet.py with clear error messages pointing to pip install extras
- Added lazy pandas import in cell_classification.py with import guard in ClassificationDataset.__init__
- Migrated cell_division_triplet.py with imports rewired from viscy.data to viscy_data prefix
- Zero references to viscy.data, viscy.transforms, or BatchedCenterSpatialCropd across all 3 files
- All files pass ruff check and ruff format

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate triplet.py with BatchedCenterSpatialCropd removal** - `a05c53d` (feat)
2. **Task 2: Migrate cell_classification.py and cell_division_triplet.py** - `97cb1e3` (feat)

## Files Created
- `packages/viscy-data/src/viscy_data/triplet.py` - TripletDataset and TripletDataModule with CenterSpatialCropd, lazy pandas/tensorstore imports
- `packages/viscy-data/src/viscy_data/cell_classification.py` - ClassificationDataset and ClassificationDataModule with lazy pandas import
- `packages/viscy-data/src/viscy_data/cell_division_triplet.py` - CellDivisionTripletDataset and CellDivisionTripletDataModule for npy-based cell division tracks

## Decisions Made
- Removed unused `DictTransform` import from triplet.py since it was only used by the utility functions now in `_utils.py`
- Added `noqa: PD013` to `ts.stack()` call since ruff incorrectly flags tensorstore's stack method as pandas `.stack()`
- Used string-literal type annotations (e.g., `"pd.DataFrame"`) for optional dependency types to avoid import-time failures when pandas/tensorstore are not installed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed unused DictTransform import**
- **Found during:** Task 1
- **Issue:** `DictTransform` was imported but unused after extracting utility functions to `_utils.py`
- **Fix:** Removed from import statement
- **Files modified:** packages/viscy-data/src/viscy_data/triplet.py
- **Verification:** ruff check passes

**2. [Rule 1 - Bug] Added noqa for tensorstore ts.stack() false positive**
- **Found during:** Task 1
- **Issue:** ruff PD013 rule flagged `ts.stack()` thinking it was pandas `.stack()`, but it is tensorstore's stack method
- **Fix:** Added `# noqa: PD013` inline comment
- **Files modified:** packages/viscy-data/src/viscy_data/triplet.py
- **Verification:** ruff check passes

**3. [Rule 2 - Missing Critical] Added docstrings for ruff D compliance**
- **Found during:** Tasks 1-2
- **Issue:** Original source code lacked docstrings on some public methods; ruff D rules enforced by pre-commit hooks
- **Fix:** Added minimal docstrings to all public classes and methods
- **Files modified:** All 3 migrated files
- **Verification:** ruff check passes

---

**Total deviations:** 3 auto-fixed (2 bug, 1 missing critical)
**Impact on plan:** All auto-fixes necessary for linting compliance. No scope creep.

## Issues Encountered
- NumPy version incompatibility in the HPC environment (NumPy 2.4.2 vs packages compiled for NumPy 1.x) prevented runtime import verification. Used AST-based parsing as alternative verification method, consistent with 07-01 approach.
- Task 1 (triplet.py) was already committed as part of a previous 07-03 execution (commit a05c53d) due to out-of-order plan execution. Verified the existing file matched plan requirements and skipped re-committing.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 3 specialized modules are in place and ready for Wave 3 (07-04: __init__.py and public API)
- The DATA-PKG-03 requirement (removing viscy-transforms dependency) is fully satisfied
- Combined with 07-01 and 07-03 modules, the full viscy-data package module set is nearly complete

## Self-Check: PASSED

- All 3 created files verified on disk
- All 2 task commits verified in git log (a05c53d, 97cb1e3)

---
*Phase: 07-code-migration*
*Completed: 2026-02-14*
