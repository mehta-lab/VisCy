---
phase: 08-test-migration-and-validation
plan: 01
subsystem: testing
tags: [pytest, ome-zarr, hcs, triplet, select, viscy-data, migration]

# Dependency graph
requires:
  - phase: 07-code-migration
    provides: "viscy_data package with all modules migrated and public API exports"
provides:
  - "conftest.py with HCS OME-Zarr fixtures for viscy-data test suite"
  - "test_hcs.py with HCSDataModule fit/predict tests using from viscy_data import"
  - "test_triplet.py with TripletDataModule/TripletDataset tests using from viscy_data import"
  - "test_select.py with SelectWell filter tests using from viscy_data import"
  - "BatchedCenterSpatialCropd in _utils.py for batch-aware spatial cropping"
affects: [08-02, testing, triplet, data-validation]

# Tech tracking
tech-stack:
  added: [tensorstore (test dep group)]
  patterns: [BatchedCenterSpatialCropd for batch-dim-aware MONAI cropping]

key-files:
  created:
    - packages/viscy-data/tests/conftest.py
    - packages/viscy-data/tests/test_hcs.py
    - packages/viscy-data/tests/test_triplet.py
    - packages/viscy-data/tests/test_select.py
  modified:
    - packages/viscy-data/src/viscy_data/_utils.py
    - packages/viscy-data/src/viscy_data/triplet.py
    - packages/viscy-data/pyproject.toml

key-decisions:
  - "Added BatchedCenterSpatialCropd to _utils.py to fix batch dimension handling in triplet crop transform"
  - "Added tensorstore to test dependency group so triplet tests can run"
  - "Replaced legacy np.random.rand with np.random.default_rng in conftest (NPY002 lint rule)"

patterns-established:
  - "BatchedCenterSpatialCropd pattern: CenterSpatialCrop subclass that operates on (B,C,*spatial) tensors by computing crop slices on shape[2:]"

# Metrics
duration: 11min
completed: 2026-02-14
---

# Phase 8 Plan 1: Data Test Migration Summary

**Migrated 3 test files (19 tests) to viscy-data package with BatchedCenterSpatialCropd fix for batch-aware spatial cropping**

## Performance

- **Duration:** 11 min
- **Started:** 2026-02-14T01:28:50Z
- **Completed:** 2026-02-14T01:39:50Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- All 19 data tests (4 HCS, 11 triplet, 4 select) pass under `from viscy_data import X`
- Shared conftest.py with 6 fixtures and `_build_hcs` helper migrated verbatim
- Fixed critical CenterSpatialCropd batch dimension bug from Phase 7 migration
- DATA-TST-01 satisfied: all existing data tests pass under new package structure

## Task Commits

Each task was committed atomically:

1. **Task 1: Create conftest.py with HCS OME-Zarr fixtures** - `819d589` (test)
2. **Task 2: Migrate test_hcs.py, test_triplet.py, test_select.py** - `ba0c499` (feat)

## Files Created/Modified
- `packages/viscy-data/tests/conftest.py` - 6 HCS OME-Zarr fixtures (preprocessed, small, labels, tracks)
- `packages/viscy-data/tests/test_hcs.py` - HCSDataModule fit/predict tests
- `packages/viscy-data/tests/test_triplet.py` - TripletDataModule/TripletDataset tests with temporal gap filtering
- `packages/viscy-data/tests/test_select.py` - SelectWell parametric filter tests
- `packages/viscy-data/src/viscy_data/_utils.py` - Added BatchedCenterSpatialCropd class
- `packages/viscy-data/src/viscy_data/triplet.py` - Switched to BatchedCenterSpatialCropd
- `packages/viscy-data/pyproject.toml` - Added tensorstore to test dep group

## Decisions Made
- Added `BatchedCenterSpatialCropd` to `_utils.py` instead of depending on `viscy.transforms` -- keeps viscy-data self-contained
- Added `tensorstore` to the test dependency group (not just the `[triplet]` extra) so tests can run without installing extras
- Replaced `np.random.rand` with `np.random.default_rng().random()` to satisfy NPY002 lint rule

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed CenterSpatialCropd batch dimension mismatch in triplet.py**
- **Found during:** Task 2 (test_triplet.py migration)
- **Issue:** Phase 7 replaced `BatchedCenterSpatialCropd` with standard MONAI `CenterSpatialCropd` (per DATA-PKG-03), but CenterSpatialCropd treats shape[1:] as spatial dimensions, failing on (B,1,Z,Y,X) tensors with "Sequence must have length 4, got 3"
- **Fix:** Implemented `BatchedCenterSpatialCropd` in `_utils.py` that computes crop slices on `img.shape[2:]`, preserving batch and channel dims
- **Files modified:** `packages/viscy-data/src/viscy_data/_utils.py`, `packages/viscy-data/src/viscy_data/triplet.py`
- **Verification:** All 11 triplet tests pass including `on_after_batch_transfer` crop assertions
- **Committed in:** ba0c499 (Task 2 commit)

**2. [Rule 3 - Blocking] Added tensorstore to test dependency group**
- **Found during:** Task 2 (test_triplet.py migration)
- **Issue:** `TripletDataset.__init__` raises ImportError when tensorstore is not installed; tensorstore was only in the `[triplet]` optional extra, not the test dep group
- **Fix:** Added `tensorstore` to `[dependency-groups] test` in pyproject.toml
- **Files modified:** `packages/viscy-data/pyproject.toml`
- **Verification:** `uv run --package viscy-data pytest` runs without ImportError
- **Committed in:** ba0c499 (Task 2 commit)

**3. [Rule 1 - Bug] Replaced legacy np.random.rand with np.random.default_rng**
- **Found during:** Task 1 (conftest.py creation)
- **Issue:** ruff NPY002 lint rule rejects `np.random.rand` (legacy NumPy random API)
- **Fix:** Changed to `np.random.default_rng().random(shape)` pattern
- **Files modified:** `packages/viscy-data/tests/conftest.py`
- **Verification:** `ruff check` passes
- **Committed in:** 819d589 (Task 1 commit)

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 blocking)
**Impact on plan:** All auto-fixes necessary for correctness and test execution. No scope creep.

## Issues Encountered
- Pre-commit hook stash conflict when unstaged files exist alongside staged files -- resolved by running ruff manually before staging
- `uv sync` with tensorstore temporarily removed `cycler` (matplotlib dependency) -- resolved by reinstalling

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All 19 data tests pass under `from viscy_data import X`
- DATA-TST-01 satisfied
- Ready for plan 08-02 (additional test validation)

## Self-Check: PASSED

- All 7 claimed files exist on disk
- Both commit hashes (819d589, ba0c499) verified in git log
- 19/19 tests pass

---
*Phase: 08-test-migration-and-validation*
*Completed: 2026-02-14*
