---
phase: 07-code-migration
plan: 03
subsystem: data
tags: [pytorch, lightning, datamodule, tensordict, pycocotools, tifffile, torchvision, lazy-import]

# Dependency graph
requires:
  - phase: 07-code-migration
    provides: "gpu_aug.py (GPUTransformDataModule, CachedOmeZarrDataset), select.py, distributed.py, _utils.py, _typing.py"
provides:
  - "mmap_cache.py: MmappedDataset, MmappedDataModule (memory-mapped tensor caching)"
  - "ctmc_v1.py: CTMCv1DataModule (CTMCv1 autoregression dataset)"
  - "livecell.py: LiveCellDataset, LiveCellTestDataset, LiveCellDataModule (LiveCell instance segmentation)"
  - "combined.py: CombinedDataModule, CombineMode, ConcatDataModule, BatchedConcatDataModule, CachedConcatDataModule, BatchedConcatDataset"
affects: [07-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy optional dependency import with try/except and None sentinel"
    - "ImportError guard in __init__ with pip install hint for extras group"

key-files:
  created:
    - packages/viscy-data/src/viscy_data/mmap_cache.py
    - packages/viscy-data/src/viscy_data/ctmc_v1.py
    - packages/viscy-data/src/viscy_data/livecell.py
    - packages/viscy-data/src/viscy_data/combined.py
  modified: []

key-decisions:
  - "Lazy import pattern: try/except at module level with None fallback, guard in __init__ with clear pip install message"
  - "combined.py preserved as-is (no split into combined.py + concat.py per REF-02 deferral)"
  - "LiveCellTestDataset also gets lazy import guard (not just LiveCellDataset) since it uses COCO and imread"

patterns-established:
  - "Lazy optional dependency: try/except ImportError at top, ClassName = None fallback, guard check in __init__ with extras hint"
  - "Import rewiring: viscy.data.X -> viscy_data.X for all internal references"

# Metrics
duration: 5min
completed: 2026-02-14
---

# Phase 7 Plan 3: Optional Dependency Module Migration Summary

**4 data modules (mmap_cache, ctmc_v1, livecell, combined) migrated with lazy imports for tensordict, pycocotools, tifffile, and torchvision optional dependencies**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-14T01:03:58Z
- **Completed:** 2026-02-14T01:08:44Z
- **Tasks:** 3
- **Files created:** 4

## Accomplishments
- Migrated 4 data modules into packages/viscy-data/src/viscy_data/
- Implemented lazy import pattern for 4 optional dependencies (tensordict, pycocotools, tifffile, torchvision) with clear error messages pointing to extras groups
- Rewired all internal imports from viscy.data.X to viscy_data.X prefix
- Preserved combined.py as-is (6 public classes, no structural refactoring per REF-02 deferral)
- All modules pass ruff check and ruff format with full D-series docstring enforcement

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate mmap_cache.py and ctmc_v1.py** - `924386b` (feat)
2. **Task 2: Migrate livecell.py with lazy imports** - `a05c53d` (feat)
3. **Task 3: Migrate combined.py as-is** - `8ddfee2` (feat)

## Files Created
- `packages/viscy-data/src/viscy_data/mmap_cache.py` - MmappedDataset and MmappedDataModule with lazy tensordict import
- `packages/viscy-data/src/viscy_data/ctmc_v1.py` - CTMCv1DataModule for autoregression on CTMCv1 dataset
- `packages/viscy-data/src/viscy_data/livecell.py` - LiveCellDataset, LiveCellTestDataset, LiveCellDataModule with lazy pycocotools/tifffile/torchvision imports
- `packages/viscy-data/src/viscy_data/combined.py` - CombinedDataModule, CombineMode, ConcatDataModule, BatchedConcatDataModule, CachedConcatDataModule, BatchedConcatDataset

## Decisions Made
- Lazy import pattern uses try/except at module level setting sentinel to None, with guard check in __init__ raising ImportError with pip install hint for the appropriate extras group
- combined.py preserved as single file (not split into combined.py + concat.py) per scope constraints and REF-02 deferral
- LiveCellTestDataset also gets the lazy import guard since it directly uses COCO and imread

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added docstrings for ruff D compliance**
- **Found during:** Tasks 1-3 (pre-commit hook enforcement)
- **Issue:** Original source code lacked docstrings on several public classes/methods; ruff D rules enforced by pre-commit hooks blocked commits
- **Fix:** Added minimal docstrings to all public classes and methods
- **Files modified:** All 4 migrated files
- **Verification:** ruff check passes, pre-commit hooks pass

**2. [Rule 3 - Blocking] triplet.py included in Task 2 commit due to pre-commit stash conflict**
- **Found during:** Task 2 (pre-commit hook execution)
- **Issue:** A pre-existing unstaged triplet.py file in the working directory caused a pre-commit stash/unstash conflict, resulting in it being included in the Task 2 commit
- **Fix:** File was already a valid migration artifact (part of broader phase 7 scope); no corrective action needed
- **Files modified:** packages/viscy-data/src/viscy_data/triplet.py (unplanned inclusion)
- **Verification:** File passes ruff check and is a valid module

---

**Total deviations:** 2 auto-fixed (1 missing critical, 1 blocking)
**Impact on plan:** Docstring additions necessary for CI compliance. triplet.py inclusion is harmless (valid migration file from phase 7 scope).

## Issues Encountered
- NumPy version incompatibility in the HPC environment (NumPy 2.4.2 vs packages compiled for NumPy 1.x) prevented runtime import verification. Used AST-based parsing as alternative verification method. All modules parse correctly with expected class/function definitions.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 4 optional-dependency modules are in place and ready for Plan 07-04
- The lazy import pattern is established and can be reused for any future optional dependency modules
- combined.py ready for future REF-02 refactoring when scope permits

## Self-Check: PASSED

- All 4 created files verified on disk
- All 3 task commits verified in git log (924386b, a05c53d, 8ddfee2)

---
*Phase: 07-code-migration*
*Completed: 2026-02-14*
