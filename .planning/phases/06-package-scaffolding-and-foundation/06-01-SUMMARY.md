---
phase: 06-package-scaffolding-and-foundation
plan: 01
subsystem: data
tags: [pyproject, hatchling, uv-dynamic-versioning, typing, monorepo, workspace]

# Dependency graph
requires:
  - phase: 02-package-structure
    provides: "Monorepo workspace layout with packages/ directory and viscy-transforms template"
provides:
  - "Installable viscy-data package skeleton with pyproject.toml"
  - "All type definitions (Sample, NormMeta, ChannelMap, etc.) importable from viscy_data"
  - "Optional dependency groups: triplet, livecell, mmap, all"
  - "PEP 561 py.typed marker for type checking support"
  - "INDEX_COLUMNS constant extracted from triplet.py"
affects: [06-02, 07-dataset-migration, 08-datamodule-migration]

# Tech tracking
tech-stack:
  added: [iohub, imageio, lightning, monai, zarr, pandas, tensorstore, pycocotools, tifffile, torchvision, tensordict]
  patterns: [src-layout package with _typing.py private module and __init__.py re-exports]

key-files:
  created:
    - packages/viscy-data/pyproject.toml
    - packages/viscy-data/src/viscy_data/__init__.py
    - packages/viscy-data/src/viscy_data/_typing.py
    - packages/viscy-data/src/viscy_data/py.typed
    - packages/viscy-data/tests/__init__.py
    - packages/viscy-data/README.md
  modified:
    - pyproject.toml

key-decisions:
  - "Updated typing_extensions.NotRequired to typing.NotRequired since requires-python >=3.11"
  - "Created README.md for viscy-data (required by hatchling build, not in original plan)"

patterns-established:
  - "viscy-data follows same src-layout as viscy-transforms: packages/viscy-data/src/viscy_data/"
  - "Type definitions in _typing.py (private), re-exported from __init__.py (public API)"
  - "Optional dependency groups for feature-gated extras (triplet, livecell, mmap, all)"

# Metrics
duration: 4min
completed: 2026-02-13
---

# Phase 6 Plan 1: Package Scaffolding Summary

**Installable viscy-data package with hatchling build, all type definitions from viscy/data/typing.py, INDEX_COLUMNS from triplet.py, and four optional dependency groups**

## Performance

- **Duration:** 3 min 47 sec
- **Started:** 2026-02-13T23:47:33Z
- **Completed:** 2026-02-13T23:51:20Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments
- Created viscy-data package skeleton with pyproject.toml, hatchling build system, and uv-dynamic-versioning
- Migrated all type definitions (Sample, NormMeta, ChannelMap, HCSStackIndex, DictTransform, etc.) into _typing.py with INDEX_COLUMNS from triplet.py
- Registered viscy-data as workspace dependency in root pyproject.toml with editable install verified
- All imports work: `from viscy_data import Sample, NormMeta` succeeds

## Task Commits

Each task was committed atomically:

1. **Task 1: Create package directory structure with pyproject.toml** - `47d8f2d` (feat)
2. **Task 2: Create _typing.py with all type definitions and __init__.py with re-exports** - `9eefb8c` (feat)
3. **Task 3: Update root pyproject.toml and verify editable install** - `f45db24` (feat)

## Files Created/Modified
- `packages/viscy-data/pyproject.toml` - Build config with hatchling, deps, optional extras, versioning
- `packages/viscy-data/src/viscy_data/__init__.py` - Package entry point re-exporting all public types
- `packages/viscy-data/src/viscy_data/_typing.py` - All type definitions plus INDEX_COLUMNS
- `packages/viscy-data/src/viscy_data/py.typed` - PEP 561 type checking marker
- `packages/viscy-data/tests/__init__.py` - Test directory initialization
- `packages/viscy-data/README.md` - Minimal package readme (required by hatchling)
- `pyproject.toml` - Root workspace updated with viscy-data source and dependency

## Decisions Made
- Updated `typing_extensions.NotRequired` to `typing.NotRequired` since the package requires Python >=3.11 where NotRequired is available in stdlib
- Created `README.md` for viscy-data package (not in original plan, but required by hatchling build system which references `readme = "README.md"` in pyproject.toml)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created README.md for hatchling build**
- **Found during:** Task 2 (editable install verification)
- **Issue:** pyproject.toml declares `readme = "README.md"` but file did not exist, causing hatchling build to fail with `OSError: Readme file does not exist: README.md`
- **Fix:** Created minimal `packages/viscy-data/README.md` with package description
- **Files modified:** packages/viscy-data/README.md
- **Verification:** `uv pip install -e packages/viscy-data` succeeds after fix
- **Committed in:** 9eefb8c (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Required for build system to function. No scope creep.

## Issues Encountered
- Pre-commit `pyproject-fmt` hook reformatted pyproject.toml on first commit (spacing normalization, alphabetical sorting of optional-deps). Re-staged and committed successfully.
- Pre-commit `ruff check` hook reordered imports in `__init__.py` (isort). Re-staged and committed successfully.
- `uv sync` failed twice due to stale `__pycache__` directories (matplotlib, scipy). Cleared and retried successfully.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- viscy-data package is installable and all types are importable
- Ready for Plan 06-02 (utility module migration) and subsequent dataset/datamodule migration
- Test infrastructure ready with tests/ directory and pytest in dependency-groups

## Self-Check: PASSED

All 7 files found. All 3 task commits verified.

---
*Phase: 06-package-scaffolding-and-foundation*
*Completed: 2026-02-13*
