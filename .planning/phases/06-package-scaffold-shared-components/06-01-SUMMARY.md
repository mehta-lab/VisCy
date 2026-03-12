---
phase: 06-package-scaffold-shared-components
plan: 01
subsystem: infra
tags: [uv-workspace, hatchling, torch, timm, monai, package-scaffold]

# Dependency graph
requires:
  - phase: 02-package-scaffold
    provides: "Workspace layout pattern (packages/*/src) and viscy-transforms example"
provides:
  - "Installable viscy-models package in uv workspace"
  - "Subpackage structure: _components, unet, unet/_layers, contrastive, vae"
  - "Test scaffolding with device fixture"
  - "PEP 561 py.typed marker for type checking"
affects: [06-02, 06-03, 07, 08, 09, 10]

# Tech tracking
tech-stack:
  added: [timm>=1.0.15]
  patterns: [src-layout, uv-dynamic-versioning, hatchling-build]

key-files:
  created:
    - packages/viscy-models/pyproject.toml
    - packages/viscy-models/src/viscy_models/__init__.py
    - packages/viscy-models/src/viscy_models/py.typed
    - packages/viscy-models/src/viscy_models/_components/__init__.py
    - packages/viscy-models/src/viscy_models/unet/__init__.py
    - packages/viscy-models/src/viscy_models/unet/_layers/__init__.py
    - packages/viscy-models/src/viscy_models/contrastive/__init__.py
    - packages/viscy-models/src/viscy_models/vae/__init__.py
    - packages/viscy-models/tests/conftest.py
  modified:
    - pyproject.toml
    - uv.lock

key-decisions:
  - "Followed viscy-transforms pattern exactly for consistency"
  - "No optional-dependencies for viscy-models (no notebook extras needed)"
  - "Dev dependency group includes only test (no jupyter for models package)"

patterns-established:
  - "Package scaffold pattern: pyproject.toml + src layout + py.typed + test fixtures"
  - "Device fixture in conftest.py for GPU-aware testing"

# Metrics
duration: 2min
completed: 2026-02-13
---

# Phase 6 Plan 1: Package Scaffold Summary

**Installable viscy-models package with src layout, torch/timm/monai/numpy deps, and subpackage structure for unet/contrastive/vae architectures**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-13T00:32:57Z
- **Completed:** 2026-02-13T00:35:09Z
- **Tasks:** 2
- **Files modified:** 15

## Accomplishments
- Created viscy-models package scaffold with 5 subpackages (_components, unet, unet/_layers, contrastive, vae)
- Registered package in uv workspace with proper dependency resolution (timm added to lockfile)
- Verified installation, import, and test runner all work correctly

## Task Commits

Each task was committed atomically:

1. **Task 1: Create viscy-models package directory structure and build config** - `9f2044f` (feat)
2. **Task 2: Register viscy-models in workspace and verify installation** - `acd56b7` (feat)

## Files Created/Modified
- `packages/viscy-models/pyproject.toml` - Build config with hatchling, torch/timm/monai/numpy deps
- `packages/viscy-models/README.md` - Brief package description
- `packages/viscy-models/src/viscy_models/__init__.py` - Package entry with version from importlib.metadata
- `packages/viscy-models/src/viscy_models/py.typed` - PEP 561 type marker
- `packages/viscy-models/src/viscy_models/_components/__init__.py` - Shared components subpackage
- `packages/viscy-models/src/viscy_models/unet/__init__.py` - UNet architectures subpackage
- `packages/viscy-models/src/viscy_models/unet/_layers/__init__.py` - UNet layers subpackage
- `packages/viscy-models/src/viscy_models/contrastive/__init__.py` - Contrastive learning subpackage
- `packages/viscy-models/src/viscy_models/vae/__init__.py` - VAE architectures subpackage
- `packages/viscy-models/tests/__init__.py` - Test package marker
- `packages/viscy-models/tests/conftest.py` - Device fixture (cuda/cpu)
- `packages/viscy-models/tests/test_components/__init__.py` - Components test subpackage
- `packages/viscy-models/tests/test_unet/__init__.py` - UNet test subpackage
- `pyproject.toml` - Added viscy-models to root deps and uv sources
- `uv.lock` - Updated lockfile with timm and viscy-models

## Decisions Made
- Followed viscy-transforms pyproject.toml pattern exactly for workspace consistency
- No optional-dependencies section (viscy-models has no notebook extras)
- Dev dependency group includes only test group (no jupyter needed for a models package)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Package scaffold is ready for Plan 02 (shared components extraction) and Plan 03 (UNet migration)
- All subpackages created and empty, awaiting module code population
- Test infrastructure in place with device fixture for GPU-aware testing

## Self-Check: PASSED

All 13 package files verified present. Both task commits (9f2044f, acd56b7) verified in git log.

---
*Phase: 06-package-scaffold-shared-components*
*Completed: 2026-02-13*
