---
phase: 10-public-api-ci-integration
plan: 01
subsystem: api, testing, infra
tags: [python, pytorch, public-api, ci, github-actions, regression-tests, state-dict]

# Dependency graph
requires:
  - phase: 07-core-unet-models
    provides: "UNeXt2 and FullyConvolutionalMAE in viscy-models unet subpackage"
  - phase: 08-representation-models
    provides: "ContrastiveEncoder, ResNet3dEncoder, BetaVae25D, BetaVaeMonai in contrastive/vae subpackages"
  - phase: 09-legacy-unet-models
    provides: "Unet2d and Unet25d in viscy-models unet subpackage"
provides:
  - "Top-level public API: from viscy_models import ModelName for all 8 architectures"
  - "State dict key regression tests for checkpoint compatibility (COMPAT-01)"
  - "CI test matrix covering viscy-models across 3 OS x 3 Python versions"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Top-level re-exports from subpackage __init__.py files"
    - "State dict regression tests: parameter count + prefix set + sentinel keys"
    - "Cross-platform CI coverage with --cov=src/ for src-layout packages"

key-files:
  created:
    - "packages/viscy-models/tests/test_state_dict_compat.py"
  modified:
    - "packages/viscy-models/src/viscy_models/__init__.py"
    - ".github/workflows/test.yml"

key-decisions:
  - "Used --cov=src/ for cross-platform CI coverage instead of named package (avoids hyphen-to-underscore conversion on Windows)"
  - "State dict tests use structural assertions (count + prefixes + sentinels) rather than freezing full key lists for maintainability"
  - "Corrected FCMAE instantiation (no backbone param) and other model configs from plan's inaccurate examples"

patterns-established:
  - "Public API pattern: subpackage __init__.py exports, top-level __init__.py re-exports, __all__ in alphabetical order"
  - "State dict compat test pattern: 3 tests per model (count, prefixes, sentinels) catching structural regressions"

# Metrics
duration: 4min
completed: 2026-02-13
---

# Phase 10 Plan 01: Public API, CI Integration Summary

**Top-level viscy_models imports for all 8 architectures, state dict regression tests, and CI matrix expansion to 18 jobs**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-13T19:03:30Z
- **Completed:** 2026-02-13T19:07:10Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- All 8 model classes importable via `from viscy_models import ModelName`
- 24 state dict compatibility regression tests (3 per model) guarding COMPAT-01
- CI test matrix expanded from 9 to 18 jobs (3 OS x 3 Python x 2 packages)
- Full test suite: 93 tests (92 passed, 1 xfailed pre-existing)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add top-level re-exports for all 8 model classes** - `86e7ebd` (feat)
2. **Task 2: Add state dict key compatibility regression tests** - `97efc88` (test)
3. **Task 3: Add viscy-models to CI test matrix** - `358c5a0` (chore)

## Files Created/Modified
- `packages/viscy-models/src/viscy_models/__init__.py` - Re-exports all 8 model classes with __all__
- `packages/viscy-models/tests/test_state_dict_compat.py` - 24 regression tests for state dict key compatibility
- `.github/workflows/test.yml` - Test matrix expanded with package dimension (viscy-transforms + viscy-models)

## Decisions Made
- Used `--cov=src/` for cross-platform CI coverage instead of named package (avoids hyphen-to-underscore conversion issue on Windows where `tr` is unavailable)
- State dict tests use structural assertions (parameter count + top-level prefix set + sentinel keys) rather than freezing full key lists -- more maintainable while still catching structural regressions
- Corrected model instantiation parameters from plan: FCMAE has no `backbone` param, ResNet3dEncoder requires explicit backbone string, BetaVae25D uses `input_spatial_size` not `input_shape`, BetaVaeMonai uses `in_shape` not `input_shape`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected model instantiation parameters in tests**
- **Found during:** Task 2 (state dict compatibility tests)
- **Issue:** Plan specified incorrect constructor parameters for several models: FCMAE(`backbone=`), BetaVae25D(`input_shape=`), BetaVaeMonai(`input_shape=`, `latent_dim=`)
- **Fix:** Used correct parameters from actual constructor signatures: FCMAE(`in_channels=`, `out_channels=`), ResNet3dEncoder(`backbone="resnet10"`), BetaVae25D(`input_spatial_size=`), BetaVaeMonai(`in_shape=`, `latent_size=`)
- **Files modified:** packages/viscy-models/tests/test_state_dict_compat.py
- **Verification:** All 24 tests pass
- **Committed in:** 97efc88 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in plan's example parameters)
**Impact on plan:** Auto-fix necessary for correctness. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- This is the FINAL plan of the v1.1 milestone
- viscy-models package is feature-complete with public API, full test coverage, and CI
- Ready for PR to main branch

## Self-Check: PASSED

All files verified present. All commit hashes verified in git log.

---
*Phase: 10-public-api-ci-integration*
*Completed: 2026-02-13*
