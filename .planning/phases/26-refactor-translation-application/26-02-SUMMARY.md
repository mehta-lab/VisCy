---
phase: 26-refactor-translation-application
plan: 02
subsystem: engine
tags: [viscy-translation, vsunet, fcmae, lightning-module, virtual-staining]

# Dependency graph
requires:
  - phase: 26-refactor-translation-application
    plan: 01
    provides: viscy-utils shared infra (HCSPredictionWriter, MixedLoss), translation app scaffold
  - phase: 12-model-migration
    provides: viscy_models package with UNeXt2, FullyConvolutionalMAE, Unet2d, Unet25d
provides:
  - VSUNet, FcmaeUNet, AugmentedPredictionVSUNet LightningModules in viscy_translation
  - MaskedMSELoss in viscy_translation.engine
  - SegmentationMetrics2D in viscy_translation.evaluation
  - Test suite with import, forward pass, state dict regression, and integration tests
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: [engine module pattern for translation LightningModules, state dict key regression testing]

key-files:
  created:
    - applications/translation/src/viscy_translation/engine.py
    - applications/translation/src/viscy_translation/evaluation.py
    - applications/translation/tests/conftest.py
    - applications/translation/tests/test_engine.py
  modified:
    - applications/translation/src/viscy_translation/__init__.py

key-decisions:
  - "Removed MixedLoss class from engine.py entirely (imported from viscy_utils.losses instead)"
  - "Removed unused ms_ssim_25d import since MixedLoss now lives in viscy_utils.losses and handles it internally"
  - "Used top-level viscy_data imports for Sample and SegmentationSample (both exported at top level)"
  - "Added numpy-style docstrings to all public methods per project convention"

patterns-established:
  - "Translation engine pattern: _UNET_ARCHITECTURE dispatch dict for architecture selection"
  - "State dict key regression: verify all keys start with model. prefix for checkpoint compatibility"

requirements-completed: []

# Metrics
duration: 10min
completed: 2026-02-28
---

# Phase 26 Plan 02: Engine Migration Summary

**VSUNet, FcmaeUNet, AugmentedPredictionVSUNet migrated to viscy_translation with all imports on new package paths and 7-test regression suite**

## Performance

- **Duration:** 10 min
- **Started:** 2026-02-27T23:59:20Z
- **Completed:** 2026-02-28T00:09:56Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- engine.py migrated with VSUNet, FcmaeUNet, AugmentedPredictionVSUNet, MaskedMSELoss using new package imports (viscy_data, viscy_models, viscy_utils)
- evaluation.py migrated with SegmentationMetrics2D using new package imports
- __init__.py updated with top-level re-exports for all public classes
- Full test suite with 7 tests: imports, init, forward pass, state dict keys, MixedLoss integration, FcmaeUNet init, no-old-imports grep

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate engine.py and evaluation.py with updated imports** - `369defa` (feat)
2. **Task 2: Create test suite with import tests, smoke tests, and state dict regression** - `4e5ca4c` (test)

## Files Created/Modified
- `applications/translation/src/viscy_translation/engine.py` - VSUNet, FcmaeUNet, AugmentedPredictionVSUNet, MaskedMSELoss LightningModules
- `applications/translation/src/viscy_translation/evaluation.py` - SegmentationMetrics2D test runner
- `applications/translation/src/viscy_translation/__init__.py` - Top-level re-exports
- `applications/translation/tests/conftest.py` - Synthetic data fixtures and dimensions
- `applications/translation/tests/test_engine.py` - 7 smoke tests for engine modules

## Decisions Made
- Removed MixedLoss class definition from engine.py since it was extracted to viscy_utils.losses in Plan 01 -- engine.py now imports it rather than defining it
- Removed ms_ssim_25d import from engine.py since it was only used by the now-removed MixedLoss class definition
- Used top-level viscy_data imports (from viscy_data import Sample, SegmentationSample) since both are exported at the package top level
- Added numpy-style docstrings to all public methods to comply with project D rules

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed unused ms_ssim_25d import**
- **Found during:** Task 1 (engine.py migration)
- **Issue:** ms_ssim_25d was imported from viscy_utils.evaluation.metrics but only used by MixedLoss which was removed from engine.py. Ruff F401 flagged it.
- **Fix:** Removed the unused import
- **Files modified:** applications/translation/src/viscy_translation/engine.py
- **Verification:** ruff check passes
- **Committed in:** 369defa (Task 1 commit)

**2. [Rule 1 - Bug] Fixed unused conftest imports in test file**
- **Found during:** Task 2 (test suite creation)
- **Issue:** SYNTH_H and SYNTH_W imported from conftest but not used in any test. Ruff F401 flagged them.
- **Fix:** Removed unused SYNTH_H, SYNTH_W from conftest import
- **Files modified:** applications/translation/tests/test_engine.py
- **Verification:** ruff check passes, all tests still pass
- **Committed in:** 4e5ca4c (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Minor import cleanup. No scope creep.

## Issues Encountered
None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Translation application is fully functional with engine and evaluation modules
- All imports use new modular package paths
- Ready for end-to-end training/prediction workflows using LightningCLI

## Self-Check: PASSED

All 5 created/modified files verified present. Both task commits (369defa, 4e5ca4c) verified in git log.

---
*Phase: 26-refactor-translation-application*
*Completed: 2026-02-28*
