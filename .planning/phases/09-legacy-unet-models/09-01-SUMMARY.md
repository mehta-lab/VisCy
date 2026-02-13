---
phase: 09-legacy-unet-models
plan: 01
subsystem: models
tags: [unet, pytorch, nn.Module, legacy-migration, virtual-staining]

# Dependency graph
requires:
  - phase: 06-package-scaffold-shared-components
    provides: "viscy-models package scaffold with unet/_layers (ConvBlock2D, ConvBlock3D)"
provides:
  - "Unet2d nn.Module in viscy_models.unet.unet2d"
  - "Unet25d nn.Module in viscy_models.unet.unet25d"
  - "All 4 UNet-family models exported from viscy_models.unet"
  - "23 pytest tests for Unet2d and Unet25d"
affects: [10-public-api]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "register_modules/add_module pattern for dynamic layer registration"
    - "squeeze(2)/unsqueeze(2) for 5D-to-4D bridging in Unet2d"
    - "Skip interruption convolution for Z-compression in Unet25d"

key-files:
  created:
    - "packages/viscy-models/src/viscy_models/unet/unet2d.py"
    - "packages/viscy-models/src/viscy_models/unet/unet25d.py"
    - "packages/viscy-models/tests/test_unet/test_unet2d.py"
    - "packages/viscy-models/tests/test_unet/test_unet25d.py"
  modified:
    - "packages/viscy-models/src/viscy_models/unet/__init__.py"

key-decisions:
  - "Convert user-provided num_filters tuple to list internally for list concatenation compatibility"
  - "Mutable default num_filters=[] changed to tuple num_filters=() in both models"
  - "Preserved register_modules/add_module pattern verbatim for state dict key compatibility"
  - "up_list kept as plain Python list (not nn.ModuleList) since nn.Upsample has no parameters"

patterns-established:
  - "list(num_filters) conversion: When mutable default is fixed to tuple, internal code using list concatenation needs explicit conversion"

# Metrics
duration: 4min
completed: 2026-02-13
---

# Phase 9 Plan 1: Legacy UNet Migration Summary

**Unet2d and Unet25d migrated from v0.3.3 to viscy-models with full state dict compatibility and 23 new pytest tests**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-13T18:35:20Z
- **Completed:** 2026-02-13T18:39:59Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Migrated Unet2d (2D UNet with variable depth) and Unet25d (2.5D UNet with Z-compression) to viscy-models
- Updated unet/__init__.py to export all 4 UNet-family models (UNeXt2, FCMAE, Unet2d, Unet25d)
- Wrote 23 parametrized pytest tests covering forward pass, state dict keys, residual/task modes, and custom filters
- Full test suite: 68 passed, 1 xfailed, 0 failures

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate Unet2d and Unet25d model files** - `1e8223e` (feat)
2. **Task 2: Write pytest tests for Unet2d and Unet25d** - `712db18` (test)

## Files Created/Modified
- `packages/viscy-models/src/viscy_models/unet/unet2d.py` - 2D UNet with squeeze/unsqueeze for 5D compatibility
- `packages/viscy-models/src/viscy_models/unet/unet25d.py` - 2.5D UNet with skip interruption convolutions for Z-compression
- `packages/viscy-models/src/viscy_models/unet/__init__.py` - Updated to export all 4 UNet models
- `packages/viscy-models/tests/test_unet/test_unet2d.py` - 12 tests: default forward, variable depth, multichannel, residual, task mode, dropout, state dict keys, custom filters
- `packages/viscy-models/tests/test_unet/test_unet25d.py` - 11 tests: Z-compression, preserved depth, variable depth, multichannel, residual, task mode, state dict keys with skip_conv_layer, custom filters

## Decisions Made
- Converted user-provided `num_filters` tuple to list internally via `list(num_filters)` to maintain compatibility with `[in_channels] + self.num_filters` list concatenation
- Preserved `up_list` as plain Python list (not nn.ModuleList) since nn.Upsample has no learnable parameters
- Preserved `register_modules`/`add_module` pattern exactly as v0.3.3 for checkpoint state dict key compatibility

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed tuple-list concatenation error when custom num_filters provided**
- **Found during:** Task 2 (test_unet2d_custom_num_filters, test_unet25d_custom_num_filters)
- **Issue:** Changing default `num_filters=[]` to `num_filters=()` caused `[in_channels] + self.num_filters` to fail with `TypeError: can only concatenate list (not "tuple") to list` when users pass a tuple
- **Fix:** Added `list(num_filters)` conversion in the assignment `self.num_filters = list(num_filters)` in both models
- **Files modified:** `unet2d.py`, `unet25d.py`
- **Verification:** `test_unet2d_custom_num_filters` and `test_unet25d_custom_num_filters` now pass
- **Committed in:** `712db18` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential fix for the mutable default change to work correctly with user-provided filter tuples. No scope creep.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All UNet-family models now available from `viscy_models.unet`: UNeXt2, FullyConvolutionalMAE, Unet2d, Unet25d
- Phase 9 complete (single plan phase) -- ready for Phase 10 public API integration
- 68 total tests passing across all model types

## Self-Check: PASSED

All 5 created/modified files verified on disk. Both task commits (1e8223e, 712db18) verified in git history.

---
*Phase: 09-legacy-unet-models*
*Completed: 2026-02-13*
