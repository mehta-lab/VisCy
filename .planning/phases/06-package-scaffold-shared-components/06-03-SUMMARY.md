---
phase: 06-package-scaffold-shared-components
plan: 03
subsystem: models
tags: [pytorch, nn-module, unet, conv-block, legacy-migration, state-dict-compat]

# Dependency graph
requires:
  - phase: 06-01
    provides: "Package scaffold with unet/_layers/ subpackage directory"
provides:
  - "ConvBlock2D nn.Module in viscy_models.unet._layers"
  - "ConvBlock3D nn.Module in viscy_models.unet._layers"
  - "State-dict-compatible layer implementations preserving add_module naming"
affects: [09-unet2d-unet25d-migration]

# Tech tracking
tech-stack:
  added: []
  patterns: [register_modules-add_module, snake-case-filenames-pascal-case-classes]

key-files:
  created:
    - packages/viscy-models/src/viscy_models/unet/_layers/conv_block_2d.py
    - packages/viscy-models/src/viscy_models/unet/_layers/conv_block_3d.py
    - packages/viscy-models/tests/test_unet/test_layers.py
  modified:
    - packages/viscy-models/src/viscy_models/unet/_layers/__init__.py

key-decisions:
  - "Preserved register_modules/add_module pattern verbatim for state dict key compatibility"
  - "Fixed only docstring formatting for ruff D-series compliance, no logic changes"
  - "Used named_modules() instead of state_dict() to test InstanceNorm (no learnable params by default)"

patterns-established:
  - "Legacy migration pattern: verbatim copy with snake_case filename, ruff docstring fixes only"
  - "State dict compatibility: always verify key naming after migration"

# Metrics
duration: 5min
completed: 2026-02-13
---

# Phase 6 Plan 3: UNet ConvBlock Layers Summary

**ConvBlock2D and ConvBlock3D migrated from v0.3.3 to viscy_models.unet._layers with state-dict-compatible add_module naming and 10 tests**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-13T00:37:36Z
- **Completed:** 2026-02-13T00:42:35Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Migrated ConvBlock2D and ConvBlock3D from v0.3.3 legacy source to new package location
- Preserved register_modules/add_module pattern ensuring state dict key compatibility (Conv2d_0, batch_norm_0, etc.)
- Full test coverage with 10 tests covering forward pass, state dict keys, residual options, filter steps, normalization variants, dropout registration, and layer ordering

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate ConvBlock2D and ConvBlock3D to unet/_layers/** - `8ef5998` (feat)
2. **Task 2: Write tests for ConvBlock2D and ConvBlock3D** - `4fa16c4` (test)

## Files Created/Modified
- `packages/viscy-models/src/viscy_models/unet/_layers/conv_block_2d.py` - ConvBlock2D nn.Module with configurable conv/norm/act ordering
- `packages/viscy-models/src/viscy_models/unet/_layers/conv_block_3d.py` - ConvBlock3D nn.Module with manual padding and dropout registration
- `packages/viscy-models/src/viscy_models/unet/_layers/__init__.py` - Public re-exports of ConvBlock2D and ConvBlock3D
- `packages/viscy-models/tests/test_unet/test_layers.py` - 10 tests covering both ConvBlock implementations

## Decisions Made
- Preserved register_modules/add_module pattern verbatim -- state dict keys like `Conv2d_0.weight`, `batch_norm_0.weight` must match original for checkpoint loading
- Fixed docstring formatting only (D400 period, D205 blank line, D401 imperative mood) -- no logic or variable name changes
- ConvBlock3D registers dropout via add_module while ConvBlock2D does not -- this asymmetry is preserved from original

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed instance norm test to use named_modules instead of state_dict**
- **Found during:** Task 2 (test writing)
- **Issue:** InstanceNorm2d has no learnable parameters by default (affine=False), so keys do not appear in state_dict()
- **Fix:** Changed test to check dict(model.named_modules()) for 'instance_norm_0' key instead
- **Files modified:** packages/viscy-models/tests/test_unet/test_layers.py
- **Verification:** Test passes correctly
- **Committed in:** 4fa16c4 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix in test)
**Impact on plan:** Minor test correction. No scope creep.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- ConvBlock2D and ConvBlock3D are stable building blocks for Phase 9 (Unet2d/Unet25d migration)
- Layer implementations are importable from `viscy_models.unet._layers`
- State dict keys verified compatible with v0.3.3 checkpoints

## Self-Check: PASSED

All 4 files verified present. Both task commits (8ef5998, 4fa16c4) verified in git log.

---
*Phase: 06-package-scaffold-shared-components*
*Completed: 2026-02-13*
