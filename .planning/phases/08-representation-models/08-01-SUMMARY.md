---
phase: 08-representation-models
plan: 01
subsystem: models
tags: [contrastive-learning, timm, monai, resnet, convnext, ssl]

# Dependency graph
requires:
  - phase: 06-package-scaffold
    provides: viscy-models package structure and contrastive/ directory
  - phase: 07-core-unet-models
    provides: components/stems.py with StemDepthtoChannels
provides:
  - ContrastiveEncoder with convnext_tiny/convnextv2_tiny/resnet50 timm backbones
  - ResNet3dEncoder with MONAI ResNetFeatures backends
  - projection_mlp shared utility function
  - 5 forward-pass tests for contrastive models
affects: [08-02-PLAN, viscy-lightning contrastive integration]

# Tech tracking
tech-stack:
  added: [timm (convnext/resnet50 backbones), monai (ResNetFeatures)]
  patterns: [projection_mlp shared utility, encoder.num_features for timm uniform API]

key-files:
  created:
    - packages/viscy-models/src/viscy_models/contrastive/encoder.py
    - packages/viscy-models/src/viscy_models/contrastive/resnet3d.py
    - packages/viscy-models/tests/test_contrastive/__init__.py
    - packages/viscy-models/tests/test_contrastive/test_encoder.py
    - packages/viscy-models/tests/test_contrastive/test_resnet3d.py
  modified:
    - packages/viscy-models/src/viscy_models/contrastive/__init__.py

key-decisions:
  - "Used encoder.num_features instead of encoder.head.fc.in_features for timm backbone-agnostic projection dim"
  - "Added pretrained parameter (default False) for pure nn.Module semantics consistent with UNeXt2 pattern"
  - "Adjusted ResNet50 test in_stack_depth from 15 to 10 for valid stem channel alignment (64 channels require even depth divisor)"

patterns-established:
  - "Contrastive model pattern: encoder + projection_mlp, forward returns (embedding, projection) tuple"
  - "Shared utility pattern: projection_mlp importable from encoder module, used by sibling modules"

# Metrics
duration: 4min
completed: 2026-02-13
---

# Phase 8 Plan 1: Contrastive Model Migration Summary

**ContrastiveEncoder (timm convnext/resnet50) and ResNet3dEncoder (MONAI) migrated with projection_mlp utility, ResNet50 bug fix, and 5 forward-pass tests**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-13T07:05:41Z
- **Completed:** 2026-02-13T07:09:21Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Migrated ContrastiveEncoder supporting convnext_tiny, convnextv2_tiny, and resnet50 timm backbones with 3D-to-2D StemDepthtoChannels
- Migrated ResNet3dEncoder using MONAI ResNetFeatures for native 3D contrastive learning
- Fixed ResNet50 backbone bug: replaced `encoder.head.fc.in_features` with `encoder.num_features` (timm uniform API)
- Added pretrained parameter (default False) to both encoders for pure nn.Module semantics
- Created 5 forward-pass tests covering both models with multiple backbone and stem configurations

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate ContrastiveEncoder and ResNet3dEncoder** - `68e7852` (feat)
2. **Task 2: Create forward-pass tests for contrastive models** - `3740e71` (test)

## Files Created/Modified
- `packages/viscy-models/src/viscy_models/contrastive/encoder.py` - ContrastiveEncoder class and projection_mlp utility
- `packages/viscy-models/src/viscy_models/contrastive/resnet3d.py` - ResNet3dEncoder class using MONAI ResNetFeatures
- `packages/viscy-models/src/viscy_models/contrastive/__init__.py` - Public re-exports for contrastive subpackage
- `packages/viscy-models/tests/test_contrastive/__init__.py` - Test package init
- `packages/viscy-models/tests/test_contrastive/test_encoder.py` - 3 forward-pass tests for ContrastiveEncoder
- `packages/viscy-models/tests/test_contrastive/test_resnet3d.py` - 2 forward-pass tests for ResNet3dEncoder

## Decisions Made
- Used `encoder.num_features` instead of `encoder.head.fc.in_features` for timm backbone-agnostic projection dimension lookup (fixes ResNet50 bug where `.head` attribute doesn't exist)
- Added `pretrained` parameter (default `False`) to both encoders, consistent with UNeXt2 pattern for pure nn.Module semantics
- Adjusted ResNet50 test `in_stack_depth` from 15 to 10 because resnet50's `conv1.out_channels=64` requires an even depth divisor from the stem computation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Adjusted ResNet50 test in_stack_depth parameter**
- **Found during:** Task 2 (forward-pass tests)
- **Issue:** Plan specified `in_stack_depth=15` for ResNet50 test, but resnet50's `conv1.out_channels=64` with default stem `(5,4,4)` produces `out_depth=3`, and `64 // 3 = 21` with remainder 1 -- triggering StemDepthtoChannels ValueError
- **Fix:** Changed `in_stack_depth` to 10 (produces `out_depth=2`, `64 // 2 = 32`, no remainder)
- **Files modified:** `packages/viscy-models/tests/test_contrastive/test_encoder.py`
- **Verification:** Test passes with correct embedding shape (2, 2048) and projection shape (2, 128)
- **Committed in:** `3740e71` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix in test parameters)
**Impact on plan:** Test parameter correction necessary for valid stem channel computation. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Contrastive models ready for use in Phase 8 Plan 2 (remaining representation models)
- `projection_mlp` utility importable for any future contrastive/SSL modules
- Test pattern established for forward-pass shape verification of contrastive models

## Self-Check: PASSED

All 7 files verified present. Both task commits (68e7852, 3740e71) found in git log.

---
*Phase: 08-representation-models*
*Completed: 2026-02-13*
