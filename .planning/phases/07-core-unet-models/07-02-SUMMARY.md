---
phase: 07-core-unet-models
plan: 02
subsystem: models
tags: [fcmae, convnextv2, masked-autoencoder, unet, pytorch]

# Dependency graph
requires:
  - phase: 07-01
    provides: "UNeXt2 migration, unet subpackage structure, components package"
  - phase: 06-viscy-models-scaffold
    provides: "viscy-models package scaffold, components/heads.py with PixelToVoxelShuffleHead"
provides:
  - "FullyConvolutionalMAE model at viscy_models.unet.fcmae"
  - "Complete unet subpackage public API (UNeXt2 + FullyConvolutionalMAE)"
  - "11 migrated FCMAE tests with updated import paths"
affects: [08-cli-configs, 09-lightning-training, 10-integration-testing]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Import shared components from components instead of duplicating"
    - "Tuple defaults for mutable Sequence parameters"
provides: []

key-files:
  created:
    - packages/viscy-models/src/viscy_models/unet/fcmae.py
    - packages/viscy-models/tests/test_unet/test_fcmae.py
  modified:
    - packages/viscy-models/src/viscy_models/unet/__init__.py

key-decisions:
  - "Removed PixelToVoxelShuffleHead duplication from fcmae.py -- imported from canonical components.heads location"
  - "Fixed mutable list defaults (encoder_blocks, dims) to tuples for safety"

patterns-established:
  - "Deduplicate shared classes by importing from components rather than redefining"

# Metrics
duration: 3min
completed: 2026-02-13
---

# Phase 7 Plan 2: FCMAE Migration Summary

**FullyConvolutionalMAE migrated to viscy_models.unet with deduped PixelToVoxelShuffleHead import and mutable defaults fixed to tuples**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-13T01:15:13Z
- **Completed:** 2026-02-13T01:18:26Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Migrated FullyConvolutionalMAE and all 10 FCMAE-specific items (5 functions, 5 classes) to viscy_models.unet.fcmae
- Removed duplicated PixelToVoxelShuffleHead class definition; imported from components.heads canonical location
- Fixed mutable list defaults (encoder_blocks, dims) to tuples in FullyConvolutionalMAE.__init__
- Migrated all 11 FCMAE tests with zero test logic changes
- All 37 tests pass across the full viscy-models test suite (no regressions)
- Finalized unet/__init__.py public API exporting both UNeXt2 and FullyConvolutionalMAE

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate FCMAE to unet/fcmae.py with mutable default fixes** - `67859a9` (feat)
2. **Task 2: Migrate 11 FCMAE tests with updated imports** - `e7f7c66` (test)

## Files Created/Modified
- `packages/viscy-models/src/viscy_models/unet/fcmae.py` - FullyConvolutionalMAE and all FCMAE helper classes/functions
- `packages/viscy-models/tests/test_unet/test_fcmae.py` - 11 migrated FCMAE tests
- `packages/viscy-models/src/viscy_models/unet/__init__.py` - Updated to export both UNeXt2 and FullyConvolutionalMAE

## Decisions Made
- Removed PixelToVoxelShuffleHead class from fcmae.py (was duplicate of components.heads version); import instead
- Fixed mutable defaults to tuples; safe because internal code uses list() conversion or iteration

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 7 complete: both core UNet models (UNeXt2, FullyConvolutionalMAE) migrated
- unet subpackage public API finalized with both exports
- All 37 tests passing, ready for Phase 8 (CLI/Configs) or Phase 9 (Lightning Training)

## Self-Check: PASSED

All files verified present. All commit hashes confirmed in git log.

---
*Phase: 07-core-unet-models*
*Completed: 2026-02-13*
