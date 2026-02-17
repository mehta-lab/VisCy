---
phase: 06-package-scaffold-shared-components
plan: 02
subsystem: models
tags: [nn-module, component-extraction, stems, heads, blocks, state-dict-compat, timm, monai]

# Dependency graph
requires:
  - phase: 06-01
    provides: "Package scaffold with components/ subpackage directory"
provides:
  - "UNeXt2Stem and StemDepthtoChannels in components/stems.py"
  - "PixelToVoxelHead, UnsqueezeHead, PixelToVoxelShuffleHead in components/heads.py"
  - "icnr_init, _get_convnext_stage, UNeXt2UpStage, UNeXt2Decoder in components/blocks.py"
  - "Full test coverage for all shared components (10 tests)"
affects: [06-03, 07, 08, 09]

# Tech tracking
tech-stack:
  added: []
  patterns: [verbatim-extraction-with-import-only-changes, intra-components-import]

key-files:
  created:
    - packages/viscy-models/src/viscy_models/components/stems.py
    - packages/viscy-models/src/viscy_models/components/heads.py
    - packages/viscy-models/src/viscy_models/components/blocks.py
    - packages/viscy-models/tests/test_components/test_stems.py
    - packages/viscy-models/tests/test_components/test_heads.py
    - packages/viscy-models/tests/test_components/test_blocks.py
  modified:
    - packages/viscy-models/src/viscy_models/components/__init__.py

key-decisions:
  - "Intra-components import allowed: heads.py imports icnr_init from blocks.py"
  - "Docstring formatting adjusted for ruff D205/D400 compliance while preserving code logic"
  - "_get_convnext_stage is private (underscore prefix) but importable, excluded from __all__"

patterns-established:
  - "Component extraction: copy verbatim from v0.3.3, only change import paths"
  - "Intra-components dependency: blocks.py is standalone, heads.py depends on blocks.py"
  - "Test pattern: forward-pass shape verification with device fixture"

# Metrics
duration: 6min
completed: 2026-02-13
---

# Phase 6 Plan 2: Shared Components Extraction Summary

**8 shared nn.Module components (2 stems, 3 heads, 2 blocks + 2 functions) extracted verbatim from v0.3.3 into components/ with 10 forward-pass tests**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-13T00:37:31Z
- **Completed:** 2026-02-13T00:43:41Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Extracted all shared architectural components from v0.3.3 unext2.py and fcmae.py into three focused modules (stems, heads, blocks)
- Zero imports from model subpackages (unet/, vae/, contrastive/) in components/ -- verified by grep
- All class names, method names, and attribute names preserved identically for state dict compatibility
- 10 forward-pass tests verify correct output shapes for all components

## Task Commits

Each task was committed atomically:

1. **Task 1: Extract shared components into components/ module** - `0a2a15c` (feat)
2. **Task 2: Write tests for all extracted components** - `29d76d9` (test)

## Files Created/Modified
- `packages/viscy-models/src/viscy_models/components/stems.py` - UNeXt2Stem, StemDepthtoChannels (from v0.3.3 unext2.py)
- `packages/viscy-models/src/viscy_models/components/heads.py` - PixelToVoxelHead, UnsqueezeHead (from unext2.py), PixelToVoxelShuffleHead (from fcmae.py)
- `packages/viscy-models/src/viscy_models/components/blocks.py` - icnr_init, _get_convnext_stage, UNeXt2UpStage, UNeXt2Decoder (from unext2.py)
- `packages/viscy-models/src/viscy_models/components/__init__.py` - Public re-exports of all 8 shared components
- `packages/viscy-models/tests/test_components/test_stems.py` - 3 tests: shape verification for both stems + mismatch error
- `packages/viscy-models/tests/test_components/test_heads.py` - 3 tests: shape verification for all 3 heads
- `packages/viscy-models/tests/test_components/test_blocks.py` - 4 tests: icnr_init, _get_convnext_stage, UNeXt2UpStage, UNeXt2Decoder

## Decisions Made
- Allowed intra-components import: heads.py imports icnr_init from blocks.py (blocks.py has no model imports, so no circular dependency risk)
- _get_convnext_stage kept as private function (underscore prefix) but still importable for model files in later phases; excluded from __all__
- Adjusted docstring formatting for ruff D205/D400 compliance without changing code semantics

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test parameters for PixelToVoxelHead and PixelToVoxelShuffleHead**
- **Found during:** Task 2 (writing tests)
- **Issue:** Plan-suggested in_channels values (96 for PixelToVoxelHead, 768 for PixelToVoxelShuffleHead) were incompatible with the heads' internal reshape operations. The heads require specific channel counts derived from out_stack_depth, out_channels, and expansion_ratio.
- **Fix:** Used actual model parameters: in_channels=224 for PixelToVoxelHead (matching UNeXt2 usage), in_channels=160 for PixelToVoxelShuffleHead (matching FCMAE usage)
- **Files modified:** test_heads.py
- **Verification:** All 10 tests pass
- **Committed in:** 29d76d9 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in test parameters)
**Impact on plan:** Test parameter fix was necessary for correctness. No scope creep.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All shared components ready for import by model files in Phases 7-9
- Import path pattern established: `from viscy_models.components.stems import UNeXt2Stem`
- Plan 03 (ConvBlock2D/3D migration to unet/_layers/) can proceed independently

## Self-Check: PASSED

All 8 files verified present. Both task commits (0a2a15c, 29d76d9) verified in git log.

---
*Phase: 06-package-scaffold-shared-components*
*Completed: 2026-02-13*
