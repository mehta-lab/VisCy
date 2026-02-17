---
phase: 09-legacy-unet-models
verified: 2026-02-13T18:45:23Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 9: Legacy UNet Models Verification Report

**Phase Goal:** Unet2d and Unet25d are importable from viscy-models with migrated test coverage
**Verified:** 2026-02-13T18:45:23Z
**Status:** passed
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | from viscy_models.unet import Unet2d works and produces correct 5D output shape | VERIFIED | Import succeeds, forward pass (1,1,1,256,256) -> (1,1,1,256,256) |
| 2 | from viscy_models.unet import Unet25d works and produces correct 5D output shape | VERIFIED | Import succeeds, forward pass (1,1,5,64,64) -> (1,1,1,64,64) with Z-compression |
| 3 | Unet2d state dict keys match legacy checkpoint format (down_conv_block_N, up_conv_block_N, etc.) | VERIFIED | State dict contains down_conv_block_0, up_conv_block_0, bottom_transition_block, terminal_block. No down_samp keys (AvgPool2d has no params) |
| 4 | Unet25d state dict keys match legacy format including skip_conv_layer_N | VERIFIED | State dict contains skip_conv_layer_0, skip_conv_layer_1 plus standard unet keys. No down_samp keys |
| 5 | Existing 46 tests still pass (no regressions) | VERIFIED | Full test suite: 68 passed, 1 xfailed (pre-existing deconv bug in UNeXt2) |
| 6 | New Unet2d tests pass for variable depth, residual, reg/seg task modes | VERIFIED | 12 tests pass covering default forward, variable depth (1,2,4 blocks), multichannel, residual, task modes, dropout, state dict keys, custom filters |
| 7 | New Unet25d tests pass for depth compression and depth preservation | VERIFIED | 11 tests pass covering Z-compression (5->1), preserved depth (5->5), variable depth, multichannel, residual, task modes, state dict keys with skip_conv_layer, custom filters |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `packages/viscy-models/src/viscy_models/unet/unet2d.py` | Unet2d nn.Module class | VERIFIED | 247 lines, contains `class Unet2d(nn.Module)`, imports ConvBlock2D from _layers |
| `packages/viscy-models/src/viscy_models/unet/unet25d.py` | Unet25d nn.Module class | VERIFIED | 276 lines, contains `class Unet25d(nn.Module)`, imports ConvBlock3D from _layers |
| `packages/viscy-models/src/viscy_models/unet/__init__.py` | Public API exporting all 4 unet models | VERIFIED | Exports UNeXt2, FullyConvolutionalMAE, Unet2d, Unet25d via `__all__` |
| `packages/viscy-models/tests/test_unet/test_unet2d.py` | Pytest tests for Unet2d | VERIFIED | 118 lines, contains 8 test functions (12 total tests with parametrization) |
| `packages/viscy-models/tests/test_unet/test_unet25d.py` | Pytest tests for Unet25d | VERIFIED | 158 lines, contains 8 test functions (11 total tests with parametrization) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `unet2d.py` | `_layers/conv_block_2d.py` | import ConvBlock2D | WIRED | `from viscy_models.unet._layers.conv_block_2d import ConvBlock2D` present |
| `unet25d.py` | `_layers/conv_block_3d.py` | import ConvBlock3D | WIRED | `from viscy_models.unet._layers.conv_block_3d import ConvBlock3D` present |
| `unet/__init__.py` | `unet2d.py` | re-export Unet2d | WIRED | `from viscy_models.unet.unet2d import Unet2d` in `__all__` |
| `unet/__init__.py` | `unet25d.py` | re-export Unet25d | WIRED | `from viscy_models.unet.unet25d import Unet25d` in `__all__` |

### Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| UNET-03: Unet2d migrated to `unet/unet2d.py` | SATISFIED | File exists at correct path with snake_case naming |
| UNET-04: Unet25d migrated to `unet/unet25d.py` | SATISFIED | File exists at correct path with snake_case naming |
| UNET-08: Unet2d/Unet25d tests migrated to pytest | SATISFIED | 23 pytest tests (12 Unet2d + 11 Unet25d) with parametrize decorators |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `unet2d.py` | 68-71,99,111,114 | TODO comments | INFO | Pre-existing from v0.3.3. Static mode variables and unimplemented parameter validation. Does not block goal - models are functional. |
| `unet25d.py` | 81-85,117,144 | TODO comments | INFO | Pre-existing from v0.3.3. Static mode variables and residual dimensionality note. Does not block goal - models are functional. |

**No blocker anti-patterns found.** All TODO comments are from the original v0.3.3 codebase and document future enhancements, not missing functionality.

### Human Verification Required

None. All observable truths verified programmatically through import tests, forward pass shape validation, state dict key inspection, and pytest execution.

---

## Summary

Phase 9 goal **fully achieved**. Both Unet2d and Unet25d are:
- Importable from `viscy_models.unet` with clean public API
- Producing correct output shapes for standard configurations (2D: 5D in/out with squeeze/unsqueeze, 2.5D: Z-compression 5->1 and preservation 5->5)
- State dict compatible with v0.3.3 checkpoint format (verified key patterns match legacy)
- Covered by 23 parametrized pytest tests (converted from unittest style)
- Integrated with shared components (_layers.ConvBlock2D/3D)

Full test suite shows no regressions: 68 passed, 1 xfailed (pre-existing UNeXt2 deconv bug from Phase 7).

Requirements UNET-03, UNET-04, UNET-08 satisfied. Phase 10 (public API integration) unblocked.

---

_Verified: 2026-02-13T18:45:23Z_
_Verifier: Claude (gsd-verifier)_
