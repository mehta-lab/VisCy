---
phase: 07-core-unet-models
verified: 2026-02-13T02:30:00Z
status: passed
score: 9/9 must-haves verified
---

# Phase 7: Core UNet Models Verification Report

**Phase Goal:** UNeXt2 and FCMAE are importable from viscy-models with forward-pass tests proving correctness

**Verified:** 2026-02-13T02:30:00Z

**Status:** passed

**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `from viscy_models.unet import UNeXt2` succeeds without error | ✓ VERIFIED | Import succeeds, model instantiates with `backbone='convnextv2_atto'`, `num_blocks` property returns 6 |
| 2 | UNeXt2 forward pass produces correct output shape (B, out_ch, out_depth, H, W) | ✓ VERIFIED | Forward pass with input (1,1,5,64,64) produces output (1,1,5,64,64); shape matches expected format |
| 3 | UNeXt2 tests cover multiple configurations: default backbone, small backbone, multichannel, different depths | ✓ VERIFIED | 6 tests exist: `test_unext2_default_forward`, `test_unext2_small_backbone`, `test_unext2_multichannel`, `test_unext2_different_stack_depths`, `test_unext2_deconv_decoder` (xfail with documented pre-existing bug), `test_unext2_stem_validation` |
| 4 | UNeXt2 constructor rejects invalid stem kernel vs stack depth combinations | ✓ VERIFIED | `test_unext2_stem_validation` tests `ValueError` with `match="not divisible"` when `in_stack_depth=7, stem_kernel_size=(5,4,4)` |
| 5 | `from viscy_models.unet import FullyConvolutionalMAE` succeeds without error | ✓ VERIFIED | Import succeeds, model instantiates with `in_channels=1, out_channels=1`, `num_blocks` property returns 8 |
| 6 | FCMAE forward pass produces correct output shape and returns (output, mask) tuple when pretraining=True | ✓ VERIFIED | Forward pass with `pretraining=True` returns tuple of length 2; verified programmatically |
| 7 | FCMAE forward pass returns output tensor only when pretraining=False | ✓ VERIFIED | Forward pass with `pretraining=False` returns single Tensor with shape (1,1,5,64,64) |
| 8 | All 11 existing FCMAE tests pass with updated import paths | ✓ VERIFIED | `test_fcmae.py` contains 11 test functions, all pass (imports updated to `viscy_models.unet.fcmae` and `viscy_models.components.heads`) |
| 9 | Mutable list defaults in FCMAE constructor are replaced with tuples | ✓ VERIFIED | `encoder_blocks` default is `(3,3,9,3)` (tuple), `dims` default is `(96,192,384,768)` (tuple); verified via `inspect.signature()` |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `packages/viscy-models/src/viscy_models/unet/unext2.py` | UNeXt2 nn.Module class | ✓ VERIFIED | Exists (85 lines), contains `class UNeXt2`, imports from `components.stems/heads/blocks`, all attribute names preserved (`encoder_stages`, `stem`, `decoder`, `head`) |
| `packages/viscy-models/tests/test_unet/test_unext2.py` | Forward-pass tests for UNeXt2 | ✓ VERIFIED | Exists (69 lines), contains 6 test functions covering required configurations |
| `packages/viscy-models/src/viscy_models/unet/fcmae.py` | FullyConvolutionalMAE and FCMAE-specific helper classes/functions | ✓ VERIFIED | Exists (441 lines), contains `class FullyConvolutionalMAE` and all 10 FCMAE-specific items (5 functions: `_init_weights`, `generate_mask`, `upsample_mask`, `masked_patchify`, `masked_unpatchify`; 5 classes: `MaskedConvNeXtV2Block`, `MaskedConvNeXtV2Stage`, `MaskedAdaptiveProjection`, `MaskedMultiscaleEncoder`, `FullyConvolutionalMAE`) |
| `packages/viscy-models/tests/test_unet/test_fcmae.py` | 11 migrated FCMAE tests | ✓ VERIFIED | Exists (136 lines), contains 11 test functions with updated import paths |
| `packages/viscy-models/src/viscy_models/unet/__init__.py` | Public exports of both UNeXt2 and FullyConvolutionalMAE | ✓ VERIFIED | Exists (6 lines), exports both models in `__all__` list |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `unext2.py` | `viscy_models.components.stems` | `from viscy_models.components.stems import UNeXt2Stem` | ✓ WIRED | Import found at line 10, `UNeXt2Stem` instantiated at line 49, used in forward pass at line 80 |
| `unext2.py` | `viscy_models.components.heads` | `from viscy_models.components.heads import PixelToVoxelHead` | ✓ WIRED | Import found at line 9, `PixelToVoxelHead` instantiated at line 65, used in forward pass at line 84 |
| `unext2.py` | `viscy_models.components.blocks` | `from viscy_models.components.blocks import UNeXt2Decoder` | ✓ WIRED | Import found at line 8, `UNeXt2Decoder` instantiated at line 57, used in forward pass at line 83 |
| `unet/__init__.py` | `viscy_models.unet.unext2` | `from viscy_models.unet.unext2 import UNeXt2` | ✓ WIRED | Import found at line 4, exported in `__all__` at line 6 |
| `fcmae.py` | `viscy_models.components.blocks` | `from viscy_models.components.blocks import UNeXt2Decoder` | ✓ WIRED | Import found at line 23, `UNeXt2Decoder` instantiated in `FullyConvolutionalMAE.__init__` |
| `fcmae.py` | `viscy_models.components.heads` | `from viscy_models.components.heads import PixelToVoxelHead, PixelToVoxelShuffleHead` | ✓ WIRED | Import found at line 24, both heads used in `FullyConvolutionalMAE.__init__` conditionally based on `head_mode` |
| `unet/__init__.py` | `viscy_models.unet.fcmae` | `from viscy_models.unet.fcmae import FullyConvolutionalMAE` | ✓ WIRED | Import found at line 3, exported in `__all__` at line 6 |

### Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| UNET-01: UNeXt2 migrated to `unet/unext2.py` with shared component imports updated | ✓ SATISFIED | None - UNeXt2 class migrated, imports from `components.*` verified, state dict attribute names preserved |
| UNET-02: FullyConvolutionalMAE migrated to `unet/fcmae.py` | ✓ SATISFIED | None - FCMAE and all 10 helper items migrated, `PixelToVoxelShuffleHead` imported from canonical `components.heads` location (not duplicated) |
| UNET-06: Forward-pass tests for UNeXt2 (NEW -- currently missing) | ✓ SATISFIED | None - 6 forward-pass tests created covering default/small backbone, multichannel I/O, different stack depths, deconv mode (xfail), stem validation |
| UNET-07: FCMAE tests migrated from existing test suite | ✓ SATISFIED | None - All 11 FCMAE tests migrated with zero logic changes, only import path updates |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `fcmae.py` | 430 | TODO comment: "replace num_blocks with explicit strides for all models" | ℹ️ Info | Architectural note documenting future API improvement. Not a stub or blocker — `num_blocks` is fully implemented with correct calculation. |

**Note:** The deconv decoder xfail test documents a pre-existing channel mismatch bug in the original code (never exercised in production). This is correctly marked as `xfail(strict=True)` to detect if/when the bug is fixed.

### Human Verification Required

None. All verification completed programmatically:

- ✓ Import paths verified via successful `import` statements
- ✓ Forward passes verified with shape assertions
- ✓ Test suite verified: 36 passed, 1 xfailed (expected), 0 failed
- ✓ Key links verified via grep pattern matching
- ✓ Mutable defaults verified via `inspect.signature()`
- ✓ Commit hashes verified in git log

---

## Summary

**Phase 7 goal achieved.** Both UNeXt2 and FullyConvolutionalMAE are importable from `viscy_models.unet` with working forward-pass tests:

1. **UNeXt2:** 6 forward-pass tests covering multiple configurations (2D/3D, varying channel counts, decoder modes, stem validation)
2. **FCMAE:** 11 existing tests pass after migration with updated import paths
3. **Public API:** `unet/__init__.py` exports both models
4. **Code quality:** All components imported from canonical `components.*` locations (no duplication), mutable defaults fixed, state dict compatibility preserved

**Test suite:** 36 tests pass, 1 xfail (documented pre-existing deconv bug), 0 failures, 0 regressions.

**Commits verified:**
- `dedaf1e` - feat(07-01): migrate UNeXt2 model class
- `58be984` - test(07-01): add 6 UNeXt2 forward-pass tests and fix deconv tuple bug
- `67859a9` - feat(07-02): migrate FullyConvolutionalMAE to viscy-models
- `e7f7c66` - test(07-02): migrate 11 FCMAE tests

All success criteria met. Ready to proceed to Phase 8 (CLI/Configs) or Phase 9 (Lightning Training).

---

_Verified: 2026-02-13T02:30:00Z_
_Verifier: Claude (gsd-verifier)_
