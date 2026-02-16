---
phase: 06-package-scaffold-shared-components
verified: 2026-02-13T00:48:10Z
status: passed
score: 17/17 must-haves verified
must_haves:
  truths:
    - "uv sync --package viscy-models succeeds without errors"
    - "python -c 'import viscy_models' runs without import errors"
    - "viscy-models appears in workspace and is importable"
    - "from viscy_models._components.stems import UNeXt2Stem, StemDepthtoChannels works"
    - "from viscy_models._components.heads import PixelToVoxelHead, UnsqueezeHead, PixelToVoxelShuffleHead works"
    - "from viscy_models._components.blocks import UNeXt2UpStage, UNeXt2Decoder, icnr_init, _get_convnext_stage works"
    - "_components/ has ZERO imports from unet/, vae/, or contrastive/"
    - "All stem, head, and block classes produce correct output shapes"
    - "All component tests pass"
    - "from viscy_models.unet._layers import ConvBlock2D, ConvBlock3D works"
    - "ConvBlock2D forward pass produces correct output shape"
    - "ConvBlock3D forward pass produces correct output shape"
    - "ConvBlock2D/3D use register_modules/add_module pattern (NOT nn.ModuleList)"
    - "State dict keys match original ConvBlock2D/3D exactly (e.g., Conv2d_0, batch_norm_0)"
    - "All layer tests pass"
  artifacts:
    - path: "packages/viscy-models/pyproject.toml"
      status: verified
    - path: "packages/viscy-models/src/viscy_models/__init__.py"
      status: verified
    - path: "packages/viscy-models/src/viscy_models/py.typed"
      status: verified
    - path: "packages/viscy-models/src/viscy_models/_components/__init__.py"
      status: verified
    - path: "packages/viscy-models/src/viscy_models/_components/stems.py"
      status: verified
    - path: "packages/viscy-models/src/viscy_models/_components/heads.py"
      status: verified
    - path: "packages/viscy-models/src/viscy_models/_components/blocks.py"
      status: verified
    - path: "packages/viscy-models/src/viscy_models/unet/__init__.py"
      status: verified
    - path: "packages/viscy-models/src/viscy_models/unet/_layers/__init__.py"
      status: verified
    - path: "packages/viscy-models/src/viscy_models/unet/_layers/conv_block_2d.py"
      status: verified
    - path: "packages/viscy-models/src/viscy_models/unet/_layers/conv_block_3d.py"
      status: verified
    - path: "packages/viscy-models/src/viscy_models/contrastive/__init__.py"
      status: verified
    - path: "packages/viscy-models/src/viscy_models/vae/__init__.py"
      status: verified
    - path: "pyproject.toml"
      status: verified
  key_links:
    - from: "pyproject.toml"
      to: "packages/viscy-models"
      status: wired
    - from: "_components/blocks.py"
      to: "timm.models.convnext.ConvNeXtStage"
      status: wired
    - from: "_components/heads.py"
      to: "monai.networks.blocks.UpSample"
      status: wired
    - from: "_components/"
      to: "model subpackages (unet/, vae/, contrastive/)"
      status: verified_zero_imports
---

# Phase 6: Package Scaffold & Shared Components Verification Report

**Phase Goal:** Users can install viscy-models and shared architectural components are available for model implementations

**Verified:** 2026-02-13T00:48:10Z

**Status:** passed

**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | uv sync --package viscy-models succeeds without errors | ✓ VERIFIED | Command exits with code 0, resolves 160 packages |
| 2 | python -c 'import viscy_models' runs without import errors | ✓ VERIFIED | Import succeeds, __version__ = 0.0.0.post207.dev0+9f2044f |
| 3 | viscy-models appears in workspace and is importable | ✓ VERIFIED | Package registered, appears in pkg_resources.working_set |
| 4 | from viscy_models._components.stems import UNeXt2Stem, StemDepthtoChannels works | ✓ VERIFIED | Import succeeds, prints "stems OK" |
| 5 | from viscy_models._components.heads import PixelToVoxelHead, UnsqueezeHead, PixelToVoxelShuffleHead works | ✓ VERIFIED | Import succeeds, prints "heads OK" |
| 6 | from viscy_models._components.blocks import UNeXt2UpStage, UNeXt2Decoder, icnr_init works | ✓ VERIFIED | Import succeeds, prints "blocks OK" |
| 7 | _components/ has ZERO imports from unet/, vae/, or contrastive/ | ✓ VERIFIED | grep returns no matches for forbidden imports |
| 8 | All stem, head, and block classes produce correct output shapes | ✓ VERIFIED | 10 component tests pass (test_stems.py: 3, test_heads.py: 3, test_blocks.py: 4) |
| 9 | All component tests pass | ✓ VERIFIED | pytest packages/viscy-models/tests/test_components/ passes 10/10 tests |
| 10 | from viscy_models.unet._layers import ConvBlock2D, ConvBlock3D works | ✓ VERIFIED | Import succeeds, prints "layers OK" |
| 11 | ConvBlock2D forward pass produces correct output shape | ✓ VERIFIED | Test passes, (1,16,64,64) -> (1,32,64,64) |
| 12 | ConvBlock3D forward pass produces correct output shape | ✓ VERIFIED | Test passes, (1,8,5,32,32) -> (1,16,5,32,32) |
| 13 | ConvBlock2D/3D use register_modules/add_module pattern (NOT nn.ModuleList) | ✓ VERIFIED | State dict keys use Conv2d_0, batch_norm_0 naming pattern |
| 14 | State dict keys match original ConvBlock2D/3D exactly | ✓ VERIFIED | Keys like Conv2d_0.weight, batch_norm_0.weight present in state dict |
| 15 | All layer tests pass | ✓ VERIFIED | pytest packages/viscy-models/tests/test_unet/test_layers.py passes 10/10 tests |
| 16 | Package directory exists with src layout | ✓ VERIFIED | packages/viscy-models/src/viscy_models/ exists with all subpackages |
| 17 | pyproject.toml has torch/timm/monai/numpy deps | ✓ VERIFIED | All dependencies present: torch>=2.10, timm>=1.0.15, monai>=1.5.2, numpy>=2.4.1 |

**Score:** 17/17 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| packages/viscy-models/pyproject.toml | Package build config with dependencies | ✓ VERIFIED | 59 lines, hatchling + uv-dynamic-versioning, torch/timm/monai/numpy deps |
| packages/viscy-models/src/viscy_models/__init__.py | Package entry point with __version__ | ✓ VERIFIED | 186 bytes, imports version from importlib.metadata |
| packages/viscy-models/src/viscy_models/py.typed | PEP 561 type marker | ✓ VERIFIED | Empty file present |
| packages/viscy-models/src/viscy_models/_components/__init__.py | Re-exports all shared components | ✓ VERIFIED | 20 lines, exports 8 components from stems/heads/blocks |
| packages/viscy-models/src/viscy_models/_components/stems.py | UNeXt2Stem, StemDepthtoChannels | ✓ VERIFIED | 80 lines, 7 classes/functions, imports from torch |
| packages/viscy-models/src/viscy_models/_components/heads.py | PixelToVoxelHead, UnsqueezeHead, PixelToVoxelShuffleHead | ✓ VERIFIED | 96 lines, 9 classes/functions, imports from monai |
| packages/viscy-models/src/viscy_models/_components/blocks.py | UNeXt2UpStage, UNeXt2Decoder, icnr_init, _get_convnext_stage | ✓ VERIFIED | 169 lines, 10 classes/functions, imports from timm |
| packages/viscy-models/src/viscy_models/unet/__init__.py | UNet subpackage marker | ✓ VERIFIED | Present with docstring |
| packages/viscy-models/src/viscy_models/unet/_layers/__init__.py | Layers subpackage with ConvBlock exports | ✓ VERIFIED | 6 lines, exports ConvBlock2D and ConvBlock3D |
| packages/viscy-models/src/viscy_models/unet/_layers/conv_block_2d.py | ConvBlock2D nn.Module | ✓ VERIFIED | 355 lines, register_modules/add_module pattern |
| packages/viscy-models/src/viscy_models/unet/_layers/conv_block_3d.py | ConvBlock3D nn.Module | ✓ VERIFIED | 312 lines, register_modules/add_module pattern |
| packages/viscy-models/src/viscy_models/contrastive/__init__.py | Contrastive subpackage marker | ✓ VERIFIED | Present with docstring |
| packages/viscy-models/src/viscy_models/vae/__init__.py | VAE subpackage marker | ✓ VERIFIED | Present with docstring |
| packages/viscy-models/tests/conftest.py | Device fixture for tests | ✓ VERIFIED | Present in summary key-files |
| packages/viscy-models/tests/test_components/test_stems.py | Stem tests | ✓ VERIFIED | 3 tests pass |
| packages/viscy-models/tests/test_components/test_heads.py | Head tests | ✓ VERIFIED | 3 tests pass |
| packages/viscy-models/tests/test_components/test_blocks.py | Block tests | ✓ VERIFIED | 4 tests pass |
| packages/viscy-models/tests/test_unet/test_layers.py | ConvBlock tests | ✓ VERIFIED | 10 tests pass |
| pyproject.toml | Root workspace with viscy-models registered | ✓ VERIFIED | Contains viscy-models in dependencies and [tool.uv.sources] |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| pyproject.toml | packages/viscy-models/pyproject.toml | uv workspace member auto-discovery | ✓ WIRED | members = ["packages/*"] matches packages/viscy-models |
| pyproject.toml | packages/viscy-models | workspace source | ✓ WIRED | viscy-models = { workspace = true } present in [tool.uv.sources] |
| _components/blocks.py | timm.models.convnext.ConvNeXtStage | import in _get_convnext_stage | ✓ WIRED | Pattern "timm.models.convnext.ConvNeXtStage" found in blocks.py line 61 |
| _components/heads.py | monai.networks.blocks.UpSample | import for PixelToVoxelHead | ✓ WIRED | "from monai.networks.blocks import Convolution, UpSample" found |
| _components/ | model subpackages (unet/, vae/, contrastive/) | MUST NOT import | ✓ VERIFIED_ZERO_IMPORTS | grep confirms zero imports from model subpackages |
| _components/blocks.py | numpy | np.linspace for filter step calculations | ✓ WIRED | "import numpy as np" present, used in code |
| _components/heads.py | _components/blocks.py | intra-component import of icnr_init | ✓ WIRED | "from viscy_models._components.blocks import icnr_init" found |

### Requirements Coverage

| Requirement | Status | Description |
|-------------|--------|-------------|
| MPKG-01 | ✓ SATISFIED | packages/viscy-models/src/viscy_models/ directory exists with src layout and __init__.py |
| MPKG-02 | ✓ SATISFIED | pyproject.toml has hatchling, uv-dynamic-versioning, torch/timm/monai/numpy deps |
| MPKG-03 | ✓ SATISFIED | uv sync --package viscy-models succeeds without errors |
| MPKG-04 | ✓ SATISFIED | viscy_models._components subpackage contains stems.py, heads.py, and blocks.py with extracted shared code |
| UNET-05 | ✓ SATISFIED | ConvBlock2D/3D layers exist in viscy_models.unet._layers and are importable with state-dict-compatible naming |
| COMPAT-02 | N/A | Mutable defaults in model constructors - deferred to Phases 7-9 (no model constructors in Phase 6 scope) |

**Note on COMPAT-02:** The roadmap success criteria notes that this requirement was deferred to Phases 7-9 since model constructors are not in Phase 6 scope. Only component/layer code was extracted in Phase 6, and no mutable defaults were found in the extracted code.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| - | - | - | - | No anti-patterns detected |

**Summary:** No TODO/FIXME/PLACEHOLDER markers, no empty implementations, no console.log-only functions, no mutable default arguments in extracted component/layer code.

### Human Verification Required

None. All verification completed programmatically.

### Gaps Summary

None. All must-haves verified. Phase goal achieved.

## Verification Details

### Plan 06-01: Package Scaffold

**Artifacts verified:**
- Package directory structure created with all 5 subpackages (_components, unet, unet/_layers, contrastive, vae)
- pyproject.toml configured with hatchling, uv-dynamic-versioning, torch/timm/monai/numpy dependencies
- py.typed marker present for PEP 561 type checking
- Root pyproject.toml updated with viscy-models in dependencies and workspace sources
- uv.lock updated with timm and viscy-models

**Tests passed:**
- uv sync --package viscy-models: SUCCESS
- import viscy_models: SUCCESS
- Package version accessible: 0.0.0.post207.dev0+9f2044f

**Commits verified:**
- 9f2044f: feat(06-01): create viscy-models package scaffold
- acd56b7: feat(06-01): register viscy-models in workspace

### Plan 06-02: Shared Components Extraction

**Artifacts verified:**
- stems.py: 80 lines, exports UNeXt2Stem and StemDepthtoChannels
- heads.py: 96 lines, exports PixelToVoxelHead, UnsqueezeHead, PixelToVoxelShuffleHead
- blocks.py: 169 lines, exports icnr_init, _get_convnext_stage, UNeXt2UpStage, UNeXt2Decoder
- _components/__init__.py: Re-exports all 8 shared components

**Import isolation verified:**
- Zero imports from viscy_models.unet, viscy_models.vae, or viscy_models.contrastive in _components/
- Only external dependencies: torch, timm, monai, numpy
- Intra-component import allowed: heads.py imports icnr_init from blocks.py

**Tests passed:**
- test_stems.py: 3/3 tests pass (shape verification + error handling)
- test_heads.py: 3/3 tests pass (all 3 heads produce correct output shapes)
- test_blocks.py: 4/4 tests pass (icnr_init, _get_convnext_stage, UNeXt2UpStage, UNeXt2Decoder)

**Commits verified:**
- 0a2a15c: feat(06-02): extract shared components into _components/ module
- 29d76d9: test(06-02): add forward-pass tests for all _components

### Plan 06-03: UNet ConvBlock Layers

**Artifacts verified:**
- conv_block_2d.py: 355 lines, ConvBlock2D with register_modules/add_module pattern
- conv_block_3d.py: 312 lines, ConvBlock3D with register_modules/add_module pattern
- _layers/__init__.py: Re-exports ConvBlock2D and ConvBlock3D

**State dict compatibility verified:**
- ConvBlock2D state dict keys: Conv2d_0.weight, batch_norm_0.weight (original naming preserved)
- ConvBlock3D state dict keys: Conv3d_0.weight, batch_norm_0.weight (original naming preserved)
- register_modules/add_module pattern preserved verbatim (NOT refactored to nn.ModuleList)

**Tests passed:**
- test_layers.py: 10/10 tests pass
  - ConvBlock2D: 5 tests (default forward, state dict keys, residual, filter_steps, instance norm)
  - ConvBlock3D: 5 tests (default forward, state dict keys, dropout registration, layer order, forward pass variants)

**Commits verified:**
- 8ef5998: feat(06-03): migrate ConvBlock2D and ConvBlock3D to unet/_layers/
- 4fa16c4: test(06-03): add tests for ConvBlock2D and ConvBlock3D

## Overall Assessment

**Phase Goal Achievement:** ✓ VERIFIED

Users can install viscy-models and shared architectural components are available for model implementations.

**Evidence:**
1. Package scaffold complete: installable via uv sync, all subpackages created
2. Shared components extracted: 8 components (2 stems, 3 heads, 2 blocks + 2 utility functions) with zero circular dependency risk
3. UNet layers migrated: ConvBlock2D/3D with state-dict-compatible naming preserved
4. Test coverage complete: 20 tests total (10 component tests + 10 layer tests), all passing
5. Workspace integration verified: package registered, imports work, dependencies resolved

**Requirements satisfied:** MPKG-01, MPKG-02, MPKG-03, MPKG-04, UNET-05

**Next phase readiness:**
- Phase 7 (Core UNet Models): Can import from viscy_models._components for shared stems/heads/blocks
- Phase 8 (Representation Models): Can import from viscy_models._components for encoder/decoder components
- Phase 9 (Legacy UNet Models): Can import ConvBlock2D/3D from viscy_models.unet._layers

---

_Verified: 2026-02-13T00:48:10Z_

_Verifier: Claude (gsd-verifier)_
