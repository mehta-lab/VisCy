# Phase 7: Core UNet Models - Research

**Researched:** 2026-02-12
**Domain:** nn.Module migration, UNeXt2 / FullyConvolutionalMAE architecture, state dict preservation
**Confidence:** HIGH

## Summary

Phase 7 migrates two model classes -- `UNeXt2` and `FullyConvolutionalMAE` -- into the `viscy-models` package at `unet/unext2.py` and `unet/fcmae.py` respectively. All shared components (stems, heads, blocks, decoder) were already extracted into `_components/` during Phase 6. The migration involves: (1) copying the model class code, (2) updating imports to point at `viscy_models._components` instead of the old monolithic `viscy.unet.networks.unext2`, (3) fixing two mutable list defaults in FCMAE's constructor, and (4) writing new tests for UNeXt2 (currently untested) and migrating 11 existing FCMAE tests.

The UNeXt2 class is thin -- it composes a `timm.create_model` encoder with the already-extracted `UNeXt2Stem`, `UNeXt2Decoder`, and `PixelToVoxelHead`. A full forward pass through the composed pipeline has been verified to work with the Phase 6 extracted components (output: `(1, 2, 5, 256, 256)` from `(1, 1, 5, 256, 256)` input with `convnextv2_tiny` backbone). The FCMAE class contains 10 additional items (5 functions + 5 classes) for masked convolution that are FCMAE-specific and stay in `fcmae.py`. FCMAE imports `UNeXt2Decoder`, `PixelToVoxelHead`, and `PixelToVoxelShuffleHead` from `_components`, which are all available. State dict key compatibility is ensured by preserving module attribute names verbatim (`self.stem`, `self.encoder_stages`, `self.decoder`, `self.head` for UNeXt2; `self.encoder`, `self.decoder`, `self.head` for FCMAE).

**Primary recommendation:** This is a straightforward copy-and-rewire migration. The critical discipline is not changing any module attribute names. The real new work is writing UNeXt2 forward-pass tests covering multiple configurations (varying backbone, channel counts, stack depths, decoder modes).

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.10 | nn.Module base, tensor ops | All models are nn.Module |
| timm | >=1.0.15 (installed: 1.0.24) | ConvNeXtV2 backbone, ConvNeXtStage, masked conv helpers | UNeXt2 uses `timm.create_model()`, FCMAE uses `timm.models.convnext.*` |
| monai | >=1.5.2 | UpSample, Convolution, ResidualUnit, get_conv_layer | Decoder and head upsampling blocks |
| numpy | >=2.4.1 | (indirect, via ConvBlock) | Not directly needed by UNeXt2/FCMAE, but in package deps |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | >=9.0.2 | Test framework | All model tests |

### Alternatives Considered
None -- all libraries are locked from Phase 6 decisions.

**Installation:**
```bash
uv sync --package viscy-models
```

## Architecture Patterns

### Target File Structure After Phase 7
```
packages/viscy-models/src/viscy_models/
    _components/           # Phase 6: DONE
        __init__.py
        stems.py           # UNeXt2Stem, StemDepthtoChannels
        heads.py           # PixelToVoxelHead, UnsqueezeHead, PixelToVoxelShuffleHead
        blocks.py          # UNeXt2UpStage, UNeXt2Decoder, icnr_init, _get_convnext_stage
    unet/
        __init__.py        # Phase 7: export UNeXt2, FullyConvolutionalMAE
        _layers/           # Phase 6: DONE
            __init__.py
            conv_block_2d.py
            conv_block_3d.py
        unext2.py          # Phase 7: UNeXt2 class ONLY (~50 lines)
        fcmae.py           # Phase 7: FCMAE + 10 FCMAE-specific items (~350 lines)
packages/viscy-models/tests/
    test_unet/
        __init__.py
        test_layers.py     # Phase 6: DONE (ConvBlock2D/3D)
        test_unext2.py     # Phase 7: NEW forward-pass tests
        test_fcmae.py      # Phase 7: MIGRATED from existing tests
```

### Pattern 1: UNeXt2 Migration (Thin Wrapper)
**What:** UNeXt2 is a composition layer that wires extracted components together. The class body is ~50 lines of init + 10 lines of forward.
**When to use:** This is the pattern for UNeXt2 specifically.

```python
# packages/viscy-models/src/viscy_models/unet/unext2.py
from typing import Literal

import timm
from torch import Tensor, nn

from viscy_models._components.blocks import UNeXt2Decoder
from viscy_models._components.heads import PixelToVoxelHead
from viscy_models._components.stems import UNeXt2Stem


class UNeXt2(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        in_stack_depth: int = 5,
        out_stack_depth: int = None,
        backbone: str = "convnextv2_tiny",
        pretrained: bool = False,
        stem_kernel_size: tuple[int, int, int] = (5, 4, 4),
        decoder_mode: Literal["deconv", "pixelshuffle"] = "pixelshuffle",
        decoder_conv_blocks: int = 2,
        decoder_norm_layer: str = "instance",
        decoder_upsample_pre_conv: bool = False,
        head_pool: bool = False,
        head_expansion_ratio: int = 4,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        # ... (body identical to original, imports changed)
```

**Key observations from source analysis:**
- `out_stack_depth` defaults to `None` and gets set to `in_stack_depth` in `__init__` body -- keep this pattern
- `self.encoder_stages` (NOT `self.encoder`) is the attribute name for the timm model
- `self.out_stack_depth` is stored as a public attribute
- `self.num_blocks` is a property returning 6 (hardcoded)
- The strides array `[2] * (len(num_channels) - 1) + [stem_kernel_size[-1]]` creates one extra entry beyond what the decoder uses -- preserve this exactly

### Pattern 2: FCMAE Migration (Self-Contained with Component Imports)
**What:** FCMAE contains 10 FCMAE-specific items plus the main class. Only 3 imports come from `_components`.
**When to use:** This is the pattern for FCMAE specifically.

```python
# packages/viscy-models/src/viscy_models/unet/fcmae.py
# Change: from viscy.unet.networks.unext2 import PixelToVoxelHead, UNeXt2Decoder
# To:
from viscy_models._components.blocks import UNeXt2Decoder
from viscy_models._components.heads import PixelToVoxelHead, PixelToVoxelShuffleHead
```

**FCMAE-specific items that stay in fcmae.py (NOT extracted):**
1. `_init_weights()` -- weight initialization function
2. `generate_mask()` -- random mask generation
3. `upsample_mask()` -- mask spatial upsampling
4. `masked_patchify()` -- apply mask to feature maps
5. `masked_unpatchify()` -- reconstruct from masked features
6. `MaskedConvNeXtV2Block` -- masked depthwise conv block
7. `MaskedConvNeXtV2Stage` -- multi-block masked stage
8. `MaskedAdaptiveProjection` -- 2D/3D adaptive stem with masking
9. `MaskedMultiscaleEncoder` -- full masked encoder
10. `FullyConvolutionalMAE` -- top-level model class

### Pattern 3: Mutable Default Fix (COMPAT-02)
**What:** FCMAE has two mutable list defaults that must be converted to tuples.
**When to use:** During FCMAE migration.

```python
# BEFORE (original fcmae.py):
class FullyConvolutionalMAE(nn.Module):
    def __init__(
        self,
        encoder_blocks: Sequence[int] = [3, 3, 9, 3],   # MUTABLE
        dims: Sequence[int] = [96, 192, 384, 768],       # MUTABLE
        ...

# AFTER (migrated):
class FullyConvolutionalMAE(nn.Module):
    def __init__(
        self,
        encoder_blocks: Sequence[int] = (3, 3, 9, 3),   # IMMUTABLE
        dims: Sequence[int] = (96, 192, 384, 768),       # IMMUTABLE
        ...
```

Internal code that does `list(dims)` or `list(encoder_blocks)` continues to work because tuples are iterable and `Sequence[int]` accepts both.

### Pattern 4: unet/__init__.py Public Exports
**What:** The `unet/__init__.py` file exports both model classes for clean imports.
**When to use:** After both models are migrated.

```python
# packages/viscy-models/src/viscy_models/unet/__init__.py
"""UNet family architectures."""

from viscy_models.unet.fcmae import FullyConvolutionalMAE
from viscy_models.unet.unext2 import UNeXt2

__all__ = ["UNeXt2", "FullyConvolutionalMAE"]
```

### Anti-Patterns to Avoid
- **Renaming module attributes:** `self.encoder_stages` must NOT become `self.encoder`. UNeXt2 uses `encoder_stages`, FCMAE uses `encoder`. They are different models with different attribute names.
- **Extracting FCMAE-specific classes to _components:** `MaskedConvNeXtV2Block`, `MaskedConvNeXtV2Stage`, etc. are ONLY used by FCMAE. They do not belong in `_components/`.
- **Changing the `num_channels` mutation pattern:** In UNeXt2.__init__, `num_channels = multi_scale_encoder.feature_info.channels()` returns a list, and later `decoder_channels = num_channels; decoder_channels.reverse()` mutates it. This is the original behavior; do not "fix" this -- it would change the decoder_channels computation.
- **Re-exporting FCMAE internal classes from unet/__init__.py:** Only `FullyConvolutionalMAE` and `UNeXt2` should be in `__all__`. Internal FCMAE helper classes/functions are implementation details.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ConvNeXtV2 encoder backbone | Custom encoder | `timm.create_model(backbone, features_only=True)` | Pretrained weights, multi-scale features, proven architecture |
| Masked depthwise convolution | Custom sparse conv | FCMAE's `MaskedConvNeXtV2Block` (from original code) | ConvNeXt V2 paper reference implementation |
| Pixel shuffle upsampling | Custom upsampler | `monai.networks.blocks.UpSample(mode="pixelshuffle")` | Handles padding, pre-conv, tested |
| Weight initialization | Manual init | `timm.models.convnext._init_weights` and `icnr_init` | ICNR prevents checkerboard artifacts |

**Key insight:** Both UNeXt2 and FCMAE are composition layers over timm + monai primitives. The migration is import-path rewiring, not architecture work.

## Common Pitfalls

### Pitfall 1: num_channels List Mutation in UNeXt2.__init__
**What goes wrong:** The original `UNeXt2.__init__` calls `multi_scale_encoder.feature_info.channels()` which returns a list `[96, 192, 384, 768]`. It then does `decoder_channels = num_channels` (aliasing, not copying), followed by `decoder_channels.reverse()` and `decoder_channels[-1] = ...`. This mutates the original list.
**Why it happens:** Python list aliasing. `decoder_channels` and `num_channels` are the same object.
**How to avoid:** Preserve this exact behavior during migration. Do NOT "fix" it by doing `decoder_channels = list(num_channels)` -- this could subtly change behavior if any downstream code references `num_channels` after the mutation. The original works because nothing reads `num_channels` after it is reversed.
**Warning signs:** If you change this pattern and see wrong decoder channel counts, this is why.

### Pitfall 2: FCMAE Import Path Change
**What goes wrong:** Original FCMAE imports `from viscy.unet.networks.unext2 import PixelToVoxelHead, UNeXt2Decoder`. The migrated version must import from `viscy_models._components`.
**Why it happens:** Components were extracted to `_components/` in Phase 6.
**How to avoid:** Change exactly three import lines:
  - `from viscy.unet.networks.unext2 import PixelToVoxelHead, UNeXt2Decoder` becomes two imports from `viscy_models._components.heads` and `viscy_models._components.blocks`
  - `PixelToVoxelShuffleHead` is already defined in fcmae.py in the original, but was also extracted to `_components/heads.py` in Phase 6. Import it from `_components` instead of redefining it.
**Warning signs:** `ImportError` at import time.

### Pitfall 3: PixelToVoxelShuffleHead Duplication
**What goes wrong:** In the original codebase, `PixelToVoxelShuffleHead` is defined in BOTH `unext2.py` and `fcmae.py`. Phase 6 extracted it to `_components/heads.py`. During FCMAE migration, you must import from `_components` and NOT copy the class definition again.
**Why it happens:** Historical code duplication in the original monolith.
**How to avoid:** Import `PixelToVoxelShuffleHead` from `viscy_models._components.heads` in fcmae.py. Verify the definition is identical (it is -- confirmed by source analysis).
**Warning signs:** Two definitions of the same class causing confusion or state dict mismatches.

### Pitfall 4: FCMAE Test Import Updates
**What goes wrong:** All 11 existing FCMAE tests import from `viscy.unet.networks.fcmae`. These must be updated to `viscy_models.unet.fcmae`.
**Why it happens:** Standard migration import update.
**How to avoid:** Systematic find-and-replace of import paths in the migrated test file. Also note that `test_pixel_to_voxel_shuffle_head` tests a class that now lives in `_components.heads` -- update its import to come from the FCMAE module (since it is re-exported there) or from `_components.heads` directly.
**Warning signs:** `ModuleNotFoundError` when running tests.

### Pitfall 5: UNeXt2 Test Memory
**What goes wrong:** UNeXt2 with `convnextv2_tiny` backbone on a 256x256x5 input creates ~273 state dict keys. Full forward-pass tests use significant memory.
**Why it happens:** ConvNeXtV2 tiny has ~28M parameters.
**How to avoid:** Use smaller spatial sizes (128x128 or 64x64) and smaller backbones (`convnextv2_atto` has 3.7M params, channels [40, 80, 160, 320]) for parametrized tests. Reserve one test with the default `convnextv2_tiny` for validation. Use `torch.no_grad()` in forward-pass tests.
**Warning signs:** OOM errors in CI, slow test execution.

### Pitfall 6: FCMAE pretraining Flag Behavior
**What goes wrong:** FCMAE's `forward()` returns a tuple `(output, mask)` when `self.pretraining=True` (default) but just `output` when `False`. Tests must handle both return types.
**Why it happens:** Design decision for training vs inference modes.
**How to avoid:** Test both modes. When `pretraining=True`: expect `(Tensor, BoolTensor|None)`. When `pretraining=False`: expect `Tensor`.
**Warning signs:** Test assertions fail because of unpacking a non-tuple return.

## Code Examples

### UNeXt2 Complete Migration (verified working)
```python
# Source: main branch viscy/unet/networks/unext2.py
# Verified: full forward pass tested with Phase 6 components (2026-02-12)

# packages/viscy-models/src/viscy_models/unet/unext2.py
from typing import Literal

import timm
from torch import Tensor, nn

from viscy_models._components.blocks import UNeXt2Decoder
from viscy_models._components.heads import PixelToVoxelHead
from viscy_models._components.stems import UNeXt2Stem


class UNeXt2(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        in_stack_depth: int = 5,
        out_stack_depth: int = None,
        backbone: str = "convnextv2_tiny",
        pretrained: bool = False,
        stem_kernel_size: tuple[int, int, int] = (5, 4, 4),
        decoder_mode: Literal["deconv", "pixelshuffle"] = "pixelshuffle",
        decoder_conv_blocks: int = 2,
        decoder_norm_layer: str = "instance",
        decoder_upsample_pre_conv: bool = False,
        head_pool: bool = False,
        head_expansion_ratio: int = 4,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if in_stack_depth % stem_kernel_size[0] != 0:
            raise ValueError(
                f"Input stack depth {in_stack_depth} is not divisible "
                f"by stem kernel depth {stem_kernel_size[0]}."
            )
        if out_stack_depth is None:
            out_stack_depth = in_stack_depth
        multi_scale_encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            drop_path_rate=drop_path_rate,
        )
        num_channels = multi_scale_encoder.feature_info.channels()
        # replace first convolution layer with a projection tokenizer
        multi_scale_encoder.stem_0 = nn.Identity()
        self.encoder_stages = multi_scale_encoder
        self.stem = UNeXt2Stem(
            in_channels, num_channels[0], stem_kernel_size, in_stack_depth
        )
        decoder_channels = num_channels
        decoder_channels.reverse()
        decoder_channels[-1] = (
            (out_stack_depth + 2) * out_channels * 2**2 * head_expansion_ratio
        )
        self.decoder = UNeXt2Decoder(
            decoder_channels,
            norm_name=decoder_norm_layer,
            mode=decoder_mode,
            conv_blocks=decoder_conv_blocks,
            strides=[2] * (len(num_channels) - 1) + [stem_kernel_size[-1]],
            upsample_pre_conv="default" if decoder_upsample_pre_conv else None,
        )
        self.head = PixelToVoxelHead(
            decoder_channels[-1],
            out_channels,
            out_stack_depth,
            head_expansion_ratio,
            pool=head_pool,
        )
        self.out_stack_depth = out_stack_depth

    @property
    def num_blocks(self) -> int:
        """2-times downscaling factor of the smallest feature map"""
        return 6

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x: list = self.encoder_stages(x)
        x.reverse()
        x = self.decoder(x)
        return self.head(x)
```

### FCMAE Import Changes (the only changes needed)
```python
# BEFORE (original fcmae.py):
from viscy.unet.networks.unext2 import PixelToVoxelHead, UNeXt2Decoder

# AFTER (migrated fcmae.py):
from viscy_models._components.blocks import UNeXt2Decoder
from viscy_models._components.heads import PixelToVoxelHead, PixelToVoxelShuffleHead

# Note: PixelToVoxelShuffleHead was DEFINED in the original fcmae.py
# but is now imported from _components.heads (extracted in Phase 6).
# Remove the class definition from fcmae.py and import instead.
```

### UNeXt2 Test Coverage (NEW -- UNET-06)
```python
# packages/viscy-models/tests/test_unet/test_unext2.py
import pytest
import torch

from viscy_models.unet import UNeXt2


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_unext2_default_forward(device):
    """Default UNeXt2: 1ch in, 1ch out, depth=5, convnextv2_tiny."""
    model = UNeXt2().to(device)
    x = torch.randn(1, 1, 5, 128, 128, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 5, 128, 128)


def test_unext2_multichannel(device):
    """Multi-channel: 3ch in, 2ch out."""
    model = UNeXt2(in_channels=3, out_channels=2).to(device)
    x = torch.randn(1, 3, 5, 128, 128, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2, 5, 128, 128)


def test_unext2_different_depths(device):
    """Different in/out stack depths."""
    model = UNeXt2(in_stack_depth=5, out_stack_depth=3).to(device)
    x = torch.randn(1, 1, 5, 128, 128, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 3, 128, 128)


def test_unext2_small_backbone(device):
    """Smaller backbone for faster testing."""
    model = UNeXt2(backbone="convnextv2_atto").to(device)
    x = torch.randn(1, 1, 5, 128, 128, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 5, 128, 128)


def test_unext2_stem_validation():
    """Raises ValueError when stack depth not divisible by stem kernel."""
    with pytest.raises(ValueError, match="not divisible"):
        UNeXt2(in_stack_depth=7, stem_kernel_size=(5, 4, 4))
```

### FCMAE Test Migration (UNET-07)
```python
# packages/viscy-models/tests/test_unet/test_fcmae.py
# Import changes only:
# BEFORE: from viscy.unet.networks.fcmae import ...
# AFTER:  from viscy_models.unet.fcmae import ...

import torch

from viscy_models.unet.fcmae import (
    FullyConvolutionalMAE,
    MaskedAdaptiveProjection,
    MaskedConvNeXtV2Block,
    MaskedConvNeXtV2Stage,
    MaskedMultiscaleEncoder,
    generate_mask,
    masked_patchify,
    masked_unpatchify,
    upsample_mask,
)
from viscy_models._components.heads import PixelToVoxelShuffleHead

# ... rest of test functions unchanged except import paths ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| All models in one `unext2.py` file | Shared components in `_components/`, models in separate files | Phase 6 (2026-02-12) | Clean separation, reusable components |
| `from viscy.unet.networks.unext2 import ...` | `from viscy_models.unet import UNeXt2` | Phase 7 (current) | Independent package, clean API |
| Mutable list defaults in FCMAE | Tuple defaults | Phase 7 (current) | Prevents shared-state bugs |
| No UNeXt2 tests | Forward-pass tests covering multiple configs | Phase 7 (current) | Catches regression |

**Deprecated/outdated:**
- `PixelToVoxelShuffleHead` defined in fcmae.py: Now imported from `_components.heads` where it was extracted.

## UNeXt2 Verified Configurations

Forward-pass tested during research (all produce correct output shapes):

| Config | Backbone | In/Out Ch | Stack Depth | Spatial | Output Shape |
|--------|----------|-----------|-------------|---------|--------------|
| Default | convnextv2_tiny | 1/2 | 5/5 | 256x256 | (1, 2, 5, 256, 256) |
| Pool head | convnextv2_tiny | 1/2 | 5/5 | 128x128 | (1, 2, 5, 128, 128) |
| Diff depth | convnextv2_tiny | 1/2 | 5/3 | 256x256 | (1, 2, 3, 256, 256) |

ConvNeXtV2 backbone channel configurations (from timm 1.0.24):

| Backbone | Channels | Approx Params |
|----------|----------|---------------|
| convnextv2_atto | [40, 80, 160, 320] | 3.7M |
| convnextv2_femto | [48, 96, 192, 384] | 5.2M |
| convnextv2_pico | [64, 128, 256, 512] | 9.1M |
| convnextv2_nano | [80, 160, 320, 640] | 15.6M |
| convnextv2_tiny | [96, 192, 384, 768] | 28.6M |

## FCMAE Migration Inventory

### Items to copy into fcmae.py (FCMAE-specific)
| Item | Type | Lines (approx) |
|------|------|----------------|
| `_init_weights` | function | 12 |
| `generate_mask` | function | 12 |
| `upsample_mask` | function | 12 |
| `masked_patchify` | function | 10 |
| `masked_unpatchify` | function | 12 |
| `MaskedConvNeXtV2Block` | class | 40 |
| `MaskedConvNeXtV2Stage` | class | 40 |
| `MaskedAdaptiveProjection` | class | 40 |
| `MaskedMultiscaleEncoder` | class | 45 |
| `FullyConvolutionalMAE` | class | 55 |

### Items imported from _components (already extracted)
| Item | Source in _components |
|------|---------------------|
| `UNeXt2Decoder` | `_components.blocks` |
| `PixelToVoxelHead` | `_components.heads` |
| `PixelToVoxelShuffleHead` | `_components.heads` |

### Mutable defaults to fix
| Parameter | Current | Fixed |
|-----------|---------|-------|
| `encoder_blocks` | `[3, 3, 9, 3]` | `(3, 3, 9, 3)` |
| `dims` | `[96, 192, 384, 768]` | `(96, 192, 384, 768)` |

### Existing tests to migrate (11 tests)
| Test Function | Tests What |
|---------------|-----------|
| `test_generate_mask` | Mask shape and ratio |
| `test_masked_patchify` | Patchify with mask |
| `test_unmasked_patchify_roundtrip` | Roundtrip without mask |
| `test_masked_patchify_roundtrip` | Roundtrip with mask |
| `test_masked_convnextv2_block` | Block forward + masking |
| `test_masked_convnextv2_stage` | Stage forward + masking |
| `test_adaptive_projection` | 2D/3D projection |
| `test_masked_multiscale_encoder` | Encoder feature shapes |
| `test_pixel_to_voxel_shuffle_head` | Shuffle head output |
| `test_fcmae` | Full model forward |
| `test_fcmae_head_conv` | Model with conv head |

## State Dict Key Verification

### UNeXt2 State Dict Prefixes (verified)
Top-level: `encoder_stages`, `stem`, `decoder`, `head`
- `stem.conv.weight`, `stem.conv.bias`
- `encoder_stages.stem_1.weight` (timm's stem_1, after stem_0=Identity)
- `encoder_stages.stages_0.blocks.0.conv_dw.weight`
- `decoder.decoder_stages.0.upsample.*`
- `decoder.decoder_stages.0.conv.blocks.*`
- `head.conv.0.conv.weight`, `head.conv.1.weight`

Total keys: 273 (convnextv2_tiny backbone)

### FCMAE State Dict Prefixes (from source analysis)
Top-level: `encoder`, `decoder`, `head`
- `encoder.stem.conv3d.weight`, `encoder.stem.conv2d.weight`, `encoder.stem.norm.weight`
- `encoder.stages.0.downsample.*`, `encoder.stages.0.blocks.*`
- `decoder.decoder_stages.*` (same structure as UNeXt2)
- `head.upsample.*` or `head.out.*`

## Open Questions

1. **Should UNeXt2 tests use `convnextv2_tiny` or smaller backbone?**
   - What we know: `convnextv2_tiny` is the default and most common config. `convnextv2_atto` is 8x smaller (3.7M vs 28.6M params).
   - Recommendation: Use `convnextv2_atto` for parametrized tests (speed), one test with `convnextv2_tiny` for default config validation. All forward-pass tests should use `torch.no_grad()` to reduce memory.

2. **Should `test_pixel_to_voxel_shuffle_head` stay in FCMAE tests?**
   - What we know: Phase 6 already has this test in `test_components/test_heads.py` but with different parameters.
   - Recommendation: Keep both. The FCMAE test uses `(240, 3, 5, 4)` params matching FCMAE usage; the component test uses `(160, 2, 5, 4)` params. Different coverage is valuable.

3. **Should FCMAE tests import `PixelToVoxelShuffleHead` from `_components.heads` or `unet.fcmae`?**
   - What we know: The class will be imported (not defined) in fcmae.py. It could be re-exported or not.
   - Recommendation: Import from `viscy_models._components.heads` in the test since that is the canonical location. The FCMAE module should NOT re-export it in `__all__`.

## Sources

### Primary (HIGH confidence)
- **Local repo main branch** -- `viscy/unet/networks/unext2.py` (UNeXt2 source, ~290 lines)
- **Local repo main branch** -- `viscy/unet/networks/fcmae.py` (FCMAE source, ~370 lines)
- **Local repo main branch** -- `tests/unet/test_fcmae.py` (existing FCMAE tests, 11 test functions)
- **Local repo modular-models branch** -- Phase 6 extracted components (stems.py, heads.py, blocks.py) verified working
- **Runtime verification** -- UNeXt2 full forward pass tested with extracted components: `(1, 1, 5, 256, 256)` -> `(1, 2, 5, 256, 256)`, confirmed 2026-02-12
- **Runtime verification** -- timm 1.0.24 API: `timm.create_model('convnextv2_tiny', features_only=True)`, ConvNeXtStage, all FCMAE timm imports verified
- **Runtime verification** -- State dict keys: 273 keys with prefixes `encoder_stages`, `stem`, `decoder`, `head`

### Secondary (MEDIUM confidence)
- None needed -- all critical claims verified by direct source analysis and runtime testing.

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All dependencies verified from lockfile and runtime imports
- Architecture: HIGH -- Full UNeXt2 forward pass verified with extracted components; FCMAE dependencies confirmed available
- Pitfalls: HIGH -- State dict keys enumerated from runtime model instantiation; mutable defaults identified from source
- Test design: HIGH -- Existing FCMAE tests read from source; UNeXt2 test configs derived from verified forward passes

**Research date:** 2026-02-12
**Valid until:** 2026-03-12 (stable domain; timm/monai APIs verified at current versions)
