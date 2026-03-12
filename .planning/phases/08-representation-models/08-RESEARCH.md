# Phase 8: Representation Models - Research

**Researched:** 2026-02-12
**Domain:** Contrastive encoder and VAE model migration, nn.Module extraction, forward-pass testing
**Confidence:** HIGH

## Summary

Phase 8 migrates four representation models from the pre-monorepo VisCy codebase into the `viscy-models` package: `ContrastiveEncoder`, `ResNet3dEncoder`, `BetaVae25D`, and `BetaVaeMonai`. The source code lives at `viscy/representation/contrastive.py` (latest version at commit fe7a5da^, which includes both `projection_mlp()`, `ContrastiveEncoder`, and `ResNet3dEncoder` from PR #285) and `viscy/representation/vae.py` (same commit, containing `VaeUpStage`, `VaeEncoder`, `VaeDecoder`, `BetaVae25D`, and `BetaVaeMonai`).

The contrastive models are straightforward wrappers: `ContrastiveEncoder` wraps a timm backbone (convnext_tiny/convnextv2_tiny/resnet50) with `StemDepthtoChannels` + a projection MLP, while `ResNet3dEncoder` wraps MONAI's `ResNetFeatures` with a projection MLP. The VAE models are more complex: `BetaVae25D` composes `VaeEncoder` (timm backbone with 3D-to-2D stem) and `VaeDecoder` (custom upsampling stages with `PixelToVoxelHead`), while `BetaVaeMonai` is a thin wrapper around MONAI's `VarAutoEncoder`. Per the Phase 6 research decision, `VaeUpStage`, `VaeEncoder`, and `VaeDecoder` stay in `vae/` module -- NOT in `_components/`.

A critical finding during research: the `ContrastiveEncoder` code uses `encoder.head.fc.in_features` unconditionally for all backbones, but timm's ResNet models expose the classifier at `encoder.fc` (not `encoder.head.fc`). ConvNeXt models DO have `head.fc`. This means the `backbone="resnet50"` path has always been broken for `ContrastiveEncoder`. Since the original code has this bug and there are no existing checkpoints using this path (no tests ever existed), the migration should fix it by using timm's uniform API (`model.num_features` / `model.get_classifier()`). However, this fix must be carefully scoped to preserve state dict keys for the working convnext backbone paths.

**Primary recommendation:** Split into two plans -- (1) contrastive models with `projection_mlp` utility, (2) VAE models with `VaeUpStage`/`VaeEncoder`/`VaeDecoder` helper classes. Use `pretrained=False` in all tests. Fix the ResNet50 backbone bug in ContrastiveEncoder using timm's `num_features` API.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.10 | Neural network framework | Locked by workspace; all models are nn.Module |
| timm | >=1.0.15 | ConvNeXt/ResNet backbones for ContrastiveEncoder and VaeEncoder | `timm.create_model()` with `features_only` and classification modes |
| monai | >=1.5.2 | ResNetFeatures for ResNet3dEncoder, VarAutoEncoder for BetaVaeMonai, UpSample/ResidualUnit for VaeUpStage | Medical imaging network blocks |
| numpy | >=2.4.1 | Array operations (indirect, via existing deps) | Already in viscy-models dependencies |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | >=9.0.2 | Test framework | Forward-pass tests for all 4 models |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| timm backbones | torchvision ResNet/ConvNeXt | Would break state dict keys; timm is already a dependency |
| MONAI VarAutoEncoder | Custom VAE | Would need to reimplement encoder/decoder/reparameterize; MONAI is already a dependency |
| MONAI ResNetFeatures | timm 3D ResNet | timm does not support 3D ResNets; MONAI provides MedicalNet pretrained weights |

## Architecture Patterns

### File Layout for Phase 8

```
packages/viscy-models/src/viscy_models/
    contrastive/
        __init__.py          # Re-exports ContrastiveEncoder, ResNet3dEncoder
        encoder.py           # ContrastiveEncoder + projection_mlp()
        resnet3d.py          # ResNet3dEncoder
    vae/
        __init__.py          # Re-exports BetaVae25D, BetaVaeMonai
        beta_vae_25d.py      # BetaVae25D, VaeEncoder, VaeDecoder, VaeUpStage
        beta_vae_monai.py    # BetaVaeMonai

packages/viscy-models/tests/
    test_contrastive/
        __init__.py
        test_encoder.py      # ContrastiveEncoder forward-pass tests
        test_resnet3d.py     # ResNet3dEncoder forward-pass tests
    test_vae/
        __init__.py
        test_beta_vae_25d.py # BetaVae25D forward-pass tests
        test_beta_vae_monai.py # BetaVaeMonai forward-pass tests
```

### Pattern 1: Contrastive Model Migration (ContrastiveEncoder)

**What:** Migrate ContrastiveEncoder from `viscy/representation/contrastive.py` (commit fe7a5da^), updating imports from `viscy.unet.networks.unext2` to `viscy_models._components.stems`.

**Import changes (the ONLY changes from original):**
```python
# BEFORE (original):
from viscy.unet.networks.unext2 import StemDepthtoChannels

# AFTER (migrated):
from viscy_models._components.stems import StemDepthtoChannels
```

**State dict keys preserved:** `self.stem`, `self.encoder`, `self.projection` -- these three module attribute names must remain identical.

### Pattern 2: Contrastive Model Migration (ResNet3dEncoder)

**What:** Migrate ResNet3dEncoder to `contrastive/resnet3d.py`. This model uses MONAI's `ResNetFeatures` instead of timm, and does NOT use `StemDepthtoChannels` (it takes native 3D input).

**Import changes:**
```python
# BEFORE (original):
from viscy.unet.networks.unext2 import StemDepthtoChannels  # NOT used by ResNet3dEncoder
# The function projection_mlp was in the same file

# AFTER (migrated):
from viscy_models.contrastive.encoder import projection_mlp  # Import from sibling
```

**State dict keys preserved:** `self.encoder`, `self.projection`

### Pattern 3: VAE Model Migration (BetaVae25D with helpers)

**What:** Migrate BetaVae25D along with its helper classes VaeUpStage, VaeEncoder, VaeDecoder to `vae/beta_vae_25d.py`. Per Phase 6 decision, these helpers stay in the vae/ module, NOT in _components/.

**Import changes:**
```python
# BEFORE (original):
from viscy.unet.networks.unext2 import PixelToVoxelHead, StemDepthtoChannels

# AFTER (migrated):
from viscy_models._components.heads import PixelToVoxelHead
from viscy_models._components.stems import StemDepthtoChannels
```

**State dict keys preserved:**
- BetaVae25D: `self.encoder` (VaeEncoder), `self.decoder` (VaeDecoder)
- VaeEncoder: `self.stem`, `self.encoder`, `self.fc`, `self.fc_mu`, `self.fc_logvar`
- VaeDecoder: `self.latent_reshape`, `self.latent_proj`, `self.decoder_stages`, `self.head`
- VaeUpStage: `self.upsample`, `self.conv`

### Pattern 4: VAE Model Migration (BetaVaeMonai)

**What:** Migrate BetaVaeMonai to `vae/beta_vae_monai.py`. This is the simplest model -- a thin wrapper around MONAI's VarAutoEncoder.

**Import changes:** None needed. BetaVaeMonai only imports from monai and torch (no viscy imports).

**State dict keys preserved:** `self.model` (VarAutoEncoder)

### Pattern 5: projection_mlp Placement

**What:** The `projection_mlp()` function is used by both `ContrastiveEncoder` and `ResNet3dEncoder`. Per Phase 6 research recommendation, it stays in the contrastive module (NOT _components/) since it is only used by contrastive models.

**Placement:** Define in `contrastive/encoder.py`, import in `contrastive/resnet3d.py`.

```python
# contrastive/encoder.py
def projection_mlp(in_dims: int, hidden_dims: int, out_dims: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dims, hidden_dims),
        nn.BatchNorm1d(hidden_dims),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dims, out_dims),
        nn.BatchNorm1d(out_dims),
    )

# contrastive/resnet3d.py
from viscy_models.contrastive.encoder import projection_mlp
```

### Pattern 6: Mutable Default Fix (COMPAT-02)

**What:** VaeDecoder has two mutable list defaults that must be converted to tuples.

```python
# BEFORE (mutable defaults):
class VaeDecoder(nn.Module):
    def __init__(
        self,
        decoder_channels: list[int] = [1024, 512, 256, 128],  # MUTABLE
        strides: list[int] = [2, 2, 2, 1],                     # MUTABLE
    ):

# AFTER (immutable defaults):
class VaeDecoder(nn.Module):
    def __init__(
        self,
        decoder_channels: Sequence[int] = (1024, 512, 256, 128),  # IMMUTABLE
        strides: Sequence[int] = (2, 2, 2, 1),                     # IMMUTABLE
    ):
```

**Note:** ContrastiveEncoder, ResNet3dEncoder, VaeEncoder, BetaVae25D, and BetaVaeMonai all use immutable defaults already (tuples or scalars). Only VaeDecoder needs fixing.

### Anti-Patterns to Avoid

- **Moving VaeUpStage/VaeEncoder/VaeDecoder to _components/:** Phase 6 research explicitly decided these stay in `vae/`. They are VAE-specific, not shared across model families.
- **Putting projection_mlp in _components/:** Only used by contrastive models. Not a shared component.
- **Using pretrained=True in tests:** Both timm and MONAI pretrained options download weights from the internet. Tests MUST use `pretrained=False` to avoid network calls and keep tests fast/deterministic.
- **Splitting VaeEncoder/VaeDecoder into separate files:** They are tightly coupled to BetaVae25D and should stay in `beta_vae_25d.py` as internal classes.
- **Changing module attribute names:** `self.stem`, `self.encoder`, `self.projection`, `self.decoder`, `self.model`, `self.head` etc. must remain exactly as-is.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 2D ConvNeXt/ResNet backbone | Custom 2D encoder | `timm.create_model()` | Pretrained weights, state dict compatibility |
| 3D ResNet feature extractor | Custom 3D ResNet | `monai.networks.nets.resnet.ResNetFeatures` | MedicalNet pretrained weights, spatial_dims=3 support |
| Full VAE architecture | Custom encoder-decoder | `monai.networks.nets.VarAutoEncoder` | Handles latent space, reparameterization, encoder/decoder symmetry |
| 2D upsampling with residuals | Custom upsampling | `monai.networks.blocks.UpSample` + `monai.networks.blocks.ResidualUnit` | Edge cases handled, medical imaging validated |
| Transposed convolution layer | Custom deconv | `monai.networks.blocks.dynunet_block.get_conv_layer` | Handles normalization, padding, transposed convolutions correctly |
| Projection MLP | Custom classifier head | `projection_mlp()` from `contrastive/encoder.py` | Consistent architecture, shared between ContrastiveEncoder and ResNet3dEncoder |

## Common Pitfalls

### Pitfall 1: ContrastiveEncoder ResNet50 Backbone Bug

**What goes wrong:** The original `ContrastiveEncoder` code accesses `encoder.head.fc.in_features` and `encoder.head.fc = nn.Identity()` unconditionally. For timm's ConvNeXt models, `head.fc` exists. For timm's ResNet models, the classifier is at `encoder.fc` -- there is NO `head` attribute.

**Why it happens:** The original code was likely only tested with ConvNeXt backbones (the default). The ResNet50 path in `ContrastiveEncoder` was never exercised (no tests existed).

**How to fix during migration:** Use timm's uniform API to handle both backbone families:
```python
# Instead of: encoder.head.fc.in_features
# Use: encoder.num_features (works for ALL timm models)
classifier_in_features = encoder.num_features

# Instead of: encoder.head.fc = nn.Identity()
# Use: encoder.reset_classifier(0) or backbone-specific replacement
if "convnext" in backbone:
    encoder.head.fc = nn.Identity()
elif "resnet" in backbone:
    encoder.fc = nn.Identity()
```

**State dict impact:** For ConvNeXt backbones (the only ones with existing checkpoints), the state dict keys are unchanged. The `encoder.head.fc` replacement with `nn.Identity()` results in an `encoder.head.fc` key that stores nothing meaningful. For ResNet50, there are no existing checkpoints to worry about.

**Confidence:** HIGH -- Verified by creating both model types and inspecting `named_children()` output.

### Pitfall 2: ResNet3dEncoder embedding_dim Mismatch

**What goes wrong:** `ResNet3dEncoder(backbone="resnet50", embedding_dim=512)` creates a projection MLP with `in_dims=512`, but ResNet-50's last feature has 2048 channels. The `projection_mlp(512, 512, 128)` would receive 2048-dim input, causing a shape mismatch at runtime.

**Why it happens:** The `embedding_dim` parameter defaults to 512, which matches ResNet-10/18 but NOT ResNet-50/101/152.

**How to avoid:** Document the correct embedding_dim values. Tests should use matching values:
- ResNet-10, ResNet-18, ResNet-34: `embedding_dim=512`
- ResNet-50, ResNet-101, ResNet-152, ResNet-200: `embedding_dim=2048`

**Warning signs:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied` during forward pass.

**Confidence:** HIGH -- Verified by running `ResNetFeatures` with different backbones and checking output dims.

### Pitfall 3: BatchNorm1d with Batch Size 1

**What goes wrong:** `projection_mlp()` uses `nn.BatchNorm1d` which requires batch_size > 1 during training (eval mode works with batch_size=1).

**Why it happens:** BatchNorm computes mean/variance across the batch dimension. With batch_size=1, there is only one sample so variance is undefined.

**How to avoid:** Tests MUST either use batch_size >= 2 OR call `model.eval()` before the forward pass. Using both is safest.

**Warning signs:** `ValueError: Expected more than 1 value per channel when training`

**Confidence:** HIGH -- Directly encountered this error during MONAI ResNetFeatures testing.

### Pitfall 4: SimpleNamespace Return Type

**What goes wrong:** `BetaVae25D.forward()` and `BetaVaeMonai.forward()` return `SimpleNamespace` objects, not tensors or named tuples. This breaks TorchScript/ONNX export.

**Why it happens:** Original design choice for flexible attribute access.

**How to avoid:** Accept this as a known limitation. Do NOT change the return type during migration -- it would break all downstream code. Document in tests that output is `SimpleNamespace`.

**Scope note:** ONNX/TorchScript export is explicitly out of scope per REQUIREMENTS.md.

**Confidence:** HIGH -- Verified from source code.

### Pitfall 5: VaeEncoder ResNet50 Feature Count

**What goes wrong:** ResNet50 in `features_only=True` mode returns 5 feature levels (channels=[64, 256, 512, 1024, 2048]) while ConvNeXt returns 4 (channels=[96, 192, 384, 768]). The `backbone_reduction = 2^(len(num_channels)-1)` formula gives different results: 16 for ResNet50, 8 for ConvNeXt.

**Why it matters:** The VaeEncoder calculates `encoder_spatial_size` used by VaeDecoder. If this is wrong, the decoder produces wrong spatial dimensions.

**Verification:** Tested that with `conv1 = Identity()`, ResNet50's effective reduction from stem output to last feature IS 16x (`2^4`), matching the formula. The code is correct.

**Confidence:** HIGH -- Verified by running actual forward passes.

### Pitfall 6: VaeDecoder Strides Has Extra Element

**What goes wrong:** `BetaVae25D.__init__` creates `strides = [2] * decoder_stages + [1]`, which has `decoder_stages + 1` elements. But `VaeDecoder` iterates `len(decoder_channels) - 1` stages, using `strides[:len(decoder_channels)-1]`. The last stride `[1]` in the strides list is NOT used by any VaeUpStage.

**Why it matters:** During testing, if someone passes mismatched strides/decoder_channels, the extra stride element is silently ignored.

**How to avoid:** Preserve the original behavior exactly. Do not "fix" the strides list length.

**Confidence:** HIGH -- Verified from source code analysis.

### Pitfall 7: Test Input Size Constraints for BetaVae25D

**What goes wrong:** BetaVae25D requires specific spatial sizes that are divisible by the stem stride AND the backbone reduction factor. For ResNet50 with `stem_stride=(2,4,4)` and `input_spatial_size=(256,256)`, the final spatial size is 2x2. Using smaller inputs like `(64,64)` would give 1x1, which causes issues with the latent space calculation.

**Why it happens:** `spatial_channels = latent_dim // (spatial_size * spatial_size)`. If spatial_size=1, spatial_channels=latent_dim. If spatial_size=2, spatial_channels=latent_dim/4.

**How to avoid:** Use input sizes that produce reasonable final spatial sizes. For tests: `input_spatial_size=(128, 128)` with `stem_stride=(2, 4, 4)` gives final spatial 2x2 for ResNet50, which works. Use `latent_dim` values divisible by `spatial_size^2`.

**Test recommendation:** Use smaller latent_dim (e.g., 64 or 256) and smaller input (e.g., 128x128) to keep tests fast while maintaining correctness.

**Confidence:** HIGH -- Verified by manual calculation and forward pass simulation.

## Code Examples

### ContrastiveEncoder Migration (contrastive/encoder.py)

Source: `viscy/representation/contrastive.py` at commit fe7a5da^ (pre-monorepo)

```python
"""Contrastive encoder using timm backbones with 3D-to-2D stem."""

from typing import Literal

import timm
import torch.nn as nn
from torch import Tensor

from viscy_models._components.stems import StemDepthtoChannels


def projection_mlp(in_dims: int, hidden_dims: int, out_dims: int) -> nn.Module:
    """MLP projection head for contrastive learning."""
    return nn.Sequential(
        nn.Linear(in_dims, hidden_dims),
        nn.BatchNorm1d(hidden_dims),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dims, out_dims),
        nn.BatchNorm1d(out_dims),
    )


class ContrastiveEncoder(nn.Module):
    def __init__(
        self,
        backbone: Literal["convnext_tiny", "convnextv2_tiny", "resnet50"],
        in_channels: int,
        in_stack_depth: int,
        stem_kernel_size: tuple[int, int, int] = (5, 4, 4),
        stem_stride: tuple[int, int, int] = (5, 4, 4),
        embedding_dim: int = 768,
        projection_dim: int = 128,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        encoder = timm.create_model(
            backbone,
            pretrained=False,  # NOTE: changed from True to False for pure nn.Module
            features_only=False,
            drop_path_rate=drop_path_rate,
            num_classes=embedding_dim,
        )
        if "convnext" in backbone:
            in_channels_encoder = encoder.stem[0].out_channels
            encoder.stem[0] = nn.Identity()
        elif "resnet" in backbone:
            in_channels_encoder = encoder.conv1.out_channels
            encoder.conv1 = nn.Identity()
        # Use timm's uniform API for classifier access
        projection = projection_mlp(encoder.num_features, embedding_dim, projection_dim)
        # Reset classifier based on backbone type
        if "convnext" in backbone:
            encoder.head.fc = nn.Identity()
        elif "resnet" in backbone:
            encoder.fc = nn.Identity()
        self.stem = StemDepthtoChannels(...)
        self.encoder = encoder
        self.projection = projection

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.stem(x)
        embedding = self.encoder(x)
        projections = self.projection(embedding)
        return (embedding, projections)
```

### ResNet3dEncoder Migration (contrastive/resnet3d.py)

Source: `viscy/representation/contrastive.py` at commit fe7a5da^

```python
"""3D ResNet encoder using MONAI's ResNetFeatures."""

from monai.networks.nets.resnet import ResNetFeatures
from torch import Tensor, nn

from viscy_models.contrastive.encoder import projection_mlp


class ResNet3dEncoder(nn.Module):
    def __init__(
        self,
        backbone: str,
        in_channels: int = 1,
        embedding_dim: int = 512,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = ResNetFeatures(
            backbone, pretrained=False, spatial_dims=3, in_channels=in_channels
        )
        self.projection = projection_mlp(embedding_dim, embedding_dim, projection_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feature_map = self.encoder(x)[-1]
        embedding = self.encoder.avgpool(feature_map)
        embedding = embedding.view(embedding.size(0), -1)
        projections = self.projection(embedding)
        return (embedding, projections)
```

### BetaVae25D Forward-Pass Test Pattern

```python
"""Forward-pass tests for BetaVae25D."""

import pytest
import torch

from viscy_models.vae import BetaVae25D


def test_beta_vae_25d_resnet50(device):
    """BetaVae25D with resnet50: small latent, 128x128 input."""
    model = BetaVae25D(
        backbone="resnet50",
        in_channels=2,
        in_stack_depth=16,
        out_stack_depth=16,
        latent_dim=256,
        input_spatial_size=(128, 128),
        stem_kernel_size=(2, 4, 4),
        stem_stride=(2, 4, 4),
        decoder_stages=3,
    ).to(device)
    model.eval()
    x = torch.randn(2, 2, 16, 128, 128, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.recon_x.shape == (2, 2, 16, 128, 128)
    assert out.mean.shape == (2, 256)
    assert out.logvar.shape == (2, 256)
    assert out.z.shape == (2, 256)
```

### BetaVaeMonai Forward-Pass Test Pattern

```python
"""Forward-pass tests for BetaVaeMonai."""

import torch

from viscy_models.vae import BetaVaeMonai


def test_beta_vae_monai_2d(device):
    """BetaVaeMonai 2D: single channel, 64x64."""
    model = BetaVaeMonai(
        spatial_dims=2,
        in_shape=(1, 64, 64),
        out_channels=1,
        latent_size=128,
        channels=(32, 64),
        strides=(2, 2),
    ).to(device)
    model.eval()
    x = torch.randn(2, 1, 64, 64, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.recon_x.shape == (2, 1, 64, 64)
    assert out.mean.shape == (2, 128)
    assert out.logvar.shape == (2, 128)
    assert out.z.shape == (2, 128)
```

### ContrastiveEncoder Test Pattern

```python
def test_contrastive_encoder_convnext(device):
    """ContrastiveEncoder with convnext_tiny backbone, 15 Z-slices."""
    model = ContrastiveEncoder(
        backbone="convnext_tiny",
        in_channels=2,
        in_stack_depth=15,
        embedding_dim=768,
        projection_dim=128,
    ).to(device)
    model.eval()
    x = torch.randn(2, 2, 15, 64, 64, device=device)
    with torch.no_grad():
        embedding, projection = model(x)
    assert embedding.shape == (2, 768)
    assert projection.shape == (2, 128)
```

### ResNet3dEncoder Test Pattern

```python
def test_resnet3d_encoder_resnet18(device):
    """ResNet3dEncoder with resnet18, 3D input."""
    model = ResNet3dEncoder(
        backbone="resnet18",
        in_channels=1,
        embedding_dim=512,
        projection_dim=128,
    ).to(device)
    model.eval()
    x = torch.randn(2, 1, 16, 16, 16, device=device)
    with torch.no_grad():
        embedding, projection = model(x)
    assert embedding.shape == (2, 512)
    assert projection.shape == (2, 128)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `pretrained=True` in model init | `pretrained=False` for pure nn.Module | Phase 8 migration | Users load pretrained weights explicitly; no side effects during construction |
| `encoder.head.fc.in_features` | `encoder.num_features` (timm uniform API) | Phase 8 migration | Fixes ResNet50 backbone bug in ContrastiveEncoder |
| Mutable list defaults in VaeDecoder | Tuple defaults (COMPAT-02) | Phase 8 migration | Prevents shared state bugs |

**Key decision: pretrained=True vs pretrained=False:**

The original code uses `pretrained=True` in ContrastiveEncoder and ResNet3dEncoder constructors. For the viscy-models migration, the decision is:
- `ContrastiveEncoder`: Change default to `pretrained=False`. The original code downloads ImageNet weights during construction, which is a side effect. Users can pass `pretrained=True` when they want it.
- `ResNet3dEncoder`: Change default to `pretrained=False`. The original code downloads MedicalNet weights. Same reasoning.
- `VaeEncoder`: Already has `pretrained` as an explicit parameter (default `True`). Change to default `False` for consistency.

**IMPORTANT:** This changes the constructor signature default, but NOT the behavior when explicitly passed. All LightningModule wrappers in applications/ explicitly set `pretrained=True/False`, so changing the default does not break downstream usage.

**Alternative:** Keep `pretrained=True` as default to match original behavior exactly. The downside is that constructing a model always triggers a network download.

**Recommendation:** Change to `pretrained=False` for clean nn.Module semantics. Document in code comments.

## Open Questions

1. **Should ContrastiveEncoder pretrained default change to False?**
   - What we know: Original uses `pretrained=True`. viscy-models aims for pure nn.Module without side effects. Phase 7's UNeXt2 already uses `pretrained=False` as default.
   - What's unclear: Whether downstream code relies on the default `True`.
   - Recommendation: Change to `pretrained=False` for consistency with UNeXt2 pattern. This is a cosmetic change since application-level code always passes the argument explicitly.

2. **Should the ResNet50 ContrastiveEncoder bug be fixed or preserved?**
   - What we know: `encoder.head.fc` does not exist for timm ResNet models. ConvNeXt path works. No checkpoints exist for the ResNet50 path.
   - What's unclear: Whether someone has a workaround in their training scripts.
   - Recommendation: Fix it. The code never worked for ResNet50 -- there is nothing to break. Use `encoder.num_features` for the projection input dim and backbone-specific classifier reset.

3. **Test input sizes for BetaVae25D**
   - What we know: Full-size inputs (256x256, 16 Z-slices) with ResNet50 backbone create large models. Tests should be fast.
   - What's unclear: The minimum viable input size that exercises all code paths.
   - Recommendation: Use `input_spatial_size=(128, 128)`, `in_stack_depth=16`, `latent_dim=256`, `decoder_stages=3` for manageable test size while preserving all code path coverage. ConvNeXt backbone tests can use even smaller inputs.

## Source Dependency Map

### ContrastiveEncoder Dependencies
```
contrastive/encoder.py
    imports: viscy_models._components.stems.StemDepthtoChannels  [EXISTING]
    imports: timm (create_model)                                  [EXTERNAL]
    imports: torch.nn                                             [EXTERNAL]
    defines: projection_mlp(), ContrastiveEncoder
```

### ResNet3dEncoder Dependencies
```
contrastive/resnet3d.py
    imports: viscy_models.contrastive.encoder.projection_mlp     [SIBLING - Plan 1]
    imports: monai.networks.nets.resnet.ResNetFeatures           [EXTERNAL]
    imports: torch.nn                                             [EXTERNAL]
    defines: ResNet3dEncoder
```

### BetaVae25D Dependencies
```
vae/beta_vae_25d.py
    imports: viscy_models._components.stems.StemDepthtoChannels  [EXISTING]
    imports: viscy_models._components.heads.PixelToVoxelHead     [EXISTING]
    imports: timm (create_model)                                  [EXTERNAL]
    imports: monai.networks.blocks (ResidualUnit, UpSample)      [EXTERNAL]
    imports: monai.networks.blocks.dynunet_block (get_conv_layer)[EXTERNAL]
    imports: torch, torch.nn                                      [EXTERNAL]
    imports: types.SimpleNamespace                                [STDLIB]
    defines: VaeUpStage, VaeEncoder, VaeDecoder, BetaVae25D
```

### BetaVaeMonai Dependencies
```
vae/beta_vae_monai.py
    imports: monai.networks.nets.VarAutoEncoder                  [EXTERNAL]
    imports: monai.networks.layers.factories.Norm                [EXTERNAL]
    imports: torch.nn                                             [EXTERNAL]
    imports: types.SimpleNamespace                                [STDLIB]
    defines: BetaVaeMonai
```

## Mutable Defaults Inventory (Phase 8 scope)

| File | Class | Parameter | Current Default | Fixed Default |
|------|-------|-----------|----------------|---------------|
| vae.py | VaeDecoder | decoder_channels | `[1024, 512, 256, 128]` | `(1024, 512, 256, 128)` |
| vae.py | VaeDecoder | strides | `[2, 2, 2, 1]` | `(2, 2, 2, 1)` |

All other model classes in Phase 8 scope use immutable defaults already.

## Plan Structure Recommendation

### Plan 08-01: Contrastive Models (ContrastiveEncoder + ResNet3dEncoder)
- Migrate `projection_mlp()` and `ContrastiveEncoder` to `contrastive/encoder.py`
- Migrate `ResNet3dEncoder` to `contrastive/resnet3d.py`
- Update `contrastive/__init__.py` with re-exports
- Create `tests/test_contrastive/` with forward-pass tests
- Fix ResNet50 backbone bug in ContrastiveEncoder

### Plan 08-02: VAE Models (BetaVae25D + BetaVaeMonai)
- Migrate VaeUpStage, VaeEncoder, VaeDecoder, BetaVae25D to `vae/beta_vae_25d.py`
- Migrate BetaVaeMonai to `vae/beta_vae_monai.py`
- Fix mutable defaults in VaeDecoder (COMPAT-02)
- Update `vae/__init__.py` with re-exports
- Create `tests/test_vae/` with forward-pass tests

## Sources

### Primary (HIGH confidence)
- **Pre-monorepo source code** (`git show fe7a5da^:viscy/representation/contrastive.py`) -- Full ContrastiveEncoder, projection_mlp, ResNet3dEncoder source
- **Pre-monorepo source code** (`git show fe7a5da^:viscy/representation/vae.py`) -- Full VaeUpStage, VaeEncoder, VaeDecoder, BetaVae25D, BetaVaeMonai source
- **Live Python verification** -- timm 1.0.24 model structure tested for resnet50, convnext_tiny, convnextv2_tiny
- **Live Python verification** -- MONAI ResNetFeatures API tested with resnet10/18/50
- **Live Python verification** -- MONAI VarAutoEncoder API tested with 2D VAE construction
- **Existing viscy-models code** -- `_components/stems.py` (StemDepthtoChannels), `_components/heads.py` (PixelToVoxelHead), `_components/blocks.py` (all verified in Phase 6/7)
- **Phase 6 RESEARCH.md** -- Component categorization, architecture patterns, confirmed VaeUpStage/VaeEncoder/VaeDecoder stay in vae/ module
- **Phase 7 PLAN files** -- Migration pattern: import changes only, state dict preservation, test patterns

### Secondary (MEDIUM confidence)
- **timm 1.0.24 API** -- `model.num_features` uniform API for classifier input features; verified for resnet50 and convnext_tiny
- **MONAI ResNetFeatures embedding dims** -- resnet10/18=512, resnet50=2048; verified via live test

### Tertiary (LOW confidence)
- None. All critical claims verified against live code execution.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All dependencies already in viscy-models lockfile, verified with live imports
- Architecture: HIGH -- Full source code read from git history, all import paths mapped
- Pitfalls: HIGH -- ResNet50 backbone bug verified by creating timm models and checking attributes; BatchNorm1d batch_size constraint verified by hitting the error; VaeEncoder spatial calculations verified by manual forward passes
- Test patterns: HIGH -- Follows established Phase 7 conventions (test_unext2.py pattern), input shapes verified via simulation

**Research date:** 2026-02-12
**Valid until:** 2026-03-12 (stable domain; dependencies pinned by lockfile)
