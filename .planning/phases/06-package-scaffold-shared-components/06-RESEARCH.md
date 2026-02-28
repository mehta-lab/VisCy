# Phase 6: Package Scaffold & Shared Components - Research

**Researched:** 2026-02-12
**Domain:** Python package scaffolding, nn.Module component extraction, uv workspace member creation
**Confidence:** HIGH

## Summary

Phase 6 creates the `viscy-models` package scaffold within the existing uv workspace and extracts shared architectural components from the original monolithic VisCy codebase. The scaffold follows the identical pattern established by `viscy-transforms` in v1.0: src layout, hatchling + uv-dynamic-versioning build system, PEP 735 dependency groups. The critical new work is component extraction -- identifying and isolating the 14+ shared nn.Module classes from `unext2.py` and related files into a `_components/` subpackage, plus migrating ConvBlock2D/3D to `unet/_layers/`.

The source code lives in the pre-migration VisCy repo (tag `v0.3.3`). Key files are `viscy/unet/networks/unext2.py` (contains UNeXt2Stem, StemDepthtoChannels, PixelToVoxelHead, UnsqueezeHead, UNeXt2Decoder, UNeXt2UpStage, UNeXt2, and helper functions), `viscy/unet/networks/layers/ConvBlock2D.py` and `ConvBlock3D.py`, and `viscy/unet/networks/fcmae.py` (which imports shared components from unext2.py). The VAE models (`viscy/representation/vae.py` on commit c591950) also import `StemDepthtoChannels` and `PixelToVoxelHead` from unext2.py, confirming these are genuinely shared. The contrastive encoder (`viscy/representation/contrastive.py`) imports `StemDepthtoChannels`. This cross-model sharing validates the `_components/` extraction approach.

Mutable defaults needing conversion to tuples exist in: `FullyConvolutionalMAE.__init__` (encoder_blocks, dims as lists), `Unet2d.__init__` (num_filters=[]), `Unet25d.__init__` (num_filters=[]), and `VaeDecoder.__init__` (decoder_channels, strides as lists). The UNeXt2 class itself uses immutable defaults already. State dict key compatibility is non-negotiable -- the module naming (`self.stem`, `self.encoder`, `self.decoder`, `self.head`, etc.) must be preserved exactly during extraction.

**Primary recommendation:** Mirror the viscy-transforms scaffold exactly (pyproject.toml, src layout, test structure), extract shared components into `_components/{stems,heads,blocks}.py`, and migrate ConvBlock2D/3D into `unet/_layers/` with snake_case filenames. Keep module attribute names identical to preserve state dict keys.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.10 | Neural network framework | Locked by workspace; all models are nn.Module |
| timm | >=1.0.15 | ConvNeXt backbones, feature extraction | UNeXt2/FCMAE/ContrastiveEncoder use `timm.create_model()` and `timm.models.convnext.ConvNeXtStage` |
| monai | >=1.5.2 | Medical imaging network blocks | UpSample, Convolution, ResidualUnit, VarAutoEncoder, ResNetFeatures |
| numpy | >=2.4.1 | Array operations | ConvBlock2D/3D use `np.linspace` for filter step calculations |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hatchling | (build-requires) | Build backend | Package build; same as viscy-transforms |
| uv-dynamic-versioning | (build-requires) | Git-based versioning | Version from tags with `viscy-models-` prefix |
| pytest | >=9.0.2 | Testing framework | Package test suite |
| pytest-cov | >=7 | Coverage reporting | Test coverage |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| hatchling | uv build backend | uv backend does not support plugins (no dynamic versioning) |
| timm ConvNeXtStage | Custom ConvNeXt blocks | Would break state dict compatibility; timm is battle-tested |
| monai UpSample/ResidualUnit | Custom upsampling | Would require reimplementing and break checkpoint loading |

**Installation (workspace sync):**
```bash
uv sync --package viscy-models
```

## Architecture Patterns

### Recommended Project Structure
```
packages/viscy-models/
    pyproject.toml
    README.md
    src/viscy_models/
        __init__.py              # Public API: from viscy_models import UNeXt2
        py.typed                 # PEP 561 marker
        _components/             # MPKG-04: Shared architectural components
            __init__.py
            stems.py             # UNeXt2Stem, StemDepthtoChannels
            heads.py             # PixelToVoxelHead, UnsqueezeHead, PixelToVoxelShuffleHead
            blocks.py            # UNeXt2UpStage, UNeXt2Decoder, VaeUpStage, VaeEncoder, VaeDecoder, _get_convnext_stage, icnr_init
        unet/                    # UNet family models (Phase 7+)
            __init__.py
            _layers/             # UNET-05: ConvBlock2D/3D layers
                __init__.py
                conv_block_2d.py # Renamed from PascalCase
                conv_block_3d.py # Renamed from PascalCase
            unext2.py            # Phase 7: UNeXt2 model
            fcmae.py             # Phase 7: FullyConvolutionalMAE
            unet2d.py            # Phase 9: Unet2d
            unet25d.py           # Phase 9: Unet25d
        contrastive/             # Contrastive models (Phase 8)
            __init__.py
            encoder.py           # ContrastiveEncoder
            resnet3d.py          # ResNet3dEncoder
        vae/                     # VAE models (Phase 8)
            __init__.py
            beta_vae_25d.py      # BetaVae25D
            beta_vae_monai.py    # BetaVaeMonai
    tests/
        __init__.py
        conftest.py
        test_components/         # Tests for _components
            __init__.py
            test_stems.py
            test_heads.py
            test_blocks.py
        test_unet/
            __init__.py
            test_layers.py       # ConvBlock2D/3D tests
```

### Pattern 1: Component Extraction with Preserved State Dict Keys
**What:** Extract shared nn.Module classes from model files into `_components/` while keeping the exact same class names and `__init__` parameter signatures.
**When to use:** Every shared component that appears in multiple models.
**Why critical:** State dict keys are derived from module attribute names. If `self.stem = UNeXt2Stem(...)` becomes `self.stem = SomethingElse(...)` with different internal names, checkpoint loading breaks.

**Example:**
```python
# Source: v0.3.3 viscy/unet/networks/unext2.py
# _components/stems.py - extracted verbatim, only imports change
from torch import Tensor, nn


class UNeXt2Stem(nn.Module):
    """Stem for UNeXt2 and ContrastiveEncoder networks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        in_stack_depth: int,
    ) -> None:
        super().__init__()
        ratio = in_stack_depth // kernel_size[0]
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels // ratio,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, x: Tensor):
        x = self.conv(x)
        b, c, d, h, w = x.shape
        return x.reshape(b, c * d, h, w)
```

### Pattern 2: Mutable Default to Tuple Conversion (COMPAT-02)
**What:** Replace mutable list defaults in `__init__` signatures with tuples.
**When to use:** Every model constructor that has `param: Sequence[int] = [...]` or `param = []`.
**Why critical:** Mutable defaults are shared across instances; modifying one affects all.

**Example:**
```python
# BEFORE (mutable default - bug-prone):
class FullyConvolutionalMAE(nn.Module):
    def __init__(
        self,
        encoder_blocks: Sequence[int] = [3, 3, 9, 3],  # mutable list!
        dims: Sequence[int] = [96, 192, 384, 768],      # mutable list!
    ) -> None:

# AFTER (immutable default - safe):
class FullyConvolutionalMAE(nn.Module):
    def __init__(
        self,
        encoder_blocks: Sequence[int] = (3, 3, 9, 3),   # immutable tuple
        dims: Sequence[int] = (96, 192, 384, 768),       # immutable tuple
    ) -> None:
```

### Pattern 3: pyproject.toml Following viscy-transforms Template
**What:** Copy the proven viscy-transforms pyproject.toml and adapt for viscy-models.
**When to use:** Package scaffolding.

**Example:**
```toml
# Source: Adapted from packages/viscy-transforms/pyproject.toml
[build-system]
build-backend = "hatchling.build"
requires = ["hatchling", "uv-dynamic-versioning"]

[project]
name = "viscy-models"
description = "Neural network architectures for virtual staining microscopy"
readme = "README.md"
keywords = ["deep learning", "microscopy", "neural networks", "pytorch", "virtual staining"]
license = "BSD-3-Clause"
authors = [{ name = "Biohub", email = "compmicro@czbiohub.org" }]
requires-python = ">=3.11"
classifiers = [
  "Development Status :: 4 - Beta",
  "Intended Audience :: Science/Research",
  "License :: OSI Approved :: BSD License",
  "Operating System :: OS Independent",
  "Programming Language :: Python :: 3 :: Only",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Programming Language :: Python :: 3.13",
  "Programming Language :: Python :: 3.14",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
  "Topic :: Scientific/Engineering :: Image Processing",
]
dynamic = ["version"]
dependencies = [
  "monai>=1.5.2",
  "numpy>=2.4.1",
  "timm>=1.0.15",
  "torch>=2.10",
]

urls.Homepage = "https://github.com/mehta-lab/VisCy"
urls.Issues = "https://github.com/mehta-lab/VisCy/issues"
urls.Repository = "https://github.com/mehta-lab/VisCy"

[dependency-groups]
dev = [{ include-group = "test" }]
test = ["pytest>=9.0.2", "pytest-cov>=7"]

[tool.hatch.version]
source = "uv-dynamic-versioning"

[tool.hatch.build.targets.wheel]
packages = ["src/viscy_models"]

[tool.uv-dynamic-versioning]
vcs = "git"
style = "pep440"
pattern-prefix = "viscy-models-"
fallback-version = "0.0.0"
```

### Pattern 4: Workspace Source Registration
**What:** Register the new package in the root pyproject.toml workspace sources.
**When to use:** When adding a new workspace member that is a dependency of the umbrella package.

**Example (root pyproject.toml changes):**
```toml
[project]
dependencies = ["viscy-transforms", "viscy-models"]

[tool.uv.sources]
viscy-transforms = { workspace = true }
viscy-models = { workspace = true }

[tool.ruff]
src = ["packages/*/src"]  # Already covers new package via glob
```

### Anti-Patterns to Avoid
- **Renaming module attributes during extraction:** `self.stem` must remain `self.stem` in the model class, even if you move `UNeXt2Stem` to a different file. Renaming breaks state dict keys.
- **Creating circular imports between _components and model subpackages:** `_components/` should have zero imports from `unet/`, `vae/`, or `contrastive/`. It should only import from torch, timm, monai, numpy.
- **Importing from viscy-transforms:** viscy-models must be independent of viscy-transforms. Only torch/timm/monai/numpy dependencies.
- **Putting model-specific code in _components:** Only truly shared code belongs in `_components/`. If something is used by only one model, keep it in that model's file.
- **Splitting ConvBlock into too many files:** conv_block_2d.py and conv_block_3d.py are sufficient. Do not over-decompose.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ConvNeXt stages | Custom ConvNeXt blocks | `timm.models.convnext.ConvNeXtStage` | Pretrained weights, proven architecture, state dict compatibility |
| Upsampling blocks | Custom pixel shuffle | `monai.networks.blocks.UpSample` | Handles edge cases (padding, pre-conv), medical imaging validated |
| 3D ResNet features | Custom ResNet backbone | `monai.networks.nets.resnet.ResNetFeatures` | Pretrained MedicalNet weights, spatial_dims=3 support |
| Feature extraction backbone | Custom encoder | `timm.create_model(..., features_only=True)` | Access to 1000+ pretrained models, multi-scale feature maps |
| Dynamic versioning | Manual version management | `uv-dynamic-versioning` with `pattern-prefix` | Tag-based, monorepo-aware, proven by viscy-transforms |
| Build system | Custom build scripts | `hatchling` with `uv-dynamic-versioning` | Same as viscy-transforms, workspace-tested |

**Key insight:** Every "shared" component in viscy-models wraps or extends third-party functionality (timm, monai). The extraction is about organizing imports and file structure, NOT reimplementing anything.

## Common Pitfalls

### Pitfall 1: State Dict Key Mismatch After Extraction
**What goes wrong:** Extracting a class to a new file and changing how models instantiate it (e.g., renaming `self.stem` to `self.projection_stem`) silently changes state dict keys, making all existing checkpoints unloadable.
**Why it happens:** PyTorch state dict keys are derived from the module hierarchy: `stem.conv.weight` comes from `self.stem = UNeXt2Stem(...)` which contains `self.conv = nn.Conv3d(...)`.
**How to avoid:** Keep module attribute names IDENTICAL. Only change the import path, never the attribute name or internal structure of extracted classes.
**Warning signs:** `model.load_state_dict(checkpoint)` raises `RuntimeError: Error(s) in loading state_dict` with "Missing key" or "Unexpected key" messages.

### Pitfall 2: Import Cycles Between _components and Model Subpackages
**What goes wrong:** `_components/blocks.py` imports from `unet/unext2.py` which imports from `_components/stems.py`, creating a circular import.
**Why it happens:** `UNeXt2UpStage` and `UNeXt2Decoder` use `_get_convnext_stage` which is a utility, not a shared component. Putting model-specific utilities in `_components` tempts circular imports.
**How to avoid:** `_components/` must have ZERO imports from model subpackages (unet/, vae/, contrastive/). It should only import from torch, timm, monai, numpy. Functions like `_get_convnext_stage` and `icnr_init` that are used by both UNeXt2 and FCMAE belong in `_components/blocks.py`, but they must not reference any model classes.
**Warning signs:** `ImportError: cannot import name 'X' from partially initialized module`.

### Pitfall 3: Mutable Default Shared State
**What goes wrong:** Two model instances share the same default list object. One mutant modifies it during `__init__`, corrupting the other.
**Why it happens:** Python evaluates default arguments once at function definition time, not at each call.
**How to avoid:** Replace all `= [...]` defaults with `= (...)` tuples. Convert to list internally if mutation is needed.
**Warning signs:** `ruff` rule B006 ("Do not use mutable data structures for argument defaults") catches this if linting is enabled.

### Pitfall 4: Missing timm Dependency in Lockfile
**What goes wrong:** `uv sync --package viscy-models` fails or imports fail at runtime because timm is not in the workspace lockfile.
**Why it happens:** viscy-transforms does not depend on timm, so it is not in the current lockfile. Adding viscy-models with timm dependency requires `uv lock` to be re-run.
**How to avoid:** Run `uv lock` after adding the viscy-models pyproject.toml, before attempting `uv sync`.
**Warning signs:** `uv sync` fails with resolution errors or timm import fails at test time.

### Pitfall 5: numpy Import in ConvBlock (Dependency Concern)
**What goes wrong:** ConvBlock2D and ConvBlock3D use `numpy.linspace` for filter step calculations. This is a runtime numpy dependency.
**Why it happens:** The original code used numpy for array math instead of pure Python or torch.
**How to avoid:** Keep numpy as a dependency (already declared). The usage is lightweight (only in `__init__`, not forward pass).
**Warning signs:** None, as long as numpy is in dependencies. Consider replacing with `torch.linspace` in a future cleanup, but not in scope for Phase 6.

### Pitfall 6: ConvBlock register_modules Not Using nn.ModuleList
**What goes wrong:** ConvBlock2D/3D use custom `register_modules()` + `add_module()` instead of `nn.ModuleList`. This works but creates non-standard state dict keys like `Conv2d_0.weight` instead of `conv_list.0.weight`.
**Why it happens:** Legacy implementation predating PyTorch's nn.ModuleList improvements.
**How to avoid:** Do NOT refactor to nn.ModuleList during migration -- this would change state dict keys and break checkpoints. Preserve the `add_module` pattern exactly.
**Warning signs:** Any changes to module registration pattern will break `load_state_dict`.

## Code Examples

### Shared Component Categorization (from v0.3.3 source analysis)

The following components from `unext2.py` are imported by multiple model files:

**stems.py** -- Used by UNeXt2, ContrastiveEncoder, BetaVae25D, VaeEncoder, FCMAE:
```python
# Source: v0.3.3 viscy/unet/networks/unext2.py
class UNeXt2Stem(nn.Module):  # Used by UNeXt2
class StemDepthtoChannels(nn.Module):  # Used by ContrastiveEncoder, VaeEncoder
```

**heads.py** -- Used by UNeXt2, FCMAE, BetaVae25D/VaeDecoder:
```python
# Source: v0.3.3 viscy/unet/networks/unext2.py, fcmae.py
class PixelToVoxelHead(nn.Module):  # Used by UNeXt2, VaeDecoder, FCMAE (head_conv=True)
class UnsqueezeHead(nn.Module):  # Used by future models that output 2D->3D
class PixelToVoxelShuffleHead(nn.Module):  # Used by FCMAE (head_conv=False)
```

**blocks.py** -- Shared encoder/decoder building blocks and initialization:
```python
# Source: v0.3.3 viscy/unet/networks/unext2.py
def icnr_init(...):  # Used by UNeXt2 stem init, _get_convnext_stage
def _get_convnext_stage(...):  # Used by UNeXt2UpStage conv blocks

class UNeXt2UpStage(nn.Module):  # Used by UNeXt2Decoder
class UNeXt2Decoder(nn.Module):  # Used by UNeXt2, FullyConvolutionalMAE

# Source: v0.3.3 viscy/representation/vae.py (commit c591950)
class VaeUpStage(nn.Module):  # Used by VaeDecoder
class VaeEncoder(nn.Module):  # Used by BetaVae25D
class VaeDecoder(nn.Module):  # Used by BetaVae25D
```

### Component Import Pattern (after extraction)
```python
# In packages/viscy-models/src/viscy_models/unet/unext2.py
from viscy_models._components.stems import UNeXt2Stem
from viscy_models._components.heads import PixelToVoxelHead
from viscy_models._components.blocks import UNeXt2Decoder
```

### ConvBlock Migration (unet/_layers/)
```python
# packages/viscy-models/src/viscy_models/unet/_layers/__init__.py
from viscy_models.unet._layers.conv_block_2d import ConvBlock2D
from viscy_models.unet._layers.conv_block_3d import ConvBlock3D

__all__ = ["ConvBlock2D", "ConvBlock3D"]
```

### Mutable Defaults Inventory (COMPAT-02)

All mutable defaults found in model constructors that must be converted:

| File | Class | Parameter | Current Default | Fixed Default |
|------|-------|-----------|----------------|---------------|
| fcmae.py | FullyConvolutionalMAE | encoder_blocks | `[3, 3, 9, 3]` | `(3, 3, 9, 3)` |
| fcmae.py | FullyConvolutionalMAE | dims | `[96, 192, 384, 768]` | `(96, 192, 384, 768)` |
| fcmae.py | FullyConvolutionalMAE | stem_kernel_size | `(5, 4, 4)` | Already tuple -- OK |
| Unet2D.py | Unet2d | num_filters | `[]` | `()` |
| Unet25D.py | Unet25d | num_filters | `[]` | `()` |
| vae.py | VaeDecoder | decoder_channels | `[1024, 512, 256, 128]` | `(1024, 512, 256, 128)` |
| vae.py | VaeDecoder | strides | `[2, 2, 2, 1]` | `(2, 2, 2, 1)` |

**Note:** Internal code that creates lists from these parameters (e.g., `list(dims)`) continues to work because tuples are iterable. The type annotation `Sequence[int]` already accepts both lists and tuples.

### Workspace Registration
```toml
# Root pyproject.toml additions:
[project]
dependencies = ["viscy-transforms", "viscy-models"]

[tool.uv.sources]
viscy-transforms = { workspace = true }
viscy-models = { workspace = true }
```

### Test Configuration
```toml
# Root pyproject.toml -- already configured for packages/*
[tool.pytest]
testpaths = ["packages/*/tests", "tests"]
addopts = ["-ra", "-q", "--import-mode=importlib"]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PascalCase filenames (ConvBlock2D.py) | snake_case filenames (conv_block_2d.py) | Python community convention | Consistent with ruff linting; modern Python style |
| Mutable list defaults | Tuple defaults | ruff B006 rule | Prevents shared state bugs |
| `hatch-vcs` | `uv-dynamic-versioning` | Early 2026 | Already used by viscy-transforms; monorepo pattern-prefix support |
| Single package with all code | Workspace packages | v1.0 migration | viscy-models is independent of viscy-transforms |
| `setuptools` | `hatchling` | v1.0 migration | Plugin support, src layout, dynamic versioning |

**Deprecated/outdated:**
- `hatch-cada`: The v1.0 research mentioned hatch-cada for workspace dependency rewriting, but the actual v1.0 implementation uses `uv-dynamic-versioning` instead. Use `uv-dynamic-versioning` with `pattern-prefix`.
- `setuptools-scm`: Replaced by `uv-dynamic-versioning` in this monorepo.

## Open Questions

1. **Where does projection_mlp belong?**
   - What we know: `projection_mlp()` is a standalone function used only by `ContrastiveEncoder` and `ResNet3dEncoder` (both in contrastive.py). It is NOT used by UNeXt2, FCMAE, or VAE models.
   - What's unclear: Whether it belongs in `_components/` or stays in the contrastive module.
   - Recommendation: Keep it in `contrastive/encoder.py` since it's only used by contrastive models. Not a shared component.

2. **Should VaeUpStage/VaeEncoder/VaeDecoder go in _components/blocks.py or vae/?**
   - What we know: VaeUpStage is similar to UNeXt2UpStage but lacks skip connections. VaeEncoder/VaeDecoder are only used by BetaVae25D.
   - What's unclear: Whether they will be reused by future models.
   - Recommendation: Keep VaeUpStage, VaeEncoder, VaeDecoder in `vae/` module (Phase 8 scope), not in `_components/`. They are specific to VAE architecture. Only truly cross-family components (stems, heads used by UNet+VAE+contrastive) belong in `_components/`.

3. **Should FCMAE's masked operation functions go in _components/?**
   - What we know: `generate_mask`, `upsample_mask`, `masked_patchify`, `masked_unpatchify` are only used by FCMAE classes.
   - What's unclear: Whether future models will need masking operations.
   - Recommendation: Keep in `unet/fcmae.py` (Phase 7 scope). Not shared.

4. **Phase 6 scope boundary: empty model dirs vs. populated dirs?**
   - What we know: Phase 6 requires _components/ and unet/_layers/ to be populated. Phases 7-9 migrate actual models.
   - What's unclear: Should `unet/__init__.py`, `vae/__init__.py`, `contrastive/__init__.py` be created empty in Phase 6?
   - Recommendation: Create the directory structure with empty `__init__.py` files in Phase 6. This allows Phase 7/8/9 to focus purely on model migration without structural work.

## Sources

### Primary (HIGH confidence)
- **v0.3.3 tag in local repo** -- All model source code read directly: `viscy/unet/networks/unext2.py`, `fcmae.py`, `Unet2D.py`, `Unet25D.py`, `layers/ConvBlock2D.py`, `layers/ConvBlock3D.py`
- **commit c591950** -- VAE model source: `viscy/representation/vae.py` (BetaVae25D, BetaVaeMonai, VaeEncoder, VaeDecoder, VaeUpStage)
- **v0.3.3 tag** -- Contrastive source: `viscy/representation/contrastive.py` (ContrastiveEncoder, projection_mlp, ResNet3dEncoder from git history)
- **packages/viscy-transforms/pyproject.toml** -- Template for package scaffold (hatchling, uv-dynamic-versioning, PEP 735 groups)
- **pyproject.toml (root)** -- Workspace configuration, ruff config, pytest config
- **uv.lock** -- Locked versions: torch 2.10.0, monai 1.5.2, numpy 2.4.2

### Secondary (MEDIUM confidence)
- [timm PyPI](https://pypi.org/project/timm/) -- Latest version 1.0.22 (Jan 5, 2026); ConvNeXtV2 support confirmed
- [uv-dynamic-versioning docs](https://github.com/ninoseki/uv-dynamic-versioning) -- pattern-prefix configuration for monorepo tag-based versioning (v0.13.0)
- [MONAI ResNetFeatures](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/resnet.py) -- ResNetFeatures class confirmed available for ResNet3dEncoder

### Tertiary (LOW confidence)
- None. All critical claims verified against source code and official documentation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- Dependencies verified from lockfile and source code imports
- Architecture: HIGH -- Component relationships mapped directly from source code analysis of 6 model files
- Pitfalls: HIGH -- State dict key preservation verified by reading PyTorch nn.Module internals; mutable defaults identified by source grep
- Component categorization: MEDIUM -- Shared vs. single-use classification based on current codebase; future models may change this

**Research date:** 2026-02-12
**Valid until:** 2026-03-12 (stable domain; dependencies pinned by lockfile)
