# Phase 3: Code Migration - Research

**Researched:** 2026-01-28
**Domain:** Python package code migration (MONAI/PyTorch transforms)
**Confidence:** HIGH

## Summary

This research analyzed the original VisCy transforms codebase to document exactly what needs to be migrated. The source repository contains **16 transform modules** (not 25 as estimated) exporting **44 public transforms** and **9 test files** covering the core functionality. All test data is synthetic (torch.rand/torch.zeros tensors), requiring no external test fixtures.

The transforms have clear dependencies: PyTorch, MONAI, kornia, and numpy. One module (`_transforms.py`) has a dependency on `viscy.data.typing` for type definitions (`Sample`, `ChannelMap`). This dependency must be handled during migration - either by extracting the needed types into viscy-transforms or by creating a minimal typing module.

**Primary recommendation:** Migrate all 16 modules in one commit, extract needed types from `viscy.data.typing` into a local `_typing.py` file, update all imports to `from viscy_transforms import X`, and verify with the existing test suite.

## Standard Stack

The transforms depend on these libraries (already in pyproject.toml):

### Core Dependencies
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.4.1 | Tensor operations, functional.interpolate, grid_sample | Core compute backend |
| monai | >=1.4 | Base transform classes (MapTransform, RandomizableTransform) | Medical imaging standard |
| kornia | (latest) | Gaussian kernels, 3D filtering, RandomAffine3D | GPU-accelerated vision ops |
| numpy | (latest) | Array operations, dtype handling | Universal array library |

### Implicit Dependencies (via MONAI)
| Library | Purpose |
|---------|---------|
| typing_extensions | Iterable, Literal, Sequence, NotRequired |
| numpy.typing | DTypeLike for type hints |

**Installation:** Already configured in `packages/viscy-transforms/pyproject.toml`

## Source Code Inventory

### Transform Modules (16 total)

| Module | Classes Exported | Dependencies |
|--------|-----------------|--------------|
| `__init__.py` | 44 re-exports | All modules |
| `_adjust_contrast.py` | BatchedRandAdjustContrast, BatchedRandAdjustContrastd | MONAI, torch |
| `_crop.py` | BatchedCenterSpatialCrop, BatchedCenterSpatialCropd, BatchedRandSpatialCrop, BatchedRandSpatialCropd | MONAI, torch |
| `_decollate.py` | Decollate | MONAI |
| `_flip.py` | BatchedRandFlip, BatchedRandFlipd | MONAI, torch |
| `_gaussian_smooth.py` | BatchedRandGaussianSmooth, BatchedRandGaussianSmoothd, filter3d_separable | kornia, MONAI, torch |
| `_noise.py` | BatchedRandGaussianNoise, BatchedRandGaussianNoised, RandGaussianNoiseTensor, RandGaussianNoiseTensord | MONAI, numpy, torch |
| `_redef.py` | 13 re-typed MONAI transforms (for jsonargparse) | MONAI, numpy.typing |
| `_scale_intensity.py` | BatchedRandScaleIntensity, BatchedRandScaleIntensityd | MONAI, torch |
| `_transforms.py` | BatchedRandAffined, BatchedScaleIntensityRangePercentiles, BatchedScaleIntensityRangePercentilesd, NormalizeSampled, RandInvertIntensityd, StackChannelsd, TiledSpatialCropSamplesd | **viscy.data.typing**, kornia, MONAI, numpy, torch |
| `_zoom.py` | BatchedZoom, BatchedZoomd | MONAI, torch |
| `batched_rand_3d_elasticd.py` | BatchedRand3DElasticd | MONAI, torch |
| `batched_rand_histogram_shiftd.py` | BatchedRandHistogramShiftd | MONAI, torch |
| `batched_rand_local_pixel_shufflingd.py` | BatchedRandLocalPixelShufflingd | MONAI, torch |
| `batched_rand_sharpend.py` | BatchedRandSharpend | MONAI, torch |
| `batched_rand_zstack_shiftd.py` | BatchedRandZStackShiftd | MONAI, torch |

### Test Files (9 total)

| Test File | Tests | Fixtures Used |
|-----------|-------|---------------|
| `test_transforms.py` | BatchedScaleIntensityRangePercentiles, Decollate | torch.rand (synthetic) |
| `test_adjust_contrast.py` | BatchedRandAdjustContrast(d), vs MONAI comparison | torch.rand (synthetic) |
| `test_crop.py` | BatchedCenterSpatialCrop(d), BatchedRandSpatialCrop(d), vs MONAI | torch.rand, torch.randint |
| `test_flip.py` | BatchedRandFlip(d) | torch.arange (deterministic) |
| `test_gaussian_smooth.py` | BatchedRandGaussianSmooth(d), separable filter, vs MONAI | torch.randn (synthetic) |
| `test_noise.py` | BatchedRandGaussianNoise(d) | torch.zeros (baseline) |
| `test_scale_intensity.py` | BatchedRandScaleIntensity(d), vs MONAI | torch.ones, torch.rand |
| `test_zoom.py` | BatchedZoom(d), roundtrip | torch.rand (synthetic) |
| `__init__.py` | Empty | - |

## Architecture Patterns

### Recommended Project Structure
```
packages/viscy-transforms/
├── src/
│   └── viscy_transforms/
│       ├── __init__.py           # Public API (44 exports)
│       ├── py.typed              # PEP 561 marker
│       ├── _typing.py            # Types extracted from viscy.data.typing
│       ├── _adjust_contrast.py   # Contrast transforms
│       ├── _crop.py              # Spatial cropping transforms
│       ├── _decollate.py         # Decollate transform
│       ├── _flip.py              # Flip transforms
│       ├── _gaussian_smooth.py   # Gaussian blur transforms
│       ├── _noise.py             # Noise injection transforms
│       ├── _redef.py             # Re-typed MONAI transforms
│       ├── _scale_intensity.py   # Intensity scaling transforms
│       ├── _transforms.py        # Mixed utility transforms
│       ├── _zoom.py              # Zoom transforms
│       ├── batched_rand_3d_elasticd.py
│       ├── batched_rand_histogram_shiftd.py
│       ├── batched_rand_local_pixel_shufflingd.py
│       ├── batched_rand_sharpend.py
│       └── batched_rand_zstack_shiftd.py
└── tests/
    ├── __init__.py
    ├── conftest.py               # Package-level fixtures (optional)
    ├── test_adjust_contrast.py
    ├── test_crop.py
    ├── test_flip.py
    ├── test_gaussian_smooth.py
    ├── test_noise.py
    ├── test_scale_intensity.py
    ├── test_transforms.py
    └── test_zoom.py
```

### Pattern 1: Import Path Transformation
**What:** Change all imports from `viscy.transforms` to `viscy_transforms`
**When to use:** Every file in the migration
**Example:**
```python
# Old (in original VisCy)
from viscy.transforms._adjust_contrast import BatchedRandAdjustContrast
from viscy.transforms import BatchedRandFlip

# New (in viscy-transforms package)
from viscy_transforms._adjust_contrast import BatchedRandAdjustContrast
from viscy_transforms import BatchedRandFlip
```

### Pattern 2: Internal Type Dependency Resolution
**What:** Extract needed types from `viscy.data.typing` into local `_typing.py`
**When to use:** For `_transforms.py` which depends on `Sample` and `ChannelMap`
**Example:**
```python
# _typing.py (new file, extracted subset)
from typing import TypedDict, TypeVar, Sequence
from torch import Tensor
from typing_extensions import NotRequired

T = TypeVar("T")
OneOrSeq = T | Sequence[T]

class ChannelMap(TypedDict):
    """Source channel names."""
    source: OneOrSeq[str]
    target: NotRequired[OneOrSeq[str]]

class Sample(TypedDict, total=False):
    """Image sample type for mini-batches."""
    source: OneOrSeq[Tensor]
    target: OneOrSeq[Tensor]
    # ... other fields as needed
```

### Pattern 3: Public API in __init__.py
**What:** Export all 44 transforms at package level with explicit __all__
**When to use:** Package `__init__.py`
**Example:**
```python
# __init__.py
from viscy_transforms._adjust_contrast import (
    BatchedRandAdjustContrast,
    BatchedRandAdjustContrastd,
)
# ... all other imports

__all__ = [
    "BatchedCenterSpatialCrop",
    "BatchedCenterSpatialCropd",
    # ... 42 more
]
```

### Anti-Patterns to Avoid
- **Importing from viscy.data.typing:** Will fail since viscy-data doesn't exist. Extract needed types locally.
- **Relative imports across modules:** Use absolute `from viscy_transforms.module import X`.
- **Missing __all__ in submodules:** Each module should declare its public exports.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Type definitions | Copy full viscy.data.typing | Extract minimal subset | Only need Sample, ChannelMap for _transforms.py |
| Test data | Generate real microscopy data | torch.rand/torch.zeros | Original tests use synthetic data successfully |
| conftest.py | Complex HCS fixtures | None needed | Transform tests don't need OME-Zarr fixtures |
| Import compatibility | Backward compat shim | Clean break | Decision from CONTEXT.md |

**Key insight:** The transforms are almost self-contained. Only `_transforms.py` has external type dependencies, and these can be extracted as a minimal `_typing.py` file.

## Common Pitfalls

### Pitfall 1: Missing viscy.data.typing Dependency
**What goes wrong:** `_transforms.py` imports `from viscy.data.typing import ChannelMap, Sample` which won't exist
**Why it happens:** Cross-package dependency in original monolith
**How to avoid:** Create `_typing.py` with extracted type definitions before migrating `_transforms.py`
**Warning signs:** ImportError on `from viscy.data.typing`

### Pitfall 2: Test Import Paths Not Updated
**What goes wrong:** Tests still import from `viscy.transforms` instead of `viscy_transforms`
**Why it happens:** Copy-paste without find-replace
**How to avoid:** Use systematic find-replace: `from viscy.transforms` -> `from viscy_transforms`
**Warning signs:** ModuleNotFoundError in test runs

### Pitfall 3: Missing Module-Level __all__
**What goes wrong:** `from viscy_transforms import *` imports unexpected names
**Why it happens:** Original modules have __all__ but it might be missed in copy
**How to avoid:** Verify each module has `__all__` listing public exports
**Warning signs:** Unexpected symbols in package namespace

### Pitfall 4: Kornia Import in _gaussian_smooth.py
**What goes wrong:** `filter3d_separable` is a module-level function, not a class
**Why it happens:** Easy to miss that it's exported in __init__.py as part of the module
**How to avoid:** Note that `_gaussian_smooth.py` exports `filter3d_separable` function
**Warning signs:** Test failures for separable filter tests

### Pitfall 5: _redef.py Class Inheritance Pattern
**What goes wrong:** _redef.py redefines MONAI classes with same names (shadowing)
**Why it happens:** Pattern for jsonargparse compatibility
**How to avoid:** Keep the pattern exactly as-is - it's intentional for type annotations
**Warning signs:** Type checker complaints about class redefinition

## Code Examples

### Public API Export Pattern
```python
# Source: https://github.com/mehta-lab/VisCy/blob/main/viscy/transforms/__init__.py
# viscy_transforms/__init__.py

from viscy_transforms._adjust_contrast import (
    BatchedRandAdjustContrast,
    BatchedRandAdjustContrastd,
)
from viscy_transforms._crop import (
    BatchedCenterSpatialCrop,
    BatchedCenterSpatialCropd,
    BatchedRandSpatialCrop,
    BatchedRandSpatialCropd,
)
from viscy_transforms._decollate import Decollate
from viscy_transforms._flip import BatchedRandFlip, BatchedRandFlipd
from viscy_transforms._gaussian_smooth import (
    BatchedRandGaussianSmooth,
    BatchedRandGaussianSmoothd,
)
from viscy_transforms._noise import (
    BatchedRandGaussianNoise,
    BatchedRandGaussianNoised,
    RandGaussianNoiseTensor,
    RandGaussianNoiseTensord,
)
from viscy_transforms._redef import (
    CenterSpatialCropd,
    Decollated,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandSpatialCropd,
    RandWeightedCropd,
    ScaleIntensityRangePercentilesd,
    ToDeviced,
)
from viscy_transforms._scale_intensity import (
    BatchedRandScaleIntensity,
    BatchedRandScaleIntensityd,
)
from viscy_transforms._transforms import (
    BatchedRandAffined,
    BatchedScaleIntensityRangePercentiles,
    BatchedScaleIntensityRangePercentilesd,
    NormalizeSampled,
    RandInvertIntensityd,
    StackChannelsd,
    TiledSpatialCropSamplesd,
)
from viscy_transforms._zoom import BatchedZoom, BatchedZoomd
from viscy_transforms.batched_rand_3d_elasticd import BatchedRand3DElasticd
from viscy_transforms.batched_rand_histogram_shiftd import BatchedRandHistogramShiftd
from viscy_transforms.batched_rand_local_pixel_shufflingd import (
    BatchedRandLocalPixelShufflingd,
)
from viscy_transforms.batched_rand_sharpend import BatchedRandSharpend
from viscy_transforms.batched_rand_zstack_shiftd import BatchedRandZStackShiftd

__all__ = [
    "BatchedCenterSpatialCrop",
    "BatchedCenterSpatialCropd",
    "BatchedRandAdjustContrast",
    "BatchedRandAdjustContrastd",
    "BatchedRandAffined",
    "BatchedRand3DElasticd",
    "BatchedRandFlip",
    "BatchedRandFlipd",
    "BatchedRandGaussianSmooth",
    "BatchedRandGaussianSmoothd",
    "BatchedRandGaussianNoise",
    "BatchedRandGaussianNoised",
    "BatchedRandHistogramShiftd",
    "BatchedRandLocalPixelShufflingd",
    "BatchedRandScaleIntensity",
    "BatchedRandScaleIntensityd",
    "BatchedRandSharpend",
    "BatchedRandSpatialCrop",
    "BatchedRandSpatialCropd",
    "BatchedRandZStackShiftd",
    "BatchedScaleIntensityRangePercentiles",
    "BatchedScaleIntensityRangePercentilesd",
    "BatchedZoom",
    "BatchedZoomd",
    "CenterSpatialCropd",
    "Decollate",
    "Decollated",
    "NormalizeSampled",
    "NormalizeIntensityd",
    "RandAdjustContrastd",
    "RandAffined",
    "RandFlipd",
    "RandGaussianNoised",
    "RandGaussianNoiseTensor",
    "RandGaussianNoiseTensord",
    "RandGaussianSmoothd",
    "RandInvertIntensityd",
    "RandScaleIntensityd",
    "RandSpatialCropd",
    "RandWeightedCropd",
    "ScaleIntensityRangePercentilesd",
    "StackChannelsd",
    "TiledSpatialCropSamplesd",
    "ToDeviced",
]
```

### Extracted Type Definitions
```python
# viscy_transforms/_typing.py
"""Type definitions for viscy-transforms.

Extracted from viscy.data.typing to avoid cross-package dependency.
"""

from typing import NamedTuple, Sequence, TypedDict, TypeVar

from torch import Tensor
from typing_extensions import NotRequired

T = TypeVar("T")
OneOrSeq = T | Sequence[T]


class HCSStackIndex(NamedTuple):
    """HCS stack index."""
    image: str
    time: int
    z: int


class LevelNormStats(TypedDict):
    mean: Tensor
    std: Tensor
    median: Tensor
    iqr: Tensor


class ChannelNormStats(TypedDict):
    dataset_statistics: LevelNormStats
    fov_statistics: LevelNormStats


NormMeta = dict[str, ChannelNormStats]


class Sample(TypedDict, total=False):
    """Image sample type for mini-batches."""
    index: HCSStackIndex
    source: OneOrSeq[Tensor]
    target: OneOrSeq[Tensor]
    weight: OneOrSeq[Tensor]
    labels: OneOrSeq[Tensor]
    norm_meta: NormMeta | None


class ChannelMap(TypedDict):
    """Source channel names."""
    source: OneOrSeq[str]
    target: NotRequired[OneOrSeq[str]]
```

### Test Import Pattern
```python
# Source: https://github.com/mehta-lab/VisCy/blob/main/tests/transforms/test_flip.py
# tests/test_flip.py (updated imports)

import pytest
import torch

from viscy_transforms import BatchedRandFlip, BatchedRandFlipd


@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("spatial_axes", [[0, 1, 2], [1, 2], [0]])
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_batched_rand_flip(device, prob, spatial_axes):
    img = (
        torch.arange(32 * 2 * 2 * 2 * 2, device=device).reshape(32, 2, 2, 2, 2).float()
    )
    transform = BatchedRandFlip(prob=prob, spatial_axes=spatial_axes)
    out = transform(img)
    # ... rest of test
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| viscy.transforms import | viscy_transforms import | This migration | Clean package boundaries |
| Monolithic deps | Package-specific deps | This migration | Minimal install footprint |
| Shared typing module | Extracted local types | This migration | Self-contained package |

**Deprecated/outdated:**
- `from viscy.transforms import X`: Use `from viscy_transforms import X` after migration

## Open Questions

1. **NormalizeSampled and StackChannelsd type dependencies**
   - What we know: These use `Sample` and `ChannelMap` types from `viscy.data.typing`
   - What's unclear: Whether these transforms are used standalone or always with viscy-data
   - Recommendation: Extract types to `_typing.py` - cleaner than optional dependency

2. **Tests for batched_rand_* modules**
   - What we know: No test files exist for `batched_rand_3d_elasticd.py`, `batched_rand_histogram_shiftd.py`, `batched_rand_local_pixel_shufflingd.py`, `batched_rand_sharpend.py`, `batched_rand_zstack_shiftd.py`
   - What's unclear: Whether these are tested elsewhere or simply untested
   - Recommendation: Migrate as-is, note coverage gap for future work

3. **_redef.py nested class bug**
   - What we know: `RandFlipd` is incorrectly nested inside `CenterSpatialCropd` class
   - What's unclear: Whether this is intentional or a bug in original
   - Recommendation: Preserve as-is during migration, flag for later review

## Sources

### Primary (HIGH confidence)
- GitHub API: `repos/mehta-lab/VisCy/contents/viscy/transforms/*` - All 16 module files fetched
- GitHub API: `repos/mehta-lab/VisCy/contents/tests/transforms/*` - All 9 test files fetched
- GitHub API: `repos/mehta-lab/VisCy/contents/viscy/data/typing.py` - Type definitions fetched
- GitHub API: `repos/mehta-lab/VisCy/contents/pyproject.toml` - Dependencies verified

### Secondary (MEDIUM confidence)
- Current package structure: `packages/viscy-transforms/pyproject.toml` - Dependencies already configured

## Metadata

**Confidence breakdown:**
- Source inventory: HIGH - Direct API access to all source files
- Dependencies: HIGH - Verified from pyproject.toml and imports
- Test patterns: HIGH - All test files fetched and analyzed
- Type extraction: MEDIUM - Types identified, extraction approach is reasonable

**Research date:** 2026-01-28
**Valid until:** 2026-02-28 (stable codebase, no expected changes)
