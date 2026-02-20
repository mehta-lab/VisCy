# Architecture Patterns: viscy-data Package Extraction

**Domain:** Python data loading subpackage for microscopy deep learning (PyTorch Lightning)
**Researched:** 2026-02-13
**Confidence:** HIGH (based on direct source code analysis of all 13 modules, existing monorepo patterns from viscy-transforms extraction, and documented dependency graph in `viscy/data/README.md`)

## Recommended Architecture

### Package Layout

```
packages/viscy-data/
    pyproject.toml
    README.md
    src/viscy_data/
        __init__.py                 # Public API exports (lazy imports for optional deps)
        py.typed                    # PEP 561 marker
        _typing.py                  # Data-specific types (Sample, NormMeta, TripletSample, etc.)
        _utils.py                   # Shared helpers extracted from hcs.py
        select.py                   # Well/FOV filtering: SelectWell mixin, _filter_wells, _filter_fovs
        distributed.py              # ShardedDistributedSampler for DDP
        hcs.py                      # HCSDataModule, SlidingWindowDataset, MaskTestDataset
        gpu_aug.py                  # GPUTransformDataModule (ABC), CachedOmeZarrDataset, CachedOmeZarrDataModule
        mmap_cache.py               # MmappedDataset, MmappedDataModule
        triplet.py                  # TripletDataset, TripletDataModule (+ helper functions)
        cell_classification.py      # ClassificationDataset, ClassificationDataModule
        cell_division_triplet.py    # CellDivisionTripletDataset, CellDivisionTripletDataModule
        ctmc_v1.py                  # CTMCv1DataModule
        livecell.py                 # LiveCellDataset, LiveCellTestDataset, LiveCellDataModule
        segmentation.py             # SegmentationDataset, SegmentationDataModule
        combined.py                 # CombinedDataModule (CombinedLoader-based wrapper)
        concat.py                   # ConcatDataModule, BatchedConcatDataModule, BatchedConcatDataset, CachedConcatDataModule
    tests/
        __init__.py
        conftest.py                 # Fixtures: preprocessed_hcs_dataset, tracks_hcs_dataset, etc.
        test_hcs.py                 # Tests for HCSDataModule + SlidingWindowDataset
        test_select.py              # Tests for SelectWell mixin (if tests exist on main)
        test_triplet.py             # Tests for TripletDataModule + TripletDataset
        test_typing.py              # Smoke tests for type definitions
```

### Rationale for This Layout

**Why flat modules, not sub-packages:** The 13 source files have a dense internal dependency graph (see README.md dependency diagram). Introducing sub-packages (e.g., `viscy_data/modules/`, `viscy_data/datasets/`) would split tightly coupled code across directories without meaningful encapsulation. The flat layout mirrors the original `viscy/data/` structure, minimizing migration risk.

**Why `_utils.py` and `_typing.py` (underscore prefix):** These are internal modules. The underscore communicates they are not part of the public API. Users import from `viscy_data` (the package), not from `viscy_data._utils`.

**Why split `combined.py` into `combined.py` + `concat.py`:** The current `combined.py` contains two distinct patterns:
1. `CombinedDataModule` -- wraps data modules using Lightning's `CombinedLoader` for multi-source training
2. `ConcatDataModule` + `BatchedConcatDataModule` + `CachedConcatDataModule` + `BatchedConcatDataset` -- concatenation-based combining using `torch.utils.data.ConcatDataset`

These serve different purposes and have different dependency profiles. `CombinedDataModule` depends on `hcs._collate_samples` (moving to `_utils.py`). The concat modules have their own inheritance hierarchy (`ConcatDataModule <|-- BatchedConcatDataModule`). Separating them improves readability and enables independent evolution.

## Component Boundaries

| Component | Responsibility | Internal Deps | External Deps |
|-----------|---------------|---------------|---------------|
| `_typing.py` | Type definitions: `Sample`, `NormMeta`, `ChannelMap`, `TripletSample`, `SegmentationSample`, label constants | None | `torch`, `typing_extensions` |
| `_utils.py` | `_ensure_channel_list()`, `_read_norm_meta()`, `_collate_samples()`, `_search_int_in_str()` | `_typing` | `torch`, `monai` |
| `select.py` | `SelectWell` mixin, `_filter_wells()`, `_filter_fovs()` | None | `iohub` |
| `distributed.py` | `ShardedDistributedSampler` | None | `torch` |
| `hcs.py` | `HCSDataModule`, `SlidingWindowDataset`, `MaskTestDataset` | `_typing`, `_utils` | `iohub`, `zarr`, `monai`, `imageio`, `lightning` |
| `gpu_aug.py` | `GPUTransformDataModule` (ABC), `CachedOmeZarrDataset`, `CachedOmeZarrDataModule` | `_typing`, `_utils`, `select`, `distributed` | `iohub`, `monai`, `lightning` |
| `mmap_cache.py` | `MmappedDataset`, `MmappedDataModule` | `_typing`, `_utils`, `gpu_aug`, `select` | `iohub`, `monai`, `tensordict`, `lightning` |
| `triplet.py` | `TripletDataset`, `TripletDataModule`, channel scatter/gather helpers | `_typing`, `_utils`, `hcs`, `select` | `iohub`, `monai`, `pandas`, `tensorstore`, `lightning` |
| `cell_classification.py` | `ClassificationDataset`, `ClassificationDataModule` | `_typing`, `_utils`, `triplet` (for `INDEX_COLUMNS`) | `iohub`, `pandas`, `lightning` |
| `cell_division_triplet.py` | `CellDivisionTripletDataset`, `CellDivisionTripletDataModule` | `_typing`, `hcs`, `triplet` (for `_transform_channel_wise`) | `monai`, `lightning` |
| `ctmc_v1.py` | `CTMCv1DataModule` | `gpu_aug` | `iohub`, `monai`, `lightning` |
| `livecell.py` | `LiveCellDataset`, `LiveCellTestDataset`, `LiveCellDataModule` | `_typing`, `gpu_aug` | `monai`, `pycocotools`, `tifffile`, `torchvision`, `lightning` |
| `segmentation.py` | `SegmentationDataset`, `SegmentationDataModule` | `_typing` | `iohub`, `lightning` |
| `combined.py` | `CombineMode`, `CombinedDataModule` | `_utils` (for `_collate_samples`) | `lightning` |
| `concat.py` | `ConcatDataModule`, `BatchedConcatDataModule`, `BatchedConcatDataset`, `CachedConcatDataModule` | `_utils`, `distributed` | `torch`, `monai`, `lightning` |

## The hcs.py Dual-Role Problem: Detailed Solution

### Problem

`hcs.py` currently serves two roles:
1. **Concrete DataModule**: `HCSDataModule`, `SlidingWindowDataset`, `MaskTestDataset` -- the translation pipeline's data loading
2. **Utility library**: `_ensure_channel_list()`, `_read_norm_meta()`, `_collate_samples()`, `_search_int_in_str()` -- used by 6+ other modules

This means other modules import from `hcs.py` for utility functions, creating unnecessary coupling. If a user only needs `CachedOmeZarrDataModule`, they still transitively depend on all of `hcs.py`.

### Solution: Extract to `_utils.py`

**Move these functions from `hcs.py` to `_utils.py`:**

| Function | Current Location | Used By |
|----------|-----------------|---------|
| `_ensure_channel_list()` | `hcs.py` | `gpu_aug.py`, `mmap_cache.py`, `hcs.py` (self) |
| `_read_norm_meta()` | `hcs.py` | `gpu_aug.py`, `mmap_cache.py`, `triplet.py`, `cell_classification.py`, `hcs.py` (self) |
| `_collate_samples()` | `hcs.py` | `combined.py`, `concat.py`, `hcs.py` (self) |
| `_search_int_in_str()` | `hcs.py` | `hcs.py` (self -- used by `MaskTestDataset`) |

**`_utils.py` content:**

```python
"""Internal utilities shared across viscy-data modules.

Extracted from hcs.py to prevent that module from being both a concrete
DataModule and a utility library.
"""

import re
from typing import Sequence

import torch
from monai.data.utils import collate_meta_tensor
from torch import Tensor

from viscy_data._typing import NormMeta, Sample


def _ensure_channel_list(str_or_seq: str | Sequence[str]) -> list[str]:
    """Ensure channel argument is a list of strings."""
    if isinstance(str_or_seq, str):
        return [str_or_seq]
    try:
        return list(str_or_seq)
    except TypeError:
        raise TypeError(
            "Channel argument must be a string or sequence of strings. "
            f"Got {str_or_seq}."
        )


def _search_int_in_str(pattern: str, file_name: str) -> str:
    """Search image indices in a file name with regex patterns."""
    match = re.search(pattern, file_name)
    if match:
        return match.group()
    else:
        raise ValueError(f"Cannot find pattern {pattern} in {file_name}.")


def _collate_samples(batch: Sequence[Sample]) -> Sample:
    """Collate samples into a batch sample."""
    collated: Sample = {}
    for key in batch[0].keys():
        data = []
        for sample in batch:
            if isinstance(sample[key], Sequence):
                data.extend(sample[key])
            else:
                data.append(sample[key])
        collated[key] = collate_meta_tensor(data)
    return collated


def _read_norm_meta(fov) -> NormMeta | None:
    """Read normalization metadata from an iohub Position.

    Convert to float32 tensors to avoid automatic casting to float64.
    The fov parameter is typed as Any to avoid importing iohub at module level
    (iohub is a required dependency but this avoids circular import risk).
    """
    norm_meta = fov.zattrs.get("normalization", None)
    if norm_meta is None:
        return None
    for channel, channel_values in norm_meta.items():
        for level, level_values in channel_values.items():
            for stat, value in level_values.items():
                if isinstance(value, Tensor):
                    value = value.clone().float()
                else:
                    value = torch.tensor(value, dtype=torch.float32)
                norm_meta[channel][level][stat] = value
    return norm_meta
```

**Updated `hcs.py` imports:**

```python
# Before (in viscy/data/hcs.py):
# Functions defined inline

# After (in viscy_data/hcs.py):
from viscy_data._utils import (
    _collate_samples,
    _ensure_channel_list,
    _read_norm_meta,
    _search_int_in_str,
)
```

**Updated consumer imports (e.g., gpu_aug.py):**

```python
# Before:
from viscy.data.hcs import _ensure_channel_list, _read_norm_meta

# After:
from viscy_data._utils import _ensure_channel_list, _read_norm_meta
```

## The Typing Overlap Problem: Detailed Solution

### Problem

`viscy/data/typing.py` defines `DictTransform` which was also copied to `viscy_transforms/_typing.py`. Both packages need this type alias.

Additionally, `viscy_transforms/_typing.py` already contains copies of `Sample`, `ChannelMap`, `NormMeta`, `HCSStackIndex`, `LevelNormStats`, `ChannelNormStats`, and `OneOrSeq` -- these were extracted during Milestone 1.

### Solution: Duplicate the type alias (Option B from README)

`DictTransform` is a single-line type alias:

```python
DictTransform = Callable[[dict[str, Tensor | dict]], dict[str, Tensor]]
```

**In `viscy_data/_typing.py`:** Keep a local copy. The duplication cost is trivial (one line), and it avoids adding viscy-transforms as a dependency just for a type alias. This aligns with the project constraint: "viscy-data must NOT depend on viscy-transforms."

**`_typing.py` for viscy-data should contain the FULL set of types from `viscy/data/typing.py`:**

```python
"""Data-specific type definitions for viscy-data.

Provides Sample, NormMeta, TripletSample, and other types used throughout
the data loading pipeline. DictTransform is duplicated from viscy-transforms
(a single-line type alias) to avoid cross-package dependency.
"""

from typing import Callable, Literal, NamedTuple, Sequence, TypedDict, TypeVar

from torch import ShortTensor, Tensor
from typing_extensions import NotRequired

# Duplicated from viscy_transforms._typing (single-line alias, not worth a dependency)
DictTransform = Callable[[dict[str, Tensor | dict]], dict[str, Tensor]]

T = TypeVar("T")
OneOrSeq = T | Sequence[T]

# ... all other types from viscy/data/typing.py unchanged ...
```

This is a verbatim copy of the original `viscy/data/typing.py`. No semantic changes needed.

## Removing the viscy-transforms Dependency

### Problem

`triplet.py` imports `BatchedCenterSpatialCropd` from `viscy.transforms`:

```python
from viscy.transforms import BatchedCenterSpatialCropd
```

This is used in `TripletDataModule._final_crop()` to create a batched center crop. The project constraint says viscy-data must NOT depend on viscy-transforms.

### Solution: Replace with shape assertion

The `BatchedCenterSpatialCropd` in `_final_crop()` performs a center spatial crop on batched data. In `TripletDataModule`, this is applied inside `on_after_batch_transfer()` via `_transform_channel_wise()`. The replacement approach:

1. **Remove the import** of `BatchedCenterSpatialCropd`
2. **Replace `_final_crop()`** with MONAI's standard `CenterSpatialCropd` (already imported in `hcs.py` which `TripletDataModule` inherits from)
3. **Add a shape assertion** in `on_after_batch_transfer()` to verify the output shape matches expectations

```python
# In TripletDataModule:
def _final_crop(self) -> CenterSpatialCropd:
    """Center crop to the target size.

    Uses MONAI's CenterSpatialCropd. The crop operates per-channel
    after _transform_channel_wise scatters the batch into individual channels.
    """
    return CenterSpatialCropd(
        keys=self.source_channel,
        roi_size=(
            self.z_window_size,
            self.yx_patch_size[0],
            self.yx_patch_size[1],
        ),
    )
```

**Why this works:** `_transform_channel_wise()` scatters the batched tensor into per-channel dictionaries, applies transforms, and gathers back. MONAI's `CenterSpatialCropd` (non-batched) works on individual channel tensors within the dict. The key insight is that `BatchedCenterSpatialCropd` was only needed because the original code path was applying the crop to a batch dimension -- but `_transform_channel_wise` already handles the batch by operating per-channel. The standard MONAI crop suffices here.

**Verification needed during implementation:** Run the triplet tests with `z_window_size` parametrization to confirm output shapes match.

## Optional Dependency Groups Structure

### Design

The package has three tiers of dependencies:

**Tier 1 - Required (always installed):**
- `torch` (core tensor ops)
- `lightning` (LightningDataModule base class)
- `numpy` (array conversion)
- `iohub` (OME-Zarr I/O -- used by 8 of 13 modules)
- `monai` (transforms, data utilities -- used by 10 of 13 modules)
- `zarr` (direct zarr operations in hcs.py caching)
- `imageio` (imread in MaskTestDataset)

**Tier 2 - Optional dependency groups (install with extras):**

| Group Name | Dependencies | Required By |
|------------|-------------|-------------|
| `triplet` | `tensorstore`, `pandas` | `triplet.py`, `cell_classification.py` |
| `livecell` | `pycocotools`, `tifffile`, `torchvision` | `livecell.py` |
| `mmap` | `tensordict` | `mmap_cache.py` |
| `all` | All of the above | Everything |

**Tier 3 - Development (dependency groups, not optional-dependencies):**
- `pytest`, `pytest-cov`

### pyproject.toml Structure

```toml
[project]
name = "viscy-data"
dynamic = ["version"]
dependencies = [
    "iohub>=0.2",
    "imageio>=2.35",
    "lightning>=2.5",
    "monai>=1.5.2",
    "numpy>=2.4.1",
    "torch>=2.10",
    "zarr>=3",
]

[project.optional-dependencies]
triplet = ["tensorstore>=0.1.68", "pandas>=2.2"]
livecell = ["pycocotools>=2.0.8", "tifffile>=2024.1", "torchvision>=0.20"]
mmap = ["tensordict>=0.6"]
all = [
    "viscy-data[triplet]",
    "viscy-data[livecell]",
    "viscy-data[mmap]",
]

[dependency-groups]
dev = [{ include-group = "test" }]
test = ["pytest>=9.0.2", "pytest-cov>=7"]
```

### Lazy Import Strategy

Modules with optional dependencies must use lazy imports with clear error messages:

```python
# In triplet.py:
def _setup_tensorstore():
    try:
        import tensorstore as ts
    except ImportError:
        raise ImportError(
            "tensorstore is required for TripletDataset. "
            "Install with: pip install 'viscy-data[triplet]'"
        ) from None
    return ts
```

**Where to apply lazy imports:**

| Module | Optional Import | Lazy Import Location |
|--------|----------------|---------------------|
| `triplet.py` | `tensorstore`, `pandas` | Module-level: guard both imports at top |
| `cell_classification.py` | `pandas` | Module-level: guard `pandas` import |
| `livecell.py` | `pycocotools`, `tifffile`, `torchvision` | Module-level: guard all three |
| `mmap_cache.py` | `tensordict` | Module-level: guard `tensordict.memmap` import |

**Pattern:** Use try/except at module level, raising `ImportError` with install instructions. Do NOT use `TYPE_CHECKING` guards for runtime dependencies.

```python
# Module-level lazy import pattern (preferred):
try:
    import pandas as pd
    import tensorstore as ts
except ImportError as e:
    _missing_dep = e
else:
    _missing_dep = None

# Then in class __init__ or function body:
if _missing_dep is not None:
    raise ImportError(
        f"Optional dependency missing: {_missing_dep}. "
        "Install with: pip install 'viscy-data[triplet]'"
    ) from _missing_dep
```

This pattern allows the module to be _imported_ without error (so `__init__.py` can reference it), but raises a clear error when a class or function is actually _used_.

## Internal Dependency Graph (Post-Extraction)

```
_typing.py          (no deps)
    |
_utils.py           (depends on: _typing)
    |
select.py           (no internal deps, external: iohub)
distributed.py      (no internal deps, external: torch)
    |
hcs.py              (depends on: _typing, _utils)
    |
    +-- gpu_aug.py   (depends on: _typing, _utils, select, distributed)
    |       |
    |       +-- mmap_cache.py  (depends on: _typing, _utils, gpu_aug, select)
    |       +-- ctmc_v1.py     (depends on: gpu_aug)
    |       +-- livecell.py    (depends on: _typing, gpu_aug)
    |
    +-- triplet.py   (depends on: _typing, _utils, hcs, select)
    |       |
    |       +-- cell_classification.py  (depends on: _typing, _utils, triplet)
    |       +-- cell_division_triplet.py (depends on: _typing, hcs, triplet)
    |
segmentation.py     (depends on: _typing; external: iohub)
combined.py         (depends on: _utils; external: lightning)
concat.py           (depends on: _utils, distributed; external: torch, monai, lightning)
```

**Key observation:** The graph is a DAG (no cycles). This confirms the package can be cleanly organized without circular imports.

## `__init__.py` Public API Design

### Principle: Export Classes, Not Internals

Users should be able to do:
```python
from viscy_data import HCSDataModule, TripletDataModule
from viscy_data import Sample, NormMeta  # Types
```

They should NOT need to know about `_utils`, `_typing`, or internal module paths.

### Recommended `__init__.py`

```python
"""VisCy Data - Data loading modules for virtual staining microscopy.

This package provides PyTorch Lightning DataModules for loading
HCS OME-Zarr microscopy data in virtual staining workflows.

Public API:
    All DataModules and types are exported at the package level.
    Example: `from viscy_data import HCSDataModule`

Optional dependencies:
    Some modules require additional packages:
    - TripletDataModule: pip install 'viscy-data[triplet]'
    - LiveCellDataModule: pip install 'viscy-data[livecell]'
    - MmappedDataModule: pip install 'viscy-data[mmap]'
    - All extras: pip install 'viscy-data[all]'
"""

from importlib.metadata import version

# Types (always available)
from viscy_data._typing import (
    AnnotationColumns,
    ChannelMap,
    DictTransform,
    HCSStackIndex,
    NormMeta,
    Sample,
    SegmentationSample,
    TripletSample,
)

# Core modules (always available -- iohub + monai required)
from viscy_data.combined import CombinedDataModule
from viscy_data.concat import (
    BatchedConcatDataModule,
    CachedConcatDataModule,
    ConcatDataModule,
)
from viscy_data.distributed import ShardedDistributedSampler
from viscy_data.gpu_aug import (
    CachedOmeZarrDataModule,
    CachedOmeZarrDataset,
    GPUTransformDataModule,
)
from viscy_data.hcs import HCSDataModule, MaskTestDataset, SlidingWindowDataset
from viscy_data.segmentation import SegmentationDataModule, SegmentationDataset
from viscy_data.select import SelectWell

# Modules with optional deps -- import will succeed,
# usage will raise ImportError with install instructions
from viscy_data.cell_classification import (
    ClassificationDataModule,
    ClassificationDataset,
)
from viscy_data.cell_division_triplet import (
    CellDivisionTripletDataModule,
    CellDivisionTripletDataset,
)
from viscy_data.ctmc_v1 import CTMCv1DataModule
from viscy_data.livecell import (
    LiveCellDataModule,
    LiveCellDataset,
    LiveCellTestDataset,
)
from viscy_data.mmap_cache import MmappedDataModule, MmappedDataset
from viscy_data.triplet import TripletDataModule, TripletDataset

__version__ = version("viscy-data")

__all__ = [
    # Types
    "AnnotationColumns",
    "ChannelMap",
    "DictTransform",
    "HCSStackIndex",
    "NormMeta",
    "Sample",
    "SegmentationSample",
    "TripletSample",
    # Core DataModules
    "CachedOmeZarrDataModule",
    "CombinedDataModule",
    "ConcatDataModule",
    "BatchedConcatDataModule",
    "CachedConcatDataModule",
    "GPUTransformDataModule",
    "HCSDataModule",
    "SegmentationDataModule",
    # Core Datasets
    "CachedOmeZarrDataset",
    "MaskTestDataset",
    "SegmentationDataset",
    "SlidingWindowDataset",
    # Core Utilities
    "SelectWell",
    "ShardedDistributedSampler",
    # Optional-dep DataModules
    "ClassificationDataModule",
    "CellDivisionTripletDataModule",
    "CTMCv1DataModule",
    "LiveCellDataModule",
    "MmappedDataModule",
    "TripletDataModule",
    # Optional-dep Datasets
    "CellDivisionTripletDataset",
    "ClassificationDataset",
    "LiveCellDataset",
    "LiveCellTestDataset",
    "MmappedDataset",
    "TripletDataset",
]
```

**Decision: Eager imports in `__init__.py`, lazy imports inside modules.** The `__init__.py` imports all modules eagerly, but modules with optional deps use the try/except pattern internally. This means `import viscy_data` works without optional deps, and users get a clear error only when they try to _use_ a class that needs them (e.g., instantiating `TripletDataModule`).

## Data Flow

### Import Resolution Chain

```
User code:
    from viscy_data import TripletDataModule
        |
__init__.py:
    from viscy_data.triplet import TripletDataModule
        |
triplet.py module-level:
    try:
        import pandas as pd          # Optional
        import tensorstore as ts      # Optional
    except ImportError:
        _missing_dep = <exception>

    from viscy_data._typing import ...    # Always works
    from viscy_data._utils import ...     # Always works
    from viscy_data.hcs import HCSDataModule  # Always works (base class)
    from viscy_data.select import ...     # Always works
        |
TripletDataModule.__init__():
    if _missing_dep is not None:
        raise ImportError("Install with: pip install 'viscy-data[triplet]'")
```

### Training Pipeline Data Flows (Unchanged)

The three training pipelines use the same data flow patterns as documented in `viscy/data/README.md`. The extraction does not change any runtime behavior:

```
FCMAE Pretrain:
    CombinedDataModule -> [CachedOmeZarrDataModule] -> CachedOmeZarrDataset
    Engine calls: dm.train_gpu_transforms()

Translation Fine-tune:
    HCSDataModule -> SlidingWindowDataset
    CPU-only transforms in __getitem__()

DynaCLR Contrastive:
    TripletDataModule -> TripletDataset
    on_after_batch_transfer() with channel scatter/gather
```

## Patterns to Follow

### Pattern 1: Mirror viscy-transforms Structure Exactly

**What:** Follow the identical packaging pattern established by viscy-transforms.

**Why:** Consistency across the monorepo. The build system, versioning, layout, and testing patterns are already proven.

**Configuration mirrors:**
```toml
[build-system]
build-backend = "hatchling.build"
requires = ["hatchling", "uv-dynamic-versioning"]

[tool.hatch.build.targets.wheel]
packages = ["src/viscy_data"]

[tool.hatch.version]
source = "uv-dynamic-versioning"

[tool.uv-dynamic-versioning]
vcs = "git"
style = "pep440"
pattern-prefix = "viscy-data-"
fallback-version = "0.0.0"
```

### Pattern 2: Internal Imports Use Package Name

**What:** All internal imports use `viscy_data.` prefix, not relative imports.

**Why:** Explicit, grep-able, consistent with viscy-transforms style.

```python
# Good:
from viscy_data._typing import Sample, NormMeta
from viscy_data._utils import _read_norm_meta

# Avoid:
from ._typing import Sample, NormMeta
from ._utils import _read_norm_meta
```

### Pattern 3: Workspace Dependency Declaration

**What:** Root `pyproject.toml` declares viscy-data as a workspace source.

```toml
# Root pyproject.toml additions:
[project]
dependencies = ["viscy-transforms", "viscy-data"]

[tool.uv.sources]
viscy-transforms = { workspace = true }
viscy-data = { workspace = true }
```

### Pattern 4: Test Fixtures Stay Local

**What:** Test fixtures (like `preprocessed_hcs_dataset`) live in `packages/viscy-data/tests/conftest.py`, not shared at workspace root.

**Why:** Package independence. Tests must work with `uv run --package viscy-data pytest`.

### Pattern 5: cell_classification.py Depends on triplet.py Only for INDEX_COLUMNS

**What:** `cell_classification.py` imports `INDEX_COLUMNS` from `triplet.py`.

**Solution:** Move `INDEX_COLUMNS` to `_typing.py` or `_utils.py` to break this fragile coupling. `INDEX_COLUMNS` is a constant (list of column names), not a function. It belongs with the type definitions.

```python
# In _typing.py:
INDEX_COLUMNS = [
    "fov_name",
    "track_id",
    "t",
    "id",
    "parent_track_id",
    "parent_id",
    "z",
    "y",
    "x",
]
```

Then both `triplet.py` and `cell_classification.py` import from `_typing`.

### Pattern 6: cell_division_triplet.py Depends on triplet.py Only for _transform_channel_wise

**What:** `cell_division_triplet.py` imports `_transform_channel_wise` from `triplet.py`.

**Solution:** Move `_transform_channel_wise`, `_scatter_channels`, and `_gather_channels` to `_utils.py`. These are general-purpose helper functions for channel-wise transform application, not specific to triplet sampling.

```python
# In _utils.py (additions):
def _scatter_channels(
    channel_names: list[str], patch: Tensor, norm_meta: NormMeta | None
) -> dict[str, Tensor | NormMeta] | dict[str, Tensor]:
    ...

def _gather_channels(
    patch_channels: dict[str, Tensor | NormMeta],
) -> list[Tensor]:
    ...

def _transform_channel_wise(
    transform: DictTransform,
    channel_names: list[str],
    patch: Tensor,
    norm_meta: NormMeta | None,
) -> list[Tensor]:
    ...
```

This reduces coupling between `cell_division_triplet.py` and `triplet.py`.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Importing from viscy.data (Old Path)

**What:** `from viscy.data.hcs import HCSDataModule`

**Why bad:** The old import path will not exist in the extracted package. Any remnant of `viscy.data` in the new code is a bug.

**Instead:** `from viscy_data.hcs import HCSDataModule` or `from viscy_data import HCSDataModule`

### Anti-Pattern 2: Making _utils.py Part of Public API

**What:** Documenting or exporting `_ensure_channel_list`, `_read_norm_meta` in `__init__.py`.

**Why bad:** These are internal helpers. Underscore prefix signals this. Users should not depend on them.

**Instead:** Only export classes and types in `__init__.py`.

### Anti-Pattern 3: Eager Import of Optional Dependencies at Package Level

**What:** `import tensorstore` at top of `triplet.py` without try/except.

**Why bad:** `import viscy_data` fails if tensorstore is not installed, even if the user only wants `HCSDataModule`.

**Instead:** Module-level try/except with deferred error at usage time.

### Anti-Pattern 4: Sub-packages for "Logical Grouping"

**What:** Creating `viscy_data/modules/`, `viscy_data/datasets/`, `viscy_data/types/`.

**Why bad:** The dependency graph shows most modules import from most other modules. Sub-packages add directory depth without real encapsulation. Migration complexity increases with no user benefit.

**Instead:** Flat module layout with `_` prefix for internal modules.

### Anti-Pattern 5: Keeping hcs.py as Utility Provider

**What:** Having other modules import utility functions from `hcs.py`.

**Why bad:** Forces loading `HCSDataModule` and all its dependencies when only a helper function is needed. Violates single-responsibility.

**Instead:** Extract to `_utils.py`.

## Build Order for Extraction

Based on the dependency graph, the recommended build order for the extraction milestone:

```
Step 1: Package scaffolding (no code)
    packages/viscy-data/pyproject.toml
    packages/viscy-data/src/viscy_data/__init__.py (empty placeholder)
    packages/viscy-data/src/viscy_data/py.typed
    packages/viscy-data/tests/__init__.py

Step 2: Foundation modules (no internal deps)
    _typing.py     -- verbatim copy from viscy/data/typing.py + DictTransform alias
    _utils.py      -- extract from hcs.py + channel scatter/gather from triplet.py
    select.py      -- verbatim copy, update imports
    distributed.py -- verbatim copy (no internal imports to change)

Step 3: Core data modules (depend on foundation)
    hcs.py         -- copy, remove extracted functions, update imports
    gpu_aug.py     -- copy, update imports to viscy_data._utils

Step 4: Derived data modules (depend on core)
    mmap_cache.py              -- copy, update imports
    triplet.py                 -- copy, remove viscy-transforms import, update imports
    cell_classification.py     -- copy, update imports (INDEX_COLUMNS from _typing)
    cell_division_triplet.py   -- copy, update imports (_transform_channel_wise from _utils)
    ctmc_v1.py                 -- copy, update imports
    livecell.py                -- copy, update imports, add lazy import guards
    segmentation.py            -- copy, update imports

Step 5: Composite modules (depend on core)
    combined.py    -- extract CombinedDataModule + CombineMode, update imports
    concat.py      -- extract Concat* classes from combined.py, update imports

Step 6: Finalize package
    __init__.py    -- full public API with all exports
    pyproject.toml -- complete with optional-dependencies and dependency-groups

Step 7: Migrate tests
    conftest.py    -- copy from tests/conftest.py, update imports
    test_hcs.py    -- copy, update imports
    test_triplet.py -- copy, update imports, handle viscy-transforms removal
    test_typing.py -- new: smoke tests for type definitions

Step 8: Workspace integration
    Update root pyproject.toml: add viscy-data to deps and sources
    Update root tool.pytest.testpaths if needed
    Verify: uv sync --package viscy-data
    Verify: uv run --package viscy-data pytest
```

**Critical path:** Steps 1-2-3-4-5-6 are sequential (each depends on prior).
**Parallelizable:** Step 7 (tests) can begin after Step 3 for modules already migrated.

## Scalability Considerations

| Concern | Now (13 modules) | After 20+ modules | Notes |
|---------|------------------|-------------------|-------|
| Import time | ~200ms (all modules) | Could grow | Lazy imports for optional deps mitigate this |
| Test time | ~30s (HCS fixture creation) | Same | Fixture scoping (session) keeps it fast |
| CI matrix | 1 job per Python version | Same | viscy-data tests one set of jobs |
| API surface | 15 classes + 8 types | Stable | New modules add to optional groups |
| Optional dep groups | 3 groups | May grow | `[all]` meta-group keeps it manageable |

## Sources

**PRIMARY (direct source code analysis -- HIGH confidence):**
- `viscy/data/README.md` on `main` branch -- comprehensive architecture documentation
- All 13 source files in `viscy/data/` on `main` branch -- full implementation review
- `packages/viscy-transforms/` -- established extraction pattern
- `.planning/PROJECT.md` -- project constraints and decisions
- `.planning/ROADMAP.md` -- milestone 1 completion status
- `.planning/research/ARCHITECTURE.md` (previous) -- workspace patterns

**SECONDARY (established patterns -- HIGH confidence):**
- `packages/viscy-transforms/pyproject.toml` -- build system configuration template
- `packages/viscy-transforms/src/viscy_transforms/__init__.py` -- public API pattern
- `packages/viscy-transforms/src/viscy_transforms/_typing.py` -- type extraction pattern
- Root `pyproject.toml` -- workspace configuration

**DECISIONS referenced:**
- No viscy-transforms dependency (PROJECT.md constraint)
- Optional dependency groups (PROJECT.md active requirement)
- Extract shared utilities from hcs.py (PROJECT.md active requirement)
