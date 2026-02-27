# Feature Research: viscy-data Subpackage

**Domain:** Scientific microscopy data loading (Lightning DataModules for HCS OME-Zarr)
**Researched:** 2026-02-13
**Confidence:** HIGH (based on direct source code analysis of 13 modules, existing data README, and comparable library patterns from MONAI)

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = package feels incomplete or broken.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **All 13 modules extracted and importable** | Users have existing configs and code referencing these classes | LOW | Straight extraction following viscy-transforms pattern |
| **Clean import paths** | `from viscy_data import HCSDataModule` not `from viscy.data import HCSDataModule` | LOW | Established pattern from viscy-transforms extraction |
| **Type exports** | `Sample`, `NormMeta`, `ChannelMap`, `DictTransform` accessible for type annotations | LOW | Keep local `_typing.py` (duplicate `DictTransform` rather than depend on viscy-transforms for one alias) |
| **Flat top-level exports for DataModules** | MONAI pattern: `monai.data.Dataset`, `monai.data.DataLoader` all top-level; users expect `from viscy_data import HCSDataModule` | MEDIUM | 15+ classes to export; need careful `__init__.py` with `__all__` |
| **Optional dependency groups** | `tensorstore` (triplet), `tensordict` (mmap), `pycocotools` (livecell) are heavy; users installing for translation pipeline should not need them | MEDIUM | Use `[project.optional-dependencies]` with `triplet`, `livecell`, `mmap`, `all` extras |
| **Workspace dependency on viscy-transforms** | `triplet.py` imports `BatchedCenterSpatialCropd`; must declare properly | LOW | `viscy-transforms = { workspace = true }` in `[tool.uv.sources]` |
| **Shared utilities extracted from hcs.py** | `_ensure_channel_list`, `_read_norm_meta`, `_collate_samples` are used by 5+ modules; must not live in hcs.py | LOW | New `_utils.py` module; internal refactor, no public API change |
| **py.typed marker** | Type checking support; viscy-transforms already has it | LOW | Established pattern |
| **Existing tests passing** | 3 test files (`test_hcs.py`, `test_triplet.py`, `test_select.py`) must pass under new import paths | MEDIUM | Test fixtures require OME-Zarr datasets; may need conftest.py adjustments |
| **src layout** | `packages/viscy-data/src/viscy_data/` | LOW | Established workspace pattern |

### Differentiators (Competitive Advantage)

Features that make viscy-data better than a raw code dump. Not required for extraction, but increase package quality.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Lazy imports for heavy optional deps** | `import viscy_data` does not fail if `tensorstore` is missing; clear error only when `TripletDataModule` is used | MEDIUM | Use `try/except ImportError` with informative messages; prevents install friction for users who only need `HCSDataModule` |
| **Submodule organization for specialized modules** | `from viscy_data.contrib import LiveCellDataModule` separates community/specialized modules from core | MEDIUM | See API Design section below; reduces cognitive load |
| **Package README with pipeline mapping table** | Existing README documents which DataModule serves which pipeline; include in package | LOW | Adapt from `viscy/data/README.md` already written |
| **Utility function exports** | `_ensure_channel_list`, `_read_norm_meta` are useful to downstream users building custom DataModules | LOW | Promote from `_utils` to public API where warranted |
| **GPU transform mixin as protocol** | `GPUTransformMixin` (Protocol) instead of ABC; enables duck typing without forced inheritance | MEDIUM | README already recommends this; enables engines to query transforms without knowing concrete type |
| **Type-safe batch structures** | Export `Sample`, `TripletSample`, `SegmentationSample` as first-class types for downstream type checking | LOW | Already defined in `typing.py`; just need proper exports |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Re-export all MONAI transforms** | "Users want one import for everything" | Creates massive import surface; hides which package owns what; version conflicts between viscy-transforms and viscy-data MONAI pins | Users import MONAI transforms directly; viscy-data only exports its own classes |
| **Backward-compatible `viscy.data` import shim** | "Don't break existing code" | Milestone 1 established clean break pattern; adding shims contradicts design decision and creates maintenance burden | Document migration: `s/from viscy.data/from viscy_data/g` in CONTRIBUTING.md |
| **Auto-detect pipeline type from config** | "Smart DataModule that figures out what you need" | Couples all pipelines together; defeats modularity; makes debugging opaque | Explicit DataModule selection in config; each module has clear purpose |
| **Split specialized modules into separate packages** | "viscy-data-triplet, viscy-data-livecell" | Over-fragmentation; 13 modules is not enough to justify 4+ packages; confuses users; multiplies release overhead | One package with optional dependency groups |
| **Abstract base class for all DataModules** | "Enforce common interface" | `HCSDataModule` and `GPUTransformDataModule` serve different inheritance chains; forcing single ABC breaks existing class hierarchy | Keep two base patterns (HCS-based, GPU-transform-based); document when to use each |
| **Unified batch structure** | "All DataModules should return same dict format" | `Sample` (source/target), `TripletSample` (anchor/pos/neg), and FCMAE nested lists are fundamentally different; forced unification hides real semantic differences | Export distinct typed dicts; let engines handle the difference |
| **Moving BatchedCenterSpatialCropd into viscy-data** | "Eliminate the cross-package dependency" | It is a general-purpose transform, not data-specific; moving it breaks the transforms package's cohesion | Keep dependency viscy-data -> viscy-transforms; one-way, clean |

## Public API Design

### Recommendation: Flat Top-Level with Logical Grouping via `__all__`

Follow the MONAI `monai.data` pattern: all public classes exported at the top level, but organized by category in `__all__` and documentation. This matches what viscy-transforms already does (44 exports, all top-level).

**Why flat, not nested submodules:**
1. viscy-transforms precedent -- users already learn `from viscy_transforms import X`
2. MONAI precedent -- `monai.data` exports 100+ symbols at top level
3. Only 20-25 public symbols total -- not enough to justify submodule navigation
4. Submodules add import path confusion (`from viscy_data.triplet import TripletDataModule` vs `from viscy_data import TripletDataModule`)

### Proposed `__init__.py` Export Categories

```python
# viscy_data/__init__.py

# --- Core DataModules (Translation Pipeline) ---
from viscy_data._hcs import HCSDataModule, SlidingWindowDataset, MaskTestDataset

# --- GPU Transform DataModules (FCMAE Pipeline) ---
from viscy_data._gpu_aug import (
    GPUTransformDataModule,
    CachedOmeZarrDataset,
    CachedOmeZarrDataModule,
)
from viscy_data._mmap_cache import MmappedDataset, MmappedDataModule

# --- Contrastive DataModules (DynaCLR Pipeline) ---
from viscy_data._triplet import TripletDataset, TripletDataModule
from viscy_data._cell_division_triplet import (
    CellDivisionTripletDataset,
    CellDivisionTripletDataModule,
)

# --- Specialized DataModules ---
from viscy_data._livecell import LiveCellDataset, LiveCellDataModule
from viscy_data._ctmc_v1 import CTMCv1DataModule
from viscy_data._cell_classification import (
    ClassificationDataset,
    ClassificationDataModule,
)
from viscy_data._segmentation import SegmentationDataset, SegmentationDataModule

# --- Composition DataModules ---
from viscy_data._combined import (
    CombinedDataModule,
    ConcatDataModule,
    BatchedConcatDataModule,
    CachedConcatDataModule,
)

# --- Utilities ---
from viscy_data._select import SelectWell
from viscy_data._distributed import ShardedDistributedSampler

# --- Types ---
from viscy_data._typing import (
    DictTransform,
    Sample,
    ChannelMap,
    NormMeta,
    HCSStackIndex,
    TripletSample,
    SegmentationSample,
)
```

### Module Naming Convention

Use underscore-prefixed private modules (matching viscy-transforms pattern):
- `_hcs.py` (not `hcs.py`) -- signals "import from package top-level, not from module"
- `_typing.py`, `_utils.py`, `_select.py`, `_distributed.py` -- internal modules
- Users always do `from viscy_data import HCSDataModule`, never `from viscy_data._hcs import HCSDataModule`

### What NOT to Export at Top Level

| Symbol | Why Private | Access Pattern |
|--------|-------------|---------------|
| `_ensure_channel_list` | Internal utility; not part of user-facing API | Used internally by multiple modules |
| `_read_norm_meta` | Internal utility; tightly coupled to iohub Position | Used internally; advanced users can access via `viscy_data._utils` |
| `_collate_samples` | Internal collation logic | Used internally by HCS and combined modules |
| `CombineMode` | Enum for combined loader modes; passed as string in configs | Keep in `_combined.py`; users pass string values |
| `BatchedConcatDataset` | Internal dataset class for concat batching | Only used by `BatchedConcatDataModule` |
| `INDEX_COLUMNS` | Internal constant for triplet indexing | Only used by triplet and classification modules |
| Label constants (`LABEL_INFECTION_STATE`, etc.) | Application-specific constants | Keep in `_typing.py`; export only if classification is actively used |

## Feature Dependencies

```
_typing.py (foundation -- no deps)
    |
    +-- _utils.py (shared helpers, depends on iohub, torch)
    |       |
    |       +-- _hcs.py (core DataModule, depends on _typing, _utils, iohub, monai)
    |       |       |
    |       |       +-- _triplet.py (extends HCSDataModule, adds pandas, tensorstore)
    |       |       |       |
    |       |       |       +-- _cell_classification.py (uses _triplet.INDEX_COLUMNS)
    |       |       |
    |       |       +-- _cell_division_triplet.py (extends HCSDataModule)
    |       |
    |       +-- _gpu_aug.py (abstract base, depends on _typing, _utils, _select, _distributed)
    |               |
    |               +-- _mmap_cache.py (extends GPUTransformDataModule + SelectWell)
    |               |
    |               +-- _livecell.py (extends GPUTransformDataModule)
    |               |
    |               +-- _ctmc_v1.py (extends GPUTransformDataModule)
    |
    +-- _select.py (SelectWell mixin, depends on iohub only)
    |
    +-- _distributed.py (ShardedDistributedSampler, depends on torch only)
    |
    +-- _segmentation.py (standalone, depends on _typing, iohub)
    |
    +-- _combined.py (wrappers, depends on _hcs._collate_samples, _distributed)

Cross-package:
    _triplet.py ──depends on──> viscy-transforms (BatchedCenterSpatialCropd)
```

### Dependency Notes

- **_triplet.py requires viscy-transforms:** Single import of `BatchedCenterSpatialCropd` for `_final_crop()`. This is a genuine runtime dependency, not removable without changing behavior. One-way dependency (data -> transforms), no circular risk.
- **_cell_classification.py requires _triplet.py:** Imports `INDEX_COLUMNS` constant. Could be extracted to `_typing.py` to remove this coupling.
- **_gpu_aug.py requires _hcs.py:** Imports `_ensure_channel_list` and `_read_norm_meta`. Refactoring these into `_utils.py` breaks this coupling.
- **_combined.py requires _hcs.py:** Imports `_collate_samples`. Same refactor to `_utils.py` resolves this.
- **Heavy optional deps are leaf-only:** `tensorstore` (triplet), `tensordict` (mmap), `pycocotools`/`tifffile` (livecell) are used only by their respective modules. Core modules (`_hcs.py`, `_gpu_aug.py`) have no heavy optional deps.

## MVP Definition

### Launch With (v1.0 -- Extraction Milestone)

Minimum viable extraction -- all existing functionality works under new import paths.

- [ ] **All 13 modules extracted to `packages/viscy-data/src/viscy_data/`** -- direct migration
- [ ] **Shared utilities refactored into `_utils.py`** -- break hcs.py's dual role as module + utility library
- [ ] **`_typing.py` with all data-specific types** -- local copy of DictTransform (no dependency on viscy-transforms for types)
- [ ] **Flat top-level exports in `__init__.py`** -- all DataModules and Datasets importable from package root
- [ ] **Optional dependency groups** -- `pip install viscy-data[triplet]`, `viscy-data[livecell]`, `viscy-data[mmap]`, `viscy-data[all]`
- [ ] **Workspace dependency on viscy-transforms** -- declared in pyproject.toml with `workspace = true`
- [ ] **All 3 existing test files passing** -- `test_hcs.py`, `test_triplet.py`, `test_select.py`
- [ ] **Package README** -- adapted from existing `viscy/data/README.md`
- [ ] **py.typed marker** -- type checking support

### Add After Validation (v1.x)

Features to add once extraction is stable and users have migrated.

- [ ] **Lazy imports for optional dependencies** -- `tensorstore`, `tensordict`, `pycocotools` imported only when needed; trigger: user complaints about install size
- [ ] **Extract INDEX_COLUMNS from _triplet.py to _typing.py** -- break cell_classification -> triplet coupling; trigger: during extraction refactor
- [ ] **GPU transform protocol/mixin** -- formalize `GPUTransformMixin` as Protocol for duck typing; trigger: when engines are extracted to viscy-models
- [ ] **Additional test coverage** -- tests for combined.py, mmap_cache.py, livecell.py (currently untested); trigger: extraction complete, need confidence in isolated behavior

### Future Consideration (v2+)

Features to defer until package is stable and actively maintained.

- [ ] **Split combined.py into combined.py + concat.py** -- reduce module size; defer because 5 classes in one file is manageable
- [ ] **Promote _read_norm_meta to public API** -- useful for custom DataModule builders; defer until there is user demand
- [ ] **Abstract cache interface** -- standardize caching across Manager.dict, tensorstore, MemoryMappedTensor patterns; defer because unification adds complexity without immediate user benefit
- [ ] **Config-driven DataModule registry** -- Lightning CLI integration for automatic class resolution; defer until viscy meta-package exists

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| All modules extracted with clean imports | HIGH | LOW | P1 |
| Shared utilities in _utils.py | HIGH | LOW | P1 |
| _typing.py with all data types | HIGH | LOW | P1 |
| Flat top-level __init__.py exports | HIGH | LOW | P1 |
| Optional dependency groups | HIGH | LOW | P1 |
| Workspace dep on viscy-transforms | HIGH | LOW | P1 |
| Existing tests passing | HIGH | MEDIUM | P1 |
| Package README | MEDIUM | LOW | P1 |
| py.typed marker | MEDIUM | LOW | P1 |
| Lazy imports for optional deps | MEDIUM | MEDIUM | P2 |
| GPU transform protocol | MEDIUM | MEDIUM | P2 |
| Additional test coverage | MEDIUM | HIGH | P2 |
| Extract INDEX_COLUMNS coupling | LOW | LOW | P2 |
| Split combined.py | LOW | LOW | P3 |
| Public _read_norm_meta | LOW | LOW | P3 |
| Abstract cache interface | LOW | HIGH | P3 |

## Comparable Package API Analysis

### MONAI (`monai.data`)

**Pattern:** Flat top-level exports. `monai.data.__init__.py` exports 100+ symbols from 20+ submodules. No subpackage nesting for data types.

**Relevance:** viscy-data follows the same domain (medical/scientific imaging data loading). MONAI's flat API works because users import specific classes, not browse the namespace.

**Key insight:** MONAI exports both Dataset classes AND utility functions at the top level. viscy-data should export DataModules and Datasets (user-facing) but keep utilities private.

### viscy-transforms (sibling package)

**Pattern:** Flat top-level with underscore-prefixed private modules. 44 exports in `__all__`. All transforms accessible via `from viscy_transforms import X`.

**Relevance:** Direct precedent. viscy-data should follow identical patterns for consistency: private modules (`_hcs.py`), flat exports, `__all__` list, `py.typed`.

### Lightning `LightningDataModule`

**Pattern:** Users subclass and configure via `__init__` params. DataModules are registered via Lightning CLI's class resolution. No special import hierarchy needed -- just the class name.

**Relevance:** viscy-data's DataModules are already Lightning DataModules. The flat export pattern enables Lightning CLI to resolve `class_path: viscy_data.HCSDataModule` directly.

## Optional Dependency Strategy

### Recommended Groups

```toml
[project.optional-dependencies]
triplet = ["tensorstore>=0.1.45", "pandas>=2.0"]
livecell = ["pycocotools>=2.0", "tifffile>=2023.0"]
mmap = ["tensordict>=0.4"]
all = [
    "viscy-data[triplet]",
    "viscy-data[livecell]",
    "viscy-data[mmap]",
]
```

### Required Dependencies (always installed)

```toml
[project]
dependencies = [
    "viscy-transforms",
    "torch>=2.1",
    "lightning>=2.0",
    "numpy>=1.24",
    "iohub>=0.1",
    "monai>=1.3",
    "zarr>=2.16",
    "imageio>=2.31",
]
```

### Import Behavior Without Optional Deps

When a user installs `pip install viscy-data` (no extras):
- `from viscy_data import HCSDataModule` -- works
- `from viscy_data import CachedOmeZarrDataModule` -- works
- `from viscy_data import TripletDataModule` -- works (import succeeds)
- `TripletDataModule(...)` -- raises `ImportError: tensorstore is required for TripletDataModule. Install with: pip install viscy-data[triplet]` only when class is instantiated and tensorstore is actually needed

This is the lazy import pattern: top-level imports always work (no try/except at import time), but runtime use of optional dependencies produces clear error messages.

## Sources

- Direct source code analysis of all 13 modules on `main` branch (HIGH confidence)
- `viscy/data/README.md` architecture document on `modular-data` branch (HIGH confidence)
- MONAI `monai.data.__init__.py` in installed package at `.venv/lib/python3.12/site-packages/monai/data/__init__.py` (HIGH confidence)
- viscy-transforms `__init__.py` pattern at `packages/viscy-transforms/src/viscy_transforms/__init__.py` (HIGH confidence)
- Existing research in `.planning/research/ARCHITECTURE.md` (HIGH confidence)

---
*Feature research for: viscy-data subpackage extraction*
*Researched: 2026-02-13*
