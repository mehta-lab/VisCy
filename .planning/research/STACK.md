# Technology Stack: viscy-data Package Extraction

**Project:** viscy-data -- data loading, Lightning DataModules, and dataset classes for VisCy
**Researched:** 2026-02-13
**Overall Confidence:** HIGH (core deps verified via lockfile and source; optional deps verified via original pyproject.toml on main)

---

## Executive Summary

This document covers the **incremental stack additions** needed to extract viscy-data as the second independent subpackage in the VisCy monorepo. The build system (hatchling + uv-dynamic-versioning), workspace structure, and CI patterns are already established by viscy-transforms (Milestone 1) and are NOT re-covered here.

viscy-data introduces three new concerns beyond viscy-transforms:

1. **Heavy I/O dependencies** -- iohub (OME-Zarr), zarr, and tifffile form the core I/O layer that nearly every module requires.
2. **Optional heavyweight deps** -- tensorstore, tensordict, and pycocotools are only needed by specific modules (triplet, mmap_cache, livecell) and require optional dependency groups.
3. **Inter-package dependency** -- viscy-data depends on viscy-transforms (for `BatchedCenterSpatialCropd` in triplet.py), making this the first workspace package with a cross-package dependency.

The core stack is: **iohub + monai + lightning + torch + numpy** as required dependencies, with **tensorstore, tensordict, pycocotools, pandas, tifffile, torchvision** available through optional extras `[triplet]`, `[livecell]`, `[mmap]`, `[all]`.

---

## Recommended Stack -- New Dependencies for viscy-data

### Core Required Dependencies

These are imported by the majority of modules and must be required (not optional).

| Technology | Version | Purpose | Why This Version | Confidence |
|------------|---------|---------|------------------|------------|
| iohub | >=0.3a2 | OME-Zarr I/O (Plate, Position, ImageArray) | Matches original VisCy pinning; provides `open_ome_zarr`, `ngff` module used by 9 of 13 data modules | HIGH |
| monai | >=1.5.2 | Transforms (Compose, MapTransform), data utilities (ThreadDataLoader, set_track_meta, collate_meta_tensor) | 1.5.2 is current (Jan 2026 release); used by 10 of 13 modules; aligns with viscy-transforms pin | HIGH |
| lightning | >=2.3 | LightningDataModule base class | Every DataModule inherits from this; matches original VisCy pin | HIGH |
| torch | >=2.10 | Tensor operations, DataLoader, Dataset, DDP | Aligns with viscy-transforms pin (>=2.10); needed by all modules | HIGH |
| numpy | >=2.4.1 | Array operations | Aligns with viscy-transforms pin; iohub returns numpy arrays | HIGH |
| zarr | * | Zarr store access | Imported directly in hcs.py for decompression caching; version managed transitively via iohub | HIGH |
| imageio | * | Image reading (imread in hcs.py) | Used in hcs.py; lightweight; version managed transitively | MEDIUM |
| viscy-transforms | (workspace) | BatchedCenterSpatialCropd | triplet.py imports this; one-way dependency (data -> transforms), no circular risk | HIGH |

**Rationale for iohub >=0.3a2:** The original VisCy on main pins `iohub[tensorstore]>=0.3a2`. This is a pre-release version (alpha), which means the API may not be fully stable. However, VisCy has been using this version in production. The `[tensorstore]` extra on iohub itself is separate from our optional `[triplet]` extra -- iohub uses tensorstore for its own OME-Zarr v0.5 sharded access. We should pin `iohub>=0.3a2` as the base dependency and let users who need tensorstore-backed I/O install `viscy-data[triplet]` which includes both.

**Rationale for monai >=1.5.2 (not >=1.4):** The original VisCy pinned >=1.4, but viscy-transforms already uses >=1.5.2. Since viscy-data depends on viscy-transforms, the effective floor is 1.5.2 anyway. Be explicit to avoid confusing lower bounds.

### Optional Dependencies (Extras)

These are imported by specific modules only and should be lazy-loaded with clear error messages.

| Extra Group | Dependencies | Used By | Why Optional |
|-------------|-------------|---------|--------------|
| `[triplet]` | tensorstore, pandas | triplet.py, cell_classification.py | tensorstore is a large C++ library (~100MB+); pandas adds weight; only needed for contrastive learning pipelines |
| `[livecell]` | pycocotools, tifffile, torchvision | livecell.py | pycocotools requires C compiler on some platforms; tifffile + torchvision only needed for LiveCell benchmark |
| `[mmap]` | tensordict | mmap_cache.py | Part of torchrl ecosystem; only needed for memory-mapped caching strategy |
| `[all]` | (union of above) | All modules | Convenience extra for users who want everything |

#### Optional Dependency Versions

| Technology | Version | Purpose | Platform Notes | Confidence |
|------------|---------|---------|----------------|------------|
| tensorstore | * | High-performance array I/O for triplet cache pool | C++ library; pre-built wheels for Linux/macOS x86_64/arm64, Windows x86_64; Python 3.11-3.12 confirmed, 3.13 support needs verification | MEDIUM |
| tensordict | * | MemoryMappedTensor for mmap_cache.py | Part of PyTorch RL ecosystem; depends on torch; Python 3.11-3.12 confirmed | MEDIUM |
| pycocotools | * | COCO annotation parsing for livecell.py | Requires C compiler for source build; pre-built wheels available on most platforms | HIGH |
| pandas | * | DataFrame operations for tracks in triplet.py and cell_classification.py | Widely available; no platform issues | HIGH |
| tifffile | * | TIFF file reading for livecell.py | Pure Python; no platform issues; version 2026.1.28 in lockfile | HIGH |
| torchvision | * | box_convert utility in livecell.py | Single function import; already commonly installed with torch | HIGH |

### Testing Dependencies

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| pytest | >=9.0.2 | Test framework | Aligns with workspace standard | HIGH |
| pytest-cov | >=7 | Coverage reporting | Aligns with workspace standard | HIGH |
| iohub | >=0.3a2 | Test fixture creation (OME-Zarr stores) | conftest.py uses `open_ome_zarr` to create test stores; required for all data tests | HIGH |
| pandas | * | Test fixtures for tracks datasets | conftest.py creates CSV track files with pandas DataFrames | HIGH |

---

## Python Version Compatibility Matrix

| Dependency | Python 3.11 | Python 3.12 | Python 3.13 | Python 3.14 | Notes |
|------------|-------------|-------------|-------------|-------------|-------|
| torch >=2.10 | Yes | Yes | Yes | Yes | Wheels for all; 3.14 is new |
| monai >=1.5.2 | Yes | Yes | Yes | Yes | Pure Python wheel (py3-none-any) |
| lightning >=2.3 | Yes | Yes | Yes | Likely | Pure Python |
| iohub >=0.3a2 | Yes | Yes | Likely | Unknown | Pre-release; limited metadata available |
| tensorstore | Yes | Yes | LOW confidence | Unknown | C++ binary; historically slow to add new Python versions |
| tensordict | Yes | Yes | LOW confidence | Unknown | C extension; tied to torch version cycle |
| pycocotools | Yes | Yes | Likely | Unknown | C extension but well-maintained |
| pandas | Yes | Yes | Yes | Yes | Broad support |
| tifffile | Yes | Yes | Yes | Yes | Pure Python |
| torchvision | Yes | Yes | Yes | Likely | Follows torch support |

**Key risk:** tensorstore and tensordict have historically lagged in Python version support. This is mitigated by making them optional -- users on Python 3.13+ can still use the core package without `[triplet]` or `[mmap]` extras.

---

## pyproject.toml Specification

```toml
[build-system]
build-backend = "hatchling.build"
requires = [ "hatchling", "uv-dynamic-versioning" ]

[project]
name = "viscy-data"
description = "Data loading and Lightning DataModules for virtual staining microscopy"
readme = "README.md"
keywords = [
  "data loading",
  "deep learning",
  "lightning",
  "microscopy",
  "ome-zarr",
  "virtual staining",
]
license = "BSD-3-Clause"
authors = [ { name = "Biohub", email = "compmicro@czbiohub.org" } ]
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
dynamic = [ "version" ]
dependencies = [
  "iohub>=0.3a2",
  "imageio",
  "lightning>=2.3",
  "monai>=1.5.2",
  "numpy>=2.4.1",
  "torch>=2.10",
  "viscy-transforms",
  "zarr",
]

[project.optional-dependencies]
triplet = [
  "pandas",
  "tensorstore",
]
livecell = [
  "pycocotools",
  "tifffile",
  "torchvision",
]
mmap = [
  "tensordict",
]
all = [
  "viscy-data[triplet,livecell,mmap]",
]

[project.urls]
Homepage = "https://github.com/mehta-lab/VisCy"
Issues = "https://github.com/mehta-lab/VisCy/issues"
Repository = "https://github.com/mehta-lab/VisCy"

[dependency-groups]
dev = [ { include-group = "jupyter" }, { include-group = "test" } ]
test = [
  "pandas",
  "pytest>=9.0.2",
  "pytest-cov>=7",
]
jupyter = [
  "ipykernel>=7.1",
  "jupyterlab>=4.5.3",
]

[tool.hatch.version]
source = "uv-dynamic-versioning"

[tool.hatch.build.targets.wheel]
packages = [ "src/viscy_data" ]

[tool.uv-dynamic-versioning]
vcs = "git"
style = "pep440"
pattern-prefix = "viscy-data-"
fallback-version = "0.0.0"
```

### Root pyproject.toml Additions

```toml
# Add to root [project] dependencies
dependencies = [ "viscy-transforms", "viscy-data" ]

# Add to root [tool.uv.sources]
[tool.uv.sources]
viscy-transforms = { workspace = true }
viscy-data = { workspace = true }

# Add to root [tool.ruff]
# src already covers packages/*/src via glob
```

---

## Lazy Import Pattern for Optional Dependencies

Modules with optional dependencies must use lazy imports with actionable error messages.

```python
# triplet.py -- example pattern
def _import_tensorstore():
    try:
        import tensorstore as ts
        return ts
    except ImportError:
        raise ImportError(
            "tensorstore is required for TripletDataset. "
            "Install with: pip install 'viscy-data[triplet]'"
        ) from None

# Usage: move import from module level into function/class that needs it
# tensorstore is currently imported at module level in triplet.py --
# this must be changed to lazy import at point of use.
```

**Modules requiring lazy import conversion:**

| Module | Current Import | Lazy Import Target |
|--------|---------------|-------------------|
| triplet.py | `import tensorstore as ts` (top-level) | Defer to `TripletDataset.__init__()` or cache pool init |
| triplet.py | `import pandas as pd` (top-level) | Defer to `TripletDataModule.setup()` |
| mmap_cache.py | `from tensordict.memmap import MemoryMappedTensor` (top-level) | Defer to `MmappedDataset.__init__()` |
| livecell.py | `from pycocotools.coco import COCO` (top-level) | Defer to `LiveCellTestDataset.__init__()` |
| livecell.py | `from tifffile import imread` (top-level) | Defer to `LiveCellDataset.__init__()` |
| livecell.py | `from torchvision.ops import box_convert` (top-level) | Defer to `LiveCellTestDataset.__init__()` |
| cell_classification.py | `import pandas as pd` (top-level) | Defer to `ClassificationDataset.__init__()` |

---

## Testing Infrastructure

### Shared Test Fixtures

The existing `tests/conftest.py` on main provides session-scoped HCS OME-Zarr fixtures that viscy-data tests will need. These must be migrated to `packages/viscy-data/tests/conftest.py`.

**Fixtures to migrate from main's conftest.py:**

| Fixture | Scope | Creates | Used By |
|---------|-------|---------|---------|
| `preprocessed_hcs_dataset` | session | 2x4x4 HCS store with norm metadata, 12x256x256, float32, multiscale | test_hcs.py, test_select.py, test_triplet.py |
| `small_hcs_dataset` | function | 2x4x4 HCS store, 12x64x64, uint16, parametrized sharded/non-sharded | test_hcs.py |
| `small_hcs_labels` | function | 2-channel labels store, 12x64x64, uint16 | test_hcs.py |
| `labels_hcs_dataset` | function | 2-channel store, 2x16x16, uint16 | test_hcs.py |
| `tracks_hcs_dataset` | function | HCS store + tracks.csv per FOV | test_triplet.py |
| `tracks_with_gaps_dataset` | function | HCS store + tracks with temporal gaps | test_triplet.py |

**Key pattern:** All fixtures use `iohub.open_ome_zarr` with `layout="hcs"` to create synthetic test stores. The `_build_hcs` helper function encapsulates the store creation logic. This helper should be part of the test conftest.

### Test Dependency Requirements

viscy-data tests require at minimum:
- `iohub>=0.3a2` -- for creating and reading test OME-Zarr stores
- `pandas` -- for creating tracks fixtures (CSV files)
- `pytest>=9.0.2` + `pytest-cov>=7` -- test runner

**pandas in test group, not just [triplet]:** Even if pandas becomes optional for runtime, the test conftest needs it to create track fixtures. It belongs in the `[dependency-groups] test` group.

### Test Categories by Dependency Tier

| Test Tier | Deps Required | Modules Covered | CI Strategy |
|-----------|---------------|----------------|-------------|
| **Core** (always run) | iohub, monai, lightning, pandas | hcs.py, select.py, distributed.py, gpu_aug.py, combined.py, typing.py, segmentation.py, cell_classification.py | Default test matrix (3 OS x 3 Python) |
| **Triplet** (conditional) | + tensorstore | triplet.py, cell_division_triplet.py | Skip with `pytest.importorskip("tensorstore")` if not installed |
| **Mmap** (conditional) | + tensordict | mmap_cache.py | Skip with `pytest.importorskip("tensordict")` |
| **LiveCell** (conditional) | + pycocotools, tifffile, torchvision | livecell.py | Skip with `pytest.importorskip("pycocotools")` |

**CI strategy:** Run core tests in the standard matrix. Run optional-dep tests in a separate CI job that installs `viscy-data[all]`, or use `pytest.importorskip()` to gracefully skip.

### Recommended CI Workflow Addition

```yaml
# In .github/workflows/test.yml -- add viscy-data job
  test-data:
    name: Test viscy-data (Python ${{ matrix.python-version }}, ${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: true
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v7
        with:
          python-version: ${{ matrix.python-version }}
          enable-cache: true
          cache-suffix: data-${{ matrix.os }}-${{ matrix.python-version }}
      - name: Install core deps
        run: uv sync --frozen --dev
        working-directory: packages/viscy-data
      - name: Run core tests
        run: uv run --frozen pytest --cov=viscy_data --cov-report=term-missing
        working-directory: packages/viscy-data

  test-data-extras:
    name: Test viscy-data extras (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]  # Narrower matrix for optional deps
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v7
        with:
          python-version: ${{ matrix.python-version }}
          enable-cache: true
      - name: Install all extras
        run: uv sync --frozen --all-extras --dev
        working-directory: packages/viscy-data
      - name: Run all tests
        run: uv run --frozen pytest --cov=viscy_data --cov-report=term-missing
        working-directory: packages/viscy-data
```

---

## Alternatives Considered

### iohub Dependency Strategy

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| iohub extras | `iohub>=0.3a2` (base) | `iohub[tensorstore]>=0.3a2` | iohub's own tensorstore extra is for its v0.5 sharded access; viscy-data's tensorstore usage in triplet.py is separate; don't force iohub's tensorstore on all users |
| iohub version | `>=0.3a2` (pre-release) | Wait for stable release | Stable release timeline unknown; VisCy has used 0.3a2 in production |

### Optional Dependency Grouping Strategy

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| Extras structure | Per-pipeline (`[triplet]`, `[livecell]`, `[mmap]`) | Per-library (`[tensorstore]`, `[pandas]`) | Pipeline-oriented groups are more user-friendly; users know which pipeline they're running |
| Extras structure | Per-pipeline | Single `[full]` extra | Loses granularity; forces heavy deps on LiveCell users who don't need tensorstore |
| pandas placement | Optional in `[triplet]`, required in test group | Always required | pandas is ~30MB; only 2 of 13 modules need it at runtime; keep install lean |

### viscy-transforms Dependency

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| Cross-package dep | Depend on viscy-transforms | Copy BatchedCenterSpatialCropd into viscy-data | Code duplication; divergence risk; the dependency is clean (one-way) |
| Cross-package dep | Depend on viscy-transforms | Move BatchedCenterSpatialCropd to viscy-data | It's a transform, not a data class; belongs in viscy-transforms |

### DictTransform Type Sharing

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| Shared type | Copy `DictTransform` alias into viscy-data `_typing.py` | Import from viscy-transforms | Adds coupling for a single type alias (`Callable`); copy is one line |
| Shared type | Copy locally | Create viscy-types micro-package | Over-engineering for a type alias |

---

## What NOT to Add

| Technology | Why Not |
|------------|---------|
| **dask** | Not used anywhere in viscy data modules; OME-Zarr access goes through iohub, not dask arrays |
| **xarray** | Was pinned in original VisCy for iohub compatibility (`<=2025.9`), but not directly imported by any data module; let iohub manage transitively |
| **anndata** | Used in other VisCy modules (preprocessing), not in data modules |
| **kornia** | Already in viscy-transforms; viscy-data does not import kornia directly |
| **scikit-image** | Not imported by any data module |
| **hypothesis** | Property-based testing is less applicable to data module tests (I/O heavy, fixture-dependent); standard pytest is sufficient |
| **pytest-xdist** | Data tests are I/O-bound and use shared session fixtures; parallel execution risks fixture conflicts with temp zarr stores |

---

## Integration Points

### viscy-transforms -> viscy-data

```
viscy_data.triplet imports:
    viscy_transforms.BatchedCenterSpatialCropd
```

This is the sole cross-package import. In the uv workspace, this is handled by:
1. Adding `"viscy-transforms"` to viscy-data's `dependencies`
2. Adding `viscy-data = { workspace = true }` to root `[tool.uv.sources]`

### viscy-data -> downstream (viscy-models, applications)

viscy-data will be consumed by:
- `viscy-models` (future) -- engines reference DataModules for GPU transforms
- `applications/` -- training configs reference DataModule classes

Import path change: `from viscy.data.hcs import HCSDataModule` becomes `from viscy_data.hcs import HCSDataModule`.

### Root Meta-package

```toml
# Root pyproject.toml
dependencies = [
  "viscy-transforms",
  "viscy-data",
]

# Optional: expose all extras through meta-package
[project.optional-dependencies]
data-triplet = ["viscy-data[triplet]"]
data-livecell = ["viscy-data[livecell]"]
data-mmap = ["viscy-data[mmap]"]
data-all = ["viscy-data[all]"]
```

---

## Version Pinning Philosophy

| Category | Strategy | Rationale |
|----------|----------|-----------|
| Core (torch, monai, numpy, lightning) | Floor pin (`>=X.Y`) | These are aligned with viscy-transforms; users may need newer versions for other packages |
| iohub | Floor pin (`>=0.3a2`) | Pre-release but stable in practice; no upper bound to allow future stable releases |
| Optional (tensorstore, tensordict, etc.) | No version pin | Let solver pick compatible version; these change independently and version conflicts are unlikely |
| Test deps (pytest, pytest-cov) | Floor pin (`>=X.Y`) | Match workspace standard |

---

## Gaps and Open Questions

| Gap | Impact | Mitigation |
|-----|--------|------------|
| iohub is pre-release (0.3a2) | API instability risk | Pin >=0.3a2; iohub is maintained by the same lab (CZ Biohub); monitor for stable release |
| tensorstore Python 3.13 support | May not have wheels | Make optional; CI extras tests only on 3.11-3.12 |
| tensordict Python 3.13 support | May not have wheels | Make optional; same mitigation as tensorstore |
| No data tests exist for livecell, mmap_cache, gpu_aug, combined | Test coverage gaps | Write new tests during extraction; existing hcs/select/triplet tests are a good foundation |
| iohub's zarr dependency version | Potential zarr v2 vs v3 conflicts | iohub manages zarr transitively; don't pin zarr version explicitly |
| xarray version pin | Original VisCy had `xarray<=2025.9` for iohub compat | Don't add xarray to viscy-data; it's iohub's transitive dep to manage |

---

## Sources

### Codebase (HIGH confidence)
- `viscy/data/README.md` (modular-data branch) -- Module inventory, dependency per module, class hierarchy
- `main:viscy/data/*.py` -- Actual import statements for all 13 modules (verified via git show)
- `main:tests/conftest.py` -- Test fixture patterns for HCS OME-Zarr stores
- `main:tests/data/test_hcs.py`, `test_select.py`, `test_triplet.py` -- Existing data test coverage
- `main:pyproject.toml` -- Original dependency pins: `iohub[tensorstore]>=0.3a2`, `monai>=1.4`, `lightning>=2.3`
- `packages/viscy-transforms/pyproject.toml` -- Precedent for package structure: `monai>=1.5.2`, `torch>=2.10`, `numpy>=2.4.1`

### Lock File (HIGH confidence)
- `uv.lock` -- Resolved versions: monai 1.5.2 (Jan 2026), torch 2.10.0, numpy 2.4.2, tifffile 2026.1.28, kornia 0.8.2

### Existing Research (HIGH confidence)
- `.planning/research/STACK.md` (v1) -- Workspace tooling decisions (hatchling, uv-dynamic-versioning, CI patterns)
- `.planning/ROADMAP.md` -- Phase structure and completion status for Milestone 1

### Unverified (LOW confidence -- web search and fetch unavailable)
- tensorstore Python 3.13 wheel availability
- tensordict Python 3.13 wheel availability
- iohub latest stable release status
- pycocotools platform wheel coverage for arm64 Linux
