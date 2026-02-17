# Project Research Summary

**Project:** viscy-data subpackage extraction
**Domain:** Scientific microscopy data loading (PyTorch Lightning DataModules for HCS OME-Zarr)
**Researched:** 2026-02-13
**Confidence:** HIGH

## Executive Summary

The viscy-data package extraction involves migrating 13 data loading modules from the VisCy monolith into a standalone workspace package. This is the second extraction milestone, building on the established patterns from viscy-transforms (Milestone 1). The extraction is architecturally straightforward — copy modules, update imports, declare dependencies — but has three significant complexities that distinguish it from viscy-transforms: (1) heavy I/O dependencies (iohub, zarr, tensorstore) with platform-specific constraints, (2) optional heavyweight dependencies requiring careful lazy loading, and (3) a one-way cross-package dependency on viscy-transforms for `BatchedCenterSpatialCropd`.

The recommended approach follows the proven viscy-transforms pattern: flat module layout with underscore-prefixed private modules, top-level exports in `__init__.py`, optional dependency groups (`[triplet]`, `[livecell]`, `[mmap]`, `[all]`), and workspace dependency declaration. The critical difference is extracting shared utilities from `hcs.py` into `_utils.py` FIRST — this breaks the current dual-role pattern where hcs.py serves as both a concrete DataModule and a utility library, preventing fragile cross-module coupling.

Key risks center on multiprocessing and cross-platform compatibility: the `Manager().dict()` shared cache pattern works on Linux (fork) but breaks on macOS/Windows (spawn), optional dependencies like pycocotools and tensorstore have platform-specific build requirements, and `ThreadDataLoader` with tensorstore creates test isolation challenges. These are mitigated by: (1) testing with `spawn` explicitly in CI, (2) using optional dependency groups with platform markers, (3) session-scoped test fixtures with explicit cleanup, and (4) a tiered CI matrix (base deps on 3x3, full extras on 1x1).

## Key Findings

### Recommended Stack

The core stack is **iohub + monai + lightning + torch + numpy** as required dependencies, with **tensorstore, tensordict, pycocotools, pandas, tifffile, torchvision** available through optional extras. This represents an incremental addition to the workspace — the build system (hatchling + uv-dynamic-versioning), Python version support (3.11-3.14), and CI patterns are already established and do not need re-research.

**Core technologies:**
- **iohub >=0.3a2**: OME-Zarr I/O layer (Plate, Position, ImageArray) used by 9 of 13 modules; pre-release version but stable in VisCy production use
- **monai >=1.5.2**: Transforms (Compose, MapTransform), data utilities (ThreadDataLoader, set_track_meta, collate_meta_tensor) used by 10 of 13 modules; aligns with viscy-transforms pin
- **lightning >=2.3**: LightningDataModule base class for all data modules; matches original VisCy pin
- **torch >=2.10**: Tensor operations, DataLoader, Dataset; aligns with viscy-transforms for consistency
- **tensorstore (optional)**: High-performance array I/O for contrastive learning triplet cache; C++ library with platform wheel limitations; only needed for `[triplet]` extra
- **tensordict (optional)**: MemoryMappedTensor for mmap caching strategy; part of PyTorch RL ecosystem; only needed for `[mmap]` extra
- **pycocotools (optional)**: COCO annotation parsing for LiveCell benchmark dataset; requires C compiler on some platforms; only needed for `[livecell]` extra

**Version pinning philosophy:** Floor pins (`>=X.Y`) for core dependencies to avoid over-constraining user environments. No upper bounds (trust semantic versioning). No version pins for optional dependencies — let the resolver pick compatible versions.

### Expected Features

All 13 modules must be extracted and importable with clean paths (`from viscy_data import HCSDataModule`). The extraction must preserve all existing functionality while establishing a cleaner architecture through utility refactoring. Optional dependency groups enable users to install only what they need for their specific pipeline (translation, FCMAE pretraining, contrastive learning, or benchmarking).

**Must have (table stakes):**
- All 13 modules extracted with clean import paths (no `viscy.data.` references)
- Flat top-level exports for all DataModules and Datasets
- Type exports (Sample, NormMeta, TripletSample, etc.) for downstream type annotations
- Optional dependency groups (`[triplet]`, `[livecell]`, `[mmap]`, `[all]`)
- Workspace dependency on viscy-transforms (for `BatchedCenterSpatialCropd` import in triplet.py)
- Shared utilities extracted from hcs.py into `_utils.py` (breaks dual-role anti-pattern)
- Existing tests passing under new import paths
- py.typed marker for type checking support

**Should have (competitive):**
- Lazy imports for heavy optional deps with clear error messages when missing
- Package README with pipeline mapping table (adapted from existing `viscy/data/README.md`)
- GPU transform mixin as protocol for duck typing without forced inheritance
- Type-safe batch structures exported as first-class types

**Defer (v2+):**
- Promoting internal utilities like `_read_norm_meta` to public API (wait for user demand)
- Abstract cache interface unifying Manager.dict, tensorstore, and MemoryMappedTensor patterns (complexity without immediate benefit)
- Config-driven DataModule registry for Lightning CLI integration (wait for viscy meta-package)

### Architecture Approach

The package uses a flat module layout following the viscy-transforms pattern. All modules use underscore-prefixed names (`_hcs.py`, `_utils.py`) to signal "import from package top-level, not from module." The internal dependency graph is a clean DAG with foundation modules (`_typing.py`, `_utils.py`) at the root, core modules (`hcs.py`, `gpu_aug.py`) in the middle, and specialized modules (triplet, livecell, mmap_cache) as leaves.

**Major components:**

1. **Foundation layer** (`_typing.py`, `_utils.py`) — Type definitions and shared utilities; no internal dependencies; imported by all other modules
2. **Core DataModules** (`hcs.py`, `gpu_aug.py`) — HCSDataModule for translation pipelines, GPUTransformDataModule (ABC) for FCMAE pretraining; depend only on foundation layer
3. **Specialized DataModules** (triplet, mmap_cache, livecell, cell_classification, cell_division_triplet, ctmc_v1, segmentation) — Pipeline-specific implementations extending core DataModules; may have optional dependencies
4. **Composition modules** (combined, concat) — Wrappers for multi-source training and dataset concatenation; depend on core modules
5. **Utilities** (select, distributed) — SelectWell mixin for well/FOV filtering, ShardedDistributedSampler for DDP; standalone with no internal dependencies

**Critical architecture decision:** Extract shared utilities (`_ensure_channel_list`, `_read_norm_meta`, `_collate_samples`) from `hcs.py` into `_utils.py` BEFORE extracting any other modules. The current code has 5 modules importing from `hcs.py` for utility functions, not for `HCSDataModule`, creating unnecessary coupling. This refactoring is the prerequisite that enables clean module boundaries.

### Critical Pitfalls

Based on analysis of the 13-module architecture, v1.0 extraction experience, and domain expertise with PyTorch/Lightning data loading patterns:

1. **Lazy Import Guard Ordering Breaks at Runtime** — Optional dependencies (tensorstore, tensordict, pycocotools) must use lazy imports that defer errors until actual usage, not module import time. If guards are missing or placed incorrectly, errors surface deep in training loops (potentially in DataLoader worker subprocesses), making them hard to trace. Prevention: centralized lazy import pattern in `_imports.py`, call import helpers at method entry points, test with base-deps-only CI job to catch unguarded imports.

2. **Manager().dict() Shared Cache Not Picklable Across spawn Contexts** — `CachedOmeZarrDataset` uses `multiprocessing.Manager().dict()` which works with `fork` (Linux) but fails with `spawn` (macOS/Windows default). Proxy objects must pickle, and depending on creation timing relative to DataLoader fork/spawn, you get either pickle errors or silently separate caches per worker. Prevention: create Manager in `setup()` (Lightning hook after multiprocessing context is configured), test with `mp_start_method="spawn"` explicitly in CI, consider replacing with file-based cache.

3. **Base Class Extraction Creates Hidden Import Cycles** — `hcs.py` is both a concrete DataModule and the base class for TripletDataModule/CellDivisionTripletDataModule. Extracting utilities into `_utils.py` without careful dependency analysis creates circular imports. Prevention: map complete import graph BEFORE moving code, follow strict layering (_typing -> _utils -> hcs -> specialized), test import order explicitly with isolated `python -c "from viscy_data import X"` calls.

4. **Optional Extras Create a 2^N CI Matrix Explosion** — With 4 optional groups across 3 Python versions and 3 OS targets, testing all combinations creates 45+ jobs. Some combinations are invalid (pycocotools doesn't build on Windows). Prevention: tiered CI strategy (base deps 3x3, full extras 1x1, per-extra smoke tests), use pytest markers to skip when deps missing, exclude known-broken combinations in matrix.

5. **pycocotools Build Failure Blocks Windows CI** — pycocotools requires C compiler; Windows has no default C compiler. When wheels are missing, pip falls back to source build which fails. Prevention: exclude livecell extra from Windows in CI matrix, mark `[livecell]` as Linux/macOS only in docs (LiveCell is HPC dataset anyway), or use `pycocotools-windows` fork.

## Implications for Roadmap

Based on the dependency analysis and architecture patterns, the extraction follows a layered build order. The critical path is: scaffolding → foundation modules → core modules → specialized modules → composition modules → finalize. Tests can begin after core modules are migrated.

### Phase 1: Package Scaffolding & Foundation
**Rationale:** Establish package structure and dependency declarations before migrating any code. Extract shared utilities first to break the hcs.py dual-role anti-pattern.

**Delivers:**
- `packages/viscy-data/pyproject.toml` with all dependencies (required and optional groups)
- Empty `__init__.py` placeholder
- `py.typed` marker
- `_typing.py` with all data types (verbatim copy from `viscy/data/typing.py` + DictTransform alias)
- `_utils.py` with extracted helpers (`_ensure_channel_list`, `_read_norm_meta`, `_collate_samples`, `_search_int_in_str`)
- Root `pyproject.toml` updated with viscy-data workspace dependency

**Addresses features:**
- Package structure (table stakes)
- Type exports (table stakes)
- Shared utilities extraction (table stakes, prevents coupling)
- Optional dependency groups (table stakes)

**Avoids pitfalls:**
- P3 (import cycles) — extracting _utils.py first prevents coupling
- P1 (lazy imports) — pyproject.toml defines extras structure
- P5 (pycocotools Windows) — optional deps with platform markers

### Phase 2: Core Data Modules
**Rationale:** Migrate the two base DataModule classes that other modules depend on. These have no optional dependencies and establish the inheritance patterns.

**Delivers:**
- `hcs.py` (HCSDataModule, SlidingWindowDataset, MaskTestDataset) — remove extracted functions, update imports
- `gpu_aug.py` (GPUTransformDataModule ABC, CachedOmeZarrDataset, CachedOmeZarrDataModule) — update imports to viscy_data._utils
- `select.py` (SelectWell mixin) — verbatim copy, update imports
- `distributed.py` (ShardedDistributedSampler) — verbatim copy, no internal imports

**Uses stack:** iohub, monai, lightning, torch (all required deps)

**Implements architecture:** Foundation → Core layer in dependency DAG

**Avoids pitfalls:**
- P2 (Manager().dict() spawn) — CachedOmeZarrDataModule addressed in this phase
- P11 (MRO fragility) — copy-first, no refactoring during extraction

### Phase 3: Specialized Data Modules
**Rationale:** Migrate pipeline-specific DataModules that extend core classes. These introduce optional dependencies and lazy loading patterns.

**Delivers:**
- `triplet.py` (TripletDataset, TripletDataModule) — add lazy imports for tensorstore/pandas
- `cell_classification.py` (ClassificationDataset, ClassificationDataModule) — lazy pandas import
- `cell_division_triplet.py` (CellDivisionTripletDataset, CellDivisionTripletDataModule)
- `mmap_cache.py` (MmappedDataset, MmappedDataModule) — lazy tensordict import
- `livecell.py` (LiveCellDataset, LiveCellTestDataset, LiveCellDataModule) — lazy pycocotools/tifffile imports
- `ctmc_v1.py` (CTMCv1DataModule)
- `segmentation.py` (SegmentationDataset, SegmentationDataModule)

**Uses stack:** tensorstore, tensordict, pycocotools (all optional)

**Implements architecture:** Specialized modules layer in DAG

**Avoids pitfalls:**
- P1 (lazy imports) — centralized pattern for all optional deps
- P7 (ThreadDataLoader leaks) — addressed in triplet.py migration
- P12 (MemoryMappedTensor cleanup) — addressed in mmap_cache.py

### Phase 4: Composition Modules & Finalize
**Rationale:** Migrate the high-level composition wrappers that depend on core modules, then finalize the package with complete exports and README.

**Delivers:**
- `combined.py` (CombinedDataModule, CombineMode) — update imports
- `concat.py` (ConcatDataModule, BatchedConcatDataModule, CachedConcatDataModule, BatchedConcatDataset) — split from combined.py
- Complete `__init__.py` with all public exports (15 classes + 8 types)
- Package README adapted from `viscy/data/README.md`

**Implements architecture:** Composition layer, public API surface

**Avoids pitfalls:**
- P8 (__init__.py eager imports) — only re-export modules with required deps

### Phase 5: Test Migration & CI
**Rationale:** Migrate existing tests after code modules are stable, establish test fixtures with proper cleanup, configure tiered CI matrix.

**Delivers:**
- `tests/conftest.py` with session-scoped OME-Zarr fixtures (migrate from main branch tests/conftest.py)
- `test_hcs.py` (update imports, verify HCSDataModule + SlidingWindowDataset)
- `test_triplet.py` (update imports, add ThreadDataLoader cleanup)
- `test_select.py` (update imports, verify SelectWell mixin)
- `test_typing.py` (new: smoke tests for type definitions)
- CI workflow with tiered matrix: base deps (3x3), full extras (1x1), per-extra smoke tests

**Verifies:** All table-stakes features work, pitfall mitigations are effective

**Avoids pitfalls:**
- P4 (CI matrix explosion) — tiered strategy keeps job count manageable
- P9 (expensive fixtures) — session-scoped, read-only fixtures
- P7 (ThreadDataLoader leaks) — explicit cleanup in fixtures
- P12 (mmap cleanup) — explicit teardown for MemoryMappedTensor

### Phase 6: Workspace Integration & Validation
**Rationale:** Verify the extracted package integrates correctly with the workspace and existing configs still reference correct import paths.

**Delivers:**
- Root `pyproject.toml` verified with viscy-data in dependencies and [tool.uv.sources]
- All YAML/JSON configs in applications/ checked for stale `viscy.data.` references
- Integration test: `uv sync --package viscy-data` + `uv run --package viscy-data pytest`
- Integration test: viscy meta-package can import from both viscy-transforms and viscy-data
- Documentation: migration guide with old → new import paths

**Verifies:** Clean workspace integration, no config breakage

**Avoids pitfalls:**
- P10 (config class_path breakage) — grep + update all in-repo configs
- P6 (iohub API coupling) — integration test verifies iohub types

### Phase Ordering Rationale

- **Sequential dependency chain:** Foundation → Core → Specialized → Composition follows the import DAG. Each phase depends on the previous phase being complete.
- **Extract _utils.py FIRST:** This is the critical prerequisite. The current hcs.py dual-role creates coupling that blocks clean extraction of downstream modules.
- **Lazy imports in Specialized phase:** Optional dependencies are leaf nodes in the dependency graph. They're isolated in Phase 3 so Phase 2 can be tested without tensorstore/tensordict/pycocotools.
- **Tests after code:** Tests require all modules to exist. Session-scoped fixtures need the full package structure. Testing comes after Phases 1-4 are stable.
- **CI after tests:** The tiered CI matrix needs to know which tests can run with base deps vs. which need extras. CI design happens after test structure is known.

### Research Flags

**Phases with standard patterns (skip research-phase):**
- **Phase 1-4:** Code extraction follows the proven viscy-transforms pattern. All architectural decisions are documented in existing research.
- **Phase 5:** Test fixture patterns are well-documented in existing tests/conftest.py. CI workflow structure mirrors existing test.yml.
- **Phase 6:** Workspace integration is standard uv workspace mechanics.

**No phases need additional research.** All architectural decisions, dependency choices, and pitfall mitigations are informed by the comprehensive upfront research (STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md) and the established viscy-transforms precedent.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Core dependencies verified via lockfile and source code analysis. Optional dependencies verified via original pyproject.toml. Python version support follows workspace standard. |
| Features | HIGH | All 13 modules analyzed with direct source code review. Import paths, class hierarchies, and dependency graph documented in existing `viscy/data/README.md`. Feature requirements are table-stakes extraction, not new development. |
| Architecture | HIGH | Package layout follows proven viscy-transforms pattern. Internal dependency graph is well-documented (README.md). Shared utilities extraction is the only new architectural element, and it's a straightforward refactoring. |
| Pitfalls | MEDIUM-HIGH | Pitfalls derived from: (1) codebase analysis, (2) v1.0 extraction experience, (3) domain expertise with PyTorch multiprocessing, Lightning DataModules, tensorstore, and C-extension packages. WebSearch unavailable during research, so some cross-platform build claims (tensorstore arm64, pycocotools Windows wheels) could not be verified against current package indices. |

**Overall confidence:** HIGH

The extraction is architecturally straightforward and follows established patterns. The three complexities (heavy I/O deps, optional deps, cross-package dependency) are well-understood and have documented mitigation strategies. The main uncertainties are platform-specific (tensorstore/pycocotools wheel availability), which are addressed by making those dependencies optional and testing in a tiered CI matrix.

### Gaps to Address

**Platform-specific dependency availability:**
- **tensorstore Python 3.13 support:** Wheels may not exist yet. Mitigated by making `[triplet]` optional and testing extras only on Python 3.11-3.12 in CI.
- **pycocotools Windows build:** No pre-built wheels for some Python versions. Mitigated by excluding Windows from livecell tests in CI or documenting as Linux/macOS only.
- **tensordict Python 3.13 support:** Same mitigation as tensorstore (optional, narrower CI matrix).

**iohub version stability:**
- iohub is pinned at pre-release (0.3a2) because stable release timeline is unknown. The API is stable in VisCy production use, but future releases may have breaking changes. Mitigated by floor pin (`>=0.3a2`), no upper bound, and integration test that verifies expected types/attributes.

**Manager().dict() cross-platform:**
- Shared cache pattern works on Linux (fork) but needs testing on macOS/Windows (spawn). Mitigated by explicit spawn testing in CI. If failures persist, fallback is to document CachedOmeZarrDataModule as Linux-only or refactor to file-based cache (tensordict MemoryMappedTensor pattern already exists in mmap_cache.py).

None of these gaps block the extraction. All have documented mitigation strategies that can be applied during Phase 2 (Manager().dict()), Phase 3 (optional deps), and Phase 5 (CI matrix).

## Sources

### Primary (HIGH confidence)
- `viscy/data/README.md` (modular-data branch) — Comprehensive module inventory, dependency graph, class hierarchy, GPU transform patterns
- All 13 source files in `viscy/data/` on main branch — Direct import statement analysis, function-level dependency tracing
- `packages/viscy-transforms/` — Established extraction pattern (pyproject.toml, __init__.py, _typing.py, CI structure)
- `main:pyproject.toml` — Original dependency pins: iohub>=0.3a2, monai>=1.4, lightning>=2.3
- `uv.lock` — Resolved versions: monai 1.5.2, torch 2.10.0, numpy 2.4.2, tifffile 2026.1.28
- `.planning/PROJECT.md` — Project constraints (no viscy-transforms dependency, clean break imports, optional extras)
- `.planning/ROADMAP.md` — Milestone 1 completion status, workspace patterns

### Secondary (HIGH confidence)
- `main:tests/conftest.py` — Test fixture patterns for HCS OME-Zarr stores, tracks datasets
- `main:tests/data/test_hcs.py`, `test_select.py`, `test_triplet.py` — Existing test coverage and patterns
- MONAI `monai.data.__init__.py` — Flat API export pattern (100+ symbols from 20+ modules)
- viscy-transforms `__init__.py` — Sibling package pattern (44 exports, underscore-prefixed private modules)

### Tertiary (MEDIUM confidence, needs validation)
- tensorstore Python 3.13 wheel availability — Claimed LIMITED based on historical lag, but not verified against current PyPI
- tensordict Python 3.13 wheel availability — Claimed LIMITED, same reason
- pycocotools Windows wheel coverage — Claimed REQUIRES C COMPILER for missing wheels, based on common CI failure pattern
- iohub latest stable release status — Pre-release 0.3a2 is used; stable release timeline unknown

---
*Research completed: 2026-02-13*
*Ready for roadmap: yes*
