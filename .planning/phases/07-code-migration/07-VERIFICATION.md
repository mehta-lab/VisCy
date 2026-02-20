---
phase: 07-code-migration
verified: 2026-02-13T18:30:00Z
status: gaps_found
score: 3/6
re_verification: false
gaps:
  - truth: "import viscy_data succeeds without any optional extras installed"
    status: failed
    reason: "__init__.py eagerly imports all modules, causing transitive dependency failures on iohub/pandas/tensorstore even though modules use try/except guards"
    artifacts:
      - path: "packages/viscy-data/src/viscy_data/__init__.py"
        issue: "Eager imports (lines 43-91) execute all module code at import time, triggering transitive dependency imports"
      - path: "packages/viscy-data/src/viscy_data/cell_classification.py"
        issue: "Imports iohub.ngff (line 17) which transitively requires pandas/xarray/dask"
      - path: "packages/viscy-data/src/viscy_data/hcs.py"
        issue: "Imports iohub.ngff (line 13) which transitively requires pandas/xarray/dask"
      - path: "packages/viscy-data/src/viscy_data/triplet.py"
        issue: "Imports iohub.ngff (line 25) which transitively requires pandas/xarray/dask"
    missing:
      - "Convert __init__.py to use lazy imports (TYPE_CHECKING or __getattr__ pattern) OR"
      - "Move iohub imports inside methods/functions so module-level import succeeds OR"
      - "Add lazy import guards for iohub (try/except at module level with None sentinel)"
  - truth: "Importing a module that requires an uninstalled optional extra produces a clear error message naming the missing package and the install command"
    status: failed
    reason: "Import fails at module import time (not class instantiation time), so custom error messages in __init__ methods are never reached"
    artifacts:
      - path: "packages/viscy-data/src/viscy_data/triplet.py"
        issue: "ImportError guards in TripletDataset.__init__ (lines 92-97) are never reached because module import fails first"
      - path: "packages/viscy-data/src/viscy_data/cell_classification.py"
        issue: "ImportError guard in ClassificationDataset.__init__ (lines 62-65) never reached because iohub import fails first"
    missing:
      - "Move optional dependency checks to module level (before other imports)"
      - "OR use lazy imports for iohub and other transitive dependencies"
  - truth: "TripletDataModule does not import or depend on viscy-transforms"
    status: verified
    reason: "Uses MONAI CenterSpatialCropd instead of BatchedCenterSpatialCropd"
    artifacts: []
    missing: []
---

# Phase 7: Code Migration Verification Report

**Phase Goal:** All 13 data modules are migrated and importable with clean paths
**Verified:** 2026-02-13T18:30:00Z
**Status:** gaps_found
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `from viscy_data import HCSDataModule` (and all other DataModules/Datasets) works for all 15+ public classes | ? UNCERTAIN | Cannot runtime-test due to NumPy incompatibility; static analysis shows all 45 exports present in __all__ |
| 2 | `import viscy_data` succeeds without any optional extras installed | ✗ FAILED | __init__.py eager imports cause transitive dependency failures (iohub requires pandas/xarray/dask) |
| 3 | All 15+ public classes are available at package top level | ✓ VERIFIED | __all__ contains 45 exports (17 types, 2 utilities, 26 DataModules/Datasets/enums) |
| 4 | TripletDataModule does not import or depend on viscy-transforms | ✓ VERIFIED | Uses MONAI CenterSpatialCropd (line 549), zero references to BatchedCenterSpatialCropd or viscy_transforms |
| 5 | All internal imports use absolute viscy_data. prefix (no relative imports) | ✓ VERIFIED | 0 relative imports found, 39 absolute viscy_data. imports across all modules |
| 6 | Importing a module that requires an uninstalled optional extra produces a clear error message | ✗ FAILED | Module-level import failures prevent reaching __init__ method error messages |

**Score:** 3/6 truths verified (2 failed, 1 uncertain)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `packages/viscy-data/src/viscy_data/__init__.py` | Complete public API with 45 exports from all 13 modules | ⚠️ ORPHANED | Exists with all exports, but eager imports cause runtime failures without optional deps |
| `packages/viscy-data/src/viscy_data/_typing.py` | Type definitions (Sample, NormMeta, etc.) | ✓ VERIFIED | Exists, 17 types exported |
| `packages/viscy-data/src/viscy_data/select.py` | SelectWell mixin | ✓ VERIFIED | Exists, 163 lines |
| `packages/viscy-data/src/viscy_data/distributed.py` | ShardedDistributedSampler | ✓ VERIFIED | Exists, 61 lines |
| `packages/viscy-data/src/viscy_data/segmentation.py` | SegmentationDataset, SegmentationDataModule | ✓ VERIFIED | Exists, 142 lines |
| `packages/viscy-data/src/viscy_data/hcs.py` | HCSDataModule, SlidingWindowDataset, MaskTestDataset | ✓ VERIFIED | Exists, 663 lines |
| `packages/viscy-data/src/viscy_data/gpu_aug.py` | GPUTransformDataModule, CachedOmeZarrDataset, CachedOmeZarrDataModule | ✓ VERIFIED | Exists, 262 lines |
| `packages/viscy-data/src/viscy_data/triplet.py` | TripletDataset, TripletDataModule | ✓ VERIFIED | Exists, 565 lines, uses CenterSpatialCropd |
| `packages/viscy-data/src/viscy_data/cell_classification.py` | ClassificationDataset, ClassificationDataModule | ✓ VERIFIED | Exists, 185 lines |
| `packages/viscy-data/src/viscy_data/cell_division_triplet.py` | CellDivisionTripletDataset, CellDivisionTripletDataModule | ✓ VERIFIED | Exists, 270 lines |
| `packages/viscy-data/src/viscy_data/mmap_cache.py` | MmappedDataset, MmappedDataModule | ✓ VERIFIED | Exists, 344 lines |
| `packages/viscy-data/src/viscy_data/ctmc_v1.py` | CTMCv1DataModule | ✓ VERIFIED | Exists, 66 lines |
| `packages/viscy-data/src/viscy_data/livecell.py` | LiveCellDataset, LiveCellTestDataset, LiveCellDataModule | ✓ VERIFIED | Exists, 319 lines |
| `packages/viscy-data/src/viscy_data/combined.py` | CombinedDataModule, CombineMode, ConcatDataModule, BatchedConcatDataModule, CachedConcatDataModule, BatchedConcatDataset | ✓ VERIFIED | Exists, 378 lines |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `__init__.py` | all 13 data modules | eager imports (lines 22-91) | ⚠️ PARTIAL | Imports exist but cause runtime failures due to transitive dependencies |
| `triplet.py` | `_final_crop()` | `CenterSpatialCropd` | ✓ WIRED | Line 549 uses MONAI CenterSpatialCropd, not viscy-transforms BatchedCenterSpatialCropd |
| All modules | `_typing.py`, `_utils.py` | absolute imports | ✓ WIRED | 39 internal imports using viscy_data. prefix |

### Requirements Coverage

No REQUIREMENTS.md entries mapped to Phase 7.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `__init__.py` | 43-91 | Eager imports of all modules | 🛑 Blocker | Prevents `import viscy_data` without optional extras; violates success criterion 2 |
| `cell_classification.py` | 17 | Eager import of iohub.ngff (transitive dep on pandas/xarray/dask) | 🛑 Blocker | Module import fails without pandas even though pandas has try/except guard |
| `hcs.py` | 13 | Eager import of iohub.ngff | 🛑 Blocker | Core module fails to import without pandas (iohub transitive dep) |
| `triplet.py` | 25 | Eager import of iohub.ngff | 🛑 Blocker | Module import fails before reaching ImportError guard in __init__ |
| `gpu_aug.py` | 19 | Eager import of iohub.ngff | 🛑 Blocker | Core module fails to import without pandas |
| `mmap_cache.py` | 13 | Eager import of iohub.ngff | 🛑 Blocker | Module import fails without pandas |
| `segmentation.py` | 9 | Eager import of iohub.ngff | 🛑 Blocker | Core module fails to import without pandas |
| `livecell.py` | - | Lazy imports for pycocotools/tifffile/torchvision | ✓ Good pattern | Correctly uses try/except with None sentinel |

### Human Verification Required

None - all verification criteria can be tested programmatically.

### Gaps Summary

**Root cause:** The plan assumed each module's try/except guards for optional dependencies (pandas, tensorstore, tensordict) would be sufficient. However, nearly ALL modules (including core modules like hcs.py) import `iohub.ngff` eagerly at the module level, and iohub has transitive dependencies on pandas/xarray/dask. This means:

1. `import viscy_data` → `from viscy_data.hcs import ...` → executes hcs.py → `from iohub.ngff import ...` → fails without pandas
2. The ImportError guards in class `__init__` methods (e.g., `TripletDataset.__init__`) are never reached because the module import fails first
3. Even "core" modules (hcs, gpu_aug, segmentation) that don't need optional extras for their basic functionality cannot be imported without pandas installed

**Fix required:** Either:
- Option A: Make __init__.py use lazy imports (TYPE_CHECKING pattern or `__getattr__` pattern)
- Option B: Make iohub imports lazy (move inside methods, or add try/except with None sentinel at module level)
- Option C: Declare iohub/pandas as a base dependency (not optional), which defeats the purpose of optional extras

**Recommendation:** Option B (lazy iohub imports) is most aligned with the phase goal. Move iohub imports inside methods/functions or use try/except at module level.

---

_Verified: 2026-02-13T18:30:00Z_
_Verifier: Claude (gsd-verifier)_
