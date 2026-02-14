# Roadmap: VisCy Modularization

## Milestones

- Completed **v1.0 Extract viscy-transforms** - Phases 1-5 (shipped 2026-01-29)
- Active **v1.1 Extract viscy-data** - Phases 6-9 (in progress)

## Phases

<details>
<summary>Completed: v1.0 Extract viscy-transforms (Phases 1-5) - SHIPPED 2026-01-29</summary>

### Phase 1: Workspace Foundation
**Goal**: Establish a clean uv workspace with shared tooling configuration
**Depends on**: Nothing (first phase)
**Requirements**: WORK-00, WORK-01, WORK-02, WORK-03, WORK-04, WORK-05
**Success Criteria** (what must be TRUE):
  1. Repository contains only LICENSE, CITATION.cff, .gitignore, and new workspace structure
  2. `uv sync` runs successfully at workspace root
  3. `uvx prek` passes with ruff and mypy hooks configured
  4. Python 3.11+ constraint enforced in root pyproject.toml
  5. Empty `packages/` directory exists and is a workspace member
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md -- Clean slate + workspace pyproject.toml with uv configuration
- [x] 01-02-PLAN.md -- Pre-commit hooks with ruff and ty

### Phase 2: Package Structure
**Goal**: Create viscy-transforms package skeleton with modern build system
**Depends on**: Phase 1
**Requirements**: PKG-01, PKG-02, PKG-03, PKG-04
**Success Criteria** (what must be TRUE):
  1. `packages/viscy-transforms/src/viscy_transforms/__init__.py` exists with proper structure
  2. Package pyproject.toml uses hatchling with uv-dynamic-versioning
  3. `uv pip install -e packages/viscy-transforms` succeeds
  4. Package README.md documents installation and basic usage
**Plans**: 1 plan

Plans:
- [x] 02-01-PLAN.md -- Package skeleton with hatchling, uv-dynamic-versioning, and README

### Phase 3: Code Migration
**Goal**: Migrate all transforms code and tests with passing test suite
**Depends on**: Phase 2
**Requirements**: MIG-01, MIG-02, MIG-03, MIG-04, MIG-05
**Success Criteria** (what must be TRUE):
  1. All 16 transform modules exist in `packages/viscy-transforms/src/viscy_transforms/`
  2. `from viscy_transforms import X` works for all public exports
  3. `uv run --package viscy-transforms pytest` passes all tests
  4. No `viscy/transforms/` directory exists in repository
  5. Import paths in tests updated to `viscy_transforms`
**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md -- Extract types from viscy.data.typing to _typing.py
- [x] 03-02-PLAN.md -- Migrate 16 transform modules with updated imports
- [x] 03-03-PLAN.md -- Migrate tests and verify full test suite passes

### Phase 4: Documentation
**Goal**: Zensical documentation deployed to GitHub Pages
**Depends on**: Phase 3
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04
**Status**: Deferred
**Plans**: TBD

### Phase 5: CI/CD
**Goal**: Automated testing and linting via GitHub Actions
**Depends on**: Phase 3
**Requirements**: CI-01, CI-03, CI-04
**Success Criteria** (what must be TRUE):
  1. Push to main triggers test workflow for viscy-transforms
  2. Tests run against Python 3.11, 3.12, 3.13 on Ubuntu, macOS, Windows
  3. `uvx prek` linting passes in CI
  4. alls-green check job aggregates matrix results for branch protection
**Plans**: 1 plan

Plans:
- [x] 05-01-PLAN.md -- Test matrix (9 jobs) + lint workflow with prek

</details>

### Active: v1.1 Extract viscy-data

**Milestone Goal:** Extract all 13 data modules into an independent `viscy-data` package with optional dependency groups, clean import paths, and no cross-package dependencies.

**Phase Numbering:**
- Integer phases (6, 7, 8, 9): Planned milestone work
- Decimal phases (6.1, 7.1): Urgent insertions (marked with INSERTED)

- [x] **Phase 6: Package Scaffolding and Foundation** - Package structure, dependency declarations, and shared utility extraction
- [x] **Phase 7: Code Migration** - Migrate all 13 data modules with updated imports and lazy loading
- [ ] **Phase 8: Test Migration and Validation** - Migrate tests and verify package works correctly
- [ ] **Phase 9: CI Integration** - Extend CI workflows with viscy-data jobs and tiered matrix

## Phase Details

### Phase 6: Package Scaffolding and Foundation
**Goal**: Users can install viscy-data and import foundational types and utilities
**Depends on**: Phase 5 (v1.0 workspace established)
**Requirements**: DATA-PKG-01, DATA-PKG-02, DATA-PKG-04
**Success Criteria** (what must be TRUE):
  1. `uv pip install -e packages/viscy-data` succeeds from workspace root
  2. `from viscy_data import Sample, NormMeta` imports type definitions without error
  3. Optional dependency groups (`[triplet]`, `[livecell]`, `[mmap]`, `[all]`) are declared in pyproject.toml and installable
  4. `_utils.py` contains shared helpers (`_ensure_channel_list`, `_read_norm_meta`, `_collate_samples`) extracted from hcs.py, importable as `from viscy_data._utils import X`
  5. `py.typed` marker exists for type checking support
**Plans**: 2 plans

Plans:
- [x] 06-01-PLAN.md -- Package skeleton with pyproject.toml, type definitions, and workspace integration
- [x] 06-02-PLAN.md -- Extract shared utilities from hcs.py and triplet.py into _utils.py

### Phase 7: Code Migration
**Goal**: All 13 data modules are migrated and importable with clean paths
**Depends on**: Phase 6
**Requirements**: DATA-PKG-03, DATA-MIG-01, DATA-MIG-02, DATA-MIG-03, DATA-MIG-04
**Success Criteria** (what must be TRUE):
  1. `from viscy_data import HCSDataModule` (and all other DataModules/Datasets) works for all 15+ public classes
  2. `import viscy_data` succeeds without any optional extras installed (tensorstore, tensordict, pycocotools are not required at import time)
  3. `TripletDataModule` does not import or depend on viscy-transforms; batch shape is asserted directly instead of using `BatchedCenterSpatialCropd`
  4. All internal imports use absolute `viscy_data.` prefix (no relative imports)
  5. Importing a module that requires an uninstalled optional extra produces a clear error message naming the missing package and the install command
**Plans**: 4 plans

Plans:
- [x] 07-01-PLAN.md -- Migrate core modules (select, distributed, segmentation, hcs, gpu_aug)
- [x] 07-02-PLAN.md -- Migrate triplet family (triplet with BatchedCenterSpatialCropd removal, cell_classification, cell_division_triplet)
- [x] 07-03-PLAN.md -- Migrate optional dep modules + composition (mmap_cache, ctmc_v1, livecell, combined)
- [x] 07-04-PLAN.md -- Complete __init__.py exports and full package verification

### Phase 8: Test Migration and Validation
**Goal**: All existing data tests pass under the new package structure
**Depends on**: Phase 7
**Requirements**: DATA-TST-01, DATA-TST-02
**Success Criteria** (what must be TRUE):
  1. `uv run --package viscy-data pytest` passes all tests (test_hcs.py, test_triplet.py, test_select.py)
  2. A smoke test verifies `import viscy_data` works in an environment with only base dependencies (no optional extras)
  3. Smoke tests verify that accessing optional-dependency modules without the extra installed raises an error with the correct install instruction
**Plans**: 2 plans

Plans:
- [ ] 08-01-PLAN.md -- Migrate conftest.py and 3 test files (test_hcs, test_triplet, test_select) with updated imports
- [ ] 08-02-PLAN.md -- Smoke tests for import, __all__ completeness, and optional dep error messages

### Phase 9: CI Integration
**Goal**: CI automatically tests viscy-data on every push with tiered dependency coverage
**Depends on**: Phase 8
**Requirements**: DATA-CI-01, DATA-CI-02
**Success Criteria** (what must be TRUE):
  1. Push to main or PR triggers viscy-data test jobs in GitHub Actions
  2. Base dependency tests run across 3 Python versions (3.11, 3.12, 3.13) and 3 operating systems (Ubuntu, macOS, Windows)
  3. Full extras tests run on a narrower matrix (1 Python version, 1 OS) to verify optional dependency integration
  4. alls-green aggregation job includes viscy-data results alongside viscy-transforms results
**Plans**: TBD

Plans:
- [ ] 09-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 6 -> 7 -> 8 -> 9

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Workspace Foundation | v1.0 | 2/2 | Complete | 2026-01-28 |
| 2. Package Structure | v1.0 | 1/1 | Complete | 2026-01-28 |
| 3. Code Migration | v1.0 | 3/3 | Complete | 2026-01-28 |
| 4. Documentation | v1.0 | 0/TBD | Deferred | - |
| 5. CI/CD | v1.0 | 1/1 | Complete | 2026-01-29 |
| 6. Package Scaffolding and Foundation | v1.1 | 2/2 | Complete | 2026-02-13 |
| 7. Code Migration | v1.1 | 4/4 | Complete | 2026-02-14 |
| 8. Test Migration and Validation | v1.1 | 0/TBD | Not started | - |
| 9. CI Integration | v1.1 | 0/TBD | Not started | - |

---
*Roadmap created: 2025-01-27*
*v1.1 phases added: 2026-02-13*
*Last updated: 2026-02-13*
