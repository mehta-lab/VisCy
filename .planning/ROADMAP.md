# Roadmap: VisCy Modularization

## Milestones

- Shipped **v1.0 Transforms & Monorepo Skeleton** — Phases 1-5 (shipped 2026-01-29)
- Shipped **v1.1 Extract viscy-data** — Phases 6-9 (shipped 2026-02-14)
- Shipped **v1.2 Extract viscy-models** — Phases 10-14 (shipped 2026-02-13)
- Next **v2.0 Applications & Airtable** — Phases TBD

## Phases

<details>
<summary>v1.0 Transforms & Monorepo Skeleton (Phases 1-5) — SHIPPED 2026-01-29</summary>

### Phase 1: Workspace Foundation
**Goal**: Establish a clean uv workspace with shared tooling configuration
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — Clean slate + workspace pyproject.toml with uv configuration
- [x] 01-02-PLAN.md — Pre-commit hooks with ruff and ty

### Phase 2: Package Structure
**Goal**: Create viscy-transforms package skeleton with modern build system
**Plans**: 1 plan

Plans:
- [x] 02-01-PLAN.md — Package skeleton with hatchling, uv-dynamic-versioning, and README

### Phase 3: Code Migration
**Goal**: Migrate all transforms code and tests with passing test suite
**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md — Extract types from viscy.data.typing to _typing.py
- [x] 03-02-PLAN.md — Migrate 16 transform modules with updated imports
- [x] 03-03-PLAN.md — Migrate tests and verify full test suite passes

### Phase 4: Documentation (Deferred)
**Goal**: Zensical documentation deployed to GitHub Pages

### Phase 5: CI/CD
**Goal**: Automated testing and linting via GitHub Actions
**Plans**: 1 plan

Plans:
- [x] 05-01-PLAN.md — Test matrix (9 jobs) + lint workflow with prek

</details>

<details>
<summary>v1.1 Extract viscy-data (Phases 6-9) — SHIPPED 2026-02-14</summary>

### Phase 6: Package Scaffolding and Foundation
**Goal**: Users can install viscy-data and import foundational types and utilities
**Depends on**: Phase 5 (v1.0 workspace established)
**Plans**: 2 plans

Plans:
- [x] 06-01-PLAN.md — Package skeleton with pyproject.toml, type definitions, and workspace integration
- [x] 06-02-PLAN.md — Extract shared utilities from hcs.py and triplet.py into _utils.py

### Phase 7: Code Migration
**Goal**: All 13 data modules are migrated and importable with clean paths
**Depends on**: Phase 6
**Plans**: 4 plans

Plans:
- [x] 07-01-PLAN.md — Migrate core modules (select, distributed, segmentation, hcs, gpu_aug)
- [x] 07-02-PLAN.md — Migrate triplet family (triplet, cell_classification, cell_division_triplet)
- [x] 07-03-PLAN.md — Migrate optional dep modules + composition (mmap_cache, ctmc_v1, livecell, combined)
- [x] 07-04-PLAN.md — Complete __init__.py exports and full package verification

### Phase 8: Test Migration and Validation
**Goal**: All existing data tests pass under the new package structure
**Depends on**: Phase 7
**Plans**: 2 plans

Plans:
- [x] 08-01-PLAN.md — Migrate conftest.py and 3 test files with updated imports
- [x] 08-02-PLAN.md — Smoke tests for import, __all__ completeness, and optional dep error messages

### Phase 9: CI Integration
**Goal**: CI automatically tests viscy-data on every push with tiered dependency coverage
**Depends on**: Phase 8
**Plans**: 1 plan

Plans:
- [x] 09-01-PLAN.md — Add viscy-data test jobs (3x3 base + 1x1 extras) and update alls-green

</details>

<details>
<summary>v1.2 Extract viscy-models (Phases 10-14) — SHIPPED 2026-02-13</summary>

### Phase 10: Package Scaffold & Shared Components
**Goal**: Users can install viscy-models and shared architectural components are available
**Depends on**: Phase 5 (v1.0 CI infrastructure)
**Plans**: 3 plans

Plans:
- [x] 10-01-PLAN.md — Package scaffold, pyproject.toml, workspace registration
- [x] 10-02-PLAN.md — Extract shared _components (stems, heads, blocks) with tests
- [x] 10-03-PLAN.md — Migrate ConvBlock2D/3D to unet/_layers with tests

### Phase 11: Core UNet Models
**Goal**: UNeXt2 and FCMAE are importable from viscy-models with forward-pass tests
**Depends on**: Phase 10
**Plans**: 2 plans

Plans:
- [x] 11-01-PLAN.md — Migrate UNeXt2 model with new forward-pass tests (6 tests)
- [x] 11-02-PLAN.md — Migrate FCMAE model with 11 existing tests and finalize unet exports

### Phase 12: Representation Models
**Goal**: All contrastive and VAE models are importable with forward-pass tests
**Depends on**: Phase 10
**Plans**: 2 plans

Plans:
- [x] 12-01-PLAN.md — Migrate ContrastiveEncoder and ResNet3dEncoder with forward-pass tests
- [x] 12-02-PLAN.md — Migrate BetaVae25D and BetaVaeMonai with forward-pass tests

### Phase 13: Legacy UNet Models
**Goal**: Unet2d and Unet25d are importable from viscy-models with migrated test coverage
**Depends on**: Phase 10
**Plans**: 1 plan

Plans:
- [x] 13-01-PLAN.md — Migrate Unet2d and Unet25d with pytest test coverage

### Phase 14: Public API & CI Integration
**Goal**: All 8 models importable from top-level, CI verifying the full package
**Depends on**: Phases 11, 12, 13
**Plans**: 1 plan

Plans:
- [x] 14-01-PLAN.md — Public API re-exports, state dict compatibility tests, CI matrix update

</details>

### v2.0 Applications & Airtable (Phases TBD)

**Milestone Goal:** Extract application-level LightningModules and the Airtable abstraction into independent packages, composing viscy-data and viscy-models.

*(Phases to be defined during milestone planning)*

## Progress

| Phase | Milestone | Plans | Status | Completed |
|-------|-----------|-------|--------|-----------|
| 1. Workspace Foundation | v1.0 | 2/2 | Complete | 2026-01-28 |
| 2. Package Structure | v1.0 | 1/1 | Complete | 2026-01-28 |
| 3. Code Migration | v1.0 | 3/3 | Complete | 2026-01-28 |
| 4. Documentation | v1.0 | 0/TBD | Deferred | — |
| 5. CI/CD | v1.0 | 1/1 | Complete | 2026-01-29 |
| 6. Package Scaffolding | v1.1 | 2/2 | Complete | 2026-02-13 |
| 7. Data Code Migration | v1.1 | 4/4 | Complete | 2026-02-14 |
| 8. Data Test Migration | v1.1 | 2/2 | Complete | 2026-02-14 |
| 9. Data CI Integration | v1.1 | 1/1 | Complete | 2026-02-14 |
| 10. Models Scaffold | v1.2 | 3/3 | Complete | 2026-02-12 |
| 11. Core UNet Models | v1.2 | 2/2 | Complete | 2026-02-12 |
| 12. Representation Models | v1.2 | 2/2 | Complete | 2026-02-13 |
| 13. Legacy UNet Models | v1.2 | 1/1 | Complete | 2026-02-13 |
| 14. Public API & CI | v1.2 | 1/1 | Complete | 2026-02-13 |

**Total plans executed:** 25 (v1.0: 7, v1.1: 9, v1.2: 9)

---
*Roadmap created: 2025-01-27*
*Harmonized from modular-data + modular-models branches: 2026-02-16*
