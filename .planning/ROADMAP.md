# Roadmap: VisCy Modularization

## Milestones

- Shipped **v1.0 Transforms & Monorepo Skeleton** — Phases 1-5 (shipped 2026-01-29)
- Shipped **v1.1 Extract viscy-data** — Phases 6-9 (shipped 2026-02-14)
- Shipped **v1.2 Extract viscy-models** — Phases 10-14 (shipped 2026-02-13)
- Shipped **v2.0 DynaCLR Application** — Phases 15-17 (shipped 2026-02-17)
- In Progress **v2.1 DynaCLR Integration Validation** — Phases 18-19

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

<details>
<summary>v2.0 DynaCLR Application (Phases 15-17) — SHIPPED 2026-02-17</summary>

### Phase 15: Shared Infrastructure (viscy-utils)
**Goal**: Extract shared ML training infrastructure into viscy-utils package
**Plans**: Manual (no GSD plans)

Delivered:
- [x] viscy-utils package with trainer, callbacks, evaluation, cli_utils
- [x] EmbeddingWriter callback, linear classifier evaluation, visualization
- [x] cli_utils.py with format_markdown_table() and load_config()
- [x] pyyaml added to viscy-utils dependencies

### Phase 16: DynaCLR Application Core
**Goal**: Create applications/dynacrl with engine, CLI, and LightningModules
**Depends on**: Phase 15
**Plans**: Manual

Delivered:
- [x] ContrastiveModule engine (LightningModule)
- [x] MultiModalContrastiveModule for cross-modal distillation
- [x] ClassificationModule for downstream classification
- [x] vae_logging utilities
- [x] dynaclr CLI with LazyCommand pattern
- [x] pyproject.toml with workspace integration

### Phase 17: Examples & Evaluation Migration
**Goal**: Migrate examples and evaluation scripts into self-contained application
**Depends on**: Phase 16
**Plans**: Manual

Delivered:
- [x] evaluation/linear_classifiers/ — train, apply, dataset discovery, config generation
- [x] examples/configs/ — fit.yml, predict.yml, SLURM scripts with updated class_paths
- [x] examples/DynaCLR-DENV-VS-Ph/ — infection analysis demo with updated imports
- [x] examples/embedding-web-visualization/ — interactive visualizer with updated imports
- [x] examples/DynaCLR-classical-sampling/ — pseudo-track generation
- [x] examples/vcp_tutorials/ — quickstart notebook and script with updated imports
- [x] CLI commands: train-linear-classifier, apply-linear-classifier
- [x] wandb, anndata, natsort added to dynacrl [eval] optional dependencies

</details>

### v2.1 DynaCLR Integration Validation (In Progress)

**Milestone Goal:** Prove the modularized DynaCLR application produces identical results to the original monolithic VisCy, with permanent integration tests.

- [x] **Phase 18: Training Validation** - ContrastiveModule completes a full training loop via fast_dev_run with correct YAML config parsing (completed 2026-02-20)
- [x] **Phase 19: Inference Reproducibility** - Checkpoint loading and prediction produce exact match against reference outputs, with permanent test suite (completed 2026-02-20)

## Phase Details

### Phase 18: Training Validation
**Goal**: User can run a DynaCLR training loop through the modular application and confirm it completes without errors
**Depends on**: Phase 17 (v2.0 DynaCLR application exists)
**Requirements**: TRAIN-01, TRAIN-02
**Success Criteria** (what must be TRUE):
  1. `uv run --package dynacrl pytest` discovers and runs a training integration test that exercises ContrastiveModule through a complete fast_dev_run training loop (fit) without errors
  2. The training test uses a YAML config (or equivalent parametrization) that references the new modular import paths (dynacrl.engine.ContrastiveModule, viscy_models, viscy_data, viscy_transforms) and these class paths resolve correctly
  3. The fast_dev_run completes all stages (train batch, validation batch) and the trainer reports no errors
**Plans**: 1 plan

Plans:
- [ ] 18-01-PLAN.md — Training integration tests (fast_dev_run + YAML config class_path resolution)

### Phase 19: Inference Reproducibility
**Goal**: User can load a pretrained checkpoint into the modular DynaCLR application, run prediction, and get embeddings that exactly match saved reference outputs
**Depends on**: Phase 18
**Requirements**: INFER-01, INFER-02, INFER-03, TEST-01, TEST-02
**Success Criteria** (what must be TRUE):
  1. A pretrained checkpoint (from the original monolithic VisCy) loads successfully into the modular ContrastiveModule without state dict key mismatches
  2. Running the predict step with EmbeddingWriter callback writes embedding outputs to disk
  3. The predicted embeddings are numerically identical (exact match) to saved reference embeddings produced by the original monolithic code
  4. All training and inference integration tests are permanent pytest tests (not standalone scripts) living in `applications/dynacrl/tests/`
  5. The full test suite passes when invoked via `uv run --package dynacrl pytest`
**Plans**: 1 plan

Plans:
- [ ] 19-01-PLAN.md — Inference reproducibility tests (checkpoint loading, embedding prediction, exact match)

### v2.0+ Remaining Applications (Phases TBD)

**Candidates (not yet planned):**
- applications/Cytoland — VSUNet/FcmaeUNet LightningModules
- viscy-airtable — abstract from current Airtable integration
- Hydra infrastructure (viscy-hydra or integrated)

## Progress

**Execution Order:**
Phases execute in numeric order: 18 -> 19

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
| 15. Shared Infrastructure | v2.0 | manual | Complete | 2026-02-17 |
| 16. DynaCLR App Core | v2.0 | manual | Complete | 2026-02-17 |
| 17. Examples & Evaluation | v2.0 | manual | Complete | 2026-02-17 |
| 18. Training Validation | v2.1 | Complete    | 2026-02-20 | - |
| 19. Inference Reproducibility | v2.1 | Complete    | 2026-02-20 | - |

**Total plans executed:** 25 (v1.0: 7, v1.1: 9, v1.2: 9) + 3 manual phases (v2.0)

---
*Roadmap created: 2025-01-27*
*Harmonized from modular-data + modular-models branches: 2026-02-16*
*Updated for v2.0 DynaCLR: 2026-02-17*
*Updated for v2.1 DynaCLR Integration Validation: 2026-02-19*
