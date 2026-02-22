# Roadmap: VisCy Modularization

## Milestones

- Shipped **v1.0 Transforms & Monorepo Skeleton** -- Phases 1-5 (shipped 2026-01-29)
- Shipped **v1.1 Extract viscy-data** -- Phases 6-9 (shipped 2026-02-14)
- Shipped **v1.2 Extract viscy-models** -- Phases 10-14 (shipped 2026-02-13)
- Shipped **v2.0 DynaCLR Application** -- Phases 15-17 (shipped 2026-02-17)
- Shipped **v2.1 DynaCLR Integration Validation** -- Phases 18-19 (shipped 2026-02-20)
- In Progress **v2.2 Composable Sampling Framework** -- Phases 20-25

## Phases

<details>
<summary>v1.0 Transforms & Monorepo Skeleton (Phases 1-5) -- SHIPPED 2026-01-29</summary>

### Phase 1: Workspace Foundation
**Goal**: Establish a clean uv workspace with shared tooling configuration
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md -- Clean slate + workspace pyproject.toml with uv configuration
- [x] 01-02-PLAN.md -- Pre-commit hooks with ruff and ty

### Phase 2: Package Structure
**Goal**: Create viscy-transforms package skeleton with modern build system
**Plans**: 1 plan

Plans:
- [x] 02-01-PLAN.md -- Package skeleton with hatchling, uv-dynamic-versioning, and README

### Phase 3: Code Migration
**Goal**: Migrate all transforms code and tests with passing test suite
**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md -- Extract types from viscy.data.typing to _typing.py
- [x] 03-02-PLAN.md -- Migrate 16 transform modules with updated imports
- [x] 03-03-PLAN.md -- Migrate tests and verify full test suite passes

### Phase 4: Documentation (Deferred)
**Goal**: Zensical documentation deployed to GitHub Pages

### Phase 5: CI/CD
**Goal**: Automated testing and linting via GitHub Actions
**Plans**: 1 plan

Plans:
- [x] 05-01-PLAN.md -- Test matrix (9 jobs) + lint workflow with prek

</details>

<details>
<summary>v1.1 Extract viscy-data (Phases 6-9) -- SHIPPED 2026-02-14</summary>

### Phase 6: Package Scaffolding and Foundation
**Goal**: Users can install viscy-data and import foundational types and utilities
**Depends on**: Phase 5 (v1.0 workspace established)
**Plans**: 2 plans

Plans:
- [x] 06-01-PLAN.md -- Package skeleton with pyproject.toml, type definitions, and workspace integration
- [x] 06-02-PLAN.md -- Extract shared utilities from hcs.py and triplet.py into _utils.py

### Phase 7: Code Migration
**Goal**: All 13 data modules are migrated and importable with clean paths
**Depends on**: Phase 6
**Plans**: 4 plans

Plans:
- [x] 07-01-PLAN.md -- Migrate core modules (select, distributed, segmentation, hcs, gpu_aug)
- [x] 07-02-PLAN.md -- Migrate triplet family (triplet, cell_classification, cell_division_triplet)
- [x] 07-03-PLAN.md -- Migrate optional dep modules + composition (mmap_cache, ctmc_v1, livecell, combined)
- [x] 07-04-PLAN.md -- Complete __init__.py exports and full package verification

### Phase 8: Test Migration and Validation
**Goal**: All existing data tests pass under the new package structure
**Depends on**: Phase 7
**Plans**: 2 plans

Plans:
- [x] 08-01-PLAN.md -- Migrate conftest.py and 3 test files with updated imports
- [x] 08-02-PLAN.md -- Smoke tests for import, __all__ completeness, and optional dep error messages

### Phase 9: CI Integration
**Goal**: CI automatically tests viscy-data on every push with tiered dependency coverage
**Depends on**: Phase 8
**Plans**: 1 plan

Plans:
- [x] 09-01-PLAN.md -- Add viscy-data test jobs (3x3 base + 1x1 extras) and update alls-green

</details>

<details>
<summary>v1.2 Extract viscy-models (Phases 10-14) -- SHIPPED 2026-02-13</summary>

### Phase 10: Package Scaffold & Shared Components
**Goal**: Users can install viscy-models and shared architectural components are available
**Depends on**: Phase 5 (v1.0 CI infrastructure)
**Plans**: 3 plans

Plans:
- [x] 10-01-PLAN.md -- Package scaffold, pyproject.toml, workspace registration
- [x] 10-02-PLAN.md -- Extract shared _components (stems, heads, blocks) with tests
- [x] 10-03-PLAN.md -- Migrate ConvBlock2D/3D to unet/_layers with tests

### Phase 11: Core UNet Models
**Goal**: UNeXt2 and FCMAE are importable from viscy-models with forward-pass tests
**Depends on**: Phase 10
**Plans**: 2 plans

Plans:
- [x] 11-01-PLAN.md -- Migrate UNeXt2 model with new forward-pass tests (6 tests)
- [x] 11-02-PLAN.md -- Migrate FCMAE model with 11 existing tests and finalize unet exports

### Phase 12: Representation Models
**Goal**: All contrastive and VAE models are importable with forward-pass tests
**Depends on**: Phase 10
**Plans**: 2 plans

Plans:
- [x] 12-01-PLAN.md -- Migrate ContrastiveEncoder and ResNet3dEncoder with forward-pass tests
- [x] 12-02-PLAN.md -- Migrate BetaVae25D and BetaVaeMonai with forward-pass tests

### Phase 13: Legacy UNet Models
**Goal**: Unet2d and Unet25d are importable from viscy-models with migrated test coverage
**Depends on**: Phase 10
**Plans**: 1 plan

Plans:
- [x] 13-01-PLAN.md -- Migrate Unet2d and Unet25d with pytest test coverage

### Phase 14: Public API & CI Integration
**Goal**: All 8 models importable from top-level, CI verifying the full package
**Depends on**: Phases 11, 12, 13
**Plans**: 1 plan

Plans:
- [x] 14-01-PLAN.md -- Public API re-exports, state dict compatibility tests, CI matrix update

</details>

<details>
<summary>v2.0 DynaCLR Application (Phases 15-17) -- SHIPPED 2026-02-17</summary>

### Phase 15: Shared Infrastructure (viscy-utils)
**Goal**: Extract shared ML training infrastructure into viscy-utils package
**Plans**: Manual (no GSD plans)

Delivered:
- [x] viscy-utils package with trainer, callbacks, evaluation, cli_utils
- [x] EmbeddingWriter callback, linear classifier evaluation, visualization
- [x] cli_utils.py with format_markdown_table() and load_config()
- [x] pyyaml added to viscy-utils dependencies

### Phase 16: DynaCLR Application Core
**Goal**: Create applications/dynaclr with engine, CLI, and LightningModules
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
- [x] evaluation/linear_classifiers/ -- train, apply, dataset discovery, config generation
- [x] examples/configs/ -- fit.yml, predict.yml, SLURM scripts with updated class_paths
- [x] examples/DynaCLR-DENV-VS-Ph/ -- infection analysis demo with updated imports
- [x] examples/embedding-web-visualization/ -- interactive visualizer with updated imports
- [x] examples/DynaCLR-classical-sampling/ -- pseudo-track generation
- [x] examples/vcp_tutorials/ -- quickstart notebook and script with updated imports
- [x] CLI commands: train-linear-classifier, apply-linear-classifier
- [x] wandb, anndata, natsort added to dynaclr [eval] optional dependencies

</details>

<details>
<summary>v2.1 DynaCLR Integration Validation (Phases 18-19) -- SHIPPED 2026-02-20</summary>

### Phase 18: Training Validation
**Goal**: User can run a DynaCLR training loop through the modular application and confirm it completes without errors
**Depends on**: Phase 17 (v2.0 DynaCLR application exists)
**Requirements**: TRAIN-01, TRAIN-02
**Plans**: 1 plan

Plans:
- [x] 18-01-PLAN.md -- Training integration tests (fast_dev_run + YAML config class_path resolution)

### Phase 19: Inference Reproducibility
**Goal**: User can load a pretrained checkpoint into the modular DynaCLR application, run prediction, and get embeddings that exactly match saved reference outputs
**Depends on**: Phase 18
**Requirements**: INFER-01, INFER-02, INFER-03, TEST-01, TEST-02
**Plans**: 1 plan

Plans:
- [x] 19-01-PLAN.md -- Inference reproducibility tests (checkpoint loading, embedding prediction, exact match)

</details>

### v2.2 Composable Sampling Framework (In Progress)

**Milestone Goal:** Implement a composable, multi-experiment sampling framework for DynaCLR with experiment-aware batching, lineage-linked temporal positives, hard-negative concentration loss, and channel dropout -- enabling cross-experiment training that resolves heterogeneous cellular responses.

- [x] **Phase 20: Experiment Configuration** - ExperimentConfig and ExperimentRegistry with channel resolution and YAML config parsing
- [ ] **Phase 21: Cell Index & Lineage** - MultiExperimentIndex with unified tracks, lineage reconstruction, border clamping, and valid anchor computation
- [ ] **Phase 22: Batch Sampling** - FlexibleBatchSampler with experiment-aware, condition-balanced, temporal enrichment, leaky mixing, and DDP support
- [ ] **Phase 23: Loss & Augmentation** - NTXentHCL loss with hard-negative concentration plus ChannelDropout and variable tau sampling
- [ ] **Phase 24: Dataset & DataModule** - MultiExperimentTripletDataset and MultiExperimentDataModule wiring all components together
- [ ] **Phase 25: Integration** - End-to-end training validation and YAML config example for multi-experiment training

## Phase Details

### Phase 20: Experiment Configuration
**Goal**: Users can define multi-experiment training setups via dataclasses and YAML configs, with explicit source_channel lists and positional alignment across experiments
**Depends on**: Phase 19 (v2.1 validated DynaCLR application)
**Requirements**: MEXP-01, MEXP-02, MEXP-03, MEXP-04
**Success Criteria** (what must be TRUE):
  1. User can instantiate an ExperimentConfig with experiment metadata (name, data_path, tracks_path, channel_names, source_channel, condition_wells, interval_minutes) and access all fields
  2. User can create an ExperimentRegistry from multiple ExperimentConfigs and it validates channel count consistency, computes per-experiment channel_maps (source position -> zarr index)
  3. User specifies explicit source_channel list per experiment -- Registry validates source_channel membership in channel_names and positional alignment (same count across experiments)
  4. User can define experiment configs in a YAML file that ExperimentRegistry.from_yaml() parses into a valid registry
**Plans**: 2 plans

Plans:
- [x] 20-01-PLAN.md -- TDD: ExperimentConfig and ExperimentRegistry with validation, from_yaml, tau_range_frames
- [x] 20-02-PLAN.md -- Package wiring: deps, __init__.py exports, example experiments.yml

**Location**: `applications/dynaclr/src/dynaclr/`

### Phase 21: Cell Index & Lineage
**Goal**: Users have a unified cell observation index across all experiments with lineage-linked tracks, border-safe centroids, and valid anchor computation for variable tau
**Depends on**: Phase 20 (ExperimentRegistry provides experiment metadata)
**Requirements**: CELL-01, CELL-02, CELL-03, CELL-04
**Success Criteria** (what must be TRUE):
  1. MultiExperimentIndex builds a single tracks DataFrame from all registered experiments with columns: experiment, condition, global_track_id, hours_post_infection, well_name, fluorescence_channel -- and each row represents one cell observation at one timepoint
  2. Lineage is reconstructed from parent_track_id -- when a cell divides, daughter tracks are linked to the parent track so that temporal positive sampling can follow through division events
  3. Border cells are retained by clamping crop centroids inward rather than excluding them -- cells near the image boundary still appear as valid observations with shifted patch origins
  4. valid_anchors is computed accounting for variable tau range and lineage continuity -- an anchor is valid only if at least one tau in the configured range yields a same-track or daughter-track positive
**Plans**: TBD
**Location**: `applications/dynaclr/src/dynaclr/`

### Phase 22: Batch Sampling
**Goal**: Users can compose experiment-aware, condition-balanced, and temporally enriched batch sampling strategies via a single configurable FlexibleBatchSampler
**Depends on**: Phase 21 (MultiExperimentIndex provides valid_anchors DataFrame)
**Requirements**: SAMP-01, SAMP-02, SAMP-03, SAMP-04, SAMP-05
**Success Criteria** (what must be TRUE):
  1. With experiment_aware=True, every batch contains cells from only a single experiment (verified over many batches)
  2. With condition_balanced=True, each batch has approximately 50/50 infected vs uninfected cells (within statistical tolerance across many batches)
  3. With temporal_enrichment=True, batches concentrate cells around a focal HPI with a configurable window, while still including a global fraction from all timepoints
  4. FlexibleBatchSampler supports DDP via set_epoch() for deterministic shuffling and rank-aware iteration that composes with the existing ShardedDistributedSampler pattern
  5. Leaky experiment mixing (leaky > 0.0) allows a configurable fraction of cross-experiment samples in otherwise experiment-restricted batches
**Plans**: TBD
**Location**: `packages/viscy-data/src/viscy_data/`

### Phase 23: Loss & Augmentation
**Goal**: Users have an HCL-enhanced contrastive loss, channel dropout augmentation, and variable tau sampling -- all independent modules that plug into the existing DynaCLR training pipeline
**Depends on**: Phase 20 (ExperimentRegistry for channel information); independent of Phases 21-22
**Requirements**: LOSS-01, LOSS-02, LOSS-03, AUG-01, AUG-02, AUG-03
**Success Criteria** (what must be TRUE):
  1. NTXentHCL computes NT-Xent loss with hard-negative concentration (beta parameter), returns a scalar loss with gradients, and produces numerically identical results to standard NT-Xent when beta=0.0
  2. NTXentHCL is an nn.Module that works as a drop-in replacement via ContrastiveModule(loss_function=NTXentHCL(...)) without any changes to the training step
  3. ChannelDropout randomly zeros specified channels with configurable probability on batched (B,C,Z,Y,X) tensors and integrates into on_after_batch_transfer after the existing scatter/gather augmentation chain
  4. Variable tau sampling uses exponential decay within tau_range, favoring small temporal offsets -- verified by statistical distribution test
**Plans**: TBD
**Location**: NTXentHCL in `applications/dynaclr/src/dynaclr/`, ChannelDropout in `packages/viscy-data/src/viscy_data/`

### Phase 24: Dataset & DataModule
**Goal**: Users can train DynaCLR across multiple experiments using MultiExperimentTripletDataset and MultiExperimentDataModule, which wire together all sampling, loss, and augmentation components with full Lightning CLI configurability
**Depends on**: Phase 21 (MultiExperimentIndex), Phase 22 (FlexibleBatchSampler), Phase 23 (NTXentHCL, ChannelDropout, variable tau)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05
**Success Criteria** (what must be TRUE):
  1. MultiExperimentTripletDataset.__getitems__ returns a batch dict with anchor, positive keys and norm_meta that is directly compatible with the existing ContrastiveModule.training_step -- no changes to the engine
  2. Positive sampling follows lineage through division events -- when an anchor track ends at a division, the daughter track at t+tau is selected as a valid positive
  3. MultiExperimentDataModule wires FlexibleBatchSampler + Dataset + ChannelDropout + ThreadDataLoader with collate_fn=lambda x: x, and train/val split is by whole experiments (not individual FOVs)
  4. All sampling, loss, and augmentation hyperparameters (tau_range, tau_decay, experiment_aware, condition_balanced, temporal_enrichment, hcl_beta, channel_dropout_prob) are exposed as __init__ parameters for Lightning CLI YAML configuration
**Plans**: TBD
**Location**: `applications/dynaclr/src/dynaclr/`

### Phase 25: Integration
**Goal**: Users can run an end-to-end multi-experiment DynaCLR training loop with all composable sampling axes enabled, validated by a fast_dev_run integration test and a complete YAML config example
**Depends on**: Phase 24 (all components wired)
**Requirements**: INTG-01, INTG-02
**Success Criteria** (what must be TRUE):
  1. A fast_dev_run integration test completes without errors using MultiExperimentDataModule + ContrastiveModule + NTXentHCL with synthetic multi-experiment data (at least 2 experiments with different channel sets)
  2. A YAML config example demonstrates multi-experiment training with all sampling axes (experiment_aware, condition_balanced, temporal_enrichment) and is parseable by Lightning CLI
**Plans**: TBD
**Location**: `applications/dynaclr/tests/`, `applications/dynaclr/examples/configs/`

### v2.3+ Future Applications (Phases TBD)

**Candidates (not yet planned):**
- applications/Cytoland -- VSUNet/FcmaeUNet LightningModules
- viscy-airtable -- abstract from current Airtable integration
- Hydra infrastructure (viscy-hydra or integrated)
- Zero-padding for missing channels (MEXP-05)
- kNN-based hard negative mining in sampler (SAMP-06)
- Prediction/inference mode for MultiExperimentDataModule (DATA-06)
- Multi-GPU benchmark (INTG-03)

## Progress

**Execution Order:**
Phases execute in numeric order: 20 -> 21 -> 22 -> 23 -> 24 -> 25
(Phase 23 can execute in parallel with Phase 22 since they are independent)

| Phase | Milestone | Plans | Status | Completed |
|-------|-----------|-------|--------|-----------|
| 1. Workspace Foundation | v1.0 | 2/2 | Complete | 2026-01-28 |
| 2. Package Structure | v1.0 | 1/1 | Complete | 2026-01-28 |
| 3. Code Migration | v1.0 | 3/3 | Complete | 2026-01-28 |
| 4. Documentation | v1.0 | 0/TBD | Deferred | -- |
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
| 18. Training Validation | v2.1 | 1/1 | Complete | 2026-02-20 |
| 19. Inference Reproducibility | v2.1 | 1/1 | Complete | 2026-02-20 |
| 20. Experiment Configuration | v2.2 | 2/2 | Complete | 2026-02-22 |
| 21. Cell Index & Lineage | v2.2 | 0/TBD | Not started | -- |
| 22. Batch Sampling | v2.2 | 0/TBD | Not started | -- |
| 23. Loss & Augmentation | v2.2 | 0/TBD | Not started | -- |
| 24. Dataset & DataModule | v2.2 | 0/TBD | Not started | -- |
| 25. Integration | v2.2 | 0/TBD | Not started | -- |

**Total plans executed:** 29 (v1.0: 7, v1.1: 9, v1.2: 9, v2.1: 2, v2.2: 2) + 3 manual phases (v2.0)

---
*Roadmap created: 2025-01-27*
*Harmonized from modular-data + modular-models branches: 2026-02-16*
*Updated for v2.0 DynaCLR: 2026-02-17*
*Updated for v2.1 DynaCLR Integration Validation: 2026-02-19*
*Updated for v2.2 Composable Sampling Framework: 2026-02-21*
*Phase 20 planned: 2026-02-21*
