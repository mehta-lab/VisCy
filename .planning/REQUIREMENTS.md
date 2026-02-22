# Requirements: VisCy Modularization

**Defined:** 2025-01-27
**Core Value:** Independent, reusable subpackages with clean import paths

## v1.0 Requirements (Complete)

### Workspace Foundation

- [x] **WORK-00**: Clean slate setup — wipe repo keeping only LICENSE, CITATION.cff, .gitignore
- [x] **WORK-01**: Virtual workspace root with `[tool.uv.workspace]` and `members = ["packages/*"]`
- [x] **WORK-02**: Shared lockfile (`uv.lock`) at repository root
- [x] **WORK-03**: Python version floor (>=3.11) enforced in root pyproject.toml
- [x] **WORK-04**: Pre-commit hooks configured (ruff, prek) for local development
- [x] **WORK-05**: Shared pytest configuration in root pyproject.toml

### Package Structure (viscy-transforms)

- [x] **PKG-01**: src layout for viscy-transforms (`packages/viscy-transforms/src/viscy_transforms/`)
- [x] **PKG-02**: Package pyproject.toml with hatchling build backend
- [x] **PKG-03**: uv-dynamic-versioning configured for git-based versioning
- [x] **PKG-04**: Package README.md with installation and usage instructions

### Code Migration (viscy-transforms)

- [x] **MIG-01**: All transform modules migrated from `viscy/transforms/` to package
- [x] **MIG-02**: All transform tests migrated to `packages/viscy-transforms/tests/`
- [x] **MIG-03**: Import path updated to `from viscy_transforms import X`
- [x] **MIG-04**: All migrated tests passing with `uv run --package viscy-transforms pytest`
- [x] **MIG-05**: Original `viscy/transforms/` directory removed

### CI/CD

- [x] **CI-01**: GitHub Actions workflow for testing viscy-transforms package
- [x] **CI-03**: Matrix testing across Python 3.11, 3.12, 3.13 on 3 OSes
- [x] **CI-04**: Linting via prek (uvx prek) in CI workflows

## v1.1 Requirements (Complete)

### Data — Package Structure

- [x] **DATA-PKG-01**: viscy-data package at `packages/viscy-data/src/viscy_data/` with hatchling + uv-dynamic-versioning
- [x] **DATA-PKG-02**: Optional dependency groups `[triplet]`, `[livecell]`, `[mmap]`, `[all]` in pyproject.toml
- [x] **DATA-PKG-03**: No dependency on viscy-transforms; BatchedCenterSpatialCropd in _utils.py
- [x] **DATA-PKG-04**: Shared utilities extracted from hcs.py and triplet.py into `_utils.py`

### Data — Code Migration

- [x] **DATA-MIG-01**: All 13 data modules migrated with updated import paths
- [x] **DATA-MIG-02**: Flat top-level exports in `__init__.py` (45 public exports)
- [x] **DATA-MIG-03**: Lazy imports for optional dependencies with clear error messages
- [x] **DATA-MIG-04**: Internal imports use absolute `viscy_data.` prefix

### Data — Testing

- [x] **DATA-TST-01**: All existing data tests passing under new import paths
- [x] **DATA-TST-02**: Smoke tests for import without extras and correct error messages

### Data — CI/CD

- [x] **DATA-CI-01**: GitHub Actions test workflow extended with viscy-data jobs
- [x] **DATA-CI-02**: Tiered CI matrix: base deps (3x3) + full extras (1x1)

## v1.2 Requirements (Complete)

### Models — Package Infrastructure

- [x] **MPKG-01**: Package directory `packages/viscy-models/` with src layout
- [x] **MPKG-02**: pyproject.toml with hatchling, uv-dynamic-versioning, torch/timm/monai/numpy deps
- [x] **MPKG-03**: `uv sync --package viscy-models` succeeds in workspace
- [x] **MPKG-04**: `_components/` module with stems.py, heads.py, blocks.py

### Models — UNet Architectures

- [x] **UNET-01**: UNeXt2 migrated to `unet/unext2.py`
- [x] **UNET-02**: FullyConvolutionalMAE migrated to `unet/fcmae.py`
- [x] **UNET-03**: Unet2d migrated to `unet/unet2d.py`
- [x] **UNET-04**: Unet25d migrated to `unet/unet25d.py`
- [x] **UNET-05**: ConvBlock2D/3D migrated to `unet/_layers/`
- [x] **UNET-06**: Forward-pass tests for UNeXt2
- [x] **UNET-07**: FCMAE tests migrated from existing test suite
- [x] **UNET-08**: Unet2d/Unet25d tests migrated and converted to pytest

### Models — Variational Autoencoders

- [x] **VAE-01**: BetaVae25D migrated to `vae/beta_vae_25d.py`
- [x] **VAE-02**: BetaVaeMonai migrated to `vae/beta_vae_monai.py`
- [x] **VAE-03**: Forward-pass tests for both VAE models

### Models — Contrastive Learning

- [x] **CONT-01**: ContrastiveEncoder migrated to `contrastive/encoder.py`
- [x] **CONT-02**: ResNet3dEncoder migrated to `contrastive/resnet3d.py`
- [x] **CONT-03**: Forward-pass tests for contrastive models

### Models — Public API & CI

- [x] **API-01**: `from viscy_models import UNeXt2` works for all 8 model classes
- [x] **API-02**: `uv run --package viscy-models pytest` passes all tests
- [x] **API-03**: CI test matrix updated to include viscy-models
- [x] **API-04**: Root pyproject.toml updated with viscy-models workspace dependency

### Models — Compatibility

- [x] **COMPAT-01**: State dict keys preserved identically for all migrated models
- [x] **COMPAT-02**: Mutable default arguments fixed to tuples in model constructors

## v2.0 Requirements (Complete)

### Shared Infrastructure (viscy-utils)

- [x] **UTIL-PKG-01**: viscy-utils package at `packages/viscy-utils/src/viscy_utils/` with hatchling + uv-dynamic-versioning
- [x] **UTIL-PKG-02**: Shared training infrastructure (trainer, callbacks, evaluation modules)
- [x] **UTIL-PKG-03**: cli_utils.py with format_markdown_table() and load_config()
- [x] **UTIL-PKG-04**: pyyaml added as dependency for config loading
- [x] **UTIL-PKG-05**: Optional dependency groups `[anndata]`, `[eval]`, `[all]`

### DynaCLR Application — Core

- [x] **APP-01-PKG**: applications/dynaclr package at `applications/dynaclr/src/dynaclr/`
- [x] **APP-01-ENG**: ContrastiveModule LightningModule in engine.py
- [x] **APP-01-MM**: MultiModalContrastiveModule for cross-modal distillation
- [x] **APP-01-CLS**: ClassificationModule for downstream classification
- [x] **APP-01-CLI**: dynaclr CLI entry point with LazyCommand pattern

### DynaCLR Application — Evaluation

- [x] **APP-01-EVAL-01**: Linear classifier training CLI (train_linear_classifier.py)
- [x] **APP-01-EVAL-02**: Linear classifier inference CLI (apply_linear_classifier.py)
- [x] **APP-01-EVAL-03**: Dataset discovery for predictions/annotations (dataset_discovery.py)
- [x] **APP-01-EVAL-04**: SLURM prediction script generation (generate_prediction_scripts.py)
- [x] **APP-01-EVAL-05**: Training config generation (generate_train_config.py)
- [x] **APP-01-EVAL-06**: CLI commands registered: train-linear-classifier, apply-linear-classifier

### DynaCLR Application — Examples

- [x] **APP-01-EX-01**: Training configs (fit.yml, predict.yml) with updated class_path imports
- [x] **APP-01-EX-02**: ONNX export config (dynaclr_microglia_onnx.yml) with updated imports
- [x] **APP-01-EX-03**: SLURM scripts (fit_slurm.sh, predict_slurm.sh)
- [x] **APP-01-EX-04**: Infection analysis demo (DynaCLR-DENV-VS-Ph/) with updated Python imports
- [x] **APP-01-EX-05**: Interactive embedding visualizer with updated imports
- [x] **APP-01-EX-06**: Classical sampling pseudo-track generation
- [x] **APP-01-EX-07**: VCP quickstart tutorial (notebook + script) with updated imports

### DynaCLR Application — Dependencies

- [x] **APP-01-DEP-01**: wandb, anndata, natsort in dynaclr [eval] optional dependencies
- [x] **APP-01-DEP-02**: Workspace-level uv configuration updated

## v2.1 Requirements (Complete)

### Training

- [x] **TRAIN-01**: ContrastiveModule completes a training loop via `fast_dev_run` without errors
- [x] **TRAIN-02**: YAML training configs (fit.yml, predict.yml) parse and instantiate correctly with new import paths

### Inference

- [x] **INFER-01**: ContrastiveModule loads a pretrained checkpoint in the modular structure
- [x] **INFER-02**: Prediction (predict step) writes embeddings via EmbeddingWriter callback
- [x] **INFER-03**: Predicted embeddings are an exact match against saved reference outputs

### Test Infrastructure

- [x] **TEST-01**: Training and inference checks are permanent pytest integration tests
- [x] **TEST-02**: Tests are runnable via `uv run --package dynaclr pytest`

## v2.2 Requirements

Requirements for composable, multi-experiment sampling framework. Each maps to roadmap phases.

### Multi-Experiment Configuration

- [ ] **MEXP-01**: User can define an experiment via `ExperimentConfig` dataclass (name, data_path, tracks_path, channel_names, condition_wells, interval_minutes, organelle, date, moi)
- [ ] **MEXP-02**: User can register multiple experiments in an `ExperimentRegistry` with automatic shared/union channel resolution and per-experiment channel index mapping
- [ ] **MEXP-03**: User can specify `training_channels` as "shared", "all", or explicit list — Registry resolves per-experiment channel mapping (active_channels, channel_maps)
- [ ] **MEXP-04**: User can configure experiments via YAML config that Lightning CLI parses into ExperimentRegistry

### Cell Indexing & Lineage

- [ ] **CELL-01**: `MultiExperimentIndex` builds a unified tracks DataFrame across all experiments with columns: experiment, condition, global_track_id, hours_post_infection, well_name, fluorescence_channel
- [ ] **CELL-02**: `MultiExperimentIndex` reconstructs cell lineage from `parent_track_id`/`track_id` — linking daughter cells to their parent track so positives can follow through division
- [ ] **CELL-03**: `MultiExperimentIndex` retains border cells by clamping crop centroids — if a cell's center is within `yx_patch_size/2` of the image boundary, the patch origin is shifted inward so the full patch fits within the image (only excludes cells with centroid completely outside)
- [ ] **CELL-04**: `MultiExperimentIndex` computes `valid_anchors` accounting for variable τ range and lineage continuity (anchor valid if any τ in range has a same-track or daughter-track positive)

### Batch Sampling

- [ ] **SAMP-01**: `FlexibleBatchSampler` restricts each batch to a single experiment when `experiment_aware=True`
- [ ] **SAMP-02**: `FlexibleBatchSampler` balances conditions within each batch (~50% infected/uninfected) when `condition_balanced=True`
- [ ] **SAMP-03**: `FlexibleBatchSampler` concentrates batches around a focal HPI with configurable window when `temporal_enrichment=True`
- [ ] **SAMP-04**: `FlexibleBatchSampler` supports DDP via `set_epoch()` and rank-aware iteration, composing with existing `ShardedDistributedSampler`
- [ ] **SAMP-05**: `FlexibleBatchSampler` supports leaky experiment mixing (configurable fraction of cross-experiment samples)

### Loss Function

- [ ] **LOSS-01**: `NTXentHCL` implements NT-Xent with hard-negative concentration (beta parameter), returning scalar loss with gradients
- [ ] **LOSS-02**: `NTXentHCL` is an `nn.Module` drop-in compatible with `ContrastiveModule(loss_function=NTXentHCL(...))`
- [ ] **LOSS-03**: `NTXentHCL` with `beta=0.0` produces numerically identical results to standard NT-Xent

### Augmentation

- [ ] **AUG-01**: `ChannelDropout` randomly zeros specified channels with configurable probability, compatible with batched (B,C,Z,Y,X) tensors
- [ ] **AUG-02**: `ChannelDropout` integrates into `on_after_batch_transfer` pipeline after the existing scatter/gather augmentation chain
- [ ] **AUG-03**: Variable τ sampling uses exponential decay distribution within `tau_range`, favoring small temporal offsets

### Dataset & DataModule

- [ ] **DATA-01**: `MultiExperimentTripletDataset.__getitems__` returns batch dict compatible with existing `ContrastiveModule.training_step` (anchor, positive keys + norm_meta)
- [ ] **DATA-02**: Positive sampling follows lineage through division events — when anchor track ends at division, daughter track at t+τ is a valid positive
- [ ] **DATA-03**: `MultiExperimentDataModule` wires FlexibleBatchSampler + Dataset + ChannelDropout + ThreadDataLoader with `collate_fn=lambda x: x`
- [ ] **DATA-04**: `MultiExperimentDataModule` performs train/val split by experiment (whole experiments, not FOVs)
- [ ] **DATA-05**: `MultiExperimentDataModule` exposes all sampling/loss/augmentation hyperparameters for Lightning CLI YAML configuration

### Integration

- [ ] **INTG-01**: End-to-end training loop (fast_dev_run) completes with MultiExperimentDataModule + ContrastiveModule + NTXentHCL
- [ ] **INTG-02**: YAML config example for multi-experiment training with all sampling axes enabled

## Future Requirements

Deferred to future milestones. Tracked but not in current roadmap.

### Composable Sampling (v2.3+)

- **MEXP-05**: Zero-padding for missing channels when `training_channels="all"` (experiment lacks a channel → pad with zeros)
- **SAMP-06**: kNN-based hard negative mining in the sampler (currently HCL handles this in the loss)
- **DATA-06**: Prediction/inference mode for MultiExperimentDataModule (sequential per-experiment)
- **INTG-03**: Multi-GPU benchmark comparing FlexibleBatchSampler throughput vs standard shuffled batches

### Applications (v3.0+)

- **APP-02**: applications/Cytoland with VSUNet/FcmaeUNet LightningModules
- **APP-03**: viscy-airtable package abstracted from current Airtable integration

### Hydra Integration (future viscy-hydra package)

- **HYDRA-01**: BaseModelMeta metaclass or `__init_subclass__` registry for model discovery
- **HYDRA-02**: BaseModel(LightningModule) base class with auto Hydra instantiation
- **HYDRA-03**: Hydra ConfigStore integration (optional dependency)
- **HYDRA-04**: `get_model("unext2")` factory function for name-based lookup

### Documentation (deferred from v1.0)

- **DOC-01**: Zensical configuration (`zensical.toml`) at repository root
- **DOC-02**: API reference auto-generated from docstrings
- **DOC-03**: Documentation site structure with navigation
- **DOC-04**: GitHub Pages deployment

### Refactoring

- **REF-01**: GPU transform protocol/mixin (GPUTransformMixin) for interface standardization
- **REF-02**: Split combined.py into combined.py + concat.py
- **REF-03**: Abstract cache interface across Manager.dict, tensorstore, MemoryMappedTensor

## Out of Scope

| Feature | Reason |
|---------|--------|
| Backward-compatible imports | Clean break established in v1.0 |
| Meta-package with re-exports | Decided against, clean break approach |
| Hydra/BaseModel registry in viscy-models | Deferred to separate viscy-hydra package |
| LightningModule wrappers in model package | Training logic stays in applications/ |
| ONNX/TorchScript export | VAE models return SimpleNamespace, incompatible |
| Unified batch structure across pipelines | Different pipelines have fundamentally different batch semantics |
| Split into multiple data packages | Over-fragmentation for 13 modules |
| Zensical documentation | Deferred |
| Modifying triplet.py | Backward compatibility — new composable sampling code in new files only |
| Bag-of-single-channels input | Design decision: 2-channel input (Phase + Fluorescence) with channel dropout |
| kNN sampler for hard negatives | HCL in loss is sufficient; sampler handles experiment/condition/temporal axes |

## Traceability

### v1.0 (18/18 complete)

| Requirement | Phase | Status |
|-------------|-------|--------|
| WORK-00 through WORK-05 | Phase 1 | Complete |
| PKG-01 through PKG-04 | Phase 2 | Complete |
| MIG-01 through MIG-05 | Phase 3 | Complete |
| CI-01, CI-03, CI-04 | Phase 5 | Complete |

### v1.1 (12/12 complete)

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-PKG-01, DATA-PKG-02, DATA-PKG-04 | Phase 6 | Complete |
| DATA-PKG-03, DATA-MIG-01 through DATA-MIG-04 | Phase 7 | Complete |
| DATA-TST-01, DATA-TST-02 | Phase 8 | Complete |
| DATA-CI-01, DATA-CI-02 | Phase 9 | Complete |

### v1.2 (24/24 complete)

| Requirement | Phase | Status |
|-------------|-------|--------|
| MPKG-01 through MPKG-04, UNET-05 | Phase 10 | Complete |
| UNET-01, UNET-02, UNET-06, UNET-07 | Phase 11 | Complete |
| CONT-01 through CONT-03, VAE-01 through VAE-03 | Phase 12 | Complete |
| UNET-03, UNET-04, UNET-08 | Phase 13 | Complete |
| API-01 through API-04, COMPAT-01, COMPAT-02 | Phase 14 | Complete |

### v2.0 (22/22 complete)

| Requirement | Phase | Status |
|-------------|-------|--------|
| UTIL-PKG-01 through UTIL-PKG-05 | Phase 15 | Complete |
| APP-01-PKG, APP-01-ENG, APP-01-MM, APP-01-CLS, APP-01-CLI | Phase 16 | Complete |
| APP-01-EVAL-01 through APP-01-EVAL-06 | Phase 17 | Complete |
| APP-01-EX-01 through APP-01-EX-07 | Phase 17 | Complete |
| APP-01-DEP-01, APP-01-DEP-02 | Phase 17 | Complete |

### v2.1 (7/7 complete)

| Requirement | Phase | Status |
|-------------|-------|--------|
| TRAIN-01 | Phase 18 | Complete |
| TRAIN-02 | Phase 18 | Complete |
| INFER-01 | Phase 19 | Complete |
| INFER-02 | Phase 19 | Complete |
| INFER-03 | Phase 19 | Complete |
| TEST-01 | Phase 19 | Complete |
| TEST-02 | Phase 19 | Complete |

### v2.2 (0/26 pending)

| Requirement | Phase | Status |
|-------------|-------|--------|
| MEXP-01 | — | Pending |
| MEXP-02 | — | Pending |
| MEXP-03 | — | Pending |
| MEXP-04 | — | Pending |
| CELL-01 | — | Pending |
| CELL-02 | — | Pending |
| CELL-03 | — | Pending |
| CELL-04 | — | Pending |
| SAMP-01 | — | Pending |
| SAMP-02 | — | Pending |
| SAMP-03 | — | Pending |
| SAMP-04 | — | Pending |
| SAMP-05 | — | Pending |
| LOSS-01 | — | Pending |
| LOSS-02 | — | Pending |
| LOSS-03 | — | Pending |
| AUG-01 | — | Pending |
| AUG-02 | — | Pending |
| AUG-03 | — | Pending |
| DATA-01 | — | Pending |
| DATA-02 | — | Pending |
| DATA-03 | — | Pending |
| DATA-04 | — | Pending |
| DATA-05 | — | Pending |
| INTG-01 | — | Pending |
| INTG-02 | — | Pending |

**Coverage:**
- v1.0: 18 requirements, 18 complete
- v1.1: 12 requirements, 12 complete
- v1.2: 24 requirements, 24 complete
- v2.0: 22 requirements, 22 complete
- v2.1: 7 requirements, 7 complete
- v2.2: 26 requirements, 0 pending
- **Total: 109 requirements (83 shipped, 26 pending)**

---
*Requirements defined: 2025-01-27*
*Harmonized from modular-data + modular-models branches: 2026-02-16*
*Updated for v2.0 DynaCLR: 2026-02-17*
*Updated for v2.1 DynaCLR Integration Validation: 2026-02-19*
*Updated for v2.2 Composable Sampling Framework: 2026-02-21*
