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

- [x] **APP-01-PKG**: applications/dynacrl package at `applications/dynacrl/src/dynacrl/`
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

- [x] **APP-01-DEP-01**: wandb, anndata, natsort in dynacrl [eval] optional dependencies
- [x] **APP-01-DEP-02**: Workspace-level uv configuration updated

## v2.1 Requirements

Requirements for DynaCLR integration validation. Each maps to roadmap phases.

### Training

- [ ] **TRAIN-01**: ContrastiveModule completes a training loop via `fast_dev_run` without errors
- [ ] **TRAIN-02**: YAML training configs (fit.yml, predict.yml) parse and instantiate correctly with new import paths

### Inference

- [ ] **INFER-01**: ContrastiveModule loads a pretrained checkpoint in the modular structure
- [ ] **INFER-02**: Prediction (predict step) writes embeddings via EmbeddingWriter callback
- [ ] **INFER-03**: Predicted embeddings are an exact match against saved reference outputs

### Test Infrastructure

- [ ] **TEST-01**: Training and inference checks are permanent pytest integration tests
- [ ] **TEST-02**: Tests are runnable via `uv run --package dynacrl pytest`

## Future Requirements

Deferred to v2.0+ milestones. Tracked but not in current roadmap.

### Applications (v2.0+)

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

### v2.1 (0/7 complete)

| Requirement | Phase | Status |
|-------------|-------|--------|
| TRAIN-01 | — | Pending |
| TRAIN-02 | — | Pending |
| INFER-01 | — | Pending |
| INFER-02 | — | Pending |
| INFER-03 | — | Pending |
| TEST-01 | — | Pending |
| TEST-02 | — | Pending |

**Coverage:**
- v1.0: 18 requirements, 18 complete
- v1.1: 12 requirements, 12 complete
- v1.2: 24 requirements, 24 complete
- v2.0: 22 requirements, 22 complete
- v2.1: 7 requirements, 0 complete
- **Total: 83 requirements (76 shipped, 7 pending)**

---
*Requirements defined: 2025-01-27*
*Harmonized from modular-data + modular-models branches: 2026-02-16*
*Updated for v2.0 DynaCLR: 2026-02-17*
*Updated for v2.1 DynaCLR Integration Validation: 2026-02-19*
