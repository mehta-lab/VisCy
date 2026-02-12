# Requirements: VisCy Modularization

**Defined:** 2025-01-27
**Core Value:** Independent, reusable subpackages with clean import paths

## v1.0 Requirements (Complete)

### Workspace Foundation

- [x] **WORK-00**: Clean slate setup - wipe repo keeping only LICENSE, CITATION.cff, .gitignore
- [x] **WORK-01**: Virtual workspace root with `[tool.uv.workspace]` and `members = ["packages/*"]`
- [x] **WORK-02**: Shared lockfile (`uv.lock`) at repository root
- [x] **WORK-03**: Python version floor (>=3.11) enforced in root pyproject.toml
- [x] **WORK-04**: Pre-commit hooks configured (ruff, ty) for local development
- [x] **WORK-05**: Shared pytest configuration in root pyproject.toml

### Package Structure

- [x] **PKG-01**: src layout for viscy-transforms (`packages/viscy-transforms/src/viscy_transforms/`)
- [x] **PKG-02**: Package pyproject.toml with hatchling build backend
- [x] **PKG-03**: uv-dynamic-versioning configured for git-based versioning
- [x] **PKG-04**: Package README.md with installation and usage instructions

### Code Migration

- [x] **MIG-01**: All transform modules migrated from `viscy/transforms/` to package
- [x] **MIG-02**: All transform tests migrated from `tests/transforms/` to `packages/viscy-transforms/tests/`
- [x] **MIG-03**: Import path updated to `from viscy_transforms import X`
- [x] **MIG-04**: All migrated tests passing with `uv run --package viscy-transforms pytest`
- [x] **MIG-05**: Original `viscy/transforms/` directory removed

### CI/CD

- [x] **CI-01**: GitHub Actions workflow for testing viscy-transforms package
- [x] **CI-03**: Matrix testing across Python 3.11, 3.12, 3.13
- [x] **CI-04**: Linting via prek (uvx prek) in CI workflows

## v1.1 Requirements

Requirements for milestone v1.1 (Models). Each maps to roadmap phases.

### Models -- Package Infrastructure

- [ ] **MPKG-01**: Package directory `packages/viscy-models/` with src layout (`src/viscy_models/`)
- [ ] **MPKG-02**: pyproject.toml with hatchling, uv-dynamic-versioning, torch/timm/monai/numpy dependencies
- [ ] **MPKG-03**: `uv sync --package viscy-models` succeeds in workspace
- [ ] **MPKG-04**: `_components/` module with stems.py, heads.py, blocks.py extracted from shared code

### Models -- UNet Architectures

- [ ] **UNET-01**: UNeXt2 migrated to `unet/unext2.py` with shared component imports updated
- [ ] **UNET-02**: FullyConvolutionalMAE migrated to `unet/fcmae.py`
- [ ] **UNET-03**: Unet2d migrated to `unet/unet2d.py` (renamed from PascalCase)
- [ ] **UNET-04**: Unet25d migrated to `unet/unet25d.py` (renamed from PascalCase)
- [ ] **UNET-05**: ConvBlock2D/3D migrated to `unet/_layers/` (renamed from PascalCase)
- [ ] **UNET-06**: Forward-pass tests for UNeXt2 (NEW -- currently missing)
- [ ] **UNET-07**: FCMAE tests migrated from existing test suite
- [ ] **UNET-08**: Unet2d/Unet25d tests migrated and converted from unittest to pytest

### Models -- Variational Autoencoders

- [ ] **VAE-01**: BetaVae25D migrated to `vae/beta_vae_25d.py`
- [ ] **VAE-02**: BetaVaeMonai migrated to `vae/beta_vae_monai.py`
- [ ] **VAE-03**: Forward-pass tests for both VAE models (NEW -- currently missing)

### Models -- Contrastive Learning

- [ ] **CONT-01**: ContrastiveEncoder migrated to `contrastive/encoder.py`
- [ ] **CONT-02**: ResNet3dEncoder migrated to `contrastive/resnet3d.py`
- [ ] **CONT-03**: Forward-pass tests for contrastive models (NEW -- currently missing)

### Models -- Public API & CI

- [ ] **API-01**: `from viscy_models import UNeXt2` works for all 8 model classes
- [ ] **API-02**: `uv run --package viscy-models pytest` passes all tests
- [ ] **API-03**: CI test matrix updated to include viscy-models
- [ ] **API-04**: Root pyproject.toml updated with viscy-models workspace dependency

### Models -- Compatibility

- [ ] **COMPAT-01**: State dict keys preserved identically for all migrated models
- [ ] **COMPAT-02**: Mutable default arguments fixed to tuples in model constructors

## Future Requirements

Deferred to future milestones. Tracked but not in current roadmap.

### Hydra Integration (future viscy-hydra package)

- **HYDRA-01**: BaseModelMeta metaclass or `__init_subclass__` registry for model discovery
- **HYDRA-02**: BaseModel(LightningModule) base class with auto Hydra instantiation
- **HYDRA-03**: Hydra ConfigStore integration (optional dependency)
- **HYDRA-04**: `get_model("unext2")` factory function for name-based lookup

### Applications (future milestone)

- **APP-01**: applications/DynaCLR with ContrastiveModule LightningModule
- **APP-02**: applications/Cytoland with VSUNet/FcmaeUNet LightningModules

### Documentation (deferred from v1.0)

- **DOC-01**: Zensical configuration (`zensical.toml`) at repository root
- **DOC-02**: Documentation site structure with navigation
- **DOC-03**: API reference auto-generated from docstrings
- **DOC-04**: GitHub Pages deployment

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Hydra/BaseModel registry in viscy-models | Deferred to separate viscy-hydra package |
| LightningModule wrappers | Training logic stays in applications/ |
| Backward-compatible imports | Clean break approach |
| viscy-data extraction | Separate milestone |
| Documentation (Zensical) | Deferred from v1.0 |
| ONNX/TorchScript export | VAE models return SimpleNamespace, incompatible |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

### v1.0 Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| WORK-00 | Phase 1 | Complete |
| WORK-01 | Phase 1 | Complete |
| WORK-02 | Phase 1 | Complete |
| WORK-03 | Phase 1 | Complete |
| WORK-04 | Phase 1 | Complete |
| WORK-05 | Phase 1 | Complete |
| PKG-01 | Phase 2 | Complete |
| PKG-02 | Phase 2 | Complete |
| PKG-03 | Phase 2 | Complete |
| PKG-04 | Phase 2 | Complete |
| MIG-01 | Phase 3 | Complete |
| MIG-02 | Phase 3 | Complete |
| MIG-03 | Phase 3 | Complete |
| MIG-04 | Phase 3 | Complete |
| MIG-05 | Phase 3 | Complete |
| CI-01 | Phase 5 | Complete |
| CI-03 | Phase 5 | Complete |
| CI-04 | Phase 5 | Complete |

### v1.1 Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| MPKG-01 | Phase 6 | Pending |
| MPKG-02 | Phase 6 | Pending |
| MPKG-03 | Phase 6 | Pending |
| MPKG-04 | Phase 6 | Pending |
| UNET-05 | Phase 6 | Pending |
| COMPAT-02 | Phase 6 | Pending |
| UNET-01 | Phase 7 | Pending |
| UNET-02 | Phase 7 | Pending |
| UNET-06 | Phase 7 | Pending |
| UNET-07 | Phase 7 | Pending |
| CONT-01 | Phase 8 | Pending |
| CONT-02 | Phase 8 | Pending |
| CONT-03 | Phase 8 | Pending |
| VAE-01 | Phase 8 | Pending |
| VAE-02 | Phase 8 | Pending |
| VAE-03 | Phase 8 | Pending |
| UNET-03 | Phase 9 | Pending |
| UNET-04 | Phase 9 | Pending |
| UNET-08 | Phase 9 | Pending |
| API-01 | Phase 10 | Pending |
| API-02 | Phase 10 | Pending |
| API-03 | Phase 10 | Pending |
| API-04 | Phase 10 | Pending |
| COMPAT-01 | Phase 10 | Pending |

**Coverage:**
- v1.0 requirements: 18 total, 18 mapped (complete)
- v1.1 requirements: 24 total, 24 mapped
- Unmapped: 0

---
*Requirements defined: 2025-01-27*
*Last updated: 2026-02-12 after v1.1 roadmap creation*
