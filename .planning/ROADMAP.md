# Roadmap: VisCy Modularization

## Milestones

- Shipped **v1.0 Transforms & Monorepo Skeleton** - Phases 1-5 (shipped 2026-01-29)
- Current **v1.1 Models** - Phases 6-10 (in progress)

## Phases

<details>
<summary>v1.0 Transforms & Monorepo Skeleton (Phases 1-5) - SHIPPED 2026-01-29</summary>

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

### v1.1 Models (Phases 6-10)

**Milestone Goal:** Extract all 8 network architectures into `viscy-models` as pure nn.Modules with shared components, comprehensive tests, and clean public API.

- [x] **Phase 6: Package Scaffold & Shared Components** - viscy-models package structure with extracted shared layers
- [x] **Phase 7: Core UNet Models** - UNeXt2 and FCMAE migration with shared component validation
- [x] **Phase 8: Representation Models** - Contrastive encoders and VAE architectures
- [x] **Phase 9: Legacy UNet Models** - Unet2d and Unet25d migration
- [x] **Phase 10: Public API & CI Integration** - Clean imports, full test suite, CI matrix, checkpoint compatibility

## Phase Details

### Phase 6: Package Scaffold & Shared Components
**Goal**: Users can install viscy-models and shared architectural components are available for model implementations
**Depends on**: Phase 5 (v1.0 CI infrastructure)
**Requirements**: MPKG-01, MPKG-02, MPKG-03, MPKG-04, UNET-05, COMPAT-02
**Success Criteria** (what must be TRUE):
  1. `packages/viscy-models/src/viscy_models/` directory exists with src layout and `__init__.py`
  2. `uv sync --package viscy-models` succeeds in the workspace without errors
  3. `viscy_models._components` subpackage contains stems.py, heads.py, and blocks.py with extracted shared code
  4. ConvBlock2D/3D layers exist in `viscy_models.unet._layers` and are importable
  5. All model constructors use immutable defaults (tuples instead of mutable lists/dicts)
**Plans**: 3 plans

Plans:
- [x] 06-01-PLAN.md -- Package scaffold, pyproject.toml, workspace registration
- [x] 06-02-PLAN.md -- Extract shared _components (stems, heads, blocks) with tests
- [x] 06-03-PLAN.md -- Migrate ConvBlock2D/3D to unet/_layers with tests

### Phase 7: Core UNet Models
**Goal**: UNeXt2 and FCMAE are importable from viscy-models with forward-pass tests proving correctness
**Depends on**: Phase 6
**Requirements**: UNET-01, UNET-02, UNET-06, UNET-07
**Success Criteria** (what must be TRUE):
  1. `from viscy_models.unet import UNeXt2` works and the model produces correct output shapes for representative inputs
  2. `from viscy_models.unet import FullyConvolutionalMAE` works and the model produces correct output shapes
  3. UNeXt2 forward-pass test covers multiple configurations (2D/3D, varying channel counts)
  4. Existing FCMAE tests pass after migration to the new package location
**Plans**: 2 plans

Plans:
- [x] 07-01-PLAN.md -- Migrate UNeXt2 model with new forward-pass tests (6 tests)
- [x] 07-02-PLAN.md -- Migrate FCMAE model with 11 existing tests and finalize unet exports

### Phase 8: Representation Models
**Goal**: All contrastive and VAE models are importable from viscy-models with forward-pass tests
**Depends on**: Phase 6
**Requirements**: CONT-01, CONT-02, CONT-03, VAE-01, VAE-02, VAE-03
**Success Criteria** (what must be TRUE):
  1. `from viscy_models.contrastive import ContrastiveEncoder, ResNet3dEncoder` works and both produce embedding outputs
  2. `from viscy_models.vae import BetaVae25D, BetaVaeMonai` works and both produce reconstruction + latent outputs
  3. Forward-pass tests exist for ContrastiveEncoder and ResNet3dEncoder with representative input shapes
  4. Forward-pass tests exist for BetaVae25D and BetaVaeMonai verifying output structure (reconstruction, mu, logvar)
**Plans**: 2 plans

Plans:
- [x] 08-01-PLAN.md -- Migrate ContrastiveEncoder and ResNet3dEncoder with forward-pass tests (5 tests)
- [x] 08-02-PLAN.md -- Migrate BetaVae25D and BetaVaeMonai with forward-pass tests (4 tests)

### Phase 9: Legacy UNet Models
**Goal**: Unet2d and Unet25d are importable from viscy-models with migrated test coverage
**Depends on**: Phase 6
**Requirements**: UNET-03, UNET-04, UNET-08
**Success Criteria** (what must be TRUE):
  1. `from viscy_models.unet import Unet2d, Unet25d` works and both produce correct output shapes
  2. Existing unittest-style tests are migrated to pytest and pass in the new package
  3. File naming follows snake_case convention (unet2d.py, unet25d.py)
**Plans**: 1 plan

Plans:
- [x] 09-01-PLAN.md -- Migrate Unet2d and Unet25d with pytest test coverage

### Phase 10: Public API & CI Integration
**Goal**: Users can `from viscy_models import ModelName` for all 8 models, with CI verifying the full package
**Depends on**: Phases 7, 8, 9
**Requirements**: API-01, API-02, API-03, API-04, COMPAT-01
**Success Criteria** (what must be TRUE):
  1. `from viscy_models import UNeXt2, FullyConvolutionalMAE, ContrastiveEncoder, ResNet3dEncoder, BetaVae25D, BetaVaeMonai, Unet2d, Unet25d` all work from the top-level package
  2. `uv run --package viscy-models pytest` passes the complete test suite
  3. CI test matrix includes viscy-models alongside viscy-transforms
  4. State dict keys for all migrated models match their original monolithic counterparts exactly
  5. Root pyproject.toml lists viscy-models as a workspace dependency
**Plans**: 1 plan

Plans:
- [x] 10-01-PLAN.md -- Public API re-exports, state dict compatibility tests, CI matrix update

## Progress

**Execution Order:**
Phases 6 -> 7 -> 8 -> 9 -> 10 (Phases 7, 8, 9 can execute after 6; 10 depends on all)

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Workspace Foundation | v1.0 | 2/2 | Complete | 2026-01-28 |
| 2. Package Structure | v1.0 | 1/1 | Complete | 2026-01-28 |
| 3. Code Migration | v1.0 | 3/3 | Complete | 2026-01-28 |
| 4. Documentation | v1.0 | 0/TBD | Deferred | - |
| 5. CI/CD | v1.0 | 1/1 | Complete | 2026-01-29 |
| 6. Package Scaffold & Shared Components | v1.1 | 3/3 | Complete | 2026-02-12 |
| 7. Core UNet Models | v1.1 | 2/2 | Complete | 2026-02-12 |
| 8. Representation Models | v1.1 | 2/2 | Complete | 2026-02-13 |
| 9. Legacy UNet Models | v1.1 | 1/1 | Complete | 2026-02-13 |
| 10. Public API & CI Integration | v1.1 | 1/1 | Complete | 2026-02-13 |

---
*Roadmap created: 2025-01-27*
*v1.1 phases added: 2026-02-12*
