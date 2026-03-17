# VisCy Modularization

## What This Is

Restructuring VisCy from a monolithic package into a uv workspace monorepo. This enables reusing transforms, dataloaders, and models in downstream projects without requiring the entire VisCy package as a dependency. The second milestone extracts `viscy-models` as an independent package containing all 8 network architectures as pure nn.Modules, organized by function (unet/, vae/, contrastive/) with shared components factored into a `components/` module.

## Core Value

**Independent, reusable subpackages with clean import paths.** Users can `pip install viscy-models` and use `from viscy_models import UNeXt2` without pulling in the entire VisCy ecosystem.

## Current Milestone: v1.1 Models

**Goal:** Extract all network architectures into `viscy-models` as pure nn.Modules with comprehensive test coverage.

**Target features:**
- `viscy-models` package with 8 architectures organized by function (unet/, vae/, contrastive/)
- Shared components extracted to `components/` (stems, heads, decoder blocks)
- Full test coverage: migrate existing + write new for UNeXt2, ContrastiveEncoder, BetaVAE
- Independent of viscy-transforms and lightning (torch/timm/monai deps only)
- State dict key compatibility preserved for checkpoint loading

## Requirements

### Validated

- ✓ uv workspace scaffolding with `packages/` directory structure — v1.0
- ✓ `viscy-transforms` package extracted with src layout — v1.0
- ✓ Import path: `from viscy_transforms import X` (clean break) — v1.0
- ✓ hatchling build backend with uv-dynamic-versioning — v1.0
- ✓ All existing transform tests passing in new structure — v1.0
- ✓ CI updated for monorepo structure — v1.0

### Active

- [ ] `viscy-models` package with src layout and function-based organization
- [ ] All 8 architectures migrated: UNeXt2, FCMAE, ContrastiveEncoder, ResNet3dEncoder, BetaVae25D, BetaVaeMonai, Unet2d, Unet25d
- [ ] Shared components extracted to `components/` (stems, heads, blocks)
- [ ] Full test coverage: existing tests migrated + new tests for untested models
- [ ] Import path: `from viscy_models import UNeXt2` (clean break)
- [ ] State dict key compatibility preserved
- [ ] CI includes viscy-models in test matrix

### Out of Scope

- Hydra/BaseModel registry infrastructure — deferred to future `viscy-hydra` package
- BaseModelMeta metaclass — deferred to future `viscy-hydra` package
- Extracting data packages (viscy-data) — future milestone
- Application-level LightningModules — move to applications/ in future milestone
- Backward-compatible imports — not maintaining
- Documentation (Zensical + GitHub Pages) — deferred from v1.0

## Context

**Design doc:** https://github.com/mehta-lab/VisCy/issues/353

**Reference implementations:**
- flowbench BaseModel: `/home/eduardo.hirata/repos/flowbench/src/models/base_model.py` (future reference for viscy-hydra)
- lightning-hydra-template: https://github.com/ashleve/lightning-hydra-template (future reference)
- iohub pyproject.toml: modern hatchling + uv-dynamic-versioning pattern

**Current state (post v1.0):**
- uv workspace with `viscy-transforms` at `packages/viscy-transforms/`
- Root `viscy` umbrella package with dynamic versioning
- Models in monolithic `viscy/unet/networks/` and `viscy/representation/`
- 14+ shared components in unext2.py used by fcmae, contrastive, vae
- No tests for UNeXt2, ContrastiveEncoder, or BetaVAE

**Architecture vision:**
- `viscy-models`: Pure nn.Module architectures (this milestone)
- `viscy-hydra`: BaseModel, BaseModelMeta, Hydra config utilities (future milestone)
- Applications (Cytoland, DynaCLR): LightningModules composing models with training logic (future milestone)

## Constraints

- **Package naming**: `viscy-models` (hyphen) as package name, `viscy_models` (underscore) as import
- **Python version**: >=3.11 (matching current VisCy)
- **Build system**: hatchling with uv-dynamic-versioning (following viscy-transforms pattern)
- **Layout**: src layout required (`packages/viscy-models/src/viscy_models/`)
- **Independence**: viscy-models must NOT depend on viscy-transforms or lightning
- **Dependencies**: torch, timm, monai, numpy only
- **Tooling**: uv only, no pip/setuptools for package management

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Clean break on imports | Simpler architecture, no re-export complexity | ✓ Good |
| hatchling over setuptools | Modern, faster, better uv integration | ✓ Good |
| src layout | Prevents import confusion during development | ✓ Good |
| Tests inside packages | Isolated testing, `uv run --package` workflow | ✓ Good |
| Pure nn.Module in viscy-models | No Lightning/Hydra coupling; maximum reusability | — Pending |
| Hydra infra in separate package | Keeps model package lightweight; Hydra optional for consumers | — Pending |
| Function-based grouping (unet/, vae/, contrastive/) | Clean organization for 8+ models with shared components | — Pending |
| viscy-models independent of viscy-transforms | Keep packages loosely coupled | — Pending |

---
*Last updated: 2026-02-12 after requirements definition*
