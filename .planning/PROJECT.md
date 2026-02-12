# VisCy Modularization

## What This Is

Restructuring VisCy from a monolithic package into a uv workspace monorepo. This enables reusing transforms, dataloaders, and models in downstream projects without requiring the entire VisCy package as a dependency. The second milestone extracts `viscy-models` as an independent package containing all network architectures, a Hydra-ready `BaseModel` base class with a metaclass registry pattern, and comprehensive test coverage.

## Core Value

**Independent, reusable subpackages with clean import paths.** Users can `pip install viscy-models` and use `from viscy_models import UNeXt2` without pulling in the entire VisCy ecosystem.

## Current Milestone: v1.1 Models

**Goal:** Extract all network architectures into `viscy-models` with a `BaseModelMeta` registry for Hydra config discovery.

**Target features:**
- `viscy-models` package with all architectures (UNeXt2, FCMAE, ContrastiveEncoder, BetaVAE25D, BetaVaeMonai)
- `BaseModelMeta` metaclass providing Hydra registry (models register by name)
- `BaseModel(LightningModule, metaclass=BaseModelMeta)` base class
- Full test coverage (migrate existing + write missing tests)
- Independent of viscy-transforms (torch/monai deps only)

## Requirements

### Validated

- ✓ uv workspace scaffolding with `packages/` directory structure — v1.0
- ✓ `viscy-transforms` package extracted with src layout — v1.0
- ✓ Import path: `from viscy_transforms import X` (clean break) — v1.0
- ✓ hatchling build backend with uv-dynamic-versioning — v1.0
- ✓ All existing transform tests passing in new structure — v1.0
- ✓ CI updated for monorepo structure — v1.0

### Active

- [ ] `viscy-models` package with src layout (`packages/viscy-models/src/viscy_models/`)
- [ ] All architectures migrated: UNeXt2, FCMAE, ContrastiveEncoder, BetaVAE25D, BetaVaeMonai
- [ ] `BaseModelMeta` metaclass with Hydra registry pattern
- [ ] `BaseModel(LightningModule, metaclass=BaseModelMeta)` base class
- [ ] Full test coverage: existing tests migrated + new tests for untested models
- [ ] Import path: `from viscy_models import UNeXt2` (clean break)
- [ ] CI includes viscy-models in test matrix

### Out of Scope

- Extracting data packages (viscy-data) — future milestone
- Application-level LightningModules (ContrastiveModule, translation engines) — these move to applications/Cytoland and applications/DynaCLR in a future milestone
- Backward-compatible imports (`from viscy.unet.networks import X`) — not maintaining
- Documentation (Zensical + GitHub Pages) — deferred from v1.0, separate effort
- Hydra structured configs auto-generated from signatures — future enhancement on top of registry

## Context

**Design doc:** https://github.com/mehta-lab/VisCy/issues/353

**Reference implementations:**
- biahub Zensical setup: https://github.com/czbiohub-sf/biahub
- iohub pyproject.toml: modern hatchling + uv-dynamic-versioning pattern

**Current state (post v1.0):**
- uv workspace with `viscy-transforms` at `packages/viscy-transforms/`
- Root `viscy` umbrella package with dynamic versioning
- Models still in monolithic `viscy/unet/networks/` and `viscy/representation/`
- No tests for UNeXt2, ContrastiveEncoder, or BetaVAE architectures

**Architecture vision:**
- `viscy-models`: Pure architectures (nn.Module) + BaseModel base class
- Applications (Cytoland, DynaCLR): LightningModules composing viscy-models with training logic
- Models self-register via BaseModelMeta for Hydra discovery

## Constraints

- **Package naming**: `viscy-models` (hyphen) as package name, `viscy_models` (underscore) as import
- **Python version**: >=3.11 (matching current VisCy)
- **Build system**: hatchling with uv-dynamic-versioning (following viscy-transforms pattern)
- **Layout**: src layout required (`packages/viscy-models/src/viscy_models/`)
- **Independence**: viscy-models must NOT depend on viscy-transforms
- **Tooling**: uv only, no pip/setuptools for package management

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Clean break on imports | Simpler architecture, no re-export complexity | ✓ Good |
| hatchling over setuptools | Modern, faster, better uv integration | ✓ Good |
| src layout | Prevents import confusion during development | ✓ Good |
| Tests inside packages | Isolated testing, `uv run --package` workflow | ✓ Good |
| Registry metaclass for Hydra | Models self-register by name, clean Hydra integration | — Pending |
| Architectures only in viscy-models | Training logic (LightningModules) goes to applications | — Pending |
| viscy-models independent of viscy-transforms | Keep packages loosely coupled | — Pending |

---
*Last updated: 2026-02-12 after milestone v1.1 started*
