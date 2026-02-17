# VisCy Modularization

## What This Is

Restructuring VisCy from a monolithic package into a uv workspace monorepo. This enables reusing transforms, dataloaders, and models in downstream projects without requiring the entire VisCy package as a dependency. Four shared packages have been extracted: `viscy-transforms` (v1.0), `viscy-data` (v1.1), `viscy-models` (v1.2), and `viscy-utils` (v2.0). The first application, `applications/dynacrl`, composes these packages into a self-contained DynaCLR application.

## Core Value

**Independent, reusable subpackages with clean import paths.** Users can `pip install viscy-transforms`, `pip install viscy-data`, `pip install viscy-models`, or `pip install viscy-utils` and use clean imports without pulling in the entire VisCy ecosystem. Applications compose these packages into domain-specific tools (e.g., `pip install dynacrl`).

## Current Milestone: v2.0 Applications & Shared Infrastructure (In Progress)

**Goal:** Extract shared ML infrastructure (`viscy-utils`) and application-level LightningModules into independent packages, composing the extracted model and data subpackages.

**Shipped (v2.0):**
- `viscy-utils` package — shared training infrastructure (trainer, callbacks, evaluation, cli_utils)
- `applications/dynacrl` — DynaCLR application with ContrastiveModule, CLI, evaluation, examples

**Remaining (v2.0+):**
- `applications/Cytoland` with VSUNet/FcmaeUNet LightningModules
- `viscy-airtable` package abstracted from current Airtable integration
- Hydra configuration infrastructure (viscy-hydra or integrated)

## Requirements

### Validated

- uv workspace scaffolding with `packages/` directory structure — v1.0
- `viscy-transforms` package extracted with src layout (16 modules, 44 exports) — v1.0
- Import path: `from viscy_transforms import X` (clean break) — v1.0
- hatchling build backend with uv-dynamic-versioning — v1.0
- All existing transform tests passing in new structure — v1.0
- CI for monorepo (test matrix + lint) — v1.0
- `viscy-data` package extracted with src layout (15 modules, 4015 LOC) — v1.1
- All 13 data modules migrated with clean import paths — v1.1
- Import path: `from viscy_data import X` (45 public exports) — v1.1
- No dependency on viscy-transforms (BatchedCenterSpatialCropd in _utils.py) — v1.1
- Optional dependency groups: `[triplet]`, `[livecell]`, `[mmap]`, `[all]` — v1.1
- Shared utilities extracted from hcs.py into _utils.py — v1.1
- All existing data tests passing (71 tests) — v1.1
- Tiered CI for viscy-data (3x3 base + 1x1 extras) — v1.1
- `viscy-models` package with src layout and function-based organization — v1.2
- All 8 architectures migrated (UNeXt2, FCMAE, ContrastiveEncoder, ResNet3dEncoder, BetaVae25D, BetaVaeMonai, Unet2d, Unet25d) — v1.2
- Shared components in `_components/` (stems, heads, blocks) — v1.2
- Full test coverage for all models (existing + new forward-pass tests) — v1.2
- Import path: `from viscy_models import UNeXt2` (clean break) — v1.2
- State dict key compatibility preserved — v1.2
- CI includes viscy-models in test matrix — v1.2

### Active

- `viscy-utils` package extracted with shared ML infrastructure — v2.0
- `applications/dynacrl` with ContrastiveModule, MultiModalContrastiveModule, ClassificationModule — v2.0
- `dynaclr` CLI with `train-linear-classifier` and `apply-linear-classifier` commands — v2.0
- Evaluation scripts for linear classifiers on cell embeddings — v2.0
- Examples, tutorials, and training configs migrated to `applications/dynacrl/examples/` — v2.0
- `cli_utils.py` with `format_markdown_table()` and `load_config()` — v2.0

### Out of Scope

- Meta-package with re-exports — decided against, clean break approach
- Backward-compatible imports — not maintaining
- Zensical documentation / GitHub Pages — deferred
- ONNX/TorchScript export — VAE models return SimpleNamespace, incompatible

## Context

**Design doc:** https://github.com/mehta-lab/VisCy/issues/353

**Current state (after v2.0 DynaCLR):**
- uv workspace monorepo with 4 shared packages + 1 application:
  - `packages/viscy-transforms/` — 16 transform modules, 44 exports
  - `packages/viscy-data/` — 15 data modules, 45 exports, 4015 LOC source + 671 LOC tests
  - `packages/viscy-models/` — 8 architectures in unet/, vae/, contrastive/ with shared _components/
  - `packages/viscy-utils/` — shared ML infrastructure (trainer, callbacks, evaluation, cli_utils)
  - `applications/dynacrl/` — DynaCLR application (engine, CLI, evaluation, examples)
- CI: test.yml (viscy-transforms 3x3, viscy-data 3x3 + extras 1x1, viscy-models 3x3) + lint.yml
- Python >=3.11, hatchling + uv-dynamic-versioning
- Original code on `main` branch for reference

**Architecture reference:**
- `viscy/data/README.md` documents module inventory, class hierarchy, training pipeline mapping
- `.planning/codebase/` contains architecture, conventions, structure analysis

## Constraints

- **Package naming**: hyphen for package name, underscore for import
- **Python version**: >=3.11
- **Build system**: hatchling with uv-dynamic-versioning
- **Layout**: src layout (`packages/*/src/*/`, `applications/*/src/*/`)
- **Tooling**: uv only
- **No cross-package dependencies between data, transforms, and models**
- **Applications compose packages**: applications depend on shared packages, not the reverse

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Clean break on imports | Simpler architecture, no re-export complexity | Good |
| hatchling over setuptools | Modern, faster, better uv integration | Good |
| src layout | Prevents import confusion during development | Good |
| Tests inside packages | Isolated testing, `uv run --package` workflow | Good |
| No viscy-transforms dep in data | Transforms separate from data | Good |
| No viscy-transforms/data dep in models | Keep packages loosely coupled | Good |
| Optional dependency groups for data | Heavy deps as extras, lean base install | Good |
| Extract shared utils from hcs.py | Prevent dual-role module anti-pattern | Good |
| BatchedCenterSpatialCropd in _utils.py | CenterSpatialCropd can't handle batch dim | Good |
| Lazy imports for optional deps | try/except at module level, guard in __init__ | Good |
| Flat public API | MONAI pattern, consistent across packages | Good |
| combined.py preserved as-is | No split per REF-02 deferral | Good |
| Pure nn.Module in viscy-models | No Lightning/Hydra coupling; maximum reusability | Good |
| Function-based grouping (unet/, vae/, contrastive/) | Clean organization for 8+ models with shared components | Good |
| State dict key compatibility | Non-negotiable for checkpoint loading | Good |
| Applications compose packages | dynacrl depends on viscy-data, viscy-models, viscy-transforms, viscy-utils | Good |
| LazyCommand CLI pattern | Defer heavy imports until invocation; graceful fallback on missing extras | Good |
| Evaluation outside package src/ | Evaluation scripts are standalone; CLI wires them via sys.path | Good |

---
*Last updated: 2026-02-17 after v2.0 DynaCLR application migration*
