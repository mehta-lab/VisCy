# VisCy Modularization

## What This Is

Restructuring VisCy from a monolithic package into a uv workspace monorepo. This enables reusing transforms, dataloaders, and models in downstream projects without requiring the entire VisCy package as a dependency. Milestone 1 extracted `viscy-transforms`; Milestone 2 extracts `viscy-data` as the second independent subpackage.

## Core Value

**Independent, reusable subpackages with clean import paths.** Users can `pip install viscy-transforms` or `pip install viscy-data` and use clean imports without pulling in the entire VisCy ecosystem.

## Requirements

### Validated

- uv workspace scaffolding with `packages/` directory structure — v1.0
- `viscy-transforms` package extracted with src layout — v1.0
- Import path: `from viscy_transforms import X` (clean break) — v1.0
- hatchling build backend with uv-dynamic-versioning — v1.0
- All existing transform tests passing in new structure — v1.0
- CI updated for monorepo structure (9-job test matrix + lint) — v1.0

### Active

- [ ] `viscy-data` package extracted with src layout (`packages/viscy-data/src/viscy_data/`)
- [ ] All 13 data modules migrated (hcs, gpu_aug, triplet, livecell, ctmc, mmap_cache, cell_classification, cell_division_triplet, segmentation, combined, typing, select, distributed)
- [ ] Import path: `from viscy_data import X` (clean break)
- [ ] No dependency on viscy-transforms (remove BatchedCenterSpatialCropd from triplet.py, assert batch shape)
- [ ] Optional dependency groups: `[triplet]`, `[livecell]`, `[mmap]`, `[all]`
- [ ] Shared utilities extracted from hcs.py into _utils.py
- [ ] All existing data tests passing in new structure
- [ ] CI workflows extended for viscy-data package

### Out of Scope

- Extracting viscy-models, viscy-airtable — future milestones
- Meta-package with re-exports — decided against, clean break approach
- Backward-compatible imports (`from viscy.data import X`) — not maintaining
- Zensical documentation / GitHub Pages — deferred
- Fixing broken imports in applications/examples — deferred
- Hydra integration — per design doc
- GPU transform unification (GPUTransformMixin) — future refactor after extraction

## Context

**Design doc:** https://github.com/mehta-lab/VisCy/issues/353

**Reference implementations:**
- biahub Zensical setup: https://github.com/czbiohub-sf/biahub (zensical.toml, docs workflow)
- iohub pyproject.toml: modern hatchling + uv-dynamic-versioning pattern

**Current state (after v1.0):**
- uv workspace monorepo with `packages/viscy-transforms/` extracted
- `viscy/data/` has 13 modules with comprehensive architecture documentation (README.md)
- Data modules have complex dependency graph: iohub, monai, tensorstore, tensordict, pycocotools
- Three distinct training pipeline patterns (FCMAE, translation, DynaCLR) with different data flows
- Original code available on `main` branch for reference/copying

**Architecture reference:**
- `viscy/data/README.md` documents full module inventory, class hierarchy, dependency graph, training pipeline mapping, GPU transform patterns, and conversion notes

## Constraints

- **Package naming**: `viscy-data` (hyphen) as package name, `viscy_data` (underscore) as import
- **Python version**: >=3.11 (matching current VisCy and viscy-transforms)
- **Build system**: hatchling with uv-dynamic-versioning (following viscy-transforms pattern)
- **Layout**: src layout required (`packages/viscy-data/src/viscy_data/`)
- **Tooling**: uv only, no pip/setuptools for package management
- **No cross-package dependency**: viscy-data must NOT depend on viscy-transforms

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Clean break on imports | Simpler architecture, no re-export complexity | ✓ Good |
| hatchling over setuptools | Modern, faster, better uv integration | ✓ Good |
| src layout | Prevents import confusion during development | ✓ Good |
| Tests inside packages | Isolated testing, `uv run --package` workflow | ✓ Good |
| No viscy-transforms dep in data | Transforms separate from data; assert batch shape | — Pending |
| Optional dependency groups | Heavy deps (tensorstore, tensordict, pycocotools) as extras | — Pending |
| Extract shared utils from hcs.py | Prevent hcs.py from being both module and utility library | — Pending |

## Current Milestone: v1.1 Extract viscy-data

**Goal:** Extract all 13 data modules into an independent `viscy-data` package with optional dependency groups and no cross-package dependencies.

**Target features:**
- `viscy-data` package at `packages/viscy-data/src/viscy_data/`
- All data modules migrated with updated imports
- Optional dependency groups for heavy dependencies
- Remove viscy-transforms coupling (assert batch shape instead)
- CI workflows extended for viscy-data
- All existing data tests passing

---
*Last updated: 2026-02-13 after milestone v1.1 start*
