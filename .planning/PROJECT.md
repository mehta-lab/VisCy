# VisCy Modularization

## What This Is

Restructuring VisCy from a monolithic package into a uv workspace monorepo. This enables reusing transforms, dataloaders, and models in downstream projects without requiring the entire VisCy package as a dependency. Two subpackages are now extracted: `viscy-transforms` (v1.0) and `viscy-data` (v1.1).

## Core Value

**Independent, reusable subpackages with clean import paths.** Users can `pip install viscy-transforms` or `pip install viscy-data` and use clean imports without pulling in the entire VisCy ecosystem.

## Requirements

### Validated

- uv workspace scaffolding with `packages/` directory structure — v1.0
- `viscy-transforms` package extracted with src layout — v1.0
- Import path: `from viscy_transforms import X` (clean break) — v1.0
- hatchling build backend with uv-dynamic-versioning — v1.0
- All existing transform tests passing in new structure — v1.0
- CI for monorepo (9-job test matrix + lint) — v1.0
- `viscy-data` package extracted with src layout (15 modules, 4015 LOC) — v1.1
- All 13 data modules migrated with clean import paths — v1.1
- Import path: `from viscy_data import X` (45 public exports) — v1.1
- No dependency on viscy-transforms (BatchedCenterSpatialCropd in _utils.py) — v1.1
- Optional dependency groups: `[triplet]`, `[livecell]`, `[mmap]`, `[all]` — v1.1
- Shared utilities extracted from hcs.py into _utils.py — v1.1
- All existing data tests passing (71 tests) — v1.1
- Tiered CI for viscy-data (3x3 base + 1x1 extras) — v1.1

### Active

(None — next milestone not yet defined)

### Out of Scope

- Extracting viscy-models, viscy-airtable — future milestones
- Meta-package with re-exports — decided against, clean break approach
- Backward-compatible imports — not maintaining
- Zensical documentation / GitHub Pages — deferred
- Hydra integration — per design doc
- GPU transform unification (GPUTransformMixin) — future refactor

## Context

**Design doc:** https://github.com/mehta-lab/VisCy/issues/353

**Current state (after v1.1):**
- uv workspace monorepo with 2 extracted packages:
  - `packages/viscy-transforms/` — 16 transform modules, 44 exports
  - `packages/viscy-data/` — 15 data modules, 45 exports, 4015 LOC source + 671 LOC tests
- CI: test.yml (viscy-transforms 3x3, viscy-data 3x3 + extras 1x1) + lint.yml
- Python >=3.11, hatchling + uv-dynamic-versioning
- Original code on `main` branch for reference

**Architecture reference:**
- `viscy/data/README.md` documents module inventory, class hierarchy, training pipeline mapping, GPU transform patterns

## Constraints

- **Package naming**: hyphen for package name, underscore for import
- **Python version**: >=3.11
- **Build system**: hatchling with uv-dynamic-versioning
- **Layout**: src layout (`packages/*/src/*/`)
- **Tooling**: uv only
- **No cross-package dependencies between data and transforms**

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Clean break on imports | Simpler architecture, no re-export complexity | ✓ Good |
| hatchling over setuptools | Modern, faster, better uv integration | ✓ Good |
| src layout | Prevents import confusion during development | ✓ Good |
| Tests inside packages | Isolated testing, `uv run --package` workflow | ✓ Good |
| No viscy-transforms dep in data | Transforms separate from data | ✓ Good |
| Optional dependency groups | Heavy deps as extras, lean base install | ✓ Good |
| Extract shared utils from hcs.py | Prevent dual-role module anti-pattern | ✓ Good |
| BatchedCenterSpatialCropd in _utils.py | CenterSpatialCropd can't handle batch dim in on_after_batch_transfer | ✓ Good |
| Lazy imports for optional deps | try/except at module level, guard in __init__ | ✓ Good |
| Flat public API (45 exports) | MONAI pattern, consistent with viscy-transforms | ✓ Good |
| combined.py preserved as-is | No split per REF-02 deferral | ✓ Good |

---
*Last updated: 2026-02-14 after v1.1 milestone completion*
