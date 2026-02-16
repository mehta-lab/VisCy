# Milestones: VisCy Modularization

## v1.0 — Transforms & Monorepo Skeleton

**Shipped:** 2026-01-29
**Phases:** 1-5 (4 executed, 1 deferred)

**What shipped:**
- uv workspace scaffolding with `packages/` directory
- `viscy-transforms` package extracted with src layout
- Clean import paths: `from viscy_transforms import X`
- hatchling + uv-dynamic-versioning build system
- All 16 transform modules migrated with full test suite
- CI/CD: 9-job test matrix (3 OS x 3 Python) + lint workflow
- prek hooks with ruff formatting

**Deferred:**
- Phase 4: Documentation (Zensical + GitHub Pages)

**Last phase:** 5

---

## v1.1 — Extract viscy-data

**Shipped:** 2026-02-14
**Phases:** 6-9 (4 phases, 9 plans)

**What shipped:**
- `viscy-data` package extracted with src layout (15 modules, 4015 LOC)
- All 13 data modules migrated with clean import paths (45 public exports)
- No cross-package dependency on viscy-transforms
- Optional dependency groups: `[triplet]`, `[livecell]`, `[mmap]`, `[all]`
- Shared utilities extracted from hcs.py into _utils.py
- Lazy import pattern for optional deps with clear error messages
- All existing data tests passing (71 tests)
- Tiered CI: viscy-data 3x3 base + 1x1 extras

**Last phase:** 9

---

## v1.2 — Extract viscy-models

**Shipped:** 2026-02-13
**Phases:** 10-14 (5 phases, 9 plans)

**What shipped:**
- `viscy-models` package with 8 architectures organized by function (unet/, vae/, contrastive/)
- Shared components extracted to `_components/` (stems, heads, blocks)
- Full test coverage: migrated existing + new forward-pass tests for all models
- Pure nn.Module — no Lightning/Hydra coupling
- State dict key compatibility preserved for checkpoint loading
- CI includes viscy-models in test matrix

**Pending todo from v1.2:**
- Fix deconv decoder channel mismatch in UNeXt2UpStage (pre-existing bug, xfailed test)

**Last phase:** 14

---
