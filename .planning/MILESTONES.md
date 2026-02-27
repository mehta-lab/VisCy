# Milestones: VisCy Modularization

## v1.1 — Extract viscy-data

**Shipped:** 2026-02-14
**Phases:** 6–9 (4 executed)

**What shipped:**
- `viscy-data` package extracted with src layout (15 modules, 4015 LOC)
- 45 public exports with clean import paths: `from viscy_data import X`
- Optional dependency groups: `[triplet]`, `[livecell]`, `[mmap]`, `[all]`
- Shared utilities extracted from hcs.py into _utils.py
- All existing data tests passing (71 tests)
- Tiered CI for viscy-data (3x3 base + 1x1 extras)

---

## v1.0 — Transforms & Monorepo Skeleton

**Shipped:** 2026-01-29
**Phases:** 1–5 (4 executed, 1 deferred)

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
