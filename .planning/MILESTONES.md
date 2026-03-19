# Milestones: VisCy Modularization

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
