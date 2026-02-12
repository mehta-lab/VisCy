# VisCy Modularization

## What This Is

Restructuring VisCy from a monolithic package into a uv workspace monorepo. This enables reusing transforms, dataloaders, and models in downstream projects without requiring the entire VisCy package as a dependency. The first milestone extracts `viscy-transforms` as an independent package with modern Python packaging (hatchling, uv-dynamic-versioning) and sets up Zensical documentation with GitHub Pages.

## Core Value

**Independent, reusable subpackages with clean import paths.** Users can `pip install viscy-transforms` and use `from viscy_transforms import X` without pulling in the entire VisCy ecosystem.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] uv workspace scaffolding with `packages/` directory structure
- [ ] `viscy-transforms` package extracted with src layout (`packages/viscy-transforms/src/viscy_transforms/`)
- [ ] Import path: `from viscy_transforms import X` (clean break, not backward compatible)
- [ ] hatchling build backend with uv-dynamic-versioning for git-based versions
- [ ] dependency-groups (PEP 735) for test/dev dependencies
- [ ] All existing transform tests passing in new structure
- [ ] Zensical documentation replacing ReadTheDocs
- [ ] GitHub Pages deployment via GitHub Actions
- [ ] API documentation for viscy-transforms
- [ ] CI updated for monorepo structure (test packages independently)

### Out of Scope

- Extracting other packages (viscy-data, viscy-models, viscy-airtable) — Phase 2+
- Meta-package with re-exports — decided against, clean break approach
- Backward-compatible imports (`from viscy.transforms import X`) — not maintaining
- Fixing broken imports in applications/examples — deferred to later phases
- Hydra integration — Phase 6 per design doc

## Context

**Design doc:** https://github.com/mehta-lab/VisCy/issues/353

**Reference implementations:**
- biahub Zensical setup: https://github.com/czbiohub-sf/biahub (zensical.toml, docs workflow)
- iohub pyproject.toml: modern hatchling + uv-dynamic-versioning pattern

**Current state:**
- Monolithic `viscy` package with transforms at `viscy/transforms/`
- 25 transform modules with comprehensive test coverage
- Dependencies: kornia, monai, torch
- Existing ReadTheDocs setup to be replaced

**Sandbox workflow:**
- This worktree (`viscy-modular-gsd`) is a sandbox for iteration
- Final changes will be squashed and moved to `viscy-modular` branch
- Target merge: `viscy-modular` branch (not main directly)

**Clean slate approach:**
- Keep only: LICENSE, CITATION.cff, .gitignore
- Wipe everything else (viscy/, tests/, docs/, applications/, examples/, README.md, pyproject.toml)
- Rebuild from scratch with new workspace structure
- Original code available in git history for reference/copying

## Constraints

- **Package naming**: `viscy-transforms` (hyphen) as package name, `viscy_transforms` (underscore) as import
- **Python version**: >=3.11 (matching current VisCy)
- **Build system**: hatchling with uv-dynamic-versioning (following iohub pattern)
- **Layout**: src layout required (`packages/*/src/*/`)
- **Tooling**: uv only, no pip/setuptools for package management

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Clean break on imports | Simpler architecture, no re-export complexity | — Pending |
| Replace ReadTheDocs with Zensical | Modern tooling, GitHub Pages hosting | — Pending |
| hatchling over setuptools | Modern, faster, better uv integration | — Pending |
| src layout | Prevents import confusion during development | — Pending |
| Tests inside packages | Isolated testing, `uv run --package` workflow | — Pending |

---
*Last updated: 2025-01-27 after roadmap creation*
