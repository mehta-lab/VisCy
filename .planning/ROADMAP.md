# Roadmap: VisCy Modularization (Milestone 1)

## Overview

Transform VisCy from a monolithic package into a uv workspace monorepo by extracting viscy-transforms as the first independent subpackage. This milestone establishes the workspace foundation, migrates code and tests, sets up Zensical documentation with GitHub Pages, and configures CI/CD for the new monorepo structure. The repo starts with a clean slate, preserving only LICENSE, CITATION.cff, and .gitignore.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Workspace Foundation** - Clean slate setup and uv workspace scaffolding
- [x] **Phase 2: Package Structure** - viscy-transforms package scaffolding with hatchling
- [x] **Phase 3: Code Migration** - Migrate transforms code and tests to new structure
- [ ] **Phase 4: Documentation** - Zensical documentation with GitHub Pages deployment
- [x] **Phase 5: CI/CD** - GitHub Actions for testing, linting, and docs deployment

## Phase Details

### Phase 1: Workspace Foundation
**Goal**: Establish a clean uv workspace with shared tooling configuration
**Depends on**: Nothing (first phase)
**Requirements**: WORK-00, WORK-01, WORK-02, WORK-03, WORK-04, WORK-05
**Success Criteria** (what must be TRUE):
  1. Repository contains only LICENSE, CITATION.cff, .gitignore, and new workspace structure
  2. `uv sync` runs successfully at workspace root
  3. `uvx prek` passes with ruff and mypy hooks configured
  4. Python 3.11+ constraint enforced in root pyproject.toml
  5. Empty `packages/` directory exists and is a workspace member
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — Clean slate + workspace pyproject.toml with uv configuration
- [x] 01-02-PLAN.md — Pre-commit hooks with ruff and ty

### Phase 2: Package Structure
**Goal**: Create viscy-transforms package skeleton with modern build system
**Depends on**: Phase 1
**Requirements**: PKG-01, PKG-02, PKG-03, PKG-04
**Success Criteria** (what must be TRUE):
  1. `packages/viscy-transforms/src/viscy_transforms/__init__.py` exists with proper structure
  2. Package pyproject.toml uses hatchling with uv-dynamic-versioning
  3. `uv pip install -e packages/viscy-transforms` succeeds
  4. Package README.md documents installation and basic usage
**Plans**: 1 plan

Plans:
- [x] 02-01-PLAN.md — Package skeleton with hatchling, uv-dynamic-versioning, and README

### Phase 3: Code Migration
**Goal**: Migrate all transforms code and tests with passing test suite
**Depends on**: Phase 2
**Requirements**: MIG-01, MIG-02, MIG-03, MIG-04, MIG-05
**Success Criteria** (what must be TRUE):
  1. All 16 transform modules exist in `packages/viscy-transforms/src/viscy_transforms/`
  2. `from viscy_transforms import X` works for all public exports
  3. `uv run --package viscy-transforms pytest` passes all tests
  4. No `viscy/transforms/` directory exists in repository
  5. Import paths in tests updated to `viscy_transforms`
**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md — Extract types from viscy.data.typing to _typing.py
- [x] 03-02-PLAN.md — Migrate 16 transform modules with updated imports
- [x] 03-03-PLAN.md — Migrate tests and verify full test suite passes

### Phase 4: Documentation
**Goal**: Zensical documentation deployed to GitHub Pages
**Depends on**: Phase 3
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04
**Success Criteria** (what must be TRUE):
  1. `zensical.toml` or `mkdocs.yml` configured at repository root
  2. Documentation builds locally with `uvx zensical build` (or mkdocs fallback)
  3. API reference auto-generated from viscy-transforms docstrings
  4. Documentation accessible at GitHub Pages URL after push
**Plans**: TBD

Plans:
- [ ] 04-01: TBD

### Phase 5: CI/CD
**Goal**: Automated testing and linting via GitHub Actions
**Depends on**: Phase 3 (docs deployment deferred)
**Requirements**: CI-01, CI-03, CI-04 (CI-02 deferred)
**Success Criteria** (what must be TRUE):
  1. Push to main triggers test workflow for viscy-transforms
  2. Tests run against Python 3.11, 3.12, 3.13 on Ubuntu, macOS, Windows
  3. `uvx prek` linting passes in CI
  4. alls-green check job aggregates matrix results for branch protection
**Plans**: 1 plan

Plans:
- [x] 05-01-PLAN.md — Test matrix (9 jobs) + lint workflow with prek

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Workspace Foundation | 2/2 | ✓ Complete | 2026-01-28 |
| 2. Package Structure | 1/1 | ✓ Complete | 2026-01-28 |
| 3. Code Migration | 3/3 | ✓ Complete | 2026-01-28 |
| 4. Documentation | 0/TBD | Deferred | - |
| 5. CI/CD | 1/1 | ✓ Complete | 2026-01-29 |

---
*Roadmap created: 2025-01-27*
*Last updated: 2026-01-29*
