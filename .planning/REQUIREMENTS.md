# Requirements: VisCy Modularization (Milestone 1)

**Defined:** 2025-01-27
**Core Value:** Independent, reusable subpackages with clean import paths

## v1 Requirements

Requirements for Phases 0+1: Workspace scaffolding + viscy-transforms extraction.

### Workspace Foundation

- [x] **WORK-00**: Clean slate setup - wipe repo keeping only LICENSE, CITATION.cff, .gitignore
- [x] **WORK-01**: Virtual workspace root with `[tool.uv.workspace]` and `members = ["packages/*"]`
- [x] **WORK-02**: Shared lockfile (`uv.lock`) at repository root
- [x] **WORK-03**: Python version floor (>=3.11) enforced in root pyproject.toml
- [x] **WORK-04**: Pre-commit hooks configured (ruff, ty) for local development
- [x] **WORK-05**: Shared pytest configuration in root pyproject.toml

### Package Structure

- [x] **PKG-01**: src layout for viscy-transforms (`packages/viscy-transforms/src/viscy_transforms/`)
- [x] **PKG-02**: Package pyproject.toml with hatchling build backend
- [x] **PKG-03**: uv-dynamic-versioning configured for git-based versioning
- [x] **PKG-04**: Package README.md with installation and usage instructions

### Code Migration

- [x] **MIG-01**: All transform modules migrated from `viscy/transforms/` to package
- [x] **MIG-02**: All transform tests migrated from `tests/transforms/` to `packages/viscy-transforms/tests/`
- [x] **MIG-03**: Import path updated to `from viscy_transforms import X`
- [x] **MIG-04**: All migrated tests passing with `uv run --package viscy-transforms pytest`
- [x] **MIG-05**: Original `viscy/transforms/` directory removed

### Documentation

- [ ] **DOC-01**: Zensical configuration (`zensical.toml`) at repository root
- [ ] **DOC-02**: Documentation site structure with navigation (index, API reference)
- [ ] **DOC-03**: API reference for viscy-transforms auto-generated from docstrings
- [ ] **DOC-04**: GitHub Pages deployment working at project URL

### CI/CD

- [ ] **CI-01**: GitHub Actions workflow for testing viscy-transforms package
- [ ] **CI-02**: GitHub Actions workflow for building and deploying docs
- [ ] **CI-03**: Matrix testing across Python 3.11, 3.12, 3.13
- [ ] **CI-04**: Linting via prek (uvx prek) in CI workflows

## v2 Requirements

Deferred to future milestones. Tracked but not in current roadmap.

### Future Package Extractions

- **PKG-10**: Extract viscy-data package (dataloaders, Lightning DataModules)
- **PKG-11**: Extract viscy-models package (unet, representation, translation)
- **PKG-12**: Extract viscy-airtable package
- **PKG-13**: viscy meta-package with CLI and optional re-exports

### Enhanced CI/CD

- **CI-10**: Path filtering to only test changed packages
- **CI-11**: Release automation with semantic versioning
- **CI-12**: Coverage aggregation across packages

### Documentation Enhancements

- **DOC-10**: Migration guide for downstream users
- **DOC-11**: Per-package documentation sections
- **DOC-12**: Contribution guide for monorepo workflow

## Out of Scope

Explicitly excluded from this milestone. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Backward-compatible imports | Clean break decided; no `from viscy.transforms` re-exports |
| applications/ directory | Clean slate approach; restore from git history in future milestone |
| examples/ directory | Clean slate approach; restore from git history in future milestone |
| Release automation | Manual releases acceptable for v1; automate later |
| Path-based CI filtering | Added complexity; test all on every push for now |
| hatch-cada for workspace deps | No inter-package deps yet; viscy-transforms is standalone |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| WORK-00 | Phase 1 | Complete |
| WORK-01 | Phase 1 | Complete |
| WORK-02 | Phase 1 | Complete |
| WORK-03 | Phase 1 | Complete |
| WORK-04 | Phase 1 | Complete |
| WORK-05 | Phase 1 | Complete |
| PKG-01 | Phase 2 | Complete |
| PKG-02 | Phase 2 | Complete |
| PKG-03 | Phase 2 | Complete |
| PKG-04 | Phase 2 | Complete |
| MIG-01 | Phase 3 | Complete |
| MIG-02 | Phase 3 | Complete |
| MIG-03 | Phase 3 | Complete |
| MIG-04 | Phase 3 | Complete |
| MIG-05 | Phase 3 | Complete |
| DOC-01 | Phase 4 | Pending |
| DOC-02 | Phase 4 | Pending |
| DOC-03 | Phase 4 | Pending |
| DOC-04 | Phase 4 | Pending |
| CI-01 | Phase 5 | Pending |
| CI-02 | Phase 5 | Pending |
| CI-03 | Phase 5 | Pending |
| CI-04 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 23 total
- Mapped to phases: 23
- Unmapped: 0 ✓

---
*Requirements defined: 2025-01-27*
*Last updated: 2026-01-28 after Phase 3 completion*
