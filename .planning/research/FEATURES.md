# Feature Landscape: Python Monorepo with uv Workspace

**Domain:** Python scientific package monorepo (uv workspace)
**Researched:** 2026-01-27
**Overall Confidence:** HIGH (verified via uv official docs, multiple credible sources)

## Table Stakes

Features users/developers expect. Missing = monorepo feels broken or unprofessional.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Workspace member discovery** | `members = ["packages/*"]` glob pattern to auto-detect packages | Low | uv workspace standard; must have in root pyproject.toml |
| **Shared lockfile** | Single `uv.lock` for consistent dependency resolution across all packages | Low | Automatic with uv workspace; ensures reproducibility |
| **Editable inter-package dependencies** | `workspace = true` in `[tool.uv.sources]` enables editable installs between members | Low | Critical for development; changes propagate immediately |
| **Per-package pyproject.toml** | Each package has its own metadata, dependencies, build config | Low | Required by uv workspace design |
| **src layout** | `packages/*/src/*/` structure prevents import confusion | Low | pytest/pip best practice; prevents accidental local imports |
| **Independent package testing** | `uv run --package viscy-transforms pytest` to test one package | Low | Core workflow; must work from any directory in workspace |
| **Dependency groups (PEP 735)** | `[dependency-groups]` for dev/test/docs separation | Low | uv native support; `--dev` flag syncs dev group by default |
| **Git-based versioning** | Dynamic version from VCS tags (uv-dynamic-versioning) | Medium | Standard for scientific Python; avoids manual version bumps |
| **Pre-commit hooks** | Shared linting/formatting config at workspace root | Low | `prek`/pre-commit; enforces code quality |
| **Type checking config** | pyright/mypy configuration at workspace root | Low | Shared settings for consistent type checking |
| **Ruff linting/formatting** | Modern, fast Python linter/formatter | Low | Industry standard 2026; replaces flake8+black+isort |
| **pytest configuration** | Per-package or workspace-level pytest.ini/pyproject.toml | Low | Test discovery and execution |
| **CI that tests changed packages** | Path-based filtering in GitHub Actions | Medium | Essential for monorepo efficiency; don't test unchanged packages |
| **Clean import paths** | `from viscy_transforms import X` not `from viscy.transforms` | Low | User expectation for independent packages |

## Differentiators

Features that make this monorepo better than alternatives. Not expected, but valued.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Reusable CI workflows** | DRY GitHub Actions with `working-directory` parameter | Medium | Reduces CI maintenance; same workflow for all packages |
| **Package-specific documentation** | Per-package docs with cross-linking | Medium | Zensical/mkdocs-monorepo-plugin; docs close to code |
| **API documentation generation** | Auto-generated API docs from docstrings | Medium | mkdocstrings or Zensical autodoc; keeps docs in sync |
| **Matrix CI testing** | Test across Python versions per package | Medium | GitHub Actions matrix strategy |
| **Conditional package publishing** | Publish only changed packages on release | High | Requires tag-based or path-based release automation |
| **Workspace-wide type checking** | pyright with package paths configured | Medium | Catches cross-package type errors |
| **Shared test utilities** | Common test fixtures in a shared package | Medium | Avoids test code duplication |
| **Parallel test execution** | pytest-xdist for faster test runs | Low | Easy win for large test suites |
| **Coverage aggregation** | Combined coverage report across packages | Medium | Shows true coverage; pytest-cov with workspace config |
| **Dev container / devcontainer.json** | Consistent development environment | Medium | Valuable for onboarding; VS Code integration |
| **Dependabot/Renovate for workspace** | Automated dependency updates | Medium | Monorepo-aware dependency management |
| **Build caching in CI** | uv cache, dependency caching | Medium | Speeds up CI significantly |
| **Lockstep versioning option** | All packages share same version | Medium | hatch-cada or manual; good for tightly coupled packages |
| **Independent versioning option** | Each package has own version | Medium | Better for loosely coupled packages like viscy-transforms |
| **Release automation** | python-semantic-release or manual workflow | High | Reduces release friction |

## Anti-Features

Features to explicitly NOT build. Common mistakes in Python monorepos.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Single mega-package with all code** | Defeats purpose of modularity; forces users to install everything | Extract independent packages with minimal dependencies |
| **Shared requirements.txt at root** | Doesn't scale; hides per-package dependencies | Per-package pyproject.toml with proper dependencies |
| **Relative imports between packages** | Fragile; breaks when packages are installed independently | Explicit dependencies via `[tool.uv.sources]` |
| **Tests at repository root** | Causes pytest module conflicts; hard to run per-package | Tests inside each package: `packages/*/tests/` |
| **Manual version management** | Error-prone; creates release friction | uv-dynamic-versioning from git tags |
| **Circular dependencies between packages** | Indicates poor boundary design; complicates builds | Refactor to DAG structure; extract shared code |
| **Overly granular packages** | Maintenance overhead; confuses users | Group related functionality; 3-7 packages ideal |
| **God package with re-exports** | Creates import confusion; hides real dependencies | Clean break with direct imports from each package |
| **Monolithic documentation** | Hard to maintain; docs drift from code | Per-package docs with central hub linking |
| **Copy-paste CI for each package** | Maintenance nightmare; divergent workflows | Reusable workflows with parameters |
| **Development dependencies in main deps** | Bloats user installations | Use `[dependency-groups]` for dev/test/docs |
| **Path dependencies in published packages** | Breaks when installed from PyPI | Convert to version constraints at build time (hatch-cada) |
| **Ignoring Python version intersections** | uv workspace requires single requires-python | Plan packages to share compatible Python versions |
| **Tightly coupling independent packages** | Projects should not import each other directly | Use shared library for common functionality |

## Feature Dependencies

```
Workspace Setup (foundation)
    |
    +-- Per-package pyproject.toml
    |       |
    |       +-- src layout
    |       |       |
    |       |       +-- Clean import paths
    |       |       |
    |       |       +-- Independent package testing
    |       |
    |       +-- Dependency groups (PEP 735)
    |               |
    |               +-- Dev dependencies isolation
    |
    +-- Shared lockfile (uv.lock)
    |       |
    |       +-- Reproducible builds
    |       |
    |       +-- CI caching
    |
    +-- Editable inter-package deps
            |
            +-- Local development workflow

Git-based versioning
    |
    +-- uv-dynamic-versioning
            |
            +-- Release automation

CI/CD Infrastructure
    |
    +-- Path-based filtering
    |       |
    |       +-- Changed package detection
    |
    +-- Reusable workflows
    |       |
    |       +-- Matrix testing
    |
    +-- Build caching

Documentation
    |
    +-- Per-package docs
            |
            +-- API generation
            |
            +-- Central hub
```

## MVP Recommendation

For the VisCy modularization MVP (viscy-transforms extraction), prioritize:

### Phase 1: Foundation (Must Have)
1. **Workspace scaffolding** - Root pyproject.toml with `[tool.uv.workspace]`
2. **viscy-transforms package** - `packages/viscy-transforms/` with src layout
3. **Per-package pyproject.toml** - hatchling + uv-dynamic-versioning
4. **Dependency groups** - dev group for pytest, type checking
5. **Shared lockfile** - uv.lock at workspace root
6. **Basic CI** - Test viscy-transforms independently

### Phase 2: Developer Experience
7. **Pre-commit/prek hooks** - Ruff, pyright at workspace level
8. **pytest configuration** - Per-package test discovery
9. **Type checking** - pyright configuration

### Phase 3: Documentation
10. **Zensical setup** - Replace ReadTheDocs
11. **API documentation** - Auto-generated from docstrings
12. **GitHub Pages deployment** - CI workflow

Defer to post-MVP:
- **Additional package extractions** (viscy-data, viscy-models): Focus on viscy-transforms first
- **Release automation**: Manual releases acceptable initially
- **Coverage aggregation**: Nice-to-have for later
- **Dependabot/Renovate**: Can add after initial setup stabilizes
- **Dev containers**: Useful but not blocking

## Complexity Assessment

| Category | Estimated Effort | Risk Level |
|----------|------------------|------------|
| Workspace scaffolding | Low | Low |
| Package extraction (viscy-transforms) | Medium | Medium |
| CI updates | Medium | Medium |
| Documentation (Zensical) | Medium | Medium |
| Git versioning | Low | Low |
| Future package extractions | Medium each | Low (pattern established) |

## Sources

**HIGH Confidence (Official Documentation):**
- [uv Workspaces Documentation](https://docs.astral.sh/uv/concepts/projects/workspaces/)
- [uv Managing Dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [pytest Good Integration Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [uv-dynamic-versioning PyPI](https://pypi.org/project/uv-dynamic-versioning/)
- [Zensical Documentation](https://zensical.org/docs/get-started/)

**MEDIUM Confidence (Verified Community Sources):**
- [FOSDEM 2026 - Modern Python monorepo with uv](https://fosdem.org/2026/schedule/event/WE7NHM-modern-python-monorepo-apache-airflow/)
- [Tweag Python Monorepo Guide](https://www.tweag.io/blog/2023-04-04-python-monorepo-1/)
- [LlamaIndex Monorepo Overhaul](https://www.llamaindex.ai/blog/python-tooling-at-scale-llamaindex-s-monorepo-overhaul)
- [Graphite Python Monorepos Guide](https://graphite.com/guides/python-monorepos)
- [Simon Willison on Dependency Groups](https://til.simonwillison.net/uv/dependency-groups)

**LOW Confidence (Community Discussion, needs validation):**
- [uv Monorepo Best Practices Issue](https://github.com/astral-sh/uv/issues/10960) - Active discussion, no official guidance yet
