# Technology Stack: VisCy Monorepo

**Project:** VisCy uv workspace monorepo with independent subpackages
**Researched:** 2026-01-27
**Overall Confidence:** HIGH

---

## Executive Summary

This stack recommendation converts VisCy from a single-package setuptools project to a uv workspace monorepo with hatchling build backend and VCS-based dynamic versioning. The stack prioritizes:

1. **Modern tooling** (uv, hatchling) over legacy (setuptools, pip)
2. **Workspace-native versioning** (hatch-cada + hatch-vcs) for independent package releases
3. **Zensical documentation** as the successor to Material for MkDocs
4. **Minimal configuration** with sensible defaults

---

## Recommended Stack

### Package Management & Build System

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| uv | latest | Package manager, virtual env, workspace orchestration | Industry standard for 2025+; 10-100x faster than pip; native workspace support | HIGH |
| hatchling | >=1.28.0 | Build backend | Recommended by uv; extensible via plugins; better than setuptools for modern projects | HIGH |
| hatch-vcs | latest | VCS-based versioning | Derives version from git tags; eliminates manual version bumps | HIGH |
| hatch-cada | >=1.0.1 | Workspace dependency versioning | Rewrites workspace deps with version constraints at build time; enables independent releases | HIGH |

**Rationale:** The combination of hatchling + hatch-vcs + hatch-cada is specifically designed for uv workspace monorepos. This replaces setuptools-scm which the current project uses. The uv build backend (`uv_build`) does NOT support plugins yet, so hatchling is required.

### Dynamic Versioning Strategy

| Approach | When to Use | Configuration |
|----------|-------------|---------------|
| **hatch-vcs + hatch-cada** (RECOMMENDED) | Independent versioning per package | Package-specific git tags like `viscy-transforms@1.0.0` |
| uv-dynamic-versioning | Simpler single-package or lockstep versioning | Single version derived from any tag |

**Why hatch-vcs + hatch-cada over uv-dynamic-versioning:**
- hatch-cada properly handles workspace dependencies at build time
- hatch-vcs is mature and well-documented
- uv-dynamic-versioning's metadata hook is newer and less battle-tested for workspaces

### Documentation

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| zensical | >=0.0.19 | Static site generator | Successor to Material for MkDocs by same team; 4-5x faster incremental builds; Rust + Python | MEDIUM |
| mkdocstrings | latest | API documentation from docstrings | Standard for Python API docs; works with Zensical | HIGH |
| mkdocstrings-python | latest | Python handler for mkdocstrings | Required for Python docstring extraction | HIGH |

**Why Zensical over MkDocs:**
- MkDocs is unmaintained since August 2024
- Material for MkDocs entered maintenance mode (November 2025)
- Zensical is the official successor, maintains compatibility with mkdocs.yml
- New projects should use zensical.toml (not mkdocs.yml)

**Caution:** Zensical is still Alpha (0.0.x). For maximum stability, Material for MkDocs 9.7.0 works but is in maintenance mode.

### Code Quality & Linting

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| ruff | >=0.14.14 | Linting + formatting | Replaces flake8, isort, black; 200x faster; native notebook support | HIGH |
| mypy | >=1.19.1 | Static type checking | Industry standard for Python typing; catches bugs pre-runtime | HIGH |
| pre-commit | >=4.5.1 | Git hooks framework | Automates quality checks on commit | HIGH |

**Why ruff replaces black + isort + flake8:**
- Single tool, single configuration
- 200x faster (Rust-based)
- Native Jupyter notebook support (default since 0.6)
- The current project already uses ruff

### Testing

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| pytest | >=9.0.2 | Test framework | Industry standard; rich plugin ecosystem | HIGH |
| pytest-cov | latest | Coverage reporting | Integrates coverage.py with pytest | HIGH |
| hypothesis | latest | Property-based testing | Already in current project; good for scientific code | HIGH |

**Testing with uv:**
```bash
uv run pytest                           # Run all tests
uv run --package viscy-transforms pytest  # Run tests for specific package
uv run -p 3.12 pytest                   # Test against specific Python version
```

### Scientific Computing Core

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| torch | >=2.4.1 | Deep learning framework | Required for GPU acceleration; already in project | HIGH |
| kornia | latest | Differentiable image processing | GPU augmentations; integrates with PyTorch Lightning | HIGH |
| monai | >=1.4 | Medical imaging transforms | Specialized augmentations for biomedical imaging | HIGH |
| lightning | >=2.3 | Training framework | Already in project; integrates well with kornia | HIGH |

### CI/CD & Deployment

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| GitHub Actions | N/A | CI/CD pipeline | Native GitHub integration; free for open source | HIGH |
| uv in CI | latest | Fast dependency installation | 10-100x faster CI runs | HIGH |
| gh-pages | N/A | Documentation hosting | Free; integrates with Zensical | HIGH |

---

## Workspace Structure

### Root pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "viscy-workspace"
version = "0.0.0"
requires-python = ">=3.11"
description = "VisCy monorepo workspace root"
readme = "README.md"
license = "BSD-3-Clause"

[tool.uv.workspace]
members = ["packages/*"]

[tool.ruff]
line-length = 88
src = ["packages/*/src"]
extend-exclude = ["examples", "applications"]

[tool.ruff.lint]
extend-select = ["I001"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
```

### Package pyproject.toml (viscy-transforms example)

```toml
[build-system]
requires = ["hatchling", "hatch-vcs", "hatch-cada"]
build-backend = "hatchling.build"

[project]
name = "viscy-transforms"
description = "GPU augmentation transforms for VisCy"
readme = "README.md"
license = "BSD-3-Clause"
authors = [{ name = "CZ Biohub SF", email = "compmicro@czbiohub.org" }]
requires-python = ">=3.11"
classifiers = [
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
dynamic = ["version"]
dependencies = [
    "torch>=2.4.1",
    "kornia",
    "monai>=1.4",
    "numpy",
]

[project.optional-dependencies]
dev = [
    "pytest>=9.0.2",
    "pytest-cov",
    "hypothesis",
    "ruff>=0.14.14",
    "mypy>=1.19.1",
]

[tool.hatch.version]
source = "vcs"

[tool.hatch.version.raw-options]
tag_regex = "^viscy-transforms@(?P<version>.*)$"
search_parent_directories = true
git_describe_command = ["git", "describe", "--tags", "--long", "--match", "viscy-transforms@*"]

[tool.hatch.metadata.hooks.cada]
strategy = "allow-all-updates"

[tool.hatch.build.targets.wheel]
packages = ["src/viscy_transforms"]
```

### Directory Structure

Per the [design doc](https://github.com/mehta-lab/VisCy/issues/353):

```
viscy/
├── pyproject.toml              # Workspace root (also the viscy meta-package)
├── uv.lock                     # Shared lockfile
├── zensical.toml               # Documentation config
├── .pre-commit-config.yaml
├── src/
│   └── viscy/                  # Meta-package source (CLI, re-exports)
│       ├── __init__.py
│       └── cli.py
├── packages/
│   ├── viscy-transforms/       # First extraction (this milestone)
│   │   ├── pyproject.toml      # Package-specific config
│   │   ├── src/
│   │   │   └── viscy_transforms/
│   │   │       ├── __init__.py
│   │   │       └── ...
│   │   └── tests/
│   ├── viscy-data/             # Future: dataloaders, Lightning DataModules
│   ├── viscy-models/           # Future: unet, representation, translation
│   └── viscy-airtable/         # Future: Airtable integration
├── applications/               # Publications (CytoLand, DynaCLR, DynaCell)
├── tests/                      # Integration tests for meta-package
├── docs/
│   ├── index.md
│   └── api/
└── .github/
    └── workflows/
```

---

## Alternatives Considered

### Build Backend Comparison

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| Build backend | hatchling | setuptools | setuptools is legacy; less extensible; requires more config |
| Build backend | hatchling | uv_build | uv_build doesn't support plugins (yet); can't use hatch-vcs/hatch-cada |
| Build backend | hatchling | poetry-core | Poetry doesn't integrate with uv workspaces |

### Versioning Comparison

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| Versioning | hatch-vcs + hatch-cada | uv-dynamic-versioning | uv-dynamic-versioning is newer; hatch-cada handles workspace deps better |
| Versioning | hatch-vcs + hatch-cada | setuptools-scm | setuptools-scm doesn't work with hatchling |

### Documentation Comparison

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| Docs generator | zensical | mkdocs-material | MkDocs unmaintained; Material in maintenance mode |
| Docs generator | zensical | sphinx | Sphinx is complex; RST vs Markdown; worse DX |

### Monorepo Tools Comparison

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| Build/publish | hatch-cada | una | hatch-cada is simpler; una adds another tool layer |
| Build/publish | hatch-cada | pants/bazel | Massive complexity overhead for a scientific package |

---

## Installation Commands

### Initial Setup

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create workspace
uv init viscy-workspace
cd viscy-workspace
uv add --dev ruff mypy pre-commit pytest

# Create first package
mkdir -p packages/viscy-transforms/src/viscy_transforms
# ... add pyproject.toml and code

# Sync all packages
uv sync --all-packages
```

### Package Development

```bash
# Install specific package in dev mode
uv sync --package viscy-transforms

# Run tests for specific package
uv run --package viscy-transforms pytest

# Build specific package
uv build packages/viscy-transforms

# Publish (after tagging)
git tag viscy-transforms@1.0.0
uv build packages/viscy-transforms
uv publish dist/viscy_transforms-1.0.0*
```

### Documentation

```bash
# Install zensical
uv add --dev zensical mkdocstrings mkdocstrings-python

# Serve locally
uv run zensical serve

# Build for deployment
uv run zensical build

# Deploy to GitHub Pages
uv run zensical gh-deploy
```

---

## Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.14.14
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.19.1
    hooks:
      - id: mypy
        additional_dependencies: [torch, numpy]
        args: [--ignore-missing-imports]
```

---

## Migration Notes

### From Current VisCy Setup

The current VisCy uses:
- `setuptools` + `setuptools-scm` -> Replace with `hatchling` + `hatch-vcs` + `hatch-cada`
- `write_to = "viscy/_version.py"` -> Use `importlib.metadata.version()` instead
- Single package -> Workspace with multiple packages

### Key Breaking Changes

1. **Version file location**: No more `_version.py` generation; use `importlib.metadata`
2. **Import paths**: `from viscy.transforms import X` becomes `from viscy_transforms import X`
3. **Installation**: `pip install viscy` remains for full suite; individual packages available as `pip install viscy-transforms`, `pip install viscy-data`, etc.

---

## Gaps and Open Questions

| Gap | Impact | Mitigation |
|-----|--------|------------|
| Zensical is Alpha | May have bugs | Can fall back to mkdocs-material 9.7.0 |
| hatch-cada is new (v1.0.1) | Limited community testing | Well-documented; simple plugin |
| uv workspace IDE support | VSCode/Pylance may not understand workspace | Configure pyrightconfig.json |
| No official uv monorepo docs | Limited guidance | Follow patterns from pydantic-ai, MCP SDK |

---

## Sources

### Official Documentation (HIGH confidence)
- [uv Workspaces Documentation](https://docs.astral.sh/uv/concepts/projects/workspaces/)
- [Hatchling PyPI](https://pypi.org/project/hatchling/) - v1.28.0 (Nov 2025)
- [Zensical Documentation](https://zensical.org/docs/get-started/)
- [Zensical PyPI](https://pypi.org/project/zensical/) - v0.0.19 (Jan 2026)

### GitHub Repositories (HIGH confidence)
- [hatch-cada](https://github.com/bilelomrani1/hatch-cada) - v1.0.1 (Jan 2026)
- [uv-dynamic-versioning](https://github.com/ninoseki/uv-dynamic-versioning) - v0.13.0 (Jan 2026)
- [ruff-pre-commit](https://github.com/astral-sh/ruff-pre-commit) - v0.14.14

### Community Resources (MEDIUM confidence)
- [Python Workspaces (Monorepos)](https://tomasrepcik.dev/blog/2025/2025-10-26-python-workspaces/)
- [uv Monorepo Best Practices Issue](https://github.com/astral-sh/uv/issues/10960)
- [Dynamic Versioning and Automated Releases](https://slhck.info/software/2025/10/01/dynamic-versioning-uv-projects.html)
- [Modern Python Code Quality Setup](https://simone-carolini.medium.com/modern-python-code-quality-setup-uv-ruff-and-mypy-8038c6549dcc)
- [Scientific Python Development Guide](https://learn.scientific-python.org/development/guides/style/)

### Tool Version References (HIGH confidence)
- ruff v0.14.14 (Jan 22, 2026)
- pytest v9.0.2 (Dec 6, 2025)
- mypy v1.19.1 (Dec 15, 2025)
- pre-commit v4.5.1 (Dec 16, 2025)
- hatch-cada v1.0.1 (Jan 12, 2026)
