# Phase 2: Package Structure - Research

**Researched:** 2026-01-28
**Domain:** Python packaging with hatchling + uv-dynamic-versioning for monorepo
**Confidence:** HIGH (Context7 + official documentation verified)

## Summary

This phase creates the `viscy-transforms` package skeleton using hatchling as the build backend with uv-dynamic-versioning for git-based version management. Research confirms the CONTEXT.md decisions are well-supported: hatchling with src layout has automatic package discovery, uv-dynamic-versioning supports `pattern-prefix` for independent package versioning in monorepos, and PEP 735 dependency-groups can inherit from workspace roots.

Key findings:
- hatchling automatically discovers packages in `src/` layout when package name matches directory name
- uv-dynamic-versioning requires `pattern-prefix = "viscy-transforms-"` for monorepo tag filtering
- `py.typed` marker requires explicit inclusion in wheel build configuration
- Fallback version is essential for CI environments without git tags (Dependabot, shallow clones)

**Primary recommendation:** Use hatchling with explicit `packages = ["src/viscy_transforms"]` configuration (not relying on auto-discovery) to ensure predictable builds, and configure uv-dynamic-versioning with fallback-version for CI robustness.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| hatchling | >=1.27 | Build backend | Modern, fast, native src-layout support, plugin ecosystem |
| uv-dynamic-versioning | >=0.13.0 | VCS-based versioning | Designed for uv/hatch, supports monorepo pattern-prefix |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dunamai | (via uv-dynamic-versioning) | Version string parsing | Automatically used by uv-dynamic-versioning |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| uv-dynamic-versioning | hatch-vcs | hatch-vcs uses setuptools-scm under the hood; uv-dynamic-versioning is lighter and designed specifically for uv projects |
| hatchling | setuptools | setuptools is more complex; hatchling has better uv integration and simpler configuration |

**Installation:**
```bash
# Build dependencies (in pyproject.toml, not installed directly)
# These go in [build-system].requires
# - hatchling
# - uv-dynamic-versioning
```

## Architecture Patterns

### Recommended Project Structure
```
packages/viscy-transforms/
├── src/
│   └── viscy_transforms/           # Package directory (underscore)
│       ├── __init__.py             # Flat public exports with __all__
│       ├── py.typed                # PEP 561 type marker
│       ├── _crop.py                # Private modules (underscore prefix)
│       ├── _flip.py
│       └── ...
├── tests/                          # Package-specific tests
│   ├── __init__.py
│   └── test_*.py
├── pyproject.toml                  # Package metadata + hatchling config
└── README.md                       # Package documentation
```

### Pattern 1: Hatchling Build Configuration with src Layout
**What:** Configure hatchling to find packages in src/ directory and include py.typed
**When to use:** Always for src-layout packages with type hints
**Example:**
```toml
# Source: https://hatch.pypa.io/1.12/config/build/
[build-system]
requires = ["hatchling", "uv-dynamic-versioning"]
build-backend = "hatchling.build"

[project]
name = "viscy-transforms"
dynamic = ["version"]
requires-python = ">=3.11"

[tool.hatch.build.targets.wheel]
packages = ["src/viscy_transforms"]
```

### Pattern 2: uv-dynamic-versioning for Monorepo
**What:** Configure version from git tags with package-specific prefix
**When to use:** Monorepo with independent package releases
**Example:**
```toml
# Source: https://github.com/ninoseki/uv-dynamic-versioning
[tool.hatch.version]
source = "uv-dynamic-versioning"

[tool.uv-dynamic-versioning]
vcs = "git"
style = "pep440"
pattern-prefix = "viscy-transforms-"
fallback-version = "0.0.0"
```
With this config, tag `viscy-transforms-v0.1.0` produces version `0.1.0`.

### Pattern 3: PEP 735 Dependency Group Inheritance
**What:** Package test dependencies inherit from workspace root
**When to use:** Shared test tooling across workspace packages
**Example:**
```toml
# Source: https://docs.astral.sh/uv/concepts/projects/dependencies/
# In package pyproject.toml:
[dependency-groups]
test = [
    { include-group = "test" },  # Inherit from workspace root
]

# In workspace root pyproject.toml:
[dependency-groups]
test = [
    "pytest>=9.0",
    "pytest-cov",
]
```

### Pattern 4: Flat Top-Level Exports
**What:** All public transforms available at package root
**When to use:** When users expect `from viscy_transforms import Transform`
**Example:**
```python
# src/viscy_transforms/__init__.py
from viscy_transforms._crop import CropSampled
from viscy_transforms._flip import FlipSampled
# ... more imports

__all__ = [
    "CropSampled",
    "FlipSampled",
    # ... all public exports
]
```

### Anti-Patterns to Avoid
- **Relying on auto-discovery without verification:** Always explicitly configure `packages = ["src/viscy_transforms"]` in wheel build config
- **Forgetting fallback-version:** CI environments (Dependabot, shallow clones) may not have git tags
- **Boolean values as strings:** `bump = "true"` is wrong, use `bump = true` (TOML types matter since v0.9.0)
- **Creating py.typed without including it:** Hatchling may not auto-include non-Python files

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Version from git tags | Custom script reading git | uv-dynamic-versioning | Handles edge cases (dirty, distance, branches) |
| Version string in `__version__` | Manual `__version__ = "..."` | `importlib.metadata.version()` | Single source of truth from installed metadata |
| Test dep inheritance | Copy-paste deps in each package | PEP 735 `include-group` | DRY, centralized updates |
| Type marker distribution | Manual MANIFEST.in | Hatchling `packages` config | Automatic with proper build config |

**Key insight:** Python packaging has many subtle edge cases (dirty commits, commit distance, CI without tags). Using established tools prevents shipping broken builds.

## Common Pitfalls

### Pitfall 1: Dynamic Version Not Detected
**What goes wrong:** `uv-dynamic-versioning` returns 0.0.0 or fails to find version
**Why it happens:**
- No git tags exist yet (new repo)
- `.git` directory not present (Docker build, sdist)
- Tag doesn't match `pattern-prefix`
**How to avoid:**
- Always set `fallback-version = "0.0.0"`
- Verify tag matches pattern: `viscy-transforms-v0.1.0` for `pattern-prefix = "viscy-transforms-"`
**Warning signs:** Version shows as 0.0.0 when you expect something else

### Pitfall 2: py.typed Not Included in Wheel
**What goes wrong:** Type checkers ignore package types despite py.typed file existing
**Why it happens:** Hatchling doesn't automatically include all non-Python files
**How to avoid:** Use `packages = ["src/viscy_transforms"]` which includes the entire directory
**Warning signs:** `mypy` or `pyright` shows "missing type stubs" for your own package

### Pitfall 3: Package Name vs Import Name Mismatch
**What goes wrong:** `pip install viscy-transforms` but `import viscy_transforms` fails
**Why it happens:** PyPI package name uses hyphen, Python import uses underscore
**How to avoid:** Directory must be `viscy_transforms` (underscore), pyproject.toml `name` can be `viscy-transforms` (hyphen)
**Warning signs:** ModuleNotFoundError after successful install

### Pitfall 4: Lock File Version Staleness
**What goes wrong:** `uv.lock` shows old version even after new tag
**Why it happens:** Dynamic versions are computed at install time, lock file is a snapshot
**How to avoid:** This is expected behavior; version in .venv will be correct
**Warning signs:** Lock file version doesn't match `uv pip show viscy-transforms`

### Pitfall 5: Workspace Dependency Not Installed
**What goes wrong:** `uv sync` doesn't install workspace member dependencies
**Why it happens:** Missing `tool.uv.sources` configuration for workspace member
**How to avoid:** For workspace-internal deps, add `[tool.uv.sources] member = { workspace = true }`
**Warning signs:** ImportError for workspace member packages

### Pitfall 6: Build Backend Type Coercion (v0.9.0+)
**What goes wrong:** uv-dynamic-versioning fails with type errors
**Why it happens:** Since v0.9.0, TOML types must be exact (no auto-coercion)
**How to avoid:** Use `bump = true` not `bump = "true"`, `strict = false` not `strict = "false"`
**Warning signs:** Error messages about unexpected type

## Code Examples

Verified patterns from official sources:

### Complete Package pyproject.toml
```toml
# Source: Synthesized from https://hatch.pypa.io/1.12/config/build/ +
# https://github.com/ninoseki/uv-dynamic-versioning

[build-system]
requires = ["hatchling", "uv-dynamic-versioning"]
build-backend = "hatchling.build"

[project]
name = "viscy-transforms"
dynamic = ["version"]
description = "Image transforms for virtual staining microscopy"
readme = "README.md"
requires-python = ">=3.11"
license = "BSD-3-Clause"
authors = [
    { name = "CZ Biohub SF", email = "compmicro@czbiohub.org" },
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
dependencies = [
    "torch>=2.4.1",
    "kornia",
    "monai>=1.4",
    "numpy",
]

[project.urls]
Homepage = "https://github.com/mehta-lab/VisCy"
Repository = "https://github.com/mehta-lab/VisCy"
Documentation = "https://mehta-lab.github.io/VisCy/"
Issues = "https://github.com/mehta-lab/VisCy/issues"

[tool.hatch.version]
source = "uv-dynamic-versioning"

[tool.uv-dynamic-versioning]
vcs = "git"
style = "pep440"
pattern-prefix = "viscy-transforms-"
fallback-version = "0.0.0"

[tool.hatch.build.targets.wheel]
packages = ["src/viscy_transforms"]

[dependency-groups]
test = [
    { include-group = "test" },
]
```

### Package __init__.py with Flat Exports
```python
# src/viscy_transforms/__init__.py
# Source: Pattern from CONTEXT.md decisions

"""VisCy Transforms - Image transforms for virtual staining microscopy."""

# Public API - flat exports
from viscy_transforms._crop import CropSampled
from viscy_transforms._flip import FlipSampled
# ... additional imports

__all__ = [
    "CropSampled",
    "FlipSampled",
    # ... all public exports
]

# Version via importlib.metadata (no __version__ attribute)
# Users can get version with: importlib.metadata.version('viscy-transforms')
```

### py.typed Marker File
```
# src/viscy_transforms/py.typed
# This file is intentionally empty.
# Its presence indicates this package supports PEP 561 type checking.
```

### Editable Install Command
```bash
# Source: https://docs.astral.sh/uv/pip/packages/
# From workspace root:
uv pip install -e packages/viscy-transforms

# Or via uv sync (if package is in workspace):
uv sync --package viscy-transforms
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `__version__ = "X.Y.Z"` | `importlib.metadata.version()` | Python 3.8+ | Single source of truth |
| setuptools + setup.py | hatchling + pyproject.toml | PEP 517/518 (2017+) | Simpler, faster builds |
| extras_require for dev | PEP 735 dependency-groups | 2024 | Better dev/test separation |
| Manual MANIFEST.in | Hatchling file selection | With hatchling | Automatic for packages |

**Deprecated/outdated:**
- `setup.py`: Replaced by declarative pyproject.toml
- `setup.cfg`: Superseded by pyproject.toml
- `__version__` attribute: Use importlib.metadata instead
- `pkg_resources`: Deprecated, use importlib.metadata

## Open Questions

Things that couldn't be fully resolved:

1. **Lock file version staleness behavior**
   - What we know: Dynamic versions in uv.lock don't auto-update, version in .venv is correct
   - What's unclear: Whether this causes issues in CI matrix testing
   - Recommendation: Accept as expected behavior; verify with `uv pip show` if needed

2. **First tag creation timing**
   - What we know: Need tag before first real release for versioning to work
   - What's unclear: Best workflow for initial development period
   - Recommendation: Use `fallback-version = "0.0.0"` until first release tag

## Sources

### Primary (HIGH confidence)
- [/pypa/hatch](https://context7.com/pypa/hatch) - Build configuration, wheel packages, src layout
- [/ofek/hatch-vcs](https://context7.com/ofek/hatch-vcs) - VCS version source patterns (for comparison)
- [/llmstxt/astral_sh_uv_llms_txt](https://docs.astral.sh/uv/) - Workspace, dependency-groups, editable install
- [ninoseki/uv-dynamic-versioning docs](https://github.com/ninoseki/uv-dynamic-versioning) - pattern-prefix, fallback-version, all options

### Secondary (MEDIUM confidence)
- [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) - pyproject.toml reference
- [PEP 561](https://peps.python.org/pep-0561/) - py.typed marker specification
- [uv-dynamic-versioning PyPI](https://pypi.org/project/uv-dynamic-versioning/) - Version info, limitations

### Tertiary (LOW confidence)
- [GitHub Issues astral-sh/uv](https://github.com/astral-sh/uv/issues) - Known issues with dynamic versioning
- Community blog posts on uv-dynamic-versioning patterns (verified against official docs)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Context7 and official docs confirm hatchling + uv-dynamic-versioning are well-supported
- Architecture: HIGH - Patterns verified from multiple official documentation sources
- Pitfalls: HIGH - Documented in GitHub issues with confirmed workarounds

**Research date:** 2026-01-28
**Valid until:** 2026-02-28 (30 days - stable ecosystem)
