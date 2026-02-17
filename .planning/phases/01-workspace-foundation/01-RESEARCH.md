# Phase 1: Workspace Foundation - Research

**Researched:** 2026-01-27
**Domain:** uv workspace configuration, pre-commit tooling (prek), Python type checking (ty), code quality (ruff)
**Confidence:** HIGH (Context7, Official Docs) / MEDIUM (prek, ty pre-commit integration)

## Summary

This phase establishes a uv workspace with shared tooling from a clean slate. Research focused on five key areas per the CONTEXT.md decisions:

1. **uv workspace configuration**: Well-documented with clear patterns for `[tool.uv.workspace]`, member globs, and lockfile management
2. **prek pre-commit tool**: Rust-based drop-in replacement for pre-commit, 7x faster hook installation, full `.pre-commit-config.yaml` compatibility
3. **ty type checker**: Beta release (December 2025), 10-60x faster than mypy, NO official pre-commit hook yet (manual local hook required)
4. **ruff configuration**: Comprehensive rule system with the specified rules (I, NPY, D, PD, E, F, W) fully supported
5. **Dev dependency organization**: Root-level dev dependencies are the standard pattern for shared tooling

**Primary recommendation:** Use prek with a local ty hook configuration. Place all dev dependencies (ruff, ty, pytest) in the root `[dependency-groups]` section for workspace-wide sharing.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| uv | latest | Package manager, workspace orchestration | Industry standard 2025+; native workspace support via `[tool.uv.workspace]` |
| prek | latest | Pre-commit hook runner | 7x faster than pre-commit; Rust-based; drop-in replacement |
| ruff | >=0.14.14 | Linting + formatting | Replaces flake8, isort, black; 200x faster; unified Astral toolchain |
| ty | latest (Beta) | Type checking | Astral's type checker; 10-60x faster than mypy; configurable rules |
| hatchling | >=1.28.0 | Build backend | Plugin-extensible; recommended by uv; workspace-compatible |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | >=9.0.2 | Test framework | All package testing; configure in root pyproject.toml |
| pytest-cov | latest | Coverage reporting | When coverage metrics needed |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| prek | pre-commit | pre-commit is 7x slower; requires Python runtime |
| ty | mypy | mypy is 10-60x slower; ty is Astral-native (unified toolchain) |
| ty | drop type checking | Per CONTEXT.md: "if ty doesn't work out, drop type checking rather than switch to mypy" |

**Installation:**
```bash
# prek via uvx (no installation needed)
uvx prek install
uvx prek run --all-files

# Or permanent installation
uv tool install prek
```

## Architecture Patterns

### Recommended Project Structure

```
viscy/
├── pyproject.toml           # Workspace root + meta-package + ALL tool config
├── uv.lock                   # Shared lockfile (auto-generated)
├── .pre-commit-config.yaml   # prek/pre-commit configuration
├── .gitignore
├── LICENSE
├── CITATION.cff
├── packages/                 # Workspace members
│   └── [future packages]/
│       ├── pyproject.toml    # Package-specific deps only
│       ├── src/
│       │   └── package_name/
│       └── tests/
└── scripts/                  # Workspace-level utilities
```

### Pattern 1: Virtual Workspace Root

**What:** Root pyproject.toml serves as both workspace definition AND a meta-package placeholder
**When to use:** Always for this project (per CONTEXT.md decision)
**Example:**
```toml
# Source: https://docs.astral.sh/uv/concepts/projects/workspaces/
[project]
name = "viscy"
version = "0.0.0"
requires-python = ">=3.11"
description = "VisCy workspace meta-package"

[tool.uv.workspace]
members = ["packages/*"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Pattern 2: Root-Level Dev Dependencies

**What:** All shared dev tools defined in root `[dependency-groups]`
**When to use:** For tools that apply workspace-wide (linters, formatters, test runners)
**Example:**
```toml
# Source: https://docs.astral.sh/uv/concepts/projects/dependencies/
[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "pytest-cov",
    "ruff>=0.14.14",
]
```

### Pattern 3: Shared Tool Configuration in Root

**What:** All tool configuration in root `[tool.*]` sections
**When to use:** Always (per CONTEXT.md: "All tool config in root pyproject.toml")
**Example:**
```toml
# Source: https://docs.astral.sh/ruff/configuration/
[tool.ruff]
line-length = 120
indent-width = 4

[tool.ruff.format]
quote-style = "double"
docstring-code-format = true

[tool.ruff.lint]
select = ["E", "F", "W", "I", "D", "NPY", "PD"]

[tool.ty.rules]
# Configure as needed

[tool.pytest.ini_options]
testpaths = ["packages/*/tests"]
addopts = ["--import-mode=importlib"]
```

### Anti-Patterns to Avoid

- **Scattered config files:** Don't create separate `ruff.toml`, `ty.toml`, `pytest.ini` - use root pyproject.toml
- **Per-package dev dependencies:** Don't duplicate pytest/ruff in each package's deps - use root dependency-groups
- **Missing build-system:** Root pyproject.toml MUST have `[build-system]` for uv to recognize workspace

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Import sorting | Manual ordering | ruff `I` rules | Automatic, consistent, fast |
| Type checking | Skip or custom scripts | ty check | Native Astral toolchain; fast |
| Pre-commit hooks | bash scripts | prek + .pre-commit-config.yaml | Standard format, 7x faster |
| Lockfile management | Manual pip freeze | uv.lock | Automatic, deterministic |
| Python version enforcement | README notes | `requires-python = ">=3.11"` | Enforced by uv at install time |

**Key insight:** The Astral toolchain (uv + ruff + ty + prek) provides unified, fast tooling. Avoid mixing ecosystems.

## Common Pitfalls

### Pitfall 1: Missing `[build-system]` in Root

**What goes wrong:** uv doesn't recognize the directory as a project; `uv sync` fails
**Why it happens:** Workspace-only pyproject.toml files don't require build-system
**How to avoid:** Always include `[build-system]` in root, even for meta-package
**Warning signs:** "No `pyproject.toml` found" errors from uv

### Pitfall 2: Using mypy Instead of ty

**What goes wrong:** Configuration conflicts; slower CI; mixed toolchain
**Why it happens:** Habit from pre-Astral tooling
**How to avoid:** Per CONTEXT.md: use ty or drop type checking entirely
**Warning signs:** Pre-commit config references `mirrors-mypy`

### Pitfall 3: ty Pre-commit Hook Expectations

**What goes wrong:** Expecting official ty pre-commit hook that doesn't exist
**Why it happens:** ty is Beta; official pre-commit support pending
**How to avoid:** Use local hook with uvx: `entry: uvx ty check`
**Warning signs:** Looking for `repo: https://github.com/astral-sh/ty-pre-commit`

### Pitfall 4: Inconsistent Ruff Rules Across Packages

**What goes wrong:** Different linting in different packages; merge conflicts
**Why it happens:** Defining `[tool.ruff]` in package pyproject.toml
**How to avoid:** ALL ruff config in root only; per-file-ignores for exceptions
**Warning signs:** Package-level `[tool.ruff]` sections

### Pitfall 5: pytest Configuration Discovery

**What goes wrong:** pytest doesn't find tests or uses wrong config
**Why it happens:** pytest searches upward for config; workspace structure confuses it
**How to avoid:** Explicit `testpaths` in root config; use `--import-mode=importlib`
**Warning signs:** "No tests found" when tests exist

## Code Examples

### Complete Root pyproject.toml

```toml
# Source: Context7 uv docs + ruff docs + ty docs

[project]
name = "viscy"
version = "0.0.0"
requires-python = ">=3.11"
description = "VisCy workspace meta-package"
readme = "README.md"
license = "BSD-3-Clause"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv.workspace]
members = ["packages/*"]

[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "pytest-cov",
    "ruff>=0.14.14",
]

# ============== RUFF CONFIGURATION ==============
[tool.ruff]
line-length = 120
indent-width = 4
target-version = "py311"
src = ["packages/*/src"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true
skip-magic-trailing-comma = false

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "F",      # pyflakes
    "W",      # pycodestyle warnings
    "I",      # isort
    "D",      # pydocstyle
    "NPY",    # numpy-specific rules
    "PD",     # pandas-vet
]

[tool.ruff.lint.pydocstyle]
convention = "numpy"

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401", "D104"]  # Allow unused imports and missing docstrings
"tests/**" = ["D"]                 # No docstrings required in tests

# ============== TY CONFIGURATION ==============
[tool.ty.environment]
python-version = "3.11"

[tool.ty.rules]
# Start permissive, tighten over time
unresolved-import = "warn"
possibly-unresolved-reference = "warn"

# ============== PYTEST CONFIGURATION ==============
[tool.pytest.ini_options]
minversion = "9.0"
testpaths = ["packages/*/tests"]
addopts = [
    "-ra",
    "-q",
    "--import-mode=importlib",
]
pythonpath = ["."]
```

### .pre-commit-config.yaml for prek

```yaml
# Source: https://github.com/j178/prek + https://docs.astral.sh/ruff/integrations/
# Works with both prek and pre-commit

repos:
  # Ruff linting and formatting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.14.14
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format

  # ty type checking (local hook - no official pre-commit repo yet)
  - repo: local
    hooks:
      - id: ty
        name: ty type checker
        entry: uvx ty check
        language: system
        types: [python]
        pass_filenames: false
```

### Package pyproject.toml (minimal)

```toml
# Source: https://docs.astral.sh/uv/concepts/projects/workspaces/
# packages/viscy-transforms/pyproject.toml

[project]
name = "viscy-transforms"
version = "0.1.0"
description = "GPU augmentation transforms for VisCy"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4.1",
    "kornia",
    "numpy",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/viscy_transforms"]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| pre-commit (Python) | prek (Rust) | 2025 | 7x faster hook installation |
| mypy type checking | ty (Beta) | Dec 2025 | 10-60x faster type checking |
| black + isort + flake8 | ruff | 2024 | Single tool, 200x faster |
| pip + venv | uv | 2024 | 10-100x faster, native workspaces |
| setuptools | hatchling | 2024 | Simpler config, plugin support |

**Deprecated/outdated:**
- **pre-commit**: Still works but prek is faster and Rust-native
- **mypy**: Per CONTEXT.md, use ty or nothing; don't use mypy
- **Separate config files**: Modern practice is pyproject.toml-only

## Open Questions

1. **ty Pre-commit Official Support**
   - What we know: Issue #269 tracks this; manual local hook works
   - What's unclear: Timeline for official `ty-pre-commit` repo
   - Recommendation: Use local hook with `uvx ty check`; update when official support lands

2. **prek Built-in Hooks**
   - What we know: prek has `repo: builtin` for Rust-native hooks
   - What's unclear: Full list of available built-in hooks
   - Recommendation: Use standard repos (ruff-pre-commit) for now; explore built-ins later

3. **ty Configuration Completeness**
   - What we know: `[tool.ty.rules]` and `[tool.ty.environment]` documented
   - What's unclear: Full rule list and all configuration options (Beta status)
   - Recommendation: Start with permissive config, tighten as ty matures

## Sources

### Primary (HIGH confidence)
- `/llmstxt/astral_sh_uv_llms_txt` - Context7: workspace configuration, dependency-groups, lockfile
- `/websites/astral_sh_ruff` - Context7: lint rules, format config, per-file-ignores
- `/websites/astral_sh_ty` - Context7: rule configuration, environment settings, CLI usage
- [uv Workspaces Documentation](https://docs.astral.sh/uv/concepts/projects/workspaces/) - Official workspace patterns
- [Ruff Configuration](https://docs.astral.sh/ruff/configuration/) - Official ruff config

### Secondary (MEDIUM confidence)
- [prek GitHub](https://github.com/j178/prek) - Installation, configuration, performance benchmarks
- [ty Pre-commit Issue #269](https://github.com/astral-sh/ty/issues/269) - Local hook workarounds
- [Hugo van Kemenade: Ready prek go!](https://hugovk.dev/blog/2025/ready-prek-go/) - prek adoption guide
- [uv Monorepo Best Practices Issue](https://github.com/astral-sh/uv/issues/10960) - Community patterns

### Tertiary (LOW confidence)
- WebSearch results for "prek pre-commit ty ruff 2025" - General ecosystem state

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Context7 + official docs for uv, ruff, ty
- Architecture: HIGH - uv workspace patterns well-documented
- Pitfalls: MEDIUM - Gathered from issues, community experience
- prek integration: MEDIUM - GitHub docs, not Context7
- ty pre-commit: MEDIUM - Issue tracker, workarounds documented

**Research date:** 2026-01-27
**Valid until:** 2026-02-27 (30 days - ty is fast-moving Beta)
