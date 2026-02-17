# Phase 2: Package Structure - Context

**Gathered:** 2026-01-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Create viscy-transforms package skeleton with modern build system (hatchling + uv-dynamic-versioning). Package lives in `packages/viscy-transforms/` with src layout. This phase establishes the package structure only — code migration is Phase 3.

</domain>

<decisions>
## Implementation Decisions

### Public API Design
- Flat top-level exports (matches original VisCy pattern)
- `from viscy_transforms import NormalizeSampled` — all transforms at top level
- Private modules with underscore prefix (`_crop.py`, `_flip.py`, etc.)
- Explicit `__all__` list in `__init__.py` for public API control
- No `__version__` attribute — use `importlib.metadata.version('viscy-transforms')` instead
- Include `py.typed` marker for mypy/pyright type checker support

### Package Metadata
- Author: CZ Biohub SF (`compmicro@czbiohub.org`) — matches original
- License: BSD-3-Clause — same as original VisCy
- Include all project URLs (homepage, repository, documentation, issues)
- Python classifiers: 3.11, 3.12, 3.13 (conservative, skip 3.14 until released)

### Versioning Approach
- Use uv-dynamic-versioning with `pattern-prefix` for independent package versioning
- Tag convention: `viscy-transforms-v0.1.0` → version `0.1.0`
- Configuration:
  ```toml
  [tool.uv-dynamic-versioning]
  vcs = "git"
  style = "pep440"
  pattern-prefix = "viscy-transforms-"
  fallback-version = "0.0.0"
  ```
- Main `viscy` package uses unprefixed tags (`v1.0.0`)
- Each subpackage in `packages/` has its own independent release cadence

### Dependencies
- Core dependencies (minimal, only what transforms use):
  - `torch>=2.4.1`
  - `kornia`
  - `monai>=1.4`
  - `numpy`
- No optional dependency extras needed
- Test deps via PEP 735 dependency-groups:
  ```toml
  [dependency-groups]
  test = [
      { include-group = "test" },  # Inherit from workspace root
  ]
  ```

### Test Structure
- Workspace root defines shared test deps: `pytest>=8.0`, `pytest-cov`, `hypothesis`
- Per-package test groups inherit from workspace and can add package-specific deps
- Isolated tests: `uv run --package viscy-transforms pytest`
- Integration tests: `uv run pytest` (all packages)

### Claude's Discretion
- Exact pyproject.toml formatting and section ordering
- README.md structure and content
- Test configuration details (pytest.ini options, coverage settings)

</decisions>

<specifics>
## Specific Ideas

- Keep same README content from original VisCy for root package
- Follow iohub pyproject.toml pattern for hatchling + uv-dynamic-versioning setup
- Package should feel like a standalone library that happens to live in a monorepo

</specifics>

<deferred>
## Deferred Ideas

- **Root package as buildable viscy**: Change workspace root from virtual (`package=false`) to buildable `viscy` package with `src/viscy/`. This requires updating Phase 1's foundation — capture as patch phase or roadmap update.
- **Re-export from main viscy**: Later `src/viscy/` will import and re-export from subpackages like viscy-transforms. Deferred to much later stage per design doc.

</deferred>

---

*Phase: 02-package-structure*
*Context gathered: 2026-01-28*
