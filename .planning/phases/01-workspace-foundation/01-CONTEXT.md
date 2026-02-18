# Phase 1: Workspace Foundation - Context

**Gathered:** 2025-01-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish a uv workspace with shared tooling from a clean slate. The repo transitions from monolithic to workspace structure while preserving git history. Output is a working development environment ready for subpackages.

</domain>

<decisions>
## Implementation Decisions

### Clean Slate Approach
- Preserve full git history — this is a refactor, not a new repo
- Remove existing code files but keep: LICENSE, CITATION.cff, .gitignore, .github/, docs/
- Single atomic commit for the clean slate transition
- No backup branch needed — git history provides rollback capability

### Workspace Configuration
- Workspace member pattern: `packages/*` glob (automatic membership)
- Root pyproject.toml is a `viscy` meta-package (not just a container)
- Meta-package will eventually hold shared application/paper functionality; for Phase 1 it's a placeholder
- Individual packages remain primary import targets (`from viscy_transforms import X`)
- **Research flag:** Dev dependency organization (root-only vs package-specific vs mixed) — researcher to investigate real-world uv workspace patterns

### Pre-commit Setup (via prek)
- Use `ty` (Astral's type checker) instead of mypy — if it doesn't work out, drop type checking rather than switch to mypy
- Ruff configuration:
  - Format: double quotes, spaces for indent, line-length=120, docstring-code-format=true
  - Rules: I (isort), NPY (numpy), D (pydocstyle), PD (pandas-vet), E, F, W
  - Format suppression support enabled
- Hooks run on staged files only (not entire codebase)
- CI runs `uvx prek` — same checks as local

### Directory Structure
- All tool config in root pyproject.toml `[tool.*]` sections (no separate config files)
- src layout for packages: `packages/viscy-transforms/src/viscy_transforms/`
- Tests at package level: `packages/viscy-transforms/tests/`
- Workspace-level `scripts/` directory at root for utilities

### Claude's Discretion
- Exact pyproject.toml structure and metadata
- prek configuration details
- .gitignore updates for workspace structure
- Any additional workspace boilerplate

</decisions>

<specifics>
## Specific Ideas

- Unified Astral toolchain: uv + ruff + ty (experimental)
- Clean import paths: `from viscy_transforms import X` as primary usage pattern
- Git history shows clear transition point for the modularization

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-workspace-foundation*
*Context gathered: 2025-01-27*
