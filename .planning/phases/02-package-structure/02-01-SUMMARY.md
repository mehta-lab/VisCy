---
phase: 02-package-structure
plan: 01
subsystem: package-skeleton
completed: 2026-01-28
duration: ~4 min
tags: [hatchling, uv-dynamic-versioning, src-layout, transforms]
dependency-graph:
  requires: [01-workspace-foundation]
  provides: [viscy-transforms-skeleton, editable-install]
  affects: [03-code-migration]
tech-stack:
  added: [hatchling, uv-dynamic-versioning]
  patterns: [src-layout, py-typed-marker, pep735-dependency-groups]
key-files:
  created:
    - packages/viscy-transforms/src/viscy_transforms/__init__.py
    - packages/viscy-transforms/src/viscy_transforms/py.typed
    - packages/viscy-transforms/tests/__init__.py
    - packages/viscy-transforms/pyproject.toml
    - packages/viscy-transforms/README.md
  modified:
    - uv.lock
decisions:
  - id: dep-groups-fix
    choice: Removed circular dependency-groups reference
    context: Root has 'dev' group not 'test', package referenced non-existent 'test'
    alternatives: [add-test-to-root, rename-dev-to-test]
metrics:
  tasks: 3/3
  commits: 3
  deviations: 1
---

# Phase 2 Plan 01: viscy-transforms Package Skeleton Summary

**One-liner:** Created viscy-transforms package with hatchling build, uv-dynamic-versioning for independent releases, src layout structure.

## What Was Built

Created complete package skeleton at `packages/viscy-transforms/`:

1. **Directory Structure (PKG-01)**
   - `src/viscy_transforms/__init__.py` - Package entry with docstring and `__all__`
   - `src/viscy_transforms/py.typed` - Type checker marker
   - `tests/__init__.py` - Test directory initialized

2. **Build Configuration (PKG-02, PKG-03)**
   - Hatchling build backend
   - uv-dynamic-versioning with `pattern-prefix = "viscy-transforms-"`
   - Dependencies: torch>=2.4.1, kornia, monai>=1.4, numpy

3. **Documentation (PKG-04)**
   - README.md with installation and usage instructions (54 lines)
   - Project URLs, classifiers, author info

## Commits

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Create package directory structure | 0d60cfd | __init__.py, py.typed, tests/__init__.py |
| 2 | Create package pyproject.toml | 4fa07e0 | pyproject.toml |
| 3 | Create README.md and verify install | d03c3ce | README.md, pyproject.toml fix, uv.lock |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed circular dependency-groups reference**

- **Found during:** Task 3 (uv sync failed)
- **Issue:** Package pyproject.toml had `[dependency-groups] test = [{ include-group = "test" }]` which created a self-reference cycle. The workspace root defines `dev` group, not `test`.
- **Fix:** Removed the `[dependency-groups]` section from package, added comment explaining tests use workspace dev dependencies.
- **Files modified:** packages/viscy-transforms/pyproject.toml
- **Commit:** d03c3ce

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Dependency groups fix | Remove from package | Root has `dev` not `test`; tests run via workspace anyway |
| Version fallback | 0.0.0 | Safe default for CI without tags |

## Verification Results

```
Package structure: packages/viscy-transforms/src/viscy_transforms/
__init__.py: Has __all__ = []
py.typed: Present (0 bytes marker)
pyproject.toml: hatchling + uv-dynamic-versioning configured
README.md: 54 lines with installation docs
Import test: uv run python -c "import viscy_transforms" - SUCCESS
Pre-commit: All hooks passed
```

## Next Phase Readiness

Ready for Phase 3 (Code Migration):
- Package skeleton complete
- Editable install verified (`uv pip install -e packages/viscy-transforms`)
- Import path works (`from viscy_transforms import ...`)
- Version: 0.0.0.post220.dev0+4fa07e0 (fallback + commit)

## Files Changed

```
packages/viscy-transforms/
  src/viscy_transforms/
    __init__.py (created)
    py.typed (created)
  tests/
    __init__.py (created)
  pyproject.toml (created)
  README.md (created)
uv.lock (updated)
packages/.gitkeep (removed)
```

---
*Executed: 2026-01-28*
*Duration: ~4 min*
