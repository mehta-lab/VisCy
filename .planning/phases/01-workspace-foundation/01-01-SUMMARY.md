---
phase: 01-workspace-foundation
plan: 01
subsystem: infra
tags: [uv, workspace, pyproject, ruff, pytest, hatchling]

# Dependency graph
requires: []
provides:
  - uv workspace root configuration
  - shared tooling configuration (ruff, ty, pytest)
  - packages/ directory for workspace members
  - scripts/ directory for workspace utilities
  - reproducible uv.lock lockfile
affects: [02-package-structure, 03-code-migration]

# Tech tracking
tech-stack:
  added: [uv, ruff, pytest, pytest-cov, hatchling]
  patterns: [virtual-workspace-root, dependency-groups]

key-files:
  created:
    - pyproject.toml
    - uv.lock
    - packages/.gitkeep
    - scripts/.gitkeep
  modified:
    - .gitignore

key-decisions:
  - "Virtual workspace root with package=false (not a buildable package)"
  - "Project name viscy-workspace to distinguish from future viscy package"
  - "ruff version >=0.11.0 for latest features"

patterns-established:
  - "Workspace root defines shared tool config, individual packages inherit"
  - "Dev dependencies in dependency-groups, not project.dependencies"
  - "uv.lock tracked for reproducibility"

# Metrics
duration: 3min
completed: 2026-01-28
---

# Phase 01 Plan 01: Workspace Foundation Summary

**Clean slate uv workspace with virtual root, ruff/pytest/ty configuration, and packages/scripts directory structure**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-28T17:08:13Z
- **Completed:** 2026-01-28T17:10:59Z
- **Tasks:** 3
- **Files modified:** 264 deleted, 5 created/modified

## Accomplishments

- Removed all monolithic code (viscy/, tests/, docs/, applications/, examples/)
- Created uv workspace pyproject.toml with virtual root configuration
- Configured shared tooling: ruff (line-length=120, numpy docstrings), ty, pytest
- Generated uv.lock for reproducible dependency management
- Established packages/ and scripts/ directory structure

## Task Commits

All tasks were committed atomically in a single commit:

1. **Task 1: Clean slate transition** - `956337e` (feat)
2. **Task 2: Create workspace pyproject.toml** - `956337e` (feat)
3. **Task 3: Initialize workspace and verify** - `956337e` (feat)

**Combined commit:** `956337e` feat(01-01): establish uv workspace foundation

## Files Created/Modified

- `pyproject.toml` - Virtual workspace root with tool configuration
- `uv.lock` - Shared lockfile (271 lines)
- `packages/.gitkeep` - Workspace members directory placeholder
- `scripts/.gitkeep` - Workspace utilities directory placeholder
- `.gitignore` - Updated with workspace patterns (.ruff_cache/, .venv/, etc.)

## Decisions Made

1. **Virtual workspace root:** Used `[tool.uv] package = false` to mark root as non-buildable. This avoids hatchling trying to find source files and allows the root to serve purely as configuration.

2. **Project name:** Changed from `viscy` to `viscy-workspace` to distinguish from the future `viscy` meta-package that will be created in packages/.

3. **Ruff version:** Specified `ruff>=0.11.0` instead of exact version for flexibility while ensuring modern features.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed hatchling build failure for workspace root**
- **Found during:** Task 3 (uv sync)
- **Issue:** hatchling couldn't determine what files to ship since workspace root has no source code
- **Fix:** Added `[tool.uv] package = false` to mark as virtual workspace root
- **Files modified:** pyproject.toml
- **Verification:** `uv sync` completed successfully
- **Committed in:** 956337e (main commit)

**2. [Rule 3 - Blocking] Added .ruff_cache to .gitignore**
- **Found during:** Task 3 verification
- **Issue:** Ruff created cache directory that shouldn't be tracked
- **Fix:** Added `.ruff_cache/` and `.mypy_cache/` to .gitignore
- **Files modified:** .gitignore
- **Verification:** Directory excluded from git status
- **Committed in:** 956337e (main commit)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both fixes necessary for workspace functionality. No scope creep.

## Issues Encountered

None beyond the auto-fixed blocking issues.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Workspace foundation complete and verified
- Ready for Phase 01-02: Package scaffolding (viscy-transforms, viscy-networks, etc.)
- All shared tooling configuration in place for packages to inherit

---
*Phase: 01-workspace-foundation*
*Completed: 2026-01-28*
