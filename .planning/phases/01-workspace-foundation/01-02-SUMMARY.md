---
phase: 01-workspace-foundation
plan: 02
subsystem: infra
tags: [pre-commit, prek, ruff, ty, hooks, quality-gates]

# Dependency graph
requires:
  - phase: 01-01
    provides: pyproject.toml with ruff and ty configuration
provides:
  - pre-commit hook configuration (.pre-commit-config.yaml)
  - automated linting on commit (ruff-check with --fix)
  - automated formatting on commit (ruff-format)
  - automated type checking on commit (ty)
affects: [02-package-structure, 03-code-migration]

# Tech tracking
tech-stack:
  added: [prek, ruff-pre-commit]
  patterns: [local-hooks-for-uvx-tools]

key-files:
  created:
    - .pre-commit-config.yaml
  modified: []

key-decisions:
  - "Use prek (uvx prek) instead of pre-commit for faster hook execution"
  - "Use local hook for ty since no official pre-commit repo exists yet"
  - "ruff-pre-commit v0.14.14 (latest, separate version from ruff package)"

patterns-established:
  - "Local hooks use uvx for tool execution (entry: uvx ty check)"
  - "pass_filenames: false for project-wide type checkers"
  - "ruff-check with --fix runs before ruff-format"

# Metrics
duration: 2min
completed: 2026-01-28
---

# Phase 01 Plan 02: Pre-commit Configuration Summary

**Pre-commit hooks with ruff-check (--fix), ruff-format, and ty type checker using prek for fast local quality gates**

## Performance

- **Duration:** 2 min
- **Started:** 2026-01-28T17:12:00Z
- **Completed:** 2026-01-28T17:14:00Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments

- Created .pre-commit-config.yaml with ruff and ty hooks
- Installed pre-commit hooks via prek (uvx prek install)
- Verified hooks run on commit (ruff-check, ruff-format, ty)
- All hooks pass on empty workspace (no files to check status)

## Task Commits

All tasks were committed atomically:

1. **Task 1: Create pre-commit configuration** - `14e3edd` (feat)
2. **Task 2: Install and verify hooks** - no commit (runtime verification)
3. **Task 3: Commit pre-commit configuration** - `14e3edd` (feat)

**Main commit:** `14e3edd` feat(01-02): configure pre-commit hooks with ruff and ty

## Files Created/Modified

- `.pre-commit-config.yaml` - Pre-commit hook configuration for ruff and ty

## Decisions Made

1. **ruff-pre-commit v0.14.14:** Used latest version of ruff-pre-commit repo (separate from ruff package version in pyproject.toml). The plan specified v0.14.14 which is the current latest.

2. **Local hook for ty:** Since ty has no official pre-commit repo yet (per RESEARCH.md), configured as local hook using `uvx ty check`. This pattern allows using any uvx-available tool.

3. **pass_filenames: false for ty:** ty scans the entire project, not individual files, so filenames are not passed to the command.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ruff-pre-commit version**
- **Found during:** Task 1 (Create pre-commit configuration)
- **Issue:** Initially used v0.11.0 (ruff package version from pyproject.toml) but this version doesn't exist in ruff-pre-commit repo
- **Fix:** Updated to v0.14.14 as specified in the plan
- **Files modified:** .pre-commit-config.yaml
- **Verification:** `uvx prek run --all-files` succeeded
- **Committed in:** 14e3edd (main commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Fix was necessary for hooks to work. No scope creep.

## Issues Encountered

None beyond the auto-fixed version issue.

## User Setup Required

None - hooks are installed automatically via `uvx prek install`.

## Next Phase Readiness

- Phase 1 complete: workspace foundation and quality gates established
- Ready for Phase 2: Package Structure (viscy-transforms, viscy-networks, etc.)
- Pre-commit hooks will validate code as it's added to packages

### Phase 1 Complete Verification

All ROADMAP success criteria verified:

1. Clean slate with LICENSE, CITATION.cff, .gitignore, new structure
2. `uv sync` works (dry-run shows "Would make no changes")
3. `uvx prek run --all-files` passes (all hooks skip with no Python files)
4. Python 3.11+ enforced (`requires-python = ">=3.11"`)
5. packages/ is workspace member (`members = ["packages/*"]`)

---
*Phase: 01-workspace-foundation*
*Completed: 2026-01-28*
