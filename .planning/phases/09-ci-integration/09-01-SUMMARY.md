---
phase: 09-ci-integration
plan: 01
subsystem: infra
tags: [github-actions, ci, pytest, uv, coverage]

# Dependency graph
requires:
  - phase: 08-test-migration
    provides: "viscy-data test suite in packages/viscy-data"
provides:
  - "CI test-data job: 3x3 matrix (3 OS x 3 Python) for viscy-data"
  - "CI test-data-extras job: single-combo (ubuntu-latest, Python 3.13) for extras validation"
  - "Aggregated alls-green check across all test jobs"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: ["matrix CI pattern replicated for viscy-data subpackage"]

key-files:
  created: []
  modified: [".github/workflows/test.yml"]

key-decisions:
  - "Mirrored existing viscy-transforms test job pattern for viscy-data (3x3 matrix with --all-extras)"
  - "test-data-extras uses -m 'not slow' marker convention for future differentiation"

patterns-established:
  - "Per-subpackage CI jobs: each package gets its own test job with working-directory isolation"
  - "Tiered matrix: broad 3x3 for base, narrow 1x1 for extras-specific validation"

# Metrics
duration: 1min
completed: 2026-02-14
---

# Phase 9 Plan 01: CI Integration Summary

**GitHub Actions CI extended with viscy-data test jobs: 3x3 cross-platform matrix plus single-combo extras validation, aggregated via alls-green check**

## Performance

- **Duration:** 43s
- **Started:** 2026-02-14T01:51:36Z
- **Completed:** 2026-02-14T01:52:19Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Added `test-data` job with 3x3 matrix (ubuntu/macos/windows x Python 3.11/3.12/3.13) running viscy-data tests with coverage
- Added `test-data-extras` job (ubuntu-latest, Python 3.13) with `-m "not slow"` marker for future extras-specific test differentiation
- Updated `check` job to aggregate all three test jobs: `needs: [test, test-data, test-data-extras]`

## Task Commits

Each task was committed atomically:

1. **Task 1: Add viscy-data base and extras test jobs to test.yml** - `7610899` (feat)

## Files Created/Modified
- `.github/workflows/test.yml` - Added test-data (3x3 matrix) and test-data-extras (1x1) jobs; updated check job needs

## Decisions Made
- Mirrored the existing viscy-transforms `test` job pattern exactly for `test-data` (same matrix, same uv caching, same checkout/setup steps) to maintain CI consistency
- Used `-m "not slow"` pytest marker in test-data-extras as a convention placeholder -- currently runs all tests since none are marked slow, but provides the hook for future differentiation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CI workflow now tests both viscy-transforms and viscy-data on every push/PR to main
- The alls-green check aggregates all test signals for branch protection
- This completes the v1.0 milestone CI integration

## Self-Check: PASSED

- FOUND: .github/workflows/test.yml
- FOUND: .planning/phases/09-ci-integration/09-01-SUMMARY.md
- FOUND: commit 7610899

---
*Phase: 09-ci-integration*
*Completed: 2026-02-14*
