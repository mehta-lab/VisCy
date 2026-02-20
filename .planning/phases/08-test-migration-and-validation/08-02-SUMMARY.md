---
phase: 08-test-migration-and-validation
plan: 02
subsystem: testing
tags: [pytest, smoke-tests, import-validation, viscy-data, __all__]

# Dependency graph
requires:
  - phase: 07-code-migration
    provides: "viscy_data package with 45 public API exports and optional dep guards"
provides:
  - "Smoke tests verifying base import, __all__ completeness (45 names), optional dep error messages, and no legacy namespace leakage"
affects: [08-test-migration-and-validation]

# Tech tracking
tech-stack:
  added: [inspect.getsource]
  patterns: [parametrized-smoke-tests, source-inspection-for-error-messages]

key-files:
  created:
    - packages/viscy-data/tests/test_smoke.py
  modified: []

key-decisions:
  - "Used inspect.getsource() to verify optional dep error messages instead of mocking imports -- works regardless of dep installation state"
  - "Parametrized __all__ tests so each of 45 exports appears as a separate test case for clear reporting"

patterns-established:
  - "Source inspection pattern: verify error message content via inspect.getsource when import guards cannot be triggered directly"

# Metrics
duration: 5min
completed: 2026-02-14
---

# Phase 08 Plan 02: Smoke Tests Summary

**52 pytest smoke tests covering base import, all 45 public API names, 4 optional-dep error message patterns, and no-legacy-namespace assertion**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-14T01:28:55Z
- **Completed:** 2026-02-14T01:34:09Z
- **Tasks:** 1
- **Files created:** 1

## Accomplishments
- Created comprehensive smoke test suite (52 individual test cases) for viscy_data package
- All 45 names in `__all__` individually verified importable via parametrized test
- Pinned `__all__` count at 45 to catch accidental additions/removals
- Verified 4 optional-dep modules (triplet, mmap_cache, livecell, cell_classification) contain `pip install` error message hints via source inspection
- Confirmed `viscy_data` import does not pull in legacy `viscy.data` namespace

## Task Commits

Each task was committed atomically:

1. **Task 1: Create test_smoke.py with import, __all__, and error message tests** - `819d589` (test)

**Note:** test_smoke.py was bundled in the 08-01 commit due to staging overlap. Content is complete and verified.

## Files Created/Modified
- `packages/viscy-data/tests/test_smoke.py` - 5 test functions (some parametrized to 52 cases) covering import validation, __all__ completeness, optional dep error messages, and legacy namespace independence

## Decisions Made
- Used `inspect.getsource()` to verify optional dep error messages exist in module source rather than attempting to mock imports -- this approach works regardless of whether optional deps are installed in the test environment
- Parametrized `test_all_exports_importable` over `viscy_data.__all__` so each of the 45 exports shows as a separate test case for maximum visibility

## Deviations from Plan

None - plan executed exactly as written. The test file was inadvertently included in the 08-01 commit (819d589) due to staging overlap, but the content matches the plan specification exactly.

## Issues Encountered
- test_smoke.py was staged and committed together with conftest.py in the 08-01 plan commit (819d589). This is a commit-level deviation only; the file content and test coverage match the plan specification exactly. All 52 tests pass.
- Triplet test failures in test_triplet.py are pre-existing (tensorstore not installed in env) and unrelated to this plan.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- DATA-TST-02 satisfied: all smoke tests pass
- Full test suite ready for combined validation with Plan 01 tests
- 56 tests pass across smoke + hcs test files (triplet tests require optional dep)

## Self-Check: PASSED

- [x] `packages/viscy-data/tests/test_smoke.py` -- FOUND
- [x] Commit `819d589` -- FOUND
- [x] `08-02-SUMMARY.md` -- FOUND
- [x] All 52 smoke tests pass

---
*Phase: 08-test-migration-and-validation*
*Completed: 2026-02-14*
