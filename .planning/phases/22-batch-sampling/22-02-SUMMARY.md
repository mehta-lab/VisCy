---
phase: 22-batch-sampling
plan: 02
subsystem: data
tags: [sampler, batch, pytorch, numpy, ddp, temporal-enrichment, contrastive-learning]

# Dependency graph
requires:
  - phase: 22-batch-sampling/01
    provides: "FlexibleBatchSampler core with experiment-aware, condition-balanced, leaky mixing"
  - phase: 21-cell-index-lineage
    provides: "valid_anchors DataFrame with hours_post_infection column"
provides:
  - "FlexibleBatchSampler temporal enrichment (SAMP-03): focal HPI concentration with configurable window and global fraction"
  - "FlexibleBatchSampler DDP support (SAMP-04): deterministic rank-aware interleaved batch partitioning"
  - "Column validation guards: experiment/condition/hours_post_infection checked only when feature enabled"
  - "FlexibleBatchSampler importable from viscy_data top-level package"
  - "Complete 5-axis FlexibleBatchSampler satisfying all SAMP requirements"
affects: [24-datamodule]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Temporal enrichment: focal/global split sampling from experiment pool"
    - "Conditional precomputation: only groupby columns when feature enabled"
    - "Column validation guards at __init__ with descriptive error messages"

key-files:
  created: []
  modified:
    - packages/viscy-data/src/viscy_data/sampler.py
    - packages/viscy-data/tests/test_sampler.py
    - packages/viscy-data/tests/test_smoke.py

key-decisions:
  - "Temporal enrichment replaces plain sampling (not post-filter): draws focal+global directly from experiment pool for correct concentration"
  - "Conditional precomputation: groupby only runs for enabled features, avoiding KeyError on missing columns"
  - "temporal_global_fraction=0.0 means entire batch from focal window; 1.0 means no enrichment effect"

patterns-established:
  - "_enrich_temporal: picks focal HPI from pool's unique values, splits into focal/global pools, samples with replacement fallback"
  - "Validation guards pattern: check column presence in __init__ before any precomputation"

# Metrics
duration: 7min
completed: 2026-02-22
---

# Phase 22 Plan 02: FlexibleBatchSampler Temporal Enrichment + DDP Summary

**Temporal enrichment with focal HPI concentration, column validation guards, and 35-test TDD suite completing all 5 SAMP requirements**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-23T04:12:16Z
- **Completed:** 2026-02-23T04:19:39Z
- **Tasks:** 2 (TDD: RED, GREEN; no refactor needed)
- **Files modified:** 3

## Accomplishments
- FlexibleBatchSampler extended with temporal_enrichment, temporal_window_hours, temporal_global_fraction parameters (501 lines total)
- 35-test TDD suite (968 lines) covering all 5 SAMP requirements: experiment-aware, condition-balanced, temporal enrichment, DDP, leaky mixing
- Column validation guards prevent cryptic KeyError at precomputation time
- Full viscy-data test suite passes (107 tests, 0 failures)

## Task Commits

Each task was committed atomically (TDD flow):

1. **RED: Failing tests** - `7a40b6f` (test)
2. **GREEN: Implementation** - `7de55ee` (feat)

No refactor commit needed -- implementation was clean.

## Files Created/Modified
- `packages/viscy-data/src/viscy_data/sampler.py` - Added temporal enrichment, validation guards, conditional precomputation (329 -> 501 lines)
- `packages/viscy-data/tests/test_sampler.py` - Added 16 new tests for temporal enrichment, DDP coverage, validation guards, package import (569 -> 968 lines)
- `packages/viscy-data/tests/test_smoke.py` - Fixed stale __all__ count (45 -> 46)

## Decisions Made
- Temporal enrichment draws focal+global directly from the experiment pool (not post-filtering a pre-sampled primary), ensuring correct concentration even with small batch sizes
- Conditional precomputation: groupby("experiment") only runs when experiment_aware=True, avoiding KeyError on DataFrames lacking that column
- temporal_global_fraction=0.0 yields all-focal batches; temporal_global_fraction=1.0 yields effectively no enrichment

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed stale smoke test __all__ count**
- **Found during:** Task 2 (GREEN: implementation)
- **Issue:** test_smoke.py::test_all_count expected 45 names in __all__ but Plan 01 added FlexibleBatchSampler making it 46
- **Fix:** Updated expected count from 45 to 46
- **Files modified:** packages/viscy-data/tests/test_smoke.py
- **Verification:** 107/107 tests pass
- **Committed in:** 7de55ee (part of GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Stale test count from prior plan. No scope creep.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- FlexibleBatchSampler complete with all 5 SAMP axes (experiment, condition, temporal, DDP, leaky)
- Ready for Phase 24 DataModule wiring (FlexibleBatchSampler as batch_sampler in DataLoader)
- Package export verified: `from viscy_data import FlexibleBatchSampler`

## Self-Check: PASSED

- All 4 files exist (sampler.py, test_sampler.py, test_smoke.py, __init__.py)
- All 2 commits verified (7a40b6f, 7de55ee)
- sampler.py: 501 lines (min: 220)
- test_sampler.py: 968 lines (min: 350)
- Key links verified: init import, __all__ entry, hpi column, temporal_enrichment param

---
*Phase: 22-batch-sampling*
*Completed: 2026-02-22*
