---
phase: 22-batch-sampling
plan: 01
subsystem: data
tags: [sampler, batch, pytorch, numpy, ddp, contrastive-learning]

# Dependency graph
requires:
  - phase: 21-cell-index-lineage
    provides: "valid_anchors DataFrame with experiment/condition columns and reset_index(drop=True)"
provides:
  - "FlexibleBatchSampler(Sampler[list[int]]) in viscy_data.sampler"
  - "Experiment-aware batching restricting each batch to a single experiment"
  - "Condition balancing within experiment-restricted batches"
  - "Leaky mixing injecting cross-experiment samples"
  - "DDP rank-aware interleaved batch partitioning"
  - "Deterministic sampling via np.random.default_rng(seed + epoch)"
affects: [22-02-PLAN, 24-datamodule]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Cascade batch construction: experiment -> condition -> sample"
    - "Pre-computed group indices at __init__ for O(1) lookup"
    - "Interleaved DDP batch partitioning via rank slicing"
    - "Replacement sampling fallback for small groups with logged warning"

key-files:
  created:
    - packages/viscy-data/src/viscy_data/sampler.py
    - packages/viscy-data/tests/test_sampler.py
  modified:
    - packages/viscy-data/src/viscy_data/__init__.py

key-decisions:
  - "numpy RNG (np.random.default_rng) over torch Generator for weighted choice ergonomics"
  - "Proportional experiment weights by default (larger experiments sampled more often)"
  - "Condition balancing uses last-condition-gets-remainder to avoid rounding issues"
  - "DDP via interleaved batch slicing: all ranks generate same batch list, each takes rank::num_replicas"

patterns-established:
  - "FlexibleBatchSampler cascade: _build_one_batch calls _sample_condition_balanced"
  - "Pre-computed _experiment_indices, _exp_cond_indices, _condition_indices dicts at init"
  - "set_epoch(n) + seed for deterministic DDP-safe sampling"

# Metrics
duration: 6min
completed: 2026-02-22
---

# Phase 22 Plan 01: FlexibleBatchSampler Summary

**FlexibleBatchSampler with cascade batch construction: experiment-aware restriction, condition balancing, and leaky cross-experiment mixing using numpy RNG**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-23T04:03:31Z
- **Completed:** 2026-02-23T04:09:40Z
- **Tasks:** 3 (TDD: RED, GREEN, REFACTOR)
- **Files modified:** 3

## Accomplishments
- FlexibleBatchSampler(Sampler[list[int]]) with 329 lines implementing cascade batch construction
- 19-test TDD suite covering all 5 plan truths plus DDP and protocol tests
- FlexibleBatchSampler exported from viscy_data package public API

## Task Commits

Each task was committed atomically (TDD flow):

1. **RED: Failing tests** - `f12e128` (test)
2. **GREEN: Implementation** - `fe38805` (feat)
3. **REFACTOR: Package export + lint** - `4b89f53` (refactor)

## Files Created/Modified
- `packages/viscy-data/src/viscy_data/sampler.py` - FlexibleBatchSampler with experiment-aware, condition-balanced, leaky mixing
- `packages/viscy-data/tests/test_sampler.py` - 19-test TDD suite for core sampling axes
- `packages/viscy-data/src/viscy_data/__init__.py` - Added FlexibleBatchSampler to public API

## Decisions Made
- Used numpy `np.random.default_rng(seed + epoch)` over torch Generator for `rng.choice(p=weights)` ergonomics
- Default experiment weights proportional to group size (larger experiments sampled more often), not uniform
- Condition balancing assigns last condition the remainder to prevent rounding-induced batch size mismatch
- DDP interleaved batch slicing: all ranks generate identical full batch list from same seed, each rank takes every Nth batch

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- FlexibleBatchSampler ready for Plan 02 extension with temporal enrichment and DDP tests
- `valid_anchors` with `hours_post_infection` column needed for temporal enrichment (already available from Phase 21)
- Package export in place for downstream Phase 24 DataModule wiring

## Self-Check: PASSED

- All 3 files exist (sampler.py, test_sampler.py, SUMMARY.md)
- All 3 commits verified (f12e128, fe38805, 4b89f53)
- sampler.py: 329 lines (min: 150)
- test_sampler.py: 569 lines (min: 200)
- Key links verified: test import, Sampler subclass pattern

---
*Phase: 22-batch-sampling*
*Completed: 2026-02-22*
