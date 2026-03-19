---
phase: 24-dataset-datamodule
plan: 01
subsystem: data
tags: [dataset, tensorstore, lineage, contrastive, triplet, zarr]

# Dependency graph
requires:
  - phase: 21-cell-index-lineage
    provides: "MultiExperimentIndex with tracks, valid_anchors, lineage_id"
  - phase: 20-experiment-registry
    provides: "ExperimentRegistry with channel_maps, tau_range_frames"
  - phase: 23-loss-augmentation
    provides: "sample_tau for exponential decay temporal offset sampling"
provides:
  - "MultiExperimentTripletDataset class with __getitems__ returning ContrastiveModule-compatible batch dicts"
  - "Lineage-aware positive sampling following division events via shared lineage_id"
  - "Per-experiment channel remapping using registry.channel_maps"
  - "Tensorstore I/O with SLURM-aware context and per-FOV caching"
affects: [24-02-datamodule, dynaclr-training]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lineage-timepoint lookup: defaultdict((experiment, lineage_id) -> {t: [row_indices]}) for O(1) positive candidate search"
    - "Fallback tau scanning: try sampled tau first, then scan full range if not found"
    - "Per-experiment channel remapping via sorted channel_map keys"

key-files:
  created:
    - "applications/dynaclr/src/dynaclr/dataset.py"
    - "applications/dynaclr/tests/test_dataset.py"
  modified:
    - "applications/dynaclr/src/dynaclr/__init__.py"

key-decisions:
  - "Lineage-timepoint pre-built lookup indexed by (experiment, lineage_id) -> {t: [row_indices]} for O(1) positive candidate retrieval"
  - "Fallback tau strategy: sample_tau first, then linear scan of full tau range if no candidate at sampled offset"
  - "Dataset uses numpy.random.default_rng() without fixed seed; determinism delegated to external sampler"
  - "INDEX_COLUMNS optional columns (y, x, z) silently skipped in predict mode for compatibility"

patterns-established:
  - "MultiExperimentTripletDataset follows same tensorstore pattern as TripletDataset (_get_tensorstore, _slice_patches, ts.stack)"
  - "Batch dict keys match TripletSample TypedDict: anchor, positive, anchor_norm_meta, positive_norm_meta, index"

# Metrics
duration: 4min
completed: 2026-02-23
---

# Phase 24 Plan 01: MultiExperimentTripletDataset Summary

**MultiExperimentTripletDataset with lineage-aware positive sampling via pre-built (experiment, lineage_id) lookup and per-experiment channel remapping**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-23T21:49:47Z
- **Completed:** 2026-02-23T21:54:02Z
- **Tasks:** 1 (TDD: RED -> GREEN -> REFACTOR)
- **Files modified:** 3

## Accomplishments
- MultiExperimentTripletDataset.__getitems__ returns batch dicts with anchor/positive Tensors (B,C,Z,Y,X) compatible with ContrastiveModule.training_step
- Lineage-aware positive sampling follows division events naturally via shared lineage_id, using pre-built O(1) lookup structure
- Per-experiment channel remapping using registry.channel_maps ensures correct zarr index extraction across experiments with different channel orderings
- 7 TDD tests covering return format, norm_meta, lineage matching, division traversal, channel remapping, predict mode, and dataset length

## Task Commits

Each task was committed atomically (TDD):

1. **Task 1 RED: Failing tests** - `ec4aebb` (test)
2. **Task 1 GREEN: Implementation** - `835f1a8` (feat)
3. **Task 1 REFACTOR: Package exports** - `ae5a30d` (refactor)

## Files Created/Modified
- `applications/dynaclr/src/dynaclr/dataset.py` - MultiExperimentTripletDataset with __getitems__, lineage-aware positive sampling, tensorstore I/O
- `applications/dynaclr/tests/test_dataset.py` - 7 TDD tests with synthetic zarr fixtures
- `applications/dynaclr/src/dynaclr/__init__.py` - Added MultiExperimentTripletDataset to package exports

## Decisions Made
- Lineage-timepoint pre-built lookup indexed by (experiment, lineage_id) -> {t: [row_indices]} for O(1) positive candidate retrieval
- Fallback tau strategy: sample_tau first, then linear scan of full tau range if no candidate at sampled offset
- Dataset uses numpy.random.default_rng() without fixed seed; determinism delegated to external sampler
- INDEX_COLUMNS optional columns (y, x, z) silently skipped in predict mode for compatibility

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- MultiExperimentTripletDataset ready for DynaCLR DataModule (Plan 24-02)
- Dataset produces exact batch format expected by ContrastiveModule.training_step
- No blockers for next plan

## Self-Check: PASSED

All 3 files verified present. All 3 commit hashes verified in git log.

---
*Phase: 24-dataset-datamodule*
*Completed: 2026-02-23*
