---
phase: 21-cell-index-lineage
plan: 01
subsystem: data
tags: [dataframe, iohub, ome-zarr, tracking, lineage, border-clamping, pandas, multi-experiment]

# Dependency graph
requires:
  - phase: 20-experiment-configuration
    provides: ExperimentConfig dataclass and ExperimentRegistry with validation
provides:
  - MultiExperimentIndex class with unified tracks DataFrame
  - Lineage reconstruction linking daughter tracks to root ancestor via parent_track_id
  - Border clamping with y_clamp/x_clamp columns (retains border cells, excludes only out-of-image)
  - Condition resolution from well_name via condition_wells mapping
  - Global track ID uniqueness across experiments via "{exp_name}_{fov_name}_{track_id}" format
  - Hours-post-infection computation from experiment metadata
affects: [22-flexible-batch-sampler, 23-dataset-construction, 24-datamodule-assembly]

# Tech tracking
tech-stack:
  added: []
  patterns: [border clamping instead of exclusion, lineage graph traversal to root, iohub Position object storage in DataFrame]

key-files:
  created:
    - applications/dynaclr/src/dynaclr/index.py
    - applications/dynaclr/tests/test_index.py
  modified:
    - applications/dynaclr/src/dynaclr/__init__.py

key-decisions:
  - "Border clamping retains all cells within image bounds; only cells with centroid completely outside image are excluded"
  - "Lineage reconstruction chases parent_track_id to root ancestor; missing parents fall back to self"
  - "Position objects stored directly in DataFrame column for downstream data loading"
  - "Image dimensions read from position['0'] (ImageArray.height/width) for border clamping"

patterns-established:
  - "MultiExperimentIndex: builds flat DataFrame from ExperimentRegistry, enriches with metadata, reconstructs lineage, clamps borders"
  - "Global track ID format: {exp_name}_{fov_name}_{track_id} for cross-experiment uniqueness"
  - "Lineage graph per experiment+fov: child->parent map, chase-to-root traversal"
  - "Border clamping: clip(centroid, half_patch, img_dim - half_patch) preserving original coordinates"

# Metrics
duration: 5min
completed: 2026-02-22
---

# Phase 21 Plan 01: MultiExperimentIndex Summary

**MultiExperimentIndex with unified tracks DataFrame, lineage reconstruction via parent_track_id graph traversal, and border clamping retaining edge cells (23 tests)**

## Performance

- **Duration:** 5 min 0s
- **Started:** 2026-02-22T06:33:23Z
- **Completed:** 2026-02-22T06:38:23Z
- **Tasks:** 3 (TDD: RED, GREEN, REFACTOR)
- **Files created:** 2
- **Files modified:** 1

## Accomplishments
- MultiExperimentIndex builds flat tracks DataFrame from all experiments in ExperimentRegistry with enriched columns: experiment, condition, global_track_id, hours_post_infection, fluorescence_channel, well_name, fov_name, position, lineage_id, y_clamp, x_clamp
- Lineage reconstruction: parent_track_id graph traversal per experiment+fov propagates root ancestor's global_track_id as lineage_id to all descendants (daughters, granddaughters, etc.)
- Border clamping: cells near edges get clipped centroids (y_clamp, x_clamp) instead of being excluded; only cells completely outside image boundary are removed
- 23 passing tests covering CELL-01 (unified tracks, 12 tests), CELL-02 (lineage, 5 tests), CELL-03 (border clamping, 6 tests)

## Task Commits

Each task was committed atomically (TDD):

1. **RED: Failing tests** - `03bee1a` (test) - 17 test cases initially, all fail with ModuleNotFoundError
2. **GREEN: Implementation** - `680694b` (feat) - index.py with MultiExperimentIndex, all 23 tests pass
3. **REFACTOR: Cleanup** - `98dc7a6` (refactor) - Fix ruff lint issues (unused variable F841, .values->to_numpy PD011), export from __init__.py

## Files Created/Modified
- `applications/dynaclr/src/dynaclr/index.py` - MultiExperimentIndex class with tracks loading, lineage reconstruction, border clamping (238 lines)
- `applications/dynaclr/tests/test_index.py` - Comprehensive TDD test suite with 23 tests (574 lines)
- `applications/dynaclr/src/dynaclr/__init__.py` - Added MultiExperimentIndex to package exports

## Decisions Made
- Border clamping retains all cells within image bounds (y >= 0, y < height, x >= 0, x < width); only cells with centroid completely outside image are excluded -- this maximizes training data vs. the old TripletDataset._filter_tracks exclusion approach
- Lineage reconstruction chases parent_track_id to root ancestor; tracks whose parent_track_id references a track not in the data gracefully fall back to their own global_track_id as lineage_id
- iohub Position objects stored directly in DataFrame column for downstream data loading -- avoids separate lookup table
- Image dimensions (height, width) read from position["0"] (ImageArray attributes) during loading, used for clamping, then dropped as internal columns

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- MultiExperimentIndex ready for downstream consumption by Phase 21 Plan 02 and Phase 22 (FlexibleBatchSampler)
- All exports available via `from dynaclr.index import MultiExperimentIndex`
- Tracks DataFrame provides all columns needed for temporal sampling: global_track_id, lineage_id, hours_post_infection, y_clamp, x_clamp, position

## Self-Check: PASSED

- All files exist (index.py, test_index.py, SUMMARY.md)
- All 3 commits verified (03bee1a, 680694b, 98dc7a6)
- Module importable: `from dynaclr.index import MultiExperimentIndex`
- Key links verified: ExperimentRegistry import, open_ome_zarr import, test import
- Min lines met: index.py=238 (>=120), test_index.py=574 (>=150)

---
*Phase: 21-cell-index-lineage*
*Completed: 2026-02-22*
