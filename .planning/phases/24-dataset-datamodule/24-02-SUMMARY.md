---
phase: 24-dataset-datamodule
plan: 02
subsystem: data
tags: [datamodule, lightning, sampler, channel-dropout, contrastive, multi-experiment]

# Dependency graph
requires:
  - phase: 24-dataset-datamodule
    plan: 01
    provides: "MultiExperimentTripletDataset with __getitems__ returning batch dicts"
  - phase: 22-flexible-sampler
    provides: "FlexibleBatchSampler with experiment-aware, condition-balanced, temporal enrichment"
  - phase: 23-loss-augmentation
    provides: "ChannelDropout and sample_tau"
  - phase: 20-experiment-registry
    provides: "ExperimentRegistry.from_yaml and ExperimentConfig"
  - phase: 21-cell-index-lineage
    provides: "MultiExperimentIndex with valid_anchors"
provides:
  - "MultiExperimentDataModule LightningDataModule composing all sampling components"
  - "Experiment-level train/val split (whole experiments, not FOVs)"
  - "All hyperparameters exposed for Lightning CLI YAML configurability"
affects: [dynaclr-training, 25-integration-cli]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "MultiExperimentDataModule as final composition layer wiring FlexibleBatchSampler + Dataset + ChannelDropout + ThreadDataLoader"
    - "Generic channel names (ch_0, ch_1) for transform pipeline across experiments with different channel orderings"
    - "All-None norm_meta coalesced to None before _scatter_channels to avoid collation errors"

key-files:
  created:
    - "applications/dynaclr/src/dynaclr/datamodule.py"
    - "applications/dynaclr/tests/test_datamodule.py"
  modified:
    - "applications/dynaclr/src/dynaclr/__init__.py"

key-decisions:
  - "Generic channel names (ch_0, ch_1, ...) used for transform pipeline since experiments have different channel names but same count"
  - "Norm_meta all-None coalescing: list of None -> None to prevent collate_meta_tensor crash on None values"
  - "Separate ExperimentRegistry instances for train and val splits, each building their own MultiExperimentIndex"
  - "ChannelDropout applied AFTER normalizations+augmentations+final_crop (consistent with Phase 23 design)"

patterns-established:
  - "MultiExperimentDataModule follows TripletDataModule's on_after_batch_transfer pattern but with generic channel names"
  - "FlexibleBatchSampler as batch_sampler for train only; val uses simple sequential DataLoader"

# Metrics
duration: 5min
completed: 2026-02-23
---

# Phase 24 Plan 02: MultiExperimentDataModule Summary

**MultiExperimentDataModule composing FlexibleBatchSampler + Dataset + ChannelDropout + ThreadDataLoader with experiment-level train/val split and full Lightning CLI configurability**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-23T21:56:58Z
- **Completed:** 2026-02-23T22:02:10Z
- **Tasks:** 2 (Task 1: TDD RED->GREEN, Task 2: exports)
- **Files modified:** 3

## Accomplishments
- MultiExperimentDataModule wires all composable sampling components (FlexibleBatchSampler, Dataset, ChannelDropout, ThreadDataLoader) into a single LightningDataModule
- Train/val split by whole experiments verified: val_experiments parameter splits at experiment level, never at FOV level
- All sampling, augmentation, and loss hyperparameters exposed as __init__ parameters for Lightning CLI YAML configuration
- ChannelDropout correctly applied after transforms: train mode zeros specified channels, eval mode preserves them
- 6 TDD tests covering hyperparameter exposure, experiment-level split, sampler wiring, val determinism, transforms, and dropout integration

## Task Commits

Each task was committed atomically (TDD):

1. **Task 1 RED: Failing tests** - `4f03d12` (test)
2. **Task 1 GREEN: Implementation** - `d874570` (feat)
3. **Task 2: Package exports** - `5f0e743` (refactor)

## Files Created/Modified
- `applications/dynaclr/src/dynaclr/datamodule.py` - MultiExperimentDataModule with setup(), train/val dataloaders, on_after_batch_transfer, ChannelDropout
- `applications/dynaclr/tests/test_datamodule.py` - 6 TDD tests with synthetic zarr fixtures for all DataModule functionality
- `applications/dynaclr/src/dynaclr/__init__.py` - Added MultiExperimentDataModule to top-level exports

## Decisions Made
- Generic channel names (ch_0, ch_1, ...) used for transform pipeline since experiments have different channel names but same count -- enables _scatter_channels and BatchedCenterSpatialCropd to work across experiments
- Norm_meta all-None coalescing: when all norm_meta entries are None (no normalization metadata), coalesce list to None before passing to _scatter_channels to prevent collate_meta_tensor crash
- Separate ExperimentRegistry instances for train and val splits -- each builds its own MultiExperimentIndex for clean separation
- ChannelDropout applied AFTER normalizations+augmentations+final_crop, consistent with Phase 23 design

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed all-None norm_meta collation crash**
- **Found during:** Task 1 GREEN phase
- **Issue:** _scatter_channels calls collate_meta_tensor(norm_meta) which crashes when norm_meta is a list of all None values ([None, None, ...]) because it's truthy but contains uncollatable None types
- **Fix:** Added check in on_after_batch_transfer: if norm_meta is a list where all entries are None, coalesce to None before passing to _transform_channel_wise
- **Files modified:** applications/dynaclr/src/dynaclr/datamodule.py
- **Verification:** All 6 tests pass including on_after_batch_transfer tests
- **Committed in:** d874570 (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Fix necessary for correctness when experiments lack normalization metadata. No scope creep.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- MultiExperimentDataModule ready for DynaCLR CLI integration (Phase 25)
- Full pipeline: ExperimentRegistry -> MultiExperimentIndex -> Dataset -> DataModule -> ContrastiveModule
- All components importable from dynaclr top-level
- No blockers for next phase

## Self-Check: PASSED

All 3 files verified present. All 3 commit hashes verified in git log.

---
*Phase: 24-dataset-datamodule*
*Completed: 2026-02-23*
