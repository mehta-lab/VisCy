---
phase: 21-cell-index-lineage
plan: 02
subsystem: data
tags: [contrastive-sampling, temporal-positive, lineage, anchor-validation, pandas, multi-experiment]

# Dependency graph
requires:
  - phase: 21-cell-index-lineage-01
    provides: MultiExperimentIndex with tracks DataFrame, lineage_id, border clamping
  - phase: 20-experiment-configuration
    provides: ExperimentRegistry.tau_range_frames for per-experiment tau conversion
provides:
  - valid_anchors computation filtering rows to those with at least one temporal positive in same lineage
  - experiment_groups property grouping tracks indices by experiment name
  - condition_groups property grouping tracks indices by condition label
  - summary() method returning human-readable index overview with per-experiment breakdowns
  - tau_range_hours parameter on MultiExperimentIndex for variable temporal range
affects: [22-flexible-batch-sampler, 23-dataset-construction, 24-datamodule-assembly]

# Tech tracking
tech-stack:
  added: []
  patterns: [lineage-based anchor validation via set lookup, per-experiment tau conversion for variable frame rates]

key-files:
  created: []
  modified:
    - applications/dynaclr/src/dynaclr/index.py
    - applications/dynaclr/tests/test_index.py

key-decisions:
  - "Anchor validity uses lineage_id for same-track and daughter-track positive matching -- simple set lookup instead of explicit parent-child graph traversal"
  - "tau=0 is skipped to prevent anchor from being its own positive"
  - "valid_anchors is reset_index(drop=True) for clean downstream indexing"
  - "Properties (experiment_groups, condition_groups) use groupby on tracks rather than caching for simplicity and correctness"

patterns-established:
  - "Valid anchor filter: per-experiment tau conversion, lineage-based (lineage_id, t+tau) set membership check"
  - "Summary format: header line with totals, indented per-experiment lines with observation/anchor/condition counts"

# Metrics
duration: 5min
completed: 2026-02-22
---

# Phase 21 Plan 02: Valid Anchors Summary

**Valid anchor computation with per-experiment tau conversion and lineage-based temporal positive filtering, plus experiment_groups, condition_groups, and summary() (40 tests total)**

## Performance

- **Duration:** 4 min 41s
- **Started:** 2026-02-22T06:41:08Z
- **Completed:** 2026-02-22T06:45:49Z
- **Tasks:** 2 (TDD: RED, GREEN; no REFACTOR needed)
- **Files created:** 0
- **Files modified:** 2

## Accomplishments
- valid_anchors correctly filters tracks to rows with at least one temporal positive (same lineage_id at t+tau) for any tau in the per-experiment frame range
- Lineage continuity: daughter tracks satisfy parent anchor validity because they share lineage_id from Plan 01's reconstruction
- Per-experiment tau conversion via registry.tau_range_frames handles different frame intervals (30min vs 15min experiments)
- experiment_groups and condition_groups properties return dict[str, np.ndarray] of row indices
- summary() provides human-readable overview: total experiments, observations, anchors, per-experiment condition breakdowns
- 17 new tests (8 anchor + 9 property/summary), all 40 tests pass

## Task Commits

Each task was committed atomically (TDD):

1. **RED: Failing tests** - `2dbc359` (test) - 17 test cases covering valid anchors (basic, end-of-track, lineage continuity, different tau ranges, empty, gaps, self-exclusion) and properties/summary
2. **GREEN: Implementation** - `9c6408a` (feat) - tau_range_hours param, _compute_valid_anchors, experiment_groups, condition_groups, summary()

_No REFACTOR commit: code passed lint checks and met quality standards after GREEN._

## Files Created/Modified
- `applications/dynaclr/src/dynaclr/index.py` - Added tau_range_hours parameter, _compute_valid_anchors method, experiment_groups/condition_groups properties, summary() method (351 lines, +114)
- `applications/dynaclr/tests/test_index.py` - Added TestValidAnchors (8 tests) and TestMultiExperimentIndexProperties (9 tests) classes with custom track helpers (1098 lines, +524)

## Decisions Made
- Anchor validity uses lineage_id for same-track and daughter-track positive matching. This leverages Plan 01's lineage reconstruction so the check is a simple (lineage_id, t+tau) set membership rather than explicit parent-child graph traversal
- tau=0 is explicitly skipped to prevent an anchor from being its own temporal positive
- valid_anchors DataFrame is reset_index(drop=True) for clean downstream indexing (batch sampler, dataset)
- Properties use groupby on tracks rather than caching: simpler, always correct, negligible cost for typical dataset sizes

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- MultiExperimentIndex fully complete: tracks, lineage, border clamping, valid anchors, properties, summary
- Ready for Phase 22 (FlexibleBatchSampler) which will use valid_anchors and experiment_groups for sampling
- All exports available via `from dynaclr import MultiExperimentIndex` or `from dynaclr.index import MultiExperimentIndex`

## Self-Check: PASSED

- All files exist (index.py, test_index.py, __init__.py, SUMMARY.md)
- Both commits verified (2dbc359, 9c6408a)
- Module importable: `from dynaclr import MultiExperimentIndex`
- Key links verified: tau_range_frames usage in index.py, re-export in __init__.py
- Min lines met: index.py=351 (>=200), test_index.py=1098 (>=250)
- __init__.py contains MultiExperimentIndex

---
*Phase: 21-cell-index-lineage*
*Completed: 2026-02-22*
