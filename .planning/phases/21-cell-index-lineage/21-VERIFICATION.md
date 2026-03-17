---
phase: 21-cell-index-lineage
verified: 2026-02-22T06:49:57Z
status: passed
score: 11/11 must-haves verified
re_verification: false
---

# Phase 21: Cell Index & Lineage Verification Report

**Phase Goal:** Users have a unified cell observation index across all experiments with lineage-linked tracks, border-safe centroids, and valid anchor computation for variable tau
**Verified:** 2026-02-22T06:49:57Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | MultiExperimentIndex builds a flat tracks DataFrame from all registered experiments with one row per cell observation per timepoint | VERIFIED | 40 tests pass; test_all_observations_present asserts 400 rows for 2 exp x 2 wells x 2 FOVs x 5 tracks x 10 t |
| 2 | Each row has required columns: experiment, condition, global_track_id, hours_post_infection, well_name, fluorescence_channel, lineage_id, position, fov_name, track_id, t, y, x, z, y_clamp, x_clamp | VERIFIED | test_required_columns_present explicitly asserts full required set as a subset of tracks.columns; passes |
| 3 | Lineage is reconstructed — daughter tracks have lineage_id equal to their parent track's root ancestor's global_track_id | VERIFIED | Chase-to-root graph traversal implemented in _reconstruct_lineage; 5 lineage tests pass (grandchild shares grandparent lineage_id, missing parent falls back to self) |
| 4 | Border cells are retained by clamping crop centroids inward — cells near edges get shifted patch origins instead of being excluded | VERIFIED | _clamp_borders clips y/x to (half_patch, img_dim - half_patch); y_clamp/x_clamp columns present; 6 border tests pass |
| 5 | Cells whose centroids are completely outside the image boundary are excluded | VERIFIED | _clamp_borders filters out rows where y < 0 or y >= height or x < 0 or x >= width before clamping; tested with outside_cell_track=-1 |
| 6 | valid_anchors is a subset of tracks where each anchor has at least one tau in the configured range that yields a same-track or same-lineage positive | VERIFIED | _compute_valid_anchors builds (lineage_id, t) set and checks t+tau membership; 8 anchor tests pass including end-of-track (not valid) and mid-track (valid) |
| 7 | Variable tau range accounts for per-experiment frame rates — tau_range_hours is converted to frames per experiment via registry.tau_range_frames | VERIFIED | _compute_valid_anchors calls self.registry.tau_range_frames(exp.name, tau_range_hours) per experiment; ExperimentRegistry.tau_range_frames exists and is wired |
| 8 | experiment_groups property returns dict mapping experiment names to arrays of row indices in tracks | VERIFIED | Property implemented via groupby("experiment"); 3 property tests pass |
| 9 | condition_groups property returns dict mapping condition labels to arrays of row indices in tracks | VERIFIED | Property implemented via groupby("condition"); tested and passing |
| 10 | summary() returns a human-readable string with experiment counts, track counts, and anchor counts | VERIFIED | summary() method returns formatted multi-line string with header and per-experiment lines; summary test passes |
| 11 | MultiExperimentIndex is importable from dynaclr top-level package | VERIFIED | from dynaclr import MultiExperimentIndex succeeds; __init__.py exports it at line 3 |

**Score:** 11/11 truths verified

---

### Required Artifacts

| Artifact | Expected | Min Lines | Actual Lines | Status | Details |
|----------|----------|-----------|--------------|--------|---------|
| `applications/dynaclr/src/dynaclr/index.py` | MultiExperimentIndex class with tracks DataFrame, lineage reconstruction, border clamping, valid_anchors, properties, summary | 200 | 351 | VERIFIED | Fully substantive; all methods implemented with docstrings |
| `applications/dynaclr/tests/test_index.py` | TDD test suite for all CELL-01 through CELL-04 behaviors | 250 | 1098 | VERIFIED | 40 tests across 5 test classes; real fixture setup with iohub OME-Zarr |
| `applications/dynaclr/src/dynaclr/__init__.py` | Top-level export of MultiExperimentIndex | contains "MultiExperimentIndex" | 13 lines | VERIFIED | Line 3: from dynaclr.index import MultiExperimentIndex; listed in __all__ |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `index.py` | `dynaclr.experiment.ExperimentRegistry` | import and __init__ parameter | WIRED | Line 18: `from dynaclr.experiment import ExperimentRegistry`; used as registry parameter throughout |
| `index.py` | `iohub.ngff` | open_ome_zarr for reading positions | WIRED | Line 16: `from iohub.ngff import Position, open_ome_zarr`; open_ome_zarr called at line 90 |
| `test_index.py` | `dynaclr.index.MultiExperimentIndex` | import | WIRED | Line 13: `from dynaclr.index import MultiExperimentIndex`; used across all 5 test classes |
| `index.py` | `ExperimentRegistry.tau_range_frames` | method call for tau conversion | WIRED | Line 273: `self.registry.tau_range_frames(exp.name, tau_range_hours)`; tau_range_frames defined in experiment.py line 233 |
| `__init__.py` | `dynaclr.index.MultiExperimentIndex` | re-export | WIRED | Line 3: `from dynaclr.index import MultiExperimentIndex`; listed in `__all__` |

---

### Requirements Coverage

Phase 21 implements CELL-01, CELL-02, CELL-03, CELL-04 from the milestone v2.2 requirements.

| Requirement | Status | Notes |
|-------------|--------|-------|
| CELL-01: Unified tracks DataFrame | SATISFIED | 12 tests; all columns enriched |
| CELL-02: Lineage reconstruction | SATISFIED | 5 tests; chase-to-root graph traversal |
| CELL-03: Border clamping | SATISFIED | 6 tests; inward clamping + out-of-image exclusion |
| CELL-04: Valid anchors with variable tau | SATISFIED | 8 anchor tests + 9 property/summary tests |

---

### Anti-Patterns Found

No anti-patterns detected.

| File | Pattern | Severity | Result |
|------|---------|----------|--------|
| `index.py` | TODO/FIXME/placeholder | Blocker | None found |
| `index.py` | return null / stub bodies | Blocker | None found |
| `test_index.py` | TODO/FIXME/placeholder | Blocker | None found |

---

### Human Verification Required

None. All behaviors are fully verifiable programmatically.

The one item that could be considered for human review:

**Performance at scale.** The _compute_valid_anchors method uses iterrows() over the tracks DataFrame, which is O(n * tau_range) and may be slow for very large experiments (millions of cell observations). The set-lookup inner check is O(1), so the bottleneck is Python-level row iteration. This is a performance concern for HPC usage, not a correctness gap.

Expected: For typical DynaCLR experiments (thousands of cells per experiment), performance is acceptable. For foundation-model-scale datasets, a vectorized implementation may be needed.

Why human: Cannot determine acceptable latency bounds without running at scale.

---

### Gaps Summary

No gaps found. All 11 observable truths are verified against the actual codebase.

The implementation matches the plan exactly:
- No deviations from CELL-01 through CELL-04 specifications
- All 40 tests pass (23 from plan 01, 17 from plan 02)
- All 5 commits verified in git history (03bee1a, 680694b, 98dc7a6, 2dbc359, 9c6408a)
- Both plans executed with TDD RED-GREEN-REFACTOR cycle
- Module importable from both `dynaclr.index` and top-level `dynaclr` package

---

_Verified: 2026-02-22T06:49:57Z_
_Verifier: Claude (gsd-verifier)_
