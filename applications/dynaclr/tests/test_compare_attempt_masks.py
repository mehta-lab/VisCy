"""Tests for the footprint, shared-mask and tracked-nucleus counts of compare_attempt_masks."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from dynaclr.phenotyping.algorithms import map_nuclei_to_cells

spec = importlib.util.spec_from_file_location(
    "compare_attempt_masks", Path(__file__).parents[1] / "tools/compare_attempt_masks.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def synthetic():
    cells = np.zeros((20, 30), dtype=np.uint32)
    cells[:10, :10] = 1
    cells[:10, 10:20] = 2
    cells[12:14, 0:2] = 3
    nuclei = np.zeros_like(cells)
    nuclei[1:3, 1:3] = 11
    nuclei[5:7, 5:7] = 12
    nuclei[1:3, 12:14] = 13
    nuclei[4:6, 8:12] = 14
    nuclei[15:17, 25:27] = 15
    nuclei[5:7, 15:17] = 16
    return cells, nuclei


def test_counts_majority_footprints_shared_cells_and_unassigned_tracked_nuclei():
    cells, nuclei = synthetic()
    tracked = np.array([11, 12, 13, 14, 15, 99, 0])
    counts, shared, areas = m.frame_counts(cells, nuclei, tracked, 0.5)
    assert shared.tolist() == [1, 2]
    assert sorted(areas.tolist()) == [2.0, 50.0, 50.0]
    assert counts["n_cells"] == 3
    assert (counts["nuclear_footprints"], counts["assigned_footprints"]) == (6, 4)
    assert (counts["shared_masks"], counts["nuclei_in_shared_masks"]) == (2, 4)
    assert counts["shared_fraction"] == 1.0
    assert counts["tracked_total"] == 7
    assert counts["tracked_matched"] == 3
    assert (counts["tracked_ambiguous"], counts["tracked_unmatched"]) == (1, 1)
    assert counts["tracked_without_cell"] == 2
    assert counts["tracked_missing_nuclear_footprint"] == 2
    assert counts["cells_below_floor"] == 1
    assert counts["median_cell_area_um2"] == 50.0


def test_tracked_statuses_agree_with_pipeline_join():
    cells, nuclei = synthetic()
    tracks = pd.DataFrame({"nucleus_label": [11, 12, 13, 14, 15], "track_id": range(5)})
    joined = map_nuclei_to_cells(cells, nuclei, tracks).assignment_status.value_counts()
    counts, _, _ = m.frame_counts(cells, nuclei, tracks.nucleus_label.to_numpy(), 1.0)
    assert {s: counts[f"tracked_{s}"] for s in m.STATUSES} == {s: int(joined.get(s, 0)) for s in m.STATUSES}


def test_empty_frame_has_no_shared_masks():
    empty = np.zeros((8, 8), dtype=np.uint32)
    counts, shared, areas = m.frame_counts(empty, empty, np.array([], dtype=np.int64), 1.0)
    assert len(shared) == 0 and len(areas) == 0
    assert counts["n_cells"] == counts["nuclear_footprints"] == counts["tracked_total"] == 0
    assert np.isnan(counts["shared_fraction"]) and np.isnan(counts["median_cell_area_um2"])
