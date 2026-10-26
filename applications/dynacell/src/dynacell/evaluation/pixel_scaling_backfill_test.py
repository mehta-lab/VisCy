"""Tests for the dual-scaling provenance classifier.

``_classify`` decides, per cached column, which of the two freshly computed
families the cache reproduces. Everything downstream keys off that verdict -- the
write gate, the audit tally, and whether a leaf is reported as needing a re-eval
-- so its edge cases are worth pinning even though the surrounding I/O needs a
real eval tree.
"""

from __future__ import annotations

import numpy as np

from dynacell.evaluation.pixel_scaling_backfill import (
    MATCH_RTOL,
    _classify,
    _classify_pcc,
    _needed_timepoints,
)

# Representative of a real OOD case: the two families sit ~0.28 SSIM apart, which
# is why one tolerance can separate them without being brittle.
SENSITIVE = np.array([0.4420, 0.6176, 0.5672])
INVARIANT = np.array([0.7224, 0.7476, 0.6709])


def test_classify_names_the_family_the_cache_reproduces() -> None:
    """A cache matching either family is attributed to it, not to "close enough"."""
    assert _classify(SENSITIVE.copy(), SENSITIVE, INVARIANT) == "scale_sensitive"
    assert _classify(INVARIANT.copy(), SENSITIVE, INVARIANT) == "scale_invariant"


def test_classify_tolerates_reduction_order_but_not_the_other_family() -> None:
    """Float-reduction drift is absorbed; a whole-family swap never is."""
    drifted = INVARIANT * (1 + MATCH_RTOL / 10)
    assert _classify(drifted, SENSITIVE, INVARIANT) == "scale_invariant"


def test_classify_flags_a_cache_that_reproduces_neither() -> None:
    """Neither family matching means the arrays changed, not that a name did."""
    assert _classify(INVARIANT + 0.1, SENSITIVE, INVARIANT) == "mismatch"


def test_classify_reports_a_collapse_rather_than_picking_one() -> None:
    """If the two families agree, the cache carries no provenance to report.

    Real data never does this -- it takes an exactly-affine pair -- but reporting
    such a cache as either family would assert a provenance the numbers do not
    support, and would silently mean every "both scalings" table shows one number
    twice.
    """
    assert _classify(SENSITIVE.copy(), SENSITIVE, SENSITIVE) == "degenerate"


def test_classify_ignores_nan_rows_and_reports_an_all_nan_column_as_empty() -> None:
    """A degenerate timepoint is NaN in the cache; it must not decide the verdict."""
    cached = np.array([np.nan, INVARIANT[1], INVARIANT[2]])
    assert _classify(cached, SENSITIVE, INVARIANT) == "scale_invariant"
    assert _classify(np.full(3, np.nan), SENSITIVE, INVARIANT) == "empty"


def test_needed_timepoints_groups_the_cached_rows_by_fov() -> None:
    """The work list comes from the cache, so it cannot cover a different position set."""
    rows = [
        {"FOV": "0/0/fov0001", "Timepoint": 2},
        {"FOV": "0/0/fov0000", "Timepoint": 1},
        {"FOV": "0/0/fov0000", "Timepoint": 0},
    ]
    assert _needed_timepoints(rows) == {"0/0/fov0000": [0, 1], "0/0/fov0001": [2]}


def test_pcc_gate_tolerates_float32_cancellation_but_not_a_moved_mean() -> None:
    """Per-row PCC noise is not a stale cache; a moved mean is.

    Both arms are transcribed from the 2026-08-03 backfill. The near-degenerate arm
    is FNet3D on HEK mito, where PCC sits at 0.003 -- no correlation at all -- so a
    single row moves by 8.6e-4 on identical arrays purely from float32
    cancellation. The moved arm is pix2pix3d mito A549-trained on HEK, whose mean
    shifted 2.9e-3, past the 3-decimal rounding the tables publish.
    """
    degenerate_cached = np.array([0.002848, 0.003100, 0.002500, 0.003140])
    degenerate_fresh = degenerate_cached + np.array([-8.6e-4, 8.3e-4, 5.0e-5, -2.0e-5])
    assert _classify_pcc(degenerate_cached, degenerate_fresh) == "reproduced_mean"

    moved_cached = np.array([0.533478, 0.540000, 0.520000, 0.540434])
    assert _classify_pcc(moved_cached, moved_cached - 2.87e-3) == "MOVED"

    assert _classify_pcc(moved_cached, moved_cached.copy()) == "reproduced"
