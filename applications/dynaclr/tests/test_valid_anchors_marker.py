"""Regression tests for marker-aware valid_anchors in flat-parquet mode.

In flat-parquet / bag-of-channels mode, one cell observation becomes one
row per channel. ``_pick_temporal_candidate`` restricts positive candidates
to rows with the same ``marker`` as the anchor, so ``_compute_valid_anchors``
must also include ``marker`` in the validity key — otherwise an anchor can
pass validation because a different-marker row exists at ``t+tau``, then
crash at sample time with "No positive found".

These tests hit ``_compute_valid_anchors`` directly via ``object.__new__``
so they don't need real zarr stores.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from dynaclr.data.index import MultiExperimentIndex


def _make_registry(experiment_names, interval_minutes=30.0):
    """Return a minimal object that quacks like ExperimentRegistry for tau math."""
    experiments = [SimpleNamespace(name=n, interval_minutes=interval_minutes) for n in experiment_names]

    def tau_range_frames(name, tau_range_hours):
        exp = next(e for e in experiments if e.name == name)
        min_h, max_h = tau_range_hours
        frames_per_hour = 60.0 / exp.interval_minutes
        return (int(round(min_h * frames_per_hour)), int(round(max_h * frames_per_hour)))

    return SimpleNamespace(experiments=experiments, tau_range_frames=tau_range_frames)


def _make_index(tracks: pd.DataFrame, registry) -> MultiExperimentIndex:
    """Construct a bare MultiExperimentIndex without zarr I/O."""
    index = object.__new__(MultiExperimentIndex)
    index.registry = registry
    index.tracks = tracks.reset_index(drop=True)
    return index


class TestMarkerAwareValidAnchors:
    """`marker` must be part of the temporal validity key in flat-parquet mode."""

    def test_anchor_with_cross_marker_positive_rejected(self):
        """
        Anchor at (lid, marker=A, t=5) must be REJECTED when the only row
        at t+tau is (lid, marker=B, t=6). Without marker-aware validity
        this anchor would be accepted and then crash at sample time because
        `_pick_temporal_candidate` filters candidates to same marker.
        """
        tracks = pd.DataFrame(
            {
                "experiment": ["exp"] * 2,
                "lineage_id": ["L"] * 2,
                "marker": ["A", "B"],
                "t": [5, 6],
            }
        )
        registry = _make_registry(["exp"], interval_minutes=30.0)
        index = _make_index(tracks, registry)

        # tau_range 0.5h - 1.5h at 30min = (1, 3) frames.
        valid = index._compute_valid_anchors(
            tau_range_hours=(0.5, 1.5),
            positive_cell_source="lookup",
            positive_match_columns=["lineage_id"],
        )
        # Neither row is a valid anchor: A has no same-marker positive in window,
        # and B has no same-marker positive either.
        assert len(valid) == 0, f"expected 0 valid anchors, got {len(valid)}:\n{valid}"

    def test_anchor_with_same_marker_positive_accepted(self):
        """Anchor at (lid, marker=A, t=5) with (lid, marker=A, t=6) IS valid."""
        tracks = pd.DataFrame(
            {
                "experiment": ["exp"] * 3,
                "lineage_id": ["L"] * 3,
                "marker": ["A", "A", "B"],
                "t": [5, 6, 6],
            }
        )
        registry = _make_registry(["exp"], interval_minutes=30.0)
        index = _make_index(tracks, registry)

        valid = index._compute_valid_anchors(
            tau_range_hours=(0.5, 1.5),
            positive_cell_source="lookup",
            positive_match_columns=["lineage_id"],
        )
        # (A, t=5) is valid because (A, t=6) exists.
        # (A, t=6) is NOT valid because there's no (A, t=7..8).
        # (B, t=6) is NOT valid because there's no (B, t=7..8).
        assert len(valid) == 1
        row = valid.iloc[0]
        assert row["marker"] == "A"
        assert row["t"] == 5

    def test_both_markers_have_positives_both_accepted(self):
        """When each marker has its own lineage continuity, both pass."""
        tracks = pd.DataFrame(
            {
                "experiment": ["exp"] * 4,
                "lineage_id": ["L"] * 4,
                "marker": ["A", "A", "B", "B"],
                "t": [5, 6, 5, 6],
            }
        )
        registry = _make_registry(["exp"], interval_minutes=30.0)
        index = _make_index(tracks, registry)

        valid = index._compute_valid_anchors(
            tau_range_hours=(0.5, 1.5),
            positive_cell_source="lookup",
            positive_match_columns=["lineage_id"],
        )
        # (A, t=5) valid (A, t=6 exists). (B, t=5) valid (B, t=6 exists).
        # t=6 of each marker is NOT valid (no t=7 for either).
        assert len(valid) == 2
        assert set(zip(valid["marker"], valid["t"])) == {("A", 5), ("B", 5)}

    def test_no_marker_column_falls_back_to_lineage_t(self):
        """When `marker` column is absent, behavior matches legacy (lid, t) keys."""
        tracks = pd.DataFrame(
            {
                "experiment": ["exp"] * 3,
                "lineage_id": ["L"] * 3,
                "t": [5, 6, 7],
            }
        )
        registry = _make_registry(["exp"], interval_minutes=30.0)
        index = _make_index(tracks, registry)

        valid = index._compute_valid_anchors(
            tau_range_hours=(0.5, 1.5),
            positive_cell_source="lookup",
            positive_match_columns=["lineage_id"],
        )
        # tau_range_frames = (1, 3). t=5 needs t=6,7,8 (6,7 exist) -> valid.
        # t=6 needs t=7,8,9 (7 exists) -> valid. t=7 needs t=8,9,10 -> NOT valid.
        assert len(valid) == 2
        assert set(valid["t"].to_numpy()) == {5, 6}


class TestLineageCollisionDetection:
    """
    Regression for the ALFI-style bug where two FOVs share the same
    ``lineage_id`` because lineage reconstruction collapsed across FOVs.
    The marker-aware fix cannot save this — it's a data bug — so the
    test documents the failure mode: `_compute_valid_anchors` will
    accept anchors whose temporal neighbors are actually in a different
    physical FOV. Cached so we notice if lineage reconstruction ever
    starts disambiguating by FOV.
    """

    def test_cross_fov_lineage_collision_accepted_today(self):
        """Two FOVs share `lineage_id='L'`; validity check treats as one lineage."""
        # FOV1 has t=5 only; FOV2 has t=6 only. They share lineage_id.
        tracks = pd.DataFrame(
            {
                "experiment": ["exp"] * 2,
                "lineage_id": ["L", "L"],
                "fov_name": ["FOV1", "FOV2"],  # different physical fields
                "marker": ["A", "A"],
                "t": [5, 6],
            }
        )
        registry = _make_registry(["exp"], interval_minutes=30.0)
        index = _make_index(tracks, registry)

        valid = index._compute_valid_anchors(
            tau_range_hours=(0.5, 1.5),
            positive_cell_source="lookup",
            positive_match_columns=["lineage_id"],
        )
        # Today both rows pass — the fix doesn't consider fov_name in the
        # validity key. If cell_index generation ever disambiguates lineage_id
        # by fov, this test will flip and should be updated.
        assert len(valid) == 1  # (A, t=5) valid because "L" at t=6 exists
        # The surviving anchor is t=5 — at sample time it would try to
        # pull a patch from FOV2 thinking it's the same biological lineage.
        # That's still wrong biologically, but it won't raise "No positive found".


@pytest.mark.parametrize("interval_minutes", [15.0, 30.0, 60.0])
def test_marker_key_respects_per_experiment_tau(interval_minutes):
    """Marker-aware validity plays correctly with per-experiment interval_minutes."""
    tracks = pd.DataFrame(
        {
            "experiment": ["exp"] * 4,
            "lineage_id": ["L"] * 4,
            "marker": ["A", "A", "A", "A"],
            "t": [0, 1, 5, 10],
        }
    )
    registry = _make_registry(["exp"], interval_minutes=interval_minutes)
    index = _make_index(tracks, registry)

    valid = index._compute_valid_anchors(
        tau_range_hours=(0.5, 1.5),
        positive_cell_source="lookup",
        positive_match_columns=["lineage_id"],
    )
    min_f, max_f = registry.tau_range_frames("exp", (0.5, 1.5))
    # Every valid anchor t must have some other row at t+tau within [min_f, max_f].
    t_vals = set(tracks["t"].to_numpy())
    for t in valid["t"].to_numpy():
        ok = any((t + tau) in t_vals for tau in range(min_f, max_f + 1) if tau != 0)
        assert ok, f"anchor t={t} validated but no t+tau neighbor exists at interval={interval_minutes}"
