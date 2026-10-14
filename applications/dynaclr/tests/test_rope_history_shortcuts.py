from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).parents[1] / "scripts/evaluation/evaluate_rope_history_shortcuts.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("history_shortcuts", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
history_shortcuts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(history_shortcuts)


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["d1"] * 6,
            "fov_name": ["a", "a", "b", "b", "c", "c"],
            "track_id": [1, 1, 2, 2, 3, 3],
            "perturbation": ["ZIKV"] * 6,
            "hours_post_perturbation": [1.0, 2.0] * 3,
        }
    )


def test_repeat_current_preserves_current_token() -> None:
    sequences = np.arange(4 * 5 * 3).reshape(4, 5, 3)
    repeated = history_shortcuts._repeat_current(sequences)
    assert np.array_equal(repeated[:, -1], sequences[:, -1])
    for position in range(5):
        assert np.array_equal(repeated[:, position], sequences[:, -1])


def test_matched_donors_obey_matching_contract() -> None:
    labels = _labels()
    donor, audit = history_shortcuts._matched_other_track_donors(labels, seed=7)
    assert (donor >= 0).all()
    for target, source in enumerate(donor):
        assert labels.loc[target, "dataset"] == labels.loc[source, "dataset"]
        assert labels.loc[target, "perturbation"] == labels.loc[source, "perturbation"]
        assert labels.loc[target, "hours_post_perturbation"] == labels.loc[source, "hours_post_perturbation"]
        target_track = (labels.loc[target, "fov_name"], labels.loc[target, "track_id"])
        source_track = (labels.loc[source, "fov_name"], labels.loc[source, "track_id"])
        assert target_track != source_track
    assert audit["n_unmatched"].sum() == 0


def test_borrow_history_keeps_target_current_and_uses_donor_prior() -> None:
    sequences = np.arange(6 * 5 * 2).reshape(6, 5, 2)
    donor = np.roll(np.arange(6), 1)
    borrowed = history_shortcuts._borrow_history(sequences, donor)
    assert np.array_equal(borrowed[:, -1], sequences[:, -1])
    assert np.array_equal(borrowed[:, :-1], sequences[donor, :-1])


def test_singleton_match_group_is_explicitly_unmatched() -> None:
    labels = _labels().iloc[:1].copy()
    donor, audit = history_shortcuts._matched_other_track_donors(labels, seed=7)
    assert donor.tolist() == [-1]
    assert audit["n_unmatched"].sum() == 1
