"""Tests for leakage-safe temporal event forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dynaclr.evaluation.temporal.config_forecast import (
    ForecastEvent,
    TemporalForecastConfig,
    TemporalModelSettings,
)
from dynaclr.evaluation.temporal.forecast import (
    _fit_temporal_model,
    assign_fov_folds,
    build_forecast_targets,
)


def _config(**updates) -> TemporalForecastConfig:
    raw = {
        "datasets": [{"experiment": "ds", "annotation": "annotations.csv"}],
        "events": [
            {
                "name": "infection",
                "column": "infection_state",
                "negative": "uninfected",
                "positive": "infected",
            }
        ],
        "horizons_hours": [1.0],
        "n_folds": 2,
        "models": {
            "names": ["tcn", "rope_transformer"],
            "epochs": 1,
            "batch_size": 8,
            "tcn_hidden": 8,
            "tcn_dilations": [1],
            "rope_d_model": 16,
            "rope_heads": 2,
            "rope_layers": 1,
            "rope_ff_multiplier": 2,
        },
    }
    raw.update(updates)
    return TemporalForecastConfig(**raw)


def test_config_defaults_to_normalized_pca80() -> None:
    config = _config()
    assert config.feature_key == "X_normalized_pca80"
    assert config.window_observations == 5


def test_config_rejects_unsorted_or_duplicate_horizons() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        _config(horizons_hours=[2.0, 1.0, 2.0])


def test_rope_requires_even_head_dimension() -> None:
    with pytest.raises(ValueError, match="head dimension"):
        TemporalModelSettings(
            rope_d_model=12,
            rope_heads=4,
        )


def test_targets_are_strictly_pre_onset_and_right_censored() -> None:
    labels = pd.DataFrame(
        {
            "experiment": ["ds"] * 12,
            "fov_name": ["f1"] * 6 + ["f2"] * 6,
            "track_id": [1] * 6 + [2] * 6,
            "t": list(range(6)) * 2,
            "hours_post_perturbation": [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                2.5,
            ]
            * 2,
            # Track 1 transitions at 2 h. Track 2 is observed negative
            # through 2.5 h and is therefore censored near its end.
            "infection_state": [
                "uninfected",
                pd.NA,
                pd.NA,
                pd.NA,
                "infected",
                "infected",
                "uninfected",
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                "uninfected",
            ],
        }
    )
    result = build_forecast_targets(
        labels,
        [
            ForecastEvent(
                name="infection",
                column="infection_state",
                negative="uninfected",
                positive="infected",
            )
        ],
        [1.0],
    )
    y = result.values[:, 0]  # noqa: PD011

    # Fated: t=0,0.5 are negative; t=1,1.5 are positive; onset/post-onset drop.
    assert y[:6].tolist()[:4] == [0.0, 0.0, 1.0, 1.0]
    assert np.isnan(y[4:6]).all()
    # Unfated: only rows with a full observed 1 h future are valid negatives.
    assert y[6:10].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert np.isnan(y[10:12]).all()


def test_fov_folds_are_deterministic_and_keep_groups_together() -> None:
    labels = pd.DataFrame(
        {
            "experiment": ["a"] * 8 + ["b"] * 8,
            "fov_name": np.repeat(["f1", "f2", "f3", "f4"], 4),
        }
    )
    first = assign_fov_folds(labels, 2, 42)
    second = assign_fov_folds(labels, 2, 42)
    np.testing.assert_array_equal(first, second)
    for _, rows in labels.groupby(["experiment", "fov_name"]).indices.items():
        assert len(np.unique(first[np.asarray(rows)])) == 1


@pytest.mark.parametrize("model_name", ["tcn", "rope_transformer"])
def test_temporal_models_fit_multihead_probabilities(model_name: str) -> None:
    rng = np.random.default_rng(4)
    sequences = rng.normal(size=(32, 5, 6)).astype(np.float32)
    targets = np.column_stack(
        [
            np.tile([0.0, 1.0], 16),
            np.tile([0.0, 0.0, 1.0, 1.0], 8),
        ]
    ).astype(np.float32)
    config = _config(
        events=[
            {
                "name": "infection",
                "column": "infection_state",
                "negative": "uninfected",
                "positive": "infected",
            },
            {
                "name": "death",
                "column": "death_state",
                "negative": "alive",
                "positive": "dead",
            },
        ]
    )
    fit = _fit_temporal_model(
        model_name,
        sequences,
        targets,
        [np.arange(24), np.arange(4, 28)],
        config,
        seed=7,
    )
    probability = fit.predict_proba(
        sequences,
        np.arange(len(sequences)),
        batch_size=8,
    )
    assert probability.shape == targets.shape
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
