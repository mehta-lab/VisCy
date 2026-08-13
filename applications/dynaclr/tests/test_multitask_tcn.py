"""Tests for reusable compact multi-task TCN functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dynaclr.evaluation.temporal import (
    CompactTCNConfig,
    build_causal_sequences,
    fit_multitask_tcn,
)


def test_build_causal_sequences_left_pads_and_audits_context() -> None:
    labels = pd.DataFrame(
        {
            "experiment": ["a"] * 6,
            "fov_name": ["A/1/0"] * 6,
            "track_id": [1] * 6,
            "t": np.arange(6),
        }
    )
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    result = build_causal_sequences(labels, features, window=5)

    assert result.sequences.shape == (6, 5, 2)
    assert np.all(result.sequences[0] == features[0])
    assert not result.full_context[3]
    assert result.full_context[4]
    assert result.distinct_observations.tolist() == [1, 2, 3, 4, 5, 5]


def test_compact_defaults_match_validated_architecture() -> None:
    config = CompactTCNConfig()
    assert config.hidden == 64
    assert config.dilations == (1, 2)
    assert config.epochs == 15
    assert config.batch_size == 512


def test_fit_multitask_tcn_uses_training_only_scaler_and_two_heads() -> None:
    rng = np.random.default_rng(4)
    sequences = rng.normal(size=(24, 5, 4)).astype(np.float32)
    targets = np.column_stack(
        [
            np.tile([0.0, 1.0], 12),
            np.tile([0.0, 0.0, 1.0, 1.0], 6),
        ]
    ).astype(np.float32)
    train_rows = [np.arange(16), np.arange(4, 20)]
    config = CompactTCNConfig(
        hidden=8,
        epochs=2,
        batch_size=8,
        random_seed=7,
    )

    fit = fit_multitask_tcn(
        sequences,
        targets,
        train_rows,
        config=config,
        device="cpu",
    )
    union = np.unique(np.concatenate(train_rows))
    flat = sequences[union].reshape(-1, sequences.shape[-1])

    np.testing.assert_allclose(fit.center, flat.mean(axis=0), rtol=1e-5)
    probability = fit.predict_proba(sequences)
    assert probability.shape == (24, 2)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
