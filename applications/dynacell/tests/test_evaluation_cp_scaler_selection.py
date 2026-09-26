"""Each eval condition is scored with its own dataset's CP scaler.

Drives the real :func:`~dynacell.evaluation.cp_reference.eval_cp_space` and the real
grouped per-condition driver (:func:`~dynacell.evaluation.pipeline.evaluate_predictions_grouped`)
against a reference holding three datasets with different scalers plus one lite
dataset. Only the heavy per-condition work (``evaluate_predictions``,
``save_metrics``, model loading, the manifest splice) is replaced by recorders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from ._eval_fixtures import build_eval_config, live_pipeline_module, make_cp_reference

_SETS = ("set-a", "set-b", "set-c")
_LITE = {"set-b-lite": "set-b"}


def _config(tmp_path: Path):
    """Feature-metrics eval config pointed at a three-set + one-lite reference."""
    config = build_eval_config(
        tmp_path / "pred.zarr",
        tmp_path / "gt.zarr",
        tmp_path / "g",
        tmp_path / "p",
        tmp_path / "out",
        executor="serial",
        fov_workers=1,
    )
    config.compute_feature_metrics = True
    make_cp_reference(config, tmp_path / "er.json", datasets=_SETS, lite=_LITE)
    return config


def test_eval_cp_space_binds_the_config_dataset(tmp_path: Path) -> None:
    """``eval_cp_space`` returns the scaler of ``benchmark.dataset_ref.dataset``; lite gets its parent's."""
    pipeline = live_pipeline_module()
    config = _config(tmp_path)
    means = {}
    for dataset in (*_SETS, *_LITE):
        config.benchmark.dataset_ref.dataset = dataset
        space = pipeline.eval_cp_space(config)
        assert space.dataset == dataset
        means[dataset] = space.mean
    for a in _SETS:
        for b in _SETS:
            if a != b:
                assert not np.allclose(means[a], means[b])
    assert pipeline.eval_cp_space(config).scaler_dataset == "set-b"
    np.testing.assert_array_equal(means["set-b-lite"], means["set-b"])


def test_grouped_conditions_each_get_their_own_scaler(tmp_path: Path, monkeypatch) -> None:
    """The grouped driver binds every condition to its own dataset, and stamps that reference."""
    pipeline = live_pipeline_module()
    base = _config(tmp_path)
    expected = {
        d: pipeline.eval_cp_space(OmegaConf.merge(base, {"benchmark": {"dataset_ref": {"dataset": d}}}))
        for d in (*_SETS, *_LITE)
    }
    conditions = [
        {"name": d, "benchmark": {"dataset_ref": {"dataset": d}}, "save": {"save_dir": str(tmp_path / d)}}
        for d in (*_SETS, *_LITE)
    ]
    config = OmegaConf.merge(
        base,
        {
            "conditions": conditions,
            "force_recompute": {"final_metrics": True},
            "cross_condition_probe": {"enabled": False},
        },
    )
    seen: dict[str, object] = {}
    stamped: dict[str, str | None] = {}

    def _fake_evaluate_predictions(cfg, *, models, cp_space):
        seen[cfg.benchmark.dataset_ref.dataset] = cp_space
        return [], [], []

    def _fake_save_metrics(cfg, *args, cp_reference_sha256, **kwargs):
        stamped[cfg.benchmark.dataset_ref.dataset] = cp_reference_sha256

    monkeypatch.setattr(pipeline, "apply_dataset_ref", lambda cfg: None)
    monkeypatch.setattr(pipeline, "load_eval_models", lambda cfg: object())
    monkeypatch.setattr(pipeline, "evaluate_predictions", _fake_evaluate_predictions)
    monkeypatch.setattr(pipeline, "save_metrics", _fake_save_metrics)

    pipeline.evaluate_predictions_grouped(config)

    assert sorted(seen) == sorted(expected)
    for dataset, space in seen.items():
        assert space.dataset == dataset
        assert space.scaler_dataset == _LITE.get(dataset, dataset)
        np.testing.assert_array_equal(space.mean, expected[dataset].mean)
        np.testing.assert_array_equal(space.std, expected[dataset].std)
        assert stamped[dataset] == space.reference_sha256
    np.testing.assert_array_equal(seen["set-b-lite"].mean, seen["set-b"].mean)
    assert not np.allclose(seen["set-a"].mean, seen["set-c"].mean)


def test_unknown_dataset_is_refused(tmp_path: Path) -> None:
    """A condition whose dataset has no scaler fails instead of borrowing another set's."""
    pipeline = live_pipeline_module()
    config = _config(tmp_path)
    config.benchmark.dataset_ref.dataset = "unknown-set"
    with pytest.raises(KeyError, match="no scaler for dataset 'unknown-set'"):
        pipeline.eval_cp_space(config)
