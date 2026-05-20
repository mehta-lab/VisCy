"""Parity test for the grouped multi-condition eval driver.

Builds two independent fixtures (each its own pred + GT plates + mask
caches), runs them once via ``evaluate_predictions_grouped`` and once
via two back-to-back ``evaluate_predictions`` calls, and asserts the
outputs are byte-equal per condition.

Cache-only design (same shape as
``test_evaluation_pipeline_parallel_cpu.py``): ``target_name=er`` +
``io.require_complete_cache=true`` + ``compute_feature_metrics=false``
means no segmenter / extractors / cubic are loaded — the test
validates the **loop structure** of the grouped driver, not the
model-sharing speedup (which is hermetic-untestable).

Fixture-building helpers live in ``_eval_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ._eval_fixtures import (
    N_POSITIONS,
    T,
    build_eval_config,
    build_fixture,
    live_pipeline_module,
    read_position_arrays,
)


def _build_grouped_config(
    cond_a_root: Path,
    cond_b_root: Path,
    save_root: Path,
):
    """Construct the grouped config with two condition overlays."""
    pred_a, gt_a, gt_cache_a, pred_cache_a = build_fixture(cond_a_root)
    pred_b, gt_b, gt_cache_b, pred_cache_b = build_fixture(cond_b_root)

    base = build_eval_config(
        pred_a,
        gt_a,
        gt_cache_a,
        pred_cache_a,
        save_root / "_unused",
        executor="serial",
        fov_workers=1,
    )
    # Strip the placeholder io/save — conditions own those entirely.
    base["io"]["pred_path"] = None
    base["io"]["gt_path"] = None
    base["io"]["gt_cache_dir"] = None
    base["io"]["pred_cache_dir"] = None
    base["save"]["save_dir"] = None

    base["conditions"] = [
        {
            "name": "cond_a",
            "io": {
                "pred_path": str(pred_a),
                "gt_path": str(gt_a),
                "gt_cache_dir": str(gt_cache_a),
                "pred_cache_dir": str(pred_cache_a),
            },
            "save": {"save_dir": str(save_root / "cond_a_grouped")},
        },
        {
            "name": "cond_b",
            "io": {
                "pred_path": str(pred_b),
                "gt_path": str(gt_b),
                "gt_cache_dir": str(gt_cache_b),
                "pred_cache_dir": str(pred_cache_b),
            },
            "save": {"save_dir": str(save_root / "cond_b_grouped")},
        },
    ]
    return base


def test_grouped_matches_sequential_per_condition(tmp_path: Path):
    """Grouped eval produces byte-equal outputs to sequential per-condition runs."""
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()

    pred_a, gt_a, gt_cache_a, pred_cache_a = build_fixture(cond_a_root)
    pred_b, gt_b, gt_cache_b, pred_cache_b = build_fixture(cond_b_root)
    save_a_seq = save_root / "cond_a_seq"
    save_b_seq = save_root / "cond_b_seq"
    save_a_seq.mkdir()
    save_b_seq.mkdir()

    pipeline = live_pipeline_module()
    cfg_a_seq = build_eval_config(pred_a, gt_a, gt_cache_a, pred_cache_a, save_a_seq, executor="serial", fov_workers=1)
    cfg_b_seq = build_eval_config(pred_b, gt_b, gt_cache_b, pred_cache_b, save_b_seq, executor="serial", fov_workers=1)
    pixel_a_seq, mask_a_seq, _ = pipeline.evaluate_predictions(cfg_a_seq)
    pixel_b_seq, mask_b_seq, _ = pipeline.evaluate_predictions(cfg_b_seq)

    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    pipeline = live_pipeline_module()
    results = pipeline.evaluate_predictions_grouped(grouped_cfg)

    assert [name for name, _ in results] == ["cond_a", "cond_b"]
    (_, (pixel_a_grp, mask_a_grp, _)), (_, (pixel_b_grp, mask_b_grp, _)) = results

    def _sort_key(row: dict):
        return (row["FOV"], row["Timepoint"])

    assert sorted(pixel_a_grp, key=_sort_key) == sorted(pixel_a_seq, key=_sort_key)
    assert sorted(mask_a_grp, key=_sort_key) == sorted(mask_a_seq, key=_sort_key)
    assert sorted(pixel_b_grp, key=_sort_key) == sorted(pixel_b_seq, key=_sort_key)
    assert sorted(mask_b_grp, key=_sort_key) == sorted(mask_b_seq, key=_sort_key)

    for cond_name, save_seq, save_grp in [
        ("cond_a", save_a_seq, save_root / "cond_a_grouped"),
        ("cond_b", save_b_seq, save_root / "cond_b_grouped"),
    ]:
        arrs_seq = read_position_arrays(save_seq / "segmentation_results.zarr")
        arrs_grp = read_position_arrays(save_grp / "segmentation_results.zarr")
        assert set(arrs_seq.keys()) == set(arrs_grp.keys()), f"{cond_name} position set mismatch"
        for pos_name, arr_seq in arrs_seq.items():
            np.testing.assert_array_equal(arr_seq, arrs_grp[pos_name], err_msg=f"{cond_name}/{pos_name}")

    assert len(pixel_a_grp) == N_POSITIONS * T
    assert len(pixel_b_grp) == N_POSITIONS * T


def test_grouped_rejects_model_loading_field_overrides(tmp_path: Path):
    """Per-condition overrides on model-loading fields must raise — including condition 0."""
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()

    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    # Sabotage condition B's target_name.
    grouped_cfg["conditions"][1]["target_name"] = "membrane"

    pipeline = live_pipeline_module()
    with pytest.raises(ValueError, match="model-loading field"):
        pipeline.evaluate_predictions_grouped(grouped_cfg)


def test_grouped_rejects_condition0_model_field_override(tmp_path: Path):
    """A model-loading override smuggled into condition 0 must raise, not silently re-baseline.

    Earlier wiring used ``conditions[0]``-merged config as the invariant
    baseline, which would have made condition 0 implicitly trusted to
    redefine ``target_name`` / ``feature_extractor.*`` etc. Guards
    against regression of that asymmetry.
    """
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()

    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    grouped_cfg["conditions"][0]["target_name"] = "membrane"

    pipeline = live_pipeline_module()
    with pytest.raises(ValueError, match="model-loading field"):
        pipeline.evaluate_predictions_grouped(grouped_cfg)


def test_grouped_rejects_empty_conditions(tmp_path: Path):
    """Missing or empty 'conditions' list must raise an informative error."""
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()
    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    grouped_cfg["conditions"] = []

    pipeline = live_pipeline_module()
    with pytest.raises(ValueError, match="non-empty"):
        pipeline.evaluate_predictions_grouped(grouped_cfg)
