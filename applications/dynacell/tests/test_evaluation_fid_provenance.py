"""Exercise FID solver provenance through the real pipeline writer and cache gate."""

import json
from pathlib import Path

import numpy as np
import pytest

from dynacell.evaluation.cache import prediction_sources, prediction_sources_sha256_12

from ._eval_fixtures import build_eval_config, build_fixture, live_pipeline_module, make_cp_reference


@pytest.mark.parametrize(("features_on", "fid_on"), [(True, True), (True, False), (False, True)])
def test_pipeline_fid_solver_stamp_and_cache_gate(tmp_path: Path, monkeypatch, features_on: bool, fid_on: bool):
    """Only requested FID requires a known solver; cache rejection is exercised with a foreign solver."""
    pipeline = live_pipeline_module()
    pred_path, gt_path, gt_cache, pred_cache = build_fixture(tmp_path / "fixture")
    cfg = build_eval_config(
        pred_path, gt_path, gt_cache, pred_cache, tmp_path / "out", executor="serial", fov_workers=1
    )
    cfg.compute_feature_metrics = features_on
    cfg.feature_metrics.compute_fid = fid_on
    cfg.feature_metrics.compute_prc = False
    cfg.feature_metrics.compute_mind = False
    space = None
    feature = []
    if features_on:
        make_cp_reference(cfg, tmp_path / "cp_reference.json")
        space = pipeline.eval_cp_space(cfg)
        raw = np.random.default_rng(42).standard_normal((4, len(space.feature_names)))
        pred, target = pipeline._cp_row_features(raw + 0.1, raw, space)
        per_t = pipeline.compute_feature_similarity_pairwise(pred, target, "CP", compute_fid=fid_on)
        dataset = pipeline.compute_feature_similarity(
            pred, target, "CP", compute_fid=fid_on, compute_prc=False, compute_mind=False
        )
        feature = [
            {
                **per_t,
                **{f"Dataset_{key}": value for key, value in dataset.items()},
                "CP_clip_frac": 0.0,
                "Dataset_CP_clip_frac": 0.0,
            }
        ]
    # Plotting is not involved in provenance; the real CSV/NPY writer still runs.
    monkeypatch.setattr(pipeline, "plot_metrics", lambda *args: None)
    digest = prediction_sources_sha256_12(prediction_sources(pred_path, "prediction"))
    pipeline.save_metrics(
        cfg,
        [{"SI_PSNR": 10.0, "SI_SSIM": 0.5, "SI_NRMSE": 0.1}],
        [{"DICE": 0.5}],
        feature,
        cp_space=space,
        prediction_digest=digest,
    )
    stamp = Path(cfg.save.save_dir) / pipeline.PROVENANCE_FILENAME
    payload = json.loads(stamp.read_text())
    if features_on and fid_on:
        assert payload["fid_implementation"] == "sample-space-svd-v1"
    else:
        assert "fid_implementation" not in payload
    assert pipeline._final_metrics_cache_valid(cfg)
    payload["fid_implementation"] = "unknown-solver-v99"
    stamp.write_text(json.dumps(payload))
    assert pipeline._final_metrics_cache_valid(cfg) is not (features_on and fid_on)
    if features_on and fid_on:
        cfg.feature_metrics.compute_fid = False
        assert pipeline._final_metrics_cache_valid(cfg)
        cfg.feature_metrics.compute_fid = True
        del payload["fid_implementation"]
        stamp.write_text(json.dumps(payload))
        assert pipeline._final_metrics_cache_valid(cfg)  # measured-compatible legacy solver
