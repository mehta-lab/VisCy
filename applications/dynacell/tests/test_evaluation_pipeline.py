"""Regression tests for evaluation pipeline caching."""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def _write_metrics(path: Path, payload: list[dict[str, str]]) -> None:
    """Write an object-array metrics cache file."""
    np.save(path, payload)


def _import_pipeline_with_stubs(monkeypatch):
    """Import the pipeline module with lightweight dependency stubs."""
    utils_module = types.ModuleType("dynacell.evaluation.utils")
    utils_module.DinoV3FeatureExtractor = object
    utils_module.DynaCLRFeatureExtractor = object
    utils_module.plot_metrics = lambda *args, **kwargs: None

    metrics_module = types.ModuleType("dynacell.evaluation.metrics")
    metrics_module.calculate_microssim = lambda *args, **kwargs: []
    metrics_module.compute_pixel_metrics = lambda *args, **kwargs: {}
    metrics_module.evaluate_segmentations = lambda *args, **kwargs: {}
    metrics_module.cp_target_regionprops = lambda *args, **kwargs: None
    metrics_module.cp_pred_regionprops = lambda *args, **kwargs: None
    metrics_module.deep_target_features = lambda *args, **kwargs: None
    metrics_module.deep_pred_features = lambda *args, **kwargs: None

    feature_metrics_module = types.ModuleType("dynacell.evaluation.feature_metrics")
    feature_metrics_module.compute_feature_similarity = lambda *args, **kwargs: {}
    feature_metrics_module.compute_feature_similarity_pairwise = lambda *args, **kwargs: {}

    feature_select_module = types.ModuleType("dynacell.evaluation.feature_select")
    feature_select_module.select_features = lambda gt, pred, **kw: (
        gt,
        pred,
        np.ones(gt.shape[1] if gt is not None and gt.ndim >= 2 else 0, dtype=bool),
    )

    linear_probe_module = types.ModuleType("dynacell.evaluation.linear_probe")
    _nan_auroc = {"auroc_mean": float("nan"), "auroc_std": float("nan"), "n_folds": 0}
    linear_probe_module.fov_stratified_auroc = lambda *a, **kw: _nan_auroc
    linear_probe_module.paired_auroc = lambda *a, **kw: _nan_auroc
    linear_probe_module.indistinguishability = lambda auroc: float("nan")

    segmentation_module = types.ModuleType("dynacell.evaluation.segmentation")
    segmentation_module.segment = lambda *args, **kwargs: None
    segmentation_module.prepare_segmentation_model = lambda *args, **kwargs: None

    # Stub hydra if not installed
    if "hydra" not in sys.modules:
        hydra_module = types.ModuleType("hydra")
        hydra_module.main = lambda **kwargs: lambda fn: fn
        monkeypatch.setitem(sys.modules, "hydra", hydra_module)

    monkeypatch.setitem(sys.modules, "dynacell.evaluation.utils", utils_module)
    monkeypatch.setitem(sys.modules, "dynacell.evaluation.metrics", metrics_module)
    monkeypatch.setitem(sys.modules, "dynacell.evaluation.feature_metrics", feature_metrics_module)
    monkeypatch.setitem(sys.modules, "dynacell.evaluation.feature_select", feature_select_module)
    monkeypatch.setitem(sys.modules, "dynacell.evaluation.linear_probe", linear_probe_module)
    monkeypatch.setitem(sys.modules, "dynacell.evaluation.segmentation", segmentation_module)
    # Don't stub iohub globally — it's used by viscy_data in the same process
    sys.modules.pop("dynacell.evaluation.pipeline", None)

    return importlib.import_module("dynacell.evaluation.pipeline")


def test_evaluate_model_reuses_cache_without_feature_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Reuse pixel and mask caches when feature metrics are disabled."""
    pipeline = _import_pipeline_with_stubs(monkeypatch)
    config = OmegaConf.create(
        {
            "compute_feature_metrics": False,
            "force_recompute": {
                "all": False,
                "gt_masks": False,
                "gt_cp": False,
                "gt_dinov3": False,
                "gt_dynaclr": False,
                "final_metrics": False,
            },
            "save": {
                "save_dir": str(tmp_path),
                "pixel_metrics_filename": "pixel_metrics.npy",
                "mask_metrics_filename": "mask_metrics.npy",
                "feature_metrics_filename": "feature_metrics.npy",
            },
        }
    )
    expected_pixel_metrics = [{"metric": "pixel"}]
    expected_mask_metrics = [{"metric": "mask"}]
    _write_metrics(tmp_path / config.save.pixel_metrics_filename, expected_pixel_metrics)
    _write_metrics(tmp_path / config.save.mask_metrics_filename, expected_mask_metrics)

    def fail_if_recomputed(_config):
        raise AssertionError("evaluate_predictions should not run when cache is valid")

    monkeypatch.setattr(pipeline, "evaluate_predictions", fail_if_recomputed)

    # Access __wrapped__ if Hydra decorated it, otherwise call directly
    fn = getattr(pipeline.evaluate_model, "__wrapped__", pipeline.evaluate_model)
    pixel_metrics, mask_metrics, feature_metrics = fn(config)

    assert pixel_metrics == expected_pixel_metrics
    assert mask_metrics == expected_mask_metrics
    assert feature_metrics == []
