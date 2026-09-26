"""Per-FOV timepoint-count guard and flag-aware feature-cache validity.

Drives the real ``evaluate_predictions`` on tiny HCS plates whose positions carry
their own ``T``. ``_calibrate_microssim`` is the first step after the hoisted
position validation, so it is replaced by a sentinel raise: reaching it proves the
validation passed, and a mismatch must raise ``ValueError`` before it (and before
``precompute_deep_features``) can write anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from iohub.ngff import open_ome_zarr

from dynacell.evaluation.provenance import write_metrics_provenance

from ._eval_fixtures import build_eval_config, live_pipeline_module, make_cp_reference

D, H, W = 3, 8, 8


class _ReachedCalibration(Exception):
    """Raised by the stubbed calibration: validation passed and the run continued."""


def _make_plate(path: Path, channel_name: str, t_per_position: list[int]) -> None:
    """Write an HCS plate with one position per entry, position ``i`` holding ``T=t_per_position[i]``."""
    with open_ome_zarr(path, mode="w", layout="hcs", channel_names=[channel_name], version="0.5") as plate:
        for i, t in enumerate(t_per_position):
            pos = plate.create_position("A", "1", str(i))
            pos.create_image("0", np.zeros((t, 1, D, H, W), dtype=np.float32))


def _run(tmp_path: Path, monkeypatch, pred_t: list[int], gt_t: list[int], seg_t: list[int] | None):
    """Build the stores, stub calibration, and call ``evaluate_predictions``."""
    pipeline = live_pipeline_module()
    pred, gt = tmp_path / "pred.zarr", tmp_path / "gt.zarr"
    _make_plate(pred, "prediction", pred_t)
    _make_plate(gt, "target", gt_t)
    gt_cache, pred_cache = tmp_path / "gt_cache", tmp_path / "pred_cache"
    config = build_eval_config(pred, gt, gt_cache, pred_cache, tmp_path / "out", executor="serial", fov_workers=1)
    config.io.require_complete_cache = False
    config.compute_microssim = True
    if seg_t is not None:
        seg = tmp_path / "seg.zarr"
        _make_plate(seg, "cell_segmentation", seg_t)
        config.io.cell_segmentation_path = str(seg)

    def _stop(*args, **kwargs):
        raise _ReachedCalibration

    monkeypatch.setattr(pipeline, "_calibrate_microssim", _stop)
    pipeline.evaluate_predictions(config)


def _no_cache_written(tmp_path: Path) -> bool:
    """True when neither mask/feature cache dir nor any final-metrics file exists."""
    out = tmp_path / "out"
    return (
        not (tmp_path / "gt_cache").exists()
        and not (tmp_path / "pred_cache").exists()
        and not list(out.glob("*_metrics.*"))
    )


def test_pred_t_subset_of_gt_raises(tmp_path: Path, monkeypatch) -> None:
    """Pred T=2 against GT T=3 at one position raises, naming it, before any cache write."""
    with pytest.raises(ValueError, match=r"Timepoint count mismatch at position 'A/1/1'.*'pred': 2, 'gt': 3"):
        _run(tmp_path, monkeypatch, pred_t=[3, 2], gt_t=[3, 3], seg_t=None)
    assert _no_cache_written(tmp_path)


def test_seg_t_differs_from_pred_and_gt_raises(tmp_path: Path, monkeypatch) -> None:
    """GT/pred T=2 against a T=3 cell-segmentation store raises and lists all three T."""
    with pytest.raises(ValueError, match=r"'A/1/0'.*'pred': 2, 'gt': 2, 'seg': 3"):
        _run(tmp_path, monkeypatch, pred_t=[2], gt_t=[2], seg_t=[3])
    assert _no_cache_written(tmp_path)


def test_matched_per_position_t_passes(tmp_path: Path, monkeypatch) -> None:
    """T varying across positions but agreeing per position passes the guard."""
    with pytest.raises(_ReachedCalibration):
        _run(tmp_path, monkeypatch, pred_t=[2, 3], gt_t=[2, 3], seg_t=[2, 3])


def _feature_cache_config(tmp_path: Path, **flags: bool):
    """Config with feature metrics on and the given ``feature_metrics.compute_*`` flags."""
    config = build_eval_config(
        tmp_path / "pred.zarr",
        tmp_path / "gt.zarr",
        tmp_path / "gt_cache",
        tmp_path / "pred_cache",
        tmp_path,
        executor="serial",
        fov_workers=1,
    )
    config.compute_feature_metrics = True
    for name, value in flags.items():
        config.feature_metrics[name] = value
    make_cp_reference(config, tmp_path / "cp_reference.json")
    return config


def _write_final_caches(save_dir: Path, feature_row: dict) -> None:
    """Write a stamped pixel/mask/feature NPY cache set holding one feature row."""
    np.save(save_dir / "pixel_metrics.npy", [{"SI_PSNR": 1.0, "SI_SSIM": 1.0, "SI_NRMSE": 1.0}])
    np.save(save_dir / "mask_metrics.npy", [{"metric": "mask"}])
    np.save(save_dir / "feature_metrics.npy", [feature_row])
    # The stamp must carry the hash of the reference _feature_cache_config points at.
    reference_sha256 = make_cp_reference(_feature_cache_config(save_dir), save_dir / "cp_reference.json")
    write_metrics_provenance(save_dir, cp_reference_sha256=reference_sha256)


def _dataset_row(prefixes: tuple[str, ...], families: tuple[str, ...]) -> dict:
    """``Dataset_<prefix>_<family>`` columns plus the always-present KID/cosine."""
    row: dict[str, float] = {"CP_KID": 0.0}
    for prefix in prefixes:
        for family in ("KID", "KID_std", "Median_Cosine_Similarity", *families):
            row[f"Dataset_{prefix}_{family}"] = 0.0
    return row


_ALL_FAMILIES = ("FID", "Precision", "Precision_std", "Recall", "Recall_std", "F1", "F1_std", "MIND")


def test_cache_without_fid_is_invalid_once_fid_is_enabled(tmp_path: Path) -> None:
    """A dir written with FID off is recomputed when FID is on, reused while it stays off."""
    pipeline = live_pipeline_module()
    no_fid = tuple(f for f in _ALL_FAMILIES if f != "FID")
    _write_final_caches(tmp_path, _dataset_row(("CP", "DINOv3"), no_fid))

    assert not pipeline._final_metrics_cache_valid(_feature_cache_config(tmp_path))
    assert pipeline._final_metrics_cache_valid(_feature_cache_config(tmp_path, compute_fid=False))


def test_full_cache_stays_valid(tmp_path: Path) -> None:
    """A dir carrying every family for every prefix it scored remains reusable."""
    pipeline = live_pipeline_module()
    _write_final_caches(tmp_path, _dataset_row(("CP", "DINOv3", "DynaCLR"), _ALL_FAMILIES))
    assert pipeline._final_metrics_cache_valid(_feature_cache_config(tmp_path))


def test_real_full_benchmark_columns_stay_valid(tmp_path: Path) -> None:
    """The column set of a real all-flags-on benchmark dir passes the new check."""
    real = Path("/hpc/projects/virtual_staining/training/dynacell/nucleus/fnet3d_paper/ipsc/ipsc/feature_metrics.csv")
    if not real.exists():
        pytest.skip(f"{real} not reachable from this host")
    with real.open() as f:
        header = f.readline().rstrip("\n").split(",")
    pipeline = live_pipeline_module()
    _write_final_caches(tmp_path, dict.fromkeys(header, 0.0))
    assert pipeline._final_metrics_cache_valid(_feature_cache_config(tmp_path))


def test_cache_scored_in_another_cp_reference_is_invalid(tmp_path: Path) -> None:
    """Rebuilding the CP reference invalidates every final-metrics cache stamped with the old one."""
    pipeline = live_pipeline_module()
    _write_final_caches(tmp_path, _dataset_row(("CP", "DINOv3", "DynaCLR"), _ALL_FAMILIES))
    config = _feature_cache_config(tmp_path)
    assert pipeline._final_metrics_cache_valid(config)

    make_cp_reference(config, tmp_path / "cp_reference.json", seed=1)  # same path, new content
    assert not pipeline._final_metrics_cache_valid(config)


def test_missing_cp_reference_fails_before_any_work(tmp_path: Path, monkeypatch) -> None:
    """A feature-metrics eval with no CP reference raises, naming the build command, before loading models."""
    pipeline = live_pipeline_module()
    config = _feature_cache_config(tmp_path)
    config.feature_metrics.cp.reference_path = str(tmp_path / "absent.json")

    def _no_models(*args, **kwargs):
        raise AssertionError("models loaded before the CP reference was checked")

    monkeypatch.setattr(pipeline, "load_eval_models", _no_models)
    with pytest.raises(FileNotFoundError, match="build_cp_reference.py --target er"):
        pipeline.evaluate_predictions(config)


def test_cp_reference_of_another_recipe_is_refused(tmp_path: Path, monkeypatch) -> None:
    """A reference fit on GLCM-off caches cannot score a GLCM-on eval."""
    pipeline = live_pipeline_module()
    # Import after the fresh pipeline import, which re-creates dynacell.evaluation.* modules.
    from dynacell.evaluation.cache import StaleCacheError

    config = _feature_cache_config(tmp_path)  # reference built for this (GLCM-off) recipe
    config.feature_metrics.cp.glcm = {"enabled": True, "levels": 32, "distances": [1]}
    monkeypatch.setattr(pipeline, "load_eval_models", lambda *a, **k: pytest.fail("models loaded"))
    with pytest.raises(StaleCacheError, match="different CP recipe"):
        pipeline.evaluate_predictions(config)


def test_save_metrics_stamps_the_cp_reference_hash(tmp_path: Path) -> None:
    """``save_metrics`` records the hash of the reference the CP metrics were scored in."""
    import json

    from dynacell.evaluation.provenance import PROVENANCE_FILENAME

    pipeline = live_pipeline_module()
    config = _feature_cache_config(tmp_path)
    reference_sha256 = make_cp_reference(config, tmp_path / "cp_reference.json")
    pipeline.save_metrics(config, pixel_metrics=[{"FOV": "A/1/0", "Timepoint": 0, "PCC": 0.5}])
    stamp = json.loads((tmp_path / PROVENANCE_FILENAME).read_text())
    assert stamp["cp_reference_sha256"] == reference_sha256
