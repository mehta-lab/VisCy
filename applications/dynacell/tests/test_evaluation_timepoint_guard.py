"""Per-FOV timepoint-count guard and flag-aware feature-cache validity.

Drives the real ``evaluate_predictions`` on tiny HCS plates whose positions carry
their own ``T``. ``_calibrate_microssim`` is the first step after the hoisted
position validation, so it is replaced by a sentinel raise: reaching it proves the
validation passed, and a mismatch must raise ``ValueError`` before it (and before
``precompute_deep_features``) can write anything.
"""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
from iohub.ngff import open_ome_zarr
from omegaconf import OmegaConf

from dynacell.evaluation.cache import cache_paths, load_manifest, save_manifest
from dynacell.evaluation.model_loader import EvalModels
from dynacell.evaluation.provenance import PROVENANCE_FILENAME, write_metrics_provenance

from ._eval_fixtures import N_POSITIONS, build_eval_config, live_pipeline_module, make_cp_reference

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
    """Config with feature metrics on, the given ``feature_metrics.compute_*`` flags, and a CP reference.

    Returns ``(config, reference_sha256)``; the reference is built once, at
    ``tmp_path / "cp_reference.json"``.
    """
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
    if not (tmp_path / "gt.zarr").exists():  # the cache-hit path checks the GT store's positions
        _make_plate(tmp_path / "gt.zarr", "target", [2] * N_POSITIONS)
    for name, value in flags.items():
        config.feature_metrics[name] = value
    return config, make_cp_reference(config, tmp_path / "cp_reference.json")


def _write_final_caches(pipeline, save_dir: Path, feature_row: dict, config) -> None:
    """Write a pixel/mask/feature NPY cache set holding one feature row, stamped as ``config`` would score it."""
    np.save(save_dir / "pixel_metrics.npy", [{"SI_PSNR": 1.0, "SI_SSIM": 1.0, "SI_NRMSE": 1.0}])
    np.save(save_dir / "mask_metrics.npy", [{"metric": "mask"}])
    np.save(save_dir / "feature_metrics.npy", [feature_row])
    space = pipeline.eval_cp_space(config)
    write_metrics_provenance(save_dir, cp_reference_sha256=space.reference_sha256, cp_space_sha256=space.binding_sha256)


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
    config, _ = _feature_cache_config(tmp_path)
    _write_final_caches(
        pipeline, tmp_path, _dataset_row(("CP", "DINOv3"), tuple(f for f in _ALL_FAMILIES if f != "FID")), config
    )

    assert not pipeline._final_metrics_cache_valid(config)
    config.feature_metrics.compute_fid = False
    assert pipeline._final_metrics_cache_valid(config)


def test_full_cache_stays_valid(tmp_path: Path) -> None:
    """A dir carrying every family for every prefix it scored remains reusable."""
    pipeline = live_pipeline_module()
    config, _ = _feature_cache_config(tmp_path)
    _write_final_caches(pipeline, tmp_path, _dataset_row(("CP", "DINOv3", "DynaCLR"), _ALL_FAMILIES), config)
    assert pipeline._final_metrics_cache_valid(config)


def test_real_full_benchmark_columns_stay_valid(tmp_path: Path) -> None:
    """The column set of a real all-flags-on benchmark dir passes the new check."""
    real = Path("/hpc/projects/virtual_staining/training/dynacell/nucleus/fnet3d_paper/ipsc/ipsc/feature_metrics.csv")
    if not real.exists():
        pytest.skip(f"{real} not reachable from this host")
    with real.open() as f:
        header = f.readline().rstrip("\n").split(",")
    pipeline = live_pipeline_module()
    config, _ = _feature_cache_config(tmp_path)
    _write_final_caches(pipeline, tmp_path, dict.fromkeys(header, 0.0), config)
    assert pipeline._final_metrics_cache_valid(config)


def test_cache_scored_in_another_cp_reference_is_invalid(tmp_path: Path) -> None:
    """Rebuilding the CP reference invalidates every final-metrics cache stamped with the old one."""
    pipeline = live_pipeline_module()
    config, _ = _feature_cache_config(tmp_path)
    _write_final_caches(pipeline, tmp_path, _dataset_row(("CP", "DINOv3", "DynaCLR"), _ALL_FAMILIES), config)
    assert pipeline._final_metrics_cache_valid(config)

    make_cp_reference(config, tmp_path / "cp_reference.json", seed=1)  # same path, new content
    assert not pipeline._final_metrics_cache_valid(config)


@pytest.mark.parametrize(
    ("key", "value"), [("limit_positions", 1), ("io.exclude_fov_names", ["A/1/0"])], ids=["limit", "exclude"]
)
def test_partial_walk_never_reuses_cp_rows(tmp_path: Path, key: str, value) -> None:
    """A partial position walk bypasses the cache so ``check_positions`` sees its GT position set."""
    pipeline = live_pipeline_module()
    config, _ = _feature_cache_config(tmp_path)
    _write_final_caches(pipeline, tmp_path, _dataset_row(("CP", "DINOv3", "DynaCLR"), _ALL_FAMILIES), config)
    assert pipeline._final_metrics_cache_valid(config)
    OmegaConf.update(config, key, value)
    assert not pipeline._final_metrics_cache_valid(config)


def test_cache_reuse_refuses_a_gt_store_extended_since_the_fit(tmp_path: Path) -> None:
    """A GT store that gained a position the reference was not fit on is refused on the cache-hit path."""
    pipeline = live_pipeline_module()
    config, _ = _feature_cache_config(tmp_path)
    _write_final_caches(pipeline, tmp_path, _dataset_row(("CP", "DINOv3", "DynaCLR"), _ALL_FAMILIES), config)
    assert pipeline._final_metrics_cache_valid(config)
    with open_ome_zarr(tmp_path / "gt.zarr", mode="r+") as plate:
        plate.create_position("A", "1", str(N_POSITIONS)).create_image("0", np.zeros((2, 1, D, H, W), dtype=np.float32))
    with pytest.raises(ValueError):
        pipeline._final_metrics_cache_valid(config)


def test_cache_reuse_refuses_a_gt_recache(tmp_path: Path) -> None:
    """A reusable final-metrics cache is refused once the GT CP cache was re-cached after the build."""
    pipeline = live_pipeline_module()
    config, _ = _feature_cache_config(tmp_path)  # records the cache's real built_at
    _write_final_caches(pipeline, tmp_path, _dataset_row(("CP", "DINOv3", "DynaCLR"), _ALL_FAMILIES), config)
    assert pipeline._final_metrics_cache_valid(config)

    paths = cache_paths(tmp_path / "gt_cache")
    manifest = load_manifest(paths)
    manifest["artifacts"]["cp_features"]["built_at"] = "2099-01-01T00:00:00+00:00"  # re-cached after the build
    save_manifest(paths, manifest)
    with pytest.raises(Exception, match="was built at 2099-01-01") as err:
        pipeline._final_metrics_cache_valid(config)
    assert type(err.value).__name__ == "StaleCacheError"


def _two_set_cache(pipeline, tmp_path: Path):
    """A final-metrics cache scored for set-a of a two-set reference; returns ``(config, reference path)``."""
    config, _ = _feature_cache_config(tmp_path)
    reference = tmp_path / "cp_reference.json"
    make_cp_reference(config, reference, datasets=("set-a", "set-b"))  # binds config to set-a
    _write_final_caches(pipeline, tmp_path, _dataset_row(("CP", "DINOv3", "DynaCLR"), _ALL_FAMILIES), config)
    assert pipeline._final_metrics_cache_valid(config)
    return config, reference


def test_cache_scored_for_another_dataset_is_invalid(tmp_path: Path) -> None:
    """A save_dir scored with set-a's scaler is not reusable by a config bound to set-b (same reference)."""
    pipeline = live_pipeline_module()
    config, _ = _two_set_cache(pipeline, tmp_path)
    config.benchmark.dataset_ref.dataset = "set-b"
    assert not pipeline._final_metrics_cache_valid(config)


def test_cache_is_invalid_once_the_gt_matrix_changed(tmp_path: Path) -> None:
    """Same numeric reference hash, but the dataset's recorded GT matrix changed: CP rows are stale."""
    pipeline = live_pipeline_module()
    config, reference = _two_set_cache(pipeline, tmp_path)
    payload = json.loads(reference.read_text())
    payload["fit"]["datasets"]["set-a"]["gt_matrix_sha256"] = "0" * 64  # unhashed fit provenance
    reference.write_text(json.dumps(payload))
    assert pipeline.eval_cp_space(config).reference_sha256 == payload["sha256"]  # numeric hash unchanged
    assert not pipeline._final_metrics_cache_valid(config)


def test_cache_stays_valid_after_a_harmless_rebuild(tmp_path: Path) -> None:
    """A rebuild that only moves built_at (same GT matrix) keeps the cache reusable.

    check_gt_cache passes because the rebuild records the cache's new built_at, and
    the binding is unchanged because the GT-matrix sha256 is.
    """
    pipeline = live_pipeline_module()
    config, reference = _two_set_cache(pipeline, tmp_path)
    paths = cache_paths(tmp_path / "gt_cache")
    manifest = load_manifest(paths)
    manifest["artifacts"]["cp_features"]["built_at"] = "2099-01-01T00:00:00+00:00"
    save_manifest(paths, manifest)
    payload = json.loads(reference.read_text())
    for record in payload["fit"]["datasets"].values():
        record["cp_cache_built_at"] = "2099-01-01T00:00:00+00:00"
    reference.write_text(json.dumps(payload))
    assert pipeline._final_metrics_cache_valid(config)


def test_feature_less_cache_ignores_the_cp_reference(tmp_path: Path) -> None:
    """A compute_feature_metrics=false dir with no CP hash in its stamp stays reusable, with no reference."""
    pipeline = live_pipeline_module()
    config = build_eval_config(
        tmp_path / "pred.zarr",
        tmp_path / "gt.zarr",
        tmp_path / "g",
        tmp_path / "p",
        tmp_path,
        executor="serial",
        fov_workers=1,
    )
    np.save(tmp_path / "pixel_metrics.npy", [{"SI_PSNR": 1.0, "SI_SSIM": 1.0, "SI_NRMSE": 1.0}])
    np.save(tmp_path / "mask_metrics.npy", [{"metric": "mask"}])
    (tmp_path / PROVENANCE_FILENAME).write_text(json.dumps({"versions": {"cubic": version("cubic")}}))
    assert pipeline._final_metrics_cache_valid(config)


def test_missing_cp_reference_names_the_build_command(tmp_path: Path) -> None:
    """Binding an eval with no CP reference raises FileNotFoundError with the build command."""
    pipeline = live_pipeline_module()
    config, _ = _feature_cache_config(tmp_path)
    config.feature_metrics.cp.reference_path = str(tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="build_cp_reference.py --target er"):
        pipeline.eval_cp_space(config)


def test_cp_reference_of_another_recipe_is_refused(tmp_path: Path) -> None:
    """A reference fit on GLCM-off caches cannot score a GLCM-on eval."""
    pipeline = live_pipeline_module()
    config, _ = _feature_cache_config(tmp_path)  # reference built for this (GLCM-off) recipe
    config.feature_metrics.cp.glcm = {"enabled": True, "levels": 32, "distances": [1]}
    with pytest.raises(Exception, match="different CP recipe") as err:
        pipeline.eval_cp_space(config)
    # Checked by name: live_pipeline_module() re-creates dynacell.evaluation.cache, so the
    # class imported at the top of this file is not the one the fresh module raises.
    assert type(err.value).__name__ == "StaleCacheError"


def test_feature_metrics_need_a_cp_space(tmp_path: Path) -> None:
    """``evaluate_predictions`` refuses feature metrics without the caller's CP space, before any model load."""
    pipeline = live_pipeline_module()
    config, _ = _feature_cache_config(tmp_path)
    with pytest.raises(ValueError, match="cp_space must be given exactly when"):
        pipeline.evaluate_predictions(config)


def test_evaluate_model_stamps_the_reference_it_scored_with(tmp_path: Path, monkeypatch) -> None:
    """The hash stamped by save_metrics is the reference loaded before scoring, never a re-read.

    The reference file is rebuilt while "scoring" runs; the stamp must still carry the
    hash of the space evaluate_predictions received.
    """
    pipeline = live_pipeline_module()
    config, sha256 = _feature_cache_config(tmp_path)
    config.force_recompute.final_metrics = True
    seen = {}

    def _fake_evaluate_predictions(cfg, *, cp_space):
        seen["cp_space"] = cp_space
        make_cp_reference(cfg, tmp_path / "cp_reference.json", seed=7)  # rebuilt mid-run
        return [{"FOV": "A/1/0", "Timepoint": 0, "PCC": 0.5}], [], []

    monkeypatch.setattr(pipeline, "check_cubic_pin", lambda: None)
    monkeypatch.setattr(pipeline, "apply_dataset_ref", lambda cfg: None)
    monkeypatch.setattr(pipeline, "evaluate_predictions", _fake_evaluate_predictions)
    getattr(pipeline.evaluate_model, "__wrapped__", pipeline.evaluate_model)(config)

    assert seen["cp_space"].reference_sha256 == sha256
    stamp = json.loads((tmp_path / PROVENANCE_FILENAME).read_text())
    assert stamp["cp_reference_sha256"] == sha256
    assert stamp["cp_space_sha256"] == seen["cp_space"].binding_sha256


def _feature_plates(tmp_path: Path) -> None:
    """Pred/GT/segmentation plates with three positions, as the feature path needs."""
    _make_plate(tmp_path / "pred.zarr", "prediction", [2, 2, 2])
    _make_plate(tmp_path / "gt.zarr", "target", [2, 2, 2])
    _make_plate(tmp_path / "seg.zarr", "cell_segmentation", [2, 2, 2])


def _stub_models() -> EvalModels:
    """A model bundle whose extractors are never called: the run must stop before the FOV loop."""
    return EvalModels(
        seg_model=None,
        dinov3=object(),
        dynaclr=object(),
        celldino=None,
        morphem=None,
        dinov3_model_name=None,
        dynaclr_ckpt_path=None,
        dynaclr_encoder_cfg=None,
        celldino_weights_path=None,
        morphem_model_name=None,
    )


def test_partial_position_walk_is_refused_for_cp(tmp_path: Path, monkeypatch) -> None:
    """``limit_positions`` on a non-lite dataset fails before any per-FOV work: its CP cells are not the fit's."""
    pipeline = live_pipeline_module()
    _feature_plates(tmp_path)
    config, _ = _feature_cache_config(tmp_path)
    config.io.cell_segmentation_path = str(tmp_path / "seg.zarr")
    config.limit_positions = 2
    monkeypatch.setattr(pipeline, "_calibrate_microssim", lambda *a, **k: pytest.fail("reached per-FOV work"))
    cp_space = pipeline.eval_cp_space(config)
    with pytest.raises(ValueError, match="cannot be combined with compute_feature_metrics"):
        pipeline.evaluate_predictions(config, models=_stub_models(), cp_space=cp_space)


def test_gt_recache_after_the_reference_is_refused_up_front(tmp_path: Path, monkeypatch) -> None:
    """A GT CP cache whose ``built_at`` moved since the build fails before any model load."""
    pipeline = live_pipeline_module()
    config, _ = _feature_cache_config(tmp_path)  # records the cache's real built_at
    paths = cache_paths(tmp_path / "gt_cache")
    manifest = load_manifest(paths)
    manifest["artifacts"]["cp_features"]["built_at"] = "2099-01-01T00:00:00+00:00"  # re-cached after the build
    save_manifest(paths, manifest)
    monkeypatch.setattr(pipeline, "load_eval_models", lambda *a, **k: pytest.fail("models loaded"))
    cp_space = pipeline.eval_cp_space(config)
    with pytest.raises(Exception, match="was built at 2099-01-01") as err:
        pipeline.evaluate_predictions(config, cp_space=cp_space)
    assert type(err.value).__name__ == "StaleCacheError"
