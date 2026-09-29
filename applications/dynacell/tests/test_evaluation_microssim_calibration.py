"""Leaf-level MicroMS3IM calibration: degenerate GT scores NaN, resource errors fail the job.

Drives the real ``evaluate_predictions`` on the cache-only fixture from
``_eval_fixtures`` with ``compute_microssim=true``. The 2026-09-26 re-eval
swallowed a cupy OOM inside calibration, exited 0 and overwrote valid
MicroMS3IM with NaN; only a constant GT slice may produce NaN.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from iohub.ngff import open_ome_zarr
from omegaconf import OmegaConf

from ._eval_fixtures import build_eval_config, build_fixture, live_pipeline_module


def _config(tmp_path: Path):
    """Build the fixture and a serial cache-only config with MicroMS3IM on."""
    pred_path, gt_path, gt_cache_dir, pred_cache_dir = build_fixture(tmp_path / "fixture")
    save_dir = tmp_path / "out"
    save_dir.mkdir()
    config = build_eval_config(
        pred_path, gt_path, gt_cache_dir, pred_cache_dir, save_dir, executor="serial", fov_workers=1
    )
    config.compute_microssim = True
    return config, gt_path


def _cupy_oom() -> Exception:
    cupy = pytest.importorskip("cupy")
    return cupy.cuda.memory.OutOfMemoryError(1_393_542_144, 40_627_942_400)


@pytest.mark.parametrize(
    "make_error",
    [
        _cupy_oom,
        lambda: torch.OutOfMemoryError("CUDA out of memory"),
        lambda: MemoryError("host out of memory"),
        lambda: RuntimeError("RI factor failed to bracket on the right"),
    ],
    ids=["cupy_oom", "torch_oom", "host_oom", "runtime_error"],
)
def test_calibration_failure_propagates(tmp_path: Path, monkeypatch, make_error) -> None:
    """An error raised while fitting α fails the eval instead of scoring NaN."""
    config, _ = _config(tmp_path)
    error = make_error()
    pipeline = live_pipeline_module()

    def _raise(*args, **kwargs):
        raise error

    monkeypatch.setattr(pipeline, "fit_microssim", _raise)
    with pytest.raises(type(error)) as excinfo:
        pipeline.evaluate_predictions(config)
    assert excinfo.value is error


def test_constant_gt_slice_scores_nan(tmp_path: Path) -> None:
    """A constant GT z-slice in the calibration pool yields MicroMS3IM=NaN for every row."""
    config, gt_path = _config(tmp_path)
    with open_ome_zarr(gt_path, mode="r+") as plate:
        plate["A/1/0"].data[0, 0, 0] = 0.0

    pixel_rows, _, _ = live_pipeline_module().evaluate_predictions(config)

    assert pixel_rows
    assert all(math.isnan(row["MicroMS3IM"]) for row in pixel_rows)


def test_saturated_gt_scores_nan(tmp_path: Path) -> None:
    """A GT pool whose background percentile equals its max (max_val == 0) yields NaN, not a crash.

    Every slice keeps a raw range > 0 (five zero pixels in a field of 10.0), but
    cubic normalizes by ``max - percentile_3 == 0``, so its per-slice data range is NaN.
    """
    config, gt_path = _config(tmp_path)
    with open_ome_zarr(gt_path, mode="r+") as plate:
        for _, pos in plate.positions():
            data = np.full(pos.data.shape, 10.0, dtype=np.float32)
            data[..., 0, :5] = 0.0
            pos.data[:] = data

    pixel_rows, _, _ = live_pipeline_module().evaluate_predictions(config)

    assert pixel_rows
    assert all(math.isnan(row["MicroMS3IM"]) for row in pixel_rows)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf], ids=["nan", "inf"])
def test_non_finite_prediction_fails_the_eval(tmp_path: Path, bad_value: float) -> None:
    """One non-finite prediction pixel fails the eval: cubic's α bracket raises and nothing catches it."""
    config, _ = _config(tmp_path)
    with open_ome_zarr(config.io.pred_path, mode="r+") as plate:
        plate["A/1/0"].data[0, 0, 0, 3, 3] = bad_value

    with pytest.raises(RuntimeError, match="RI factor failed to bracket"):
        live_pipeline_module().evaluate_predictions(config)


def _write_plate(path: Path, channel_name: str, volumes: np.ndarray) -> None:
    """Write one HCS position per leading entry of ``volumes`` (each ``(T, D, H, W)``)."""
    with open_ome_zarr(path, mode="w", layout="hcs", channel_names=[channel_name], version="0.5") as plate:
        for i, volume in enumerate(volumes):
            plate.create_position("A", "1", str(i)).create_image("0", volume[:, None])


def test_healthy_leaf_calibrates_and_scores_finite(tmp_path: Path) -> None:
    """Real calibration on healthy GT returns a fitted sim, and scoring with it gives finite MicroMS3IM.

    MS-SSIM needs spatial dims >= 176, so this uses its own 192x192 plates rather
    than the 32x32 cache-only fixture.
    """
    rng = np.random.default_rng(0)
    gt = rng.gamma(2.0, 100.0, size=(2, 1, 2, 192, 192)).astype(np.float32)
    pred = (0.5 * gt + 20.0 + rng.normal(0.0, 10.0, size=gt.shape)).astype(np.float32)
    _write_plate(tmp_path / "gt.zarr", "target", gt)
    _write_plate(tmp_path / "pred.zarr", "prediction", pred)
    io_config = OmegaConf.create({"pred_channel_name": "prediction", "gt_channel_name": "target"})
    pipeline = live_pipeline_module()

    with (
        open_ome_zarr(tmp_path / "pred.zarr", mode="r") as pred_plate,
        open_ome_zarr(tmp_path / "gt.zarr", mode="r") as gt_plate,
    ):
        sim, _ = pipeline._calibrate_microssim(
            list(pred_plate.positions()),
            list(gt_plate.positions()),
            io_config,
            use_gpu=False,
            max_pairs=12,
            seed=42,
            cache_reads=False,
        )

    assert sim is not None
    scores = pipeline.score_microssim(
        [{"target": gt[i, 0], "predict": pred[i, 0]} for i in range(len(gt))], sim, use_gpu=False
    )
    values = [row["MicroMS3IM"] for row in scores]
    assert len(values) == len(gt)
    assert all(math.isfinite(v) and 0.0 < v <= 1.0 for v in values)
