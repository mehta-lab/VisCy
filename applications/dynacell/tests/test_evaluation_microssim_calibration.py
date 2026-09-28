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
