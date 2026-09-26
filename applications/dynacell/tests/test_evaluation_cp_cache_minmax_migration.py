"""Load-time migration of CPU-built CP caches to float32 intensity_min/max.

A CP cache written by the CPU extractor before ``cp_regionprops`` rounded
intensity_min/max to float32 holds float64 values there, while GPU-built caches
(and every fresh compute) are float32-exact. Each CP cache reader rounds the two
columns on load, so a CPU-built cache reads bit-identical to the GPU-equivalent
one and yields the same CP reference mask. These tests write both caches through
the real pipeline writer (the CPU-built one with the rounding disabled) and read
them back through every reader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from iohub.ngff import open_ome_zarr

from dynacell.evaluation import cp_reference, metrics, pipeline_cache
from dynacell.evaluation.cache import cache_paths

from ._eval_fixtures import N_POSITIONS, D, H, T, W, build_eval_config, make_mask_cache

_POSITIONS = [f"A/1/{i}" for i in range(N_POSITIONS)]


def _segmentation() -> np.ndarray:
    """``(T, D, H, W)`` labels: a 4 x 4 grid of 6 x 6 x 6 cells."""
    seg = np.zeros((T, D, H, W), dtype=np.int32)
    label = 1
    for row in range(4):
        for col in range(4):
            y, x = 1 + 8 * row, 1 + 8 * col
            seg[:, 2:8, y : y + 6, x : x + 6] = label
            label += 1
    return seg


def _saturated_gt(seed: int) -> np.ndarray:
    """``(T, D, H, W)`` GT whose every cell holds p99-clipped voxels.

    Each of the 16 cells carries a 2 x 2 x 2 hot block (128 hot voxels, > 1% of
    the volume), so the p99 clip ``hi`` is the hot value and every cell's
    normalized max is ``(hi - lo) / ((hi - lo) + eps)``: exactly 1.0 in float32,
    ``1 - eps / (hi - lo)`` in float64. The z=0 plane (9% of the volume, outside
    every cell) sits at a per-timepoint cold value, so ``lo`` -- and hence the
    float64 deviation -- differs per timepoint while barely moving the other
    features, which vary per cell. The unrounded maxima are then 6 distinct levels
    uncorrelated with the rest, which both GT-only filters keep.
    """
    rng = np.random.default_rng(seed)
    gt = rng.uniform(200.0, 800.0, size=(T, D, H, W))
    seg = _segmentation()
    for t in range(T):
        for label in range(1, 17):
            inside = seg[t] == label
            gt[t][inside] = rng.uniform(200.0, 800.0) + 50.0 * rng.standard_normal(inside.sum())
        gt[t, 0] = rng.uniform(0.0, 50.0)
    for row in range(4):
        for col in range(4):
            y, x = 2 + 8 * row, 2 + 8 * col
            gt[:, 3:5, y : y + 2, x : x + 2] = 1000.0
    return gt.astype(np.float32)


def _write_plate(path: Path, channel: str, volumes: list[np.ndarray], dtype) -> None:
    with open_ome_zarr(path, mode="w", layout="hcs", channel_names=[channel], version="0.5") as plate:
        for i, vol in enumerate(volumes):
            plate.create_position("A", "1", str(i)).create_image("0", vol[:, None].astype(dtype))


def _no_rounding(features, feature_names):
    return np.asarray(features)


class Caches:
    """One GT store with two GT CP caches: CPU-built (unrounded) and GPU-equivalent (rounded)."""

    def __init__(self, root: Path, monkeypatch):
        self.root = root
        self.gt = [_saturated_gt(seed=i) for i in range(N_POSITIONS)]
        self.seg = _segmentation()
        _write_plate(root / "gt.zarr", "target", self.gt, np.float32)
        _write_plate(root / "seg.zarr", "cell_segmentation", [self.seg] * N_POSITIONS, np.int32)
        # The CPU extractor before the fix: no min/max rounding on compute or on load.
        with monkeypatch.context() as patch:
            patch.setattr(metrics, "round_device_dependent_cp_columns", _no_rounding)
            patch.setattr(pipeline_cache, "round_device_dependent_cp_columns", _no_rounding)
            self.cpu_ctx = self._build_cache("cpu_cache")
        self.gpu_ctx = self._build_cache("gpu_cache")
        _, self.names = cp_reference.cp_space(self.config("cpu_cache"))

    def config(self, cache_name: str):
        config = build_eval_config(
            self.root / "unused_pred.zarr",
            self.root / "gt.zarr",
            self.root / cache_name,
            self.root / "unused_pred_cache",
            self.root / "unused_out",
            executor="serial",
            fov_workers=1,
        )
        config.io.require_complete_cache = False
        config.io.cell_segmentation_path = str(self.root / "seg.zarr")
        config.feature_metrics.cp = {"norm": {"p_lo": 1.0, "p_hi": 99.0}, "glcm": {"enabled": False}}
        return config

    def _build_cache(self, cache_name: str):
        make_mask_cache(self.root / cache_name, self.root / "gt.zarr", "target", side="gt")
        ctx = pipeline_cache.init_cache_context(self.config(cache_name), side="gt")
        for name, vol in zip(_POSITIONS, self.gt, strict=True):
            pipeline_cache.fov_cp_features(ctx, name, vol, self.seg)
        pipeline_cache.flush_manifest(ctx)
        return ctx

    def raw(self, cache_name: str) -> np.ndarray:
        """Every cached row of one cache, read straight from the zarr (no migration)."""
        group = zarr.open_group(str(cache_paths(self.root / cache_name).cp_features()), mode="r")
        return np.concatenate([np.asarray(group[f"{pos}/t{t}"]) for pos in _POSITIONS for t in range(T)])


def _assert_minmax_float32_exact(cells: np.ndarray, names: tuple[str, ...]) -> None:
    for col in ("intensity_min", "intensity_max"):
        values = cells[:, names.index(col)]
        np.testing.assert_array_equal(values, values.astype(np.float32).astype(np.float64), err_msg=col)


@pytest.fixture
def caches(tmp_path: Path, monkeypatch) -> Caches:
    caches = Caches(tmp_path, monkeypatch)
    # The fixture discriminates: the CPU-built cache really holds non-float32 maxima,
    # while the GPU-equivalent one holds exactly 1.0.
    col = caches.names.index("intensity_max")
    cpu_max, gpu_max = caches.raw("cpu_cache")[:, col], caches.raw("gpu_cache")[:, col]
    assert not np.array_equal(cpu_max, cpu_max.astype(np.float32).astype(np.float64))
    np.testing.assert_array_equal(gpu_max, 1.0)
    return caches


def test_fov_cp_features_rounds_cpu_built_cache_hits(caches: Caches) -> None:
    """Cache hits from a CPU-built cache come back bit-identical to the GPU-equivalent cache."""
    for name, vol in zip(_POSITIONS, caches.gt, strict=True):
        cpu = pipeline_cache.fov_cp_features(caches.cpu_ctx, name, vol, caches.seg)
        gpu = pipeline_cache.fov_cp_features(caches.gpu_ctx, name, vol, caches.seg)
        for t in range(T):
            _assert_minmax_float32_exact(cpu[t], caches.names)
            np.testing.assert_array_equal(cpu[t], gpu[t])


def test_read_gt_cp_cells_rounds_a_cpu_built_cache(caches: Caches) -> None:
    """The reference builder's reader returns the GPU-equivalent matrix from a CPU-built cache."""
    cpu, _ = cp_reference.read_gt_cp_cells(caches.cpu_ctx, caches.root / "gt.zarr", len(caches.names))
    gpu, _ = cp_reference.read_gt_cp_cells(caches.gpu_ctx, caches.root / "gt.zarr", len(caches.names))
    _assert_minmax_float32_exact(cpu, caches.names)
    np.testing.assert_array_equal(cpu, gpu)


def test_complete_cached_gt_cp_blocks_rounds_a_cpu_built_cache(caches: Caches) -> None:
    """The eval's early-gate reader returns GPU-equivalent blocks from a CPU-built cache."""
    cpu = cp_reference.complete_cached_gt_cp_blocks(caches.cpu_ctx, caches.root / "gt.zarr", len(caches.names))
    gpu = cp_reference.complete_cached_gt_cp_blocks(caches.gpu_ctx, caches.root / "gt.zarr", len(caches.names))
    assert cpu is not None and gpu is not None
    assert sorted(cpu) == sorted(gpu)
    for key in cpu:
        _assert_minmax_float32_exact(cpu[key], caches.names)
        np.testing.assert_array_equal(cpu[key], gpu[key], err_msg=str(key))


def test_reference_mask_from_cpu_built_cache_matches_gpu_equivalent(caches: Caches) -> None:
    """A CP reference fit on a CPU-built cache keeps the same features as one fit on the GPU-equivalent cache.

    intensity_max is 1.0 in every GPU-equivalent cell, so the variance filter drops
    it; the unrounded CPU values (6 distinct levels over 96 cells) would keep it.
    """
    identity, names = cp_reference.cp_space(caches.config("cpu_cache"))
    masks = {}
    for side, ctx in (("cpu", caches.cpu_ctx), ("gpu", caches.gpu_ctx)):
        cells, record = cp_reference.read_gt_cp_cells(ctx, caches.root / "gt.zarr", len(names))
        fit = cp_reference.DatasetFit(dataset="a549-mantis-sec61b-mock", cells=cells, record=record, in_mask_fit=True)
        payload = cp_reference.fit_cp_reference([fit], target_name="er", feature_names=names, cp_identity=identity)
        masks[side] = np.asarray(payload["keep_mask"], dtype=bool)
    np.testing.assert_array_equal(masks["cpu"], masks["gpu"])
    assert not masks["gpu"][names.index("intensity_max")]
