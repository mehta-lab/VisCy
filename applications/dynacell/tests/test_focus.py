"""Integration tests for focus-plane resolution (compute-at-eval-time + cache)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from iohub.ngff import open_ome_zarr

from dynacell.evaluation.focus import (
    FOCUS_FIELD,
    FocusComputeConfig,
    build_focus_slabs,
    focus_slab_from_plane,
    resolve_focus_planes,
    write_focus_slice_metadata,
)

NA_DET = 1.35
LAMBDA_ILL = 0.450
PIXEL_SIZE = 0.1494


def _compute(channel_name="Phase3D", **overrides):
    params = dict(channel_name=channel_name, na_det=NA_DET, lambda_ill=LAMBDA_ILL, pixel_size=PIXEL_SIZE, device="cpu")
    params.update(overrides)
    return FocusComputeConfig(**params)


def _make_store(path: Path, *, z: int = 16, focus_z=(5, 9), t: int = 2) -> None:
    """Two positions, T=t, one Phase3D channel, sharp texture band at a per-pos plane."""
    rng = np.random.default_rng(0)
    with open_ome_zarr(str(path), layout="hcs", mode="w", channel_names=["Phase3D"]) as plate:
        for col, fz in enumerate(focus_z):
            pos = plate.create_position("0", str(col), "0")
            data = rng.normal(0, 0.05, size=(t, 1, z, 96, 96)).astype(np.float32)
            for ti in range(t):
                for zi in range(z):
                    amp = float(np.exp(-((zi - fz) ** 2) / 2.0))
                    data[ti, 0, zi] += amp * rng.normal(0, 1.0, size=(96, 96)).astype(np.float32)
            pos.create_image("0", data)


def _resolve(pos, name, t, cache_dir, **overrides):
    return resolve_focus_planes(pos, t_count=t, compute=_compute(**overrides), cache_dir=cache_dir, pos_name=name)


def test_compute_then_cache_then_hit(tmp_path):
    """No zattrs -> compute from Phase3D + persist; second call hits the cache file."""
    store = tmp_path / "s.zarr"
    cache = tmp_path / "gt_cache"
    _make_store(store, z=16, focus_z=(5, 9), t=2)
    with open_ome_zarr(str(store), mode="r") as plate:
        name, pos = next(iter(plate.positions()))
        planes = _resolve(pos, name, 2, cache)
        assert len(planes) == 2
        assert all(0 <= p < 16 for p in planes)
        # the sharp band sits near z=5 for this first position
        assert abs(planes[0] - 5) <= 2
        cache_file = cache / "focus_planes" / "Phase3D" / f"{name.replace('/', '__')}.json"
        assert cache_file.is_file()
        # second call returns identical planes (from cache; band content unchanged)
        assert _resolve(pos, name, 2, cache) == planes


def test_param_mismatch_recomputes(tmp_path):
    """A different pixel_size must not reuse a cache entry written for another param set."""
    store = tmp_path / "s.zarr"
    cache = tmp_path / "gt_cache"
    _make_store(store, z=16, focus_z=(8, 8), t=1)
    with open_ome_zarr(str(store), mode="r") as plate:
        name, pos = next(iter(plate.positions()))
        _resolve(pos, name, 1, cache)
        cache_file = cache / "focus_planes" / "Phase3D" / f"{name.replace('/', '__')}.json"
        before = cache_file.read_text()
        _resolve(pos, name, 1, cache, pixel_size=0.250)  # different param -> overwrite
        assert cache_file.read_text() != before


def test_zattrs_take_precedence_over_cache(tmp_path):
    """Precomputed focus_slice zattrs win over both the cache and a fresh compute."""
    store = tmp_path / "s.zarr"
    cache = tmp_path / "gt_cache"
    _make_store(store, z=16, focus_z=(5, 9), t=2)
    write_focus_slice_metadata(
        str(store), channel_name="Phase3D", na_det=NA_DET, lambda_ill=LAMBDA_ILL, pixel_size=PIXEL_SIZE
    )
    with open_ome_zarr(str(store), mode="r") as plate:
        name, pos = next(iter(plate.positions()))
        assert FOCUS_FIELD in pos.zattrs
        planes = _resolve(pos, name, 2, cache)
        # came from zattrs, so no cache file is written
        assert not (cache / "focus_planes" / "Phase3D" / f"{name.replace('/', '__')}.json").is_file()
        assert len(planes) == 2


def test_build_focus_slabs_centers_and_clips(tmp_path):
    """Slabs are 2*halfwidth+1 planes centered on the focus plane, clipped to [0, Z)."""
    store = tmp_path / "s.zarr"
    cache = tmp_path / "gt_cache"
    _make_store(store, z=16, focus_z=(1, 1), t=1)  # focus near the edge -> clipping
    with open_ome_zarr(str(store), mode="r") as plate:
        name, pos = next(iter(plate.positions()))
        slabs = build_focus_slabs(pos, halfwidth=2, t_count=1, compute=_compute(), cache_dir=cache, pos_name=name)
        assert len(slabs) == 1
        sl = slabs[0]
        assert sl.start == 0  # clipped at the low edge
        assert sl.stop <= 16


def test_focus_slab_from_plane_pure():
    assert focus_slab_from_plane(10, 48, 2) == slice(8, 13)
    assert focus_slab_from_plane(0, 48, 2) == slice(0, 3)
    assert focus_slab_from_plane(47, 48, 2) == slice(45, 48)
    assert focus_slab_from_plane(10, 48, 0) == slice(10, 11)


def test_read_focus_slab_config_rejects_negative_halfwidth():
    """A negative halfwidth must fail early (it yields an empty slab → max-Z crash)."""
    from omegaconf import OmegaConf

    from dynacell.evaluation.focus import read_focus_slab_config

    bad = OmegaConf.create({"feature_metrics": {"focus_slab": {"enabled": True, "halfwidth": -1}}})
    with pytest.raises(ValueError, match="halfwidth must be >= 0"):
        read_focus_slab_config(bad)
    # halfwidth=0 is valid (single in-focus plane)
    ok = OmegaConf.create({"feature_metrics": {"focus_slab": {"enabled": True, "halfwidth": 0}}})
    assert read_focus_slab_config(ok).halfwidth == 0


def test_per_cell_similarity_z_slab_restricts_and_changes_outcome():
    """z_slab slices predict/target/seg consistently, raising per-cell PCC toward the in-focus band."""
    from dynacell.evaluation.metrics import per_cell_similarity

    rng = np.random.default_rng(0)
    z, y, x = 12, 48, 48
    target = rng.normal(size=(z, y, x)).astype(np.float32)
    predict = rng.normal(size=(z, y, x)).astype(np.float32)  # uncorrelated everywhere ...
    predict[4:8] = target[4:8]  # ... except the in-focus band, where pred == target (PCC ≈ 1)
    seg = np.zeros((z, y, x), dtype=np.int32)
    seg[:, 10:38, 10:38] = 1  # one cell spanning all z
    kw = dict(metrics=("pcc",), reduce=("mean",), use_gpu=False)
    full = per_cell_similarity(predict, target, seg, **kw)
    slab = per_cell_similarity(predict, target, seg, z_slab=slice(4, 8), **kw)
    assert slab["PerCell_PCC_mean"] > full["PerCell_PCC_mean"]
    assert slab["PerCell_PCC_mean"] > 0.9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
