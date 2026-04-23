"""Tests for pipeline_cache: per-FOV load-or-compute helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("zarr")
pytest.importorskip("iohub")

from dynacell.evaluation.cache import (  # noqa: E402
    StaleCacheError,
    cache_paths,
    load_manifest,
    read_features,
    read_mask,
    write_features,
    write_mask,
)
from dynacell.evaluation.pipeline_cache import (  # noqa: E402
    _resolve_force,
    flush_manifest,
    fov_gt_cp_features,
    fov_gt_deep_features,
    fov_gt_masks,
    init_cache_context,
)


def _make_config(**overrides: Any):
    """Produce a minimal DictConfig covering the fields init_cache_context reads."""
    base = {
        "target_name": "er",
        "io": {
            "pred_path": "/tmp/pred.zarr",
            "gt_path": "/tmp/gt.zarr",
            "cell_segmentation_path": "/tmp/seg.zarr",
            "gt_cache_dir": None,
            "pred_channel_name": "prediction",
            "gt_channel_name": "target",
            "require_complete_cache": False,
        },
        "pixel_metrics": {"spacing": [0.29, 0.108, 0.108]},
        "feature_metrics": {"patch_size": 4},
        "force_recompute": {
            "all": False,
            "gt_masks": False,
            "gt_cp": False,
            "gt_dinov3": False,
            "gt_dynaclr": False,
            "final_metrics": False,
        },
    }
    cfg = OmegaConf.create(base)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=True)
    return cfg


class _FakeSegModel:
    pass


def _seg_fn_factory(value: int):
    """Return a stand-in for ``dynacell.evaluation.segmentation.segment`` returning a constant mask."""

    def _segment(img, target_name, seg_model=None):
        del target_name, seg_model
        return np.full(img.shape, value, dtype=np.uint8)

    return _segment


class _ConstantExtractor:
    def __init__(self, dim: int, value: float):
        self.dim = dim
        self.value = value

    def extract_features(self, img):
        import torch

        return torch.full((self.dim,), self.value, dtype=torch.float32)


def test_resolve_force_all_propagates() -> None:
    """force_recompute.all=true implies every per-artifact flag is true."""
    force = OmegaConf.create(
        {
            "all": True,
            "gt_masks": False,
            "gt_cp": False,
            "gt_dinov3": False,
            "gt_dynaclr": False,
            "final_metrics": False,
        }
    )
    resolved = _resolve_force(force)
    assert all(resolved.values())


def test_resolve_force_individual() -> None:
    """Individual flags propagate without affecting their siblings."""
    force = OmegaConf.create(
        {
            "all": False,
            "gt_masks": False,
            "gt_cp": True,
            "gt_dinov3": False,
            "gt_dynaclr": False,
            "final_metrics": False,
        }
    )
    resolved = _resolve_force(force)
    assert resolved["gt_cp"] is True
    assert resolved["gt_masks"] is False


def test_init_cache_disabled_when_no_cache_dir() -> None:
    """Null gt_cache_dir produces a disabled context (enabled=False)."""
    ctx = init_cache_context(_make_config())
    assert ctx.enabled is False


def test_init_require_complete_without_cache_raises() -> None:
    """require_complete_cache=true without a cache dir raises ValueError."""
    with pytest.raises(ValueError, match="require_complete_cache"):
        init_cache_context(_make_config(**{"io.require_complete_cache": True}))


def test_init_cache_seeds_identity_on_fresh_dir(tmp_path: Path) -> None:
    """Fresh cache dir gets gt/cell_segmentation identity fields seeded."""
    ctx = init_cache_context(_make_config(**{"io.gt_cache_dir": str(tmp_path)}))
    assert ctx.enabled
    assert ctx.manifest["gt"] == {"plate_path": "/tmp/gt.zarr", "channel_name": "target"}
    assert ctx.manifest["cell_segmentation"] == {"plate_path": "/tmp/seg.zarr"}


def test_init_cache_channel_name_mismatch_raises(tmp_path: Path) -> None:
    """Cache seeded with one channel name rejects a later run with a different name."""
    init_cache_context(_make_config(**{"io.gt_cache_dir": str(tmp_path)}))
    # Simulate a prior run by flushing the manifest:
    paths = cache_paths(tmp_path)
    from dynacell.evaluation.cache import save_manifest

    save_manifest(
        paths,
        {
            "cache_schema_version": 1,
            "gt": {"plate_path": "/tmp/gt.zarr", "channel_name": "target"},
            "cell_segmentation": {"plate_path": "/tmp/seg.zarr"},
            "artifacts": {},
        },
    )
    with pytest.raises(StaleCacheError, match="gt.channel_name mismatch"):
        init_cache_context(
            _make_config(
                **{
                    "io.gt_cache_dir": str(tmp_path),
                    "io.gt_channel_name": "fluorescence",
                }
            )
        )


def test_init_cache_spacing_mismatch_raises(tmp_path: Path) -> None:
    """An existing cp_features entry with a different spacing value raises."""
    paths = cache_paths(tmp_path)
    from dynacell.evaluation.cache import save_manifest

    save_manifest(
        paths,
        {
            "cache_schema_version": 1,
            "gt": {"plate_path": "/tmp/gt.zarr", "channel_name": "target"},
            "cell_segmentation": {"plate_path": "/tmp/seg.zarr"},
            "artifacts": {"cp_features": {"spacing": [0.3, 0.108, 0.108]}},
        },
    )
    with pytest.raises(StaleCacheError, match="spacing mismatch"):
        init_cache_context(_make_config(**{"io.gt_cache_dir": str(tmp_path)}))


def test_fov_gt_masks_cache_miss_computes_and_writes(tmp_path: Path, monkeypatch) -> None:
    """First call computes masks via segment() and writes them to cache."""
    import dynacell.evaluation.segmentation as segmentation

    monkeypatch.setattr(segmentation, "segment", _seg_fn_factory(1))

    cfg = _make_config(**{"io.gt_cache_dir": str(tmp_path)})
    ctx = init_cache_context(cfg)
    target = np.zeros((2, 3, 4, 4), dtype=np.float32)

    masks = fov_gt_masks(ctx, "A/1/0", target, seg_model=_FakeSegModel())
    assert masks.shape == target.shape
    assert masks.dtype == bool
    assert masks.all()

    flush_manifest(ctx)
    cached = read_mask(cache_paths(tmp_path), "er", "A/1/0")
    assert cached is not None
    np.testing.assert_array_equal(cached, masks)


def test_fov_gt_masks_cache_hit_skips_segment(tmp_path: Path, monkeypatch) -> None:
    """Cached masks short-circuit segmentation entirely."""
    import dynacell.evaluation.segmentation as segmentation

    # Pre-populate the cache with an all-True mask:
    paths = cache_paths(tmp_path)
    masks = np.ones((2, 3, 4, 4), dtype=bool)
    write_mask(paths, "er", "A/1/0", masks)

    call_count = {"n": 0}

    def fail_segment(*args, **kwargs):
        call_count["n"] += 1
        raise AssertionError("segment() should not be called on a cache hit")

    monkeypatch.setattr(segmentation, "segment", fail_segment)

    cfg = _make_config(**{"io.gt_cache_dir": str(tmp_path)})
    ctx = init_cache_context(cfg)
    target = np.zeros((2, 3, 4, 4), dtype=np.float32)
    result = fov_gt_masks(ctx, "A/1/0", target, seg_model=_FakeSegModel())

    np.testing.assert_array_equal(result, masks)
    assert call_count["n"] == 0


def test_fov_gt_masks_force_recompute_overrides_cache(tmp_path: Path, monkeypatch) -> None:
    """force_recompute.gt_masks=true bypasses cache and calls segment() again."""
    import dynacell.evaluation.segmentation as segmentation

    paths = cache_paths(tmp_path)
    write_mask(paths, "er", "A/1/0", np.ones((1, 2, 3, 3), dtype=bool))  # stale cached value

    monkeypatch.setattr(segmentation, "segment", _seg_fn_factory(0))  # returns all zeros
    cfg = _make_config(
        **{
            "io.gt_cache_dir": str(tmp_path),
            "force_recompute.gt_masks": True,
        }
    )
    ctx = init_cache_context(cfg)
    target = np.zeros((1, 2, 3, 3), dtype=np.float32)
    result = fov_gt_masks(ctx, "A/1/0", target, seg_model=_FakeSegModel())

    # Recomputed value is all-False (segment returned zeros), overwriting the cached all-True.
    assert result.shape == target.shape
    assert not result.any()
    # Cache now holds the recomputed value.
    flush_manifest(ctx)
    np.testing.assert_array_equal(read_mask(paths, "er", "A/1/0"), result)


def test_fov_gt_masks_require_complete_raises_on_miss(tmp_path: Path, monkeypatch) -> None:
    """require_complete_cache=true turns a cache miss into StaleCacheError."""
    import dynacell.evaluation.segmentation as segmentation

    monkeypatch.setattr(segmentation, "segment", _seg_fn_factory(1))
    cfg = _make_config(
        **{
            "io.gt_cache_dir": str(tmp_path),
            "io.require_complete_cache": True,
        }
    )
    ctx = init_cache_context(cfg)
    target = np.zeros((1, 2, 3, 3), dtype=np.float32)
    with pytest.raises(StaleCacheError, match="organelle_masks"):
        fov_gt_masks(ctx, "A/1/0", target, seg_model=_FakeSegModel())


def test_fov_gt_masks_no_cache_always_computes(tmp_path: Path, monkeypatch) -> None:
    """With caching disabled (gt_cache_dir=null), masks are always recomputed."""
    import dynacell.evaluation.segmentation as segmentation

    monkeypatch.setattr(segmentation, "segment", _seg_fn_factory(1))
    ctx = init_cache_context(_make_config())
    target = np.zeros((1, 2, 3, 3), dtype=np.float32)
    masks = fov_gt_masks(ctx, "A/1/0", target, seg_model=_FakeSegModel())
    assert masks.all()


def test_flush_manifest_persists_entries(tmp_path: Path, monkeypatch) -> None:
    """flush_manifest writes accumulated artifact entries to manifest.yaml."""
    import dynacell.evaluation.segmentation as segmentation

    monkeypatch.setattr(segmentation, "segment", _seg_fn_factory(1))
    cfg = _make_config(**{"io.gt_cache_dir": str(tmp_path)})
    ctx = init_cache_context(cfg)
    fov_gt_masks(ctx, "A/1/0", np.zeros((1, 2, 3, 3), dtype=np.float32), seg_model=_FakeSegModel())
    flush_manifest(ctx)

    reloaded = load_manifest(cache_paths(tmp_path))
    er_entry = reloaded["artifacts"]["organelle_masks"]["er"]
    assert er_entry["target_name"] == "er"
    assert "A/1/0" in er_entry["positions"]
    assert "built_at" in er_entry


def test_fov_gt_deep_features_dinov3_cache_hit(tmp_path: Path) -> None:
    """Pre-populated DINOv3 cache is returned without calling the extractor."""
    cfg = _make_config(
        **{
            "io.gt_cache_dir": str(tmp_path),
            "compute_feature_metrics": True,
            "feature_extractor": {"dinov3": {"pretrained_model_name": "facebook/test-dinov3"}},
        }
    )
    # init with dinov3 model name so the ctx has it set
    ctx = init_cache_context(cfg, dinov3_model_name="facebook/test-dinov3")

    # Prime the cache:
    pos_name = "A/1/0"
    paths = cache_paths(tmp_path)
    precomputed = np.arange(6, dtype=np.float32).reshape(3, 2)
    for t in (0, 1):
        write_features(paths, "dinov3", pos_name, t, precomputed + t, model_name="facebook/test-dinov3")

    class ExplodingExtractor:
        def extract_features(self, img):
            raise AssertionError("extractor should not be called on cache hit")

    target = np.zeros((2, 1, 4, 4), dtype=np.float32)
    cell_seg = np.zeros((2, 1, 4, 4), dtype=np.int32)

    results = fov_gt_deep_features(ctx, pos_name, target, cell_seg, ExplodingExtractor(), "dinov3")
    assert len(results) == 2
    np.testing.assert_array_equal(results[0], precomputed)
    np.testing.assert_array_equal(results[1], precomputed + 1)


def test_fov_gt_cp_features_writes_on_miss(tmp_path: Path, monkeypatch) -> None:
    """CP feature miss computes via cp_target_regionprops and writes per timepoint."""

    def fake_cp(target, cell_seg, spacing):
        del cell_seg, spacing
        return np.full((2, 3), float(target.sum()), dtype=np.float32)

    # Patch the globals of fov_gt_cp_features itself — robust against sys.modules
    # churn from other tests (e.g. test_lazy_init.py) that pop dynacell modules.
    monkeypatch.setitem(fov_gt_cp_features.__globals__, "cp_target_regionprops", fake_cp)

    cfg = _make_config(**{"io.gt_cache_dir": str(tmp_path)})
    ctx = init_cache_context(cfg)
    target = np.stack([np.full((1, 2, 2), 1.0), np.full((1, 2, 2), 2.0)])
    cell_seg = np.ones_like(target, dtype=np.int32)

    results = fov_gt_cp_features(ctx, "A/1/0", target, cell_seg)
    assert len(results) == 2
    flush_manifest(ctx)
    paths = cache_paths(tmp_path)
    for t in (0, 1):
        np.testing.assert_array_equal(
            read_features(paths, "cp", "A/1/0", t),
            results[t],
        )
