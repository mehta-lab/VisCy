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
    fov_cp_features,
    fov_deep_features,
    fov_masks,
    init_cache_context,
    precompute_deep_features,
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
            "pred_cache_dir": None,
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
            "gt_celldino": False,
            "pred_masks": False,
            "pred_cp": False,
            "pred_dinov3": False,
            "pred_dynaclr": False,
            "pred_celldino": False,
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
            "gt_celldino": False,
            "pred_masks": False,
            "pred_cp": False,
            "pred_dinov3": False,
            "pred_dynaclr": False,
            "pred_celldino": False,
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
            "gt_celldino": False,
            "pred_masks": False,
            "pred_cp": False,
            "pred_dinov3": False,
            "pred_dynaclr": False,
            "pred_celldino": False,
            "final_metrics": False,
        }
    )
    resolved = _resolve_force(force)
    assert resolved["gt_cp"] is True
    assert resolved["gt_masks"] is False


def test_init_cache_disabled_when_no_cache_dir() -> None:
    """Null gt_cache_dir produces a disabled context (enabled=False)."""
    ctx = init_cache_context(_make_config(), side="gt")
    assert ctx.enabled is False


def test_init_require_complete_without_cache_raises() -> None:
    """require_complete_cache=true without a cache dir raises ValueError."""
    with pytest.raises(ValueError, match="require_complete_cache"):
        init_cache_context(_make_config(**{"io.require_complete_cache": True}), side="gt")


def test_init_cache_seeds_identity_on_fresh_dir(tmp_path: Path) -> None:
    """Fresh cache dir gets gt/cell_segmentation identity fields seeded."""
    ctx = init_cache_context(_make_config(**{"io.gt_cache_dir": str(tmp_path)}), side="gt")
    assert ctx.enabled
    assert ctx.manifest["gt"] == {"plate_path": "/tmp/gt.zarr", "channel_name": "target"}
    assert ctx.manifest["cell_segmentation"] == {"plate_path": "/tmp/seg.zarr"}


def test_init_pred_cache_disabled_when_no_cache_dir() -> None:
    """Null pred_cache_dir leaves prediction caching disabled, even in strict GT-cache runs."""
    ctx = init_cache_context(_make_config(**{"io.require_complete_cache": True}), side="pred")
    assert ctx.enabled is False
    assert ctx.require_complete is False


def test_init_pred_cache_seeds_identity_on_fresh_dir(tmp_path: Path) -> None:
    """Fresh pred cache dir gets pred/cell_segmentation identity fields seeded."""
    ctx = init_cache_context(_make_config(**{"io.pred_cache_dir": str(tmp_path)}), side="pred")
    assert ctx.enabled
    assert ctx.manifest["gt"] is None
    assert ctx.manifest["pred"] == {"plate_path": "/tmp/pred.zarr", "channel_name": "prediction"}
    assert ctx.manifest["cell_segmentation"] == {"plate_path": "/tmp/seg.zarr"}


def test_init_pred_cache_rejects_gt_cache_dir_reuse(tmp_path: Path) -> None:
    """GT and prediction caches must not share one artifact root."""
    with pytest.raises(ValueError, match="pred_cache_dir"):
        init_cache_context(
            _make_config(
                **{
                    "io.gt_cache_dir": str(tmp_path),
                    "io.pred_cache_dir": str(tmp_path),
                }
            ),
            side="pred",
        )


def test_init_cache_channel_name_mismatch_raises(tmp_path: Path) -> None:
    """Cache seeded with one channel name rejects a later run with a different name."""
    init_cache_context(_make_config(**{"io.gt_cache_dir": str(tmp_path)}), side="gt")
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
            ),
            side="gt",
        )


def test_init_pred_cache_channel_name_mismatch_raises(tmp_path: Path) -> None:
    """Prediction cache seeded with one channel name rejects a different prediction channel."""
    paths = cache_paths(tmp_path)
    from dynacell.evaluation.cache import save_manifest

    save_manifest(
        paths,
        {
            "cache_schema_version": 1,
            "gt": None,
            "pred": {"plate_path": "/tmp/pred.zarr", "channel_name": "prediction"},
            "cell_segmentation": {"plate_path": "/tmp/seg.zarr"},
            "artifacts": {},
        },
    )
    with pytest.raises(StaleCacheError, match="pred.channel_name mismatch"):
        init_cache_context(
            _make_config(
                **{
                    "io.pred_cache_dir": str(tmp_path),
                    "io.pred_channel_name": "other_prediction",
                }
            ),
            side="pred",
        )


def test_init_cache_spacing_mismatch_auto_invalidates(tmp_path: Path) -> None:
    """A cp_features entry with a stale spacing flips force_recompute, not raises.

    Pre-existing manifest has ``spacing=[0.3, 0.108, 0.108]`` but the current
    config advertises ``[0.29, 0.108, 0.108]``. The validator must warn and
    set ``force["gt_cp"] = True`` so the regionprops cache is recomputed
    with the current spacing, then the manifest entry is rewritten — letting
    spacing bumps self-heal across runs.
    """
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
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        ctx = init_cache_context(_make_config(**{"io.gt_cache_dir": str(tmp_path)}), side="gt")
    assert ctx.force["gt_cp"] is True
    assert any("artifact param mismatch" in str(w.message) and "spacing" in str(w.message) for w in caught), (
        f"expected spacing mismatch warning; got {[str(w.message) for w in caught]}"
    )


def test_fov_gt_masks_cache_miss_computes_and_writes(tmp_path: Path, monkeypatch) -> None:
    """First call computes masks via segment() and writes them to cache."""
    import dynacell.evaluation.segmentation as segmentation

    monkeypatch.setattr(segmentation, "segment", _seg_fn_factory(1))

    cfg = _make_config(**{"io.gt_cache_dir": str(tmp_path)})
    ctx = init_cache_context(cfg, side="gt")
    target = np.zeros((2, 3, 4, 4), dtype=np.float32)

    masks = fov_masks(ctx, "A/1/0", target, seg_model=_FakeSegModel())
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
    ctx = init_cache_context(cfg, side="gt")
    target = np.zeros((2, 3, 4, 4), dtype=np.float32)
    result = fov_masks(ctx, "A/1/0", target, seg_model=_FakeSegModel())

    np.testing.assert_array_equal(result, masks)
    assert call_count["n"] == 0


def test_fov_pred_masks_cache_hit_skips_segment(tmp_path: Path, monkeypatch) -> None:
    """Cached prediction masks short-circuit prediction-side segmentation."""
    import dynacell.evaluation.segmentation as segmentation

    paths = cache_paths(tmp_path)
    masks = np.ones((2, 3, 4, 4), dtype=bool)
    write_mask(paths, "er", "A/1/0", masks)

    call_count = {"n": 0}

    def fail_segment(*args, **kwargs):
        call_count["n"] += 1
        raise AssertionError("segment() should not be called on a pred-cache hit")

    monkeypatch.setattr(segmentation, "segment", fail_segment)

    cfg = _make_config(**{"io.pred_cache_dir": str(tmp_path)})
    ctx = init_cache_context(cfg, side="pred")
    prediction = np.zeros((2, 3, 4, 4), dtype=np.float32)
    result = fov_masks(ctx, "A/1/0", prediction, seg_model=_FakeSegModel())

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
    ctx = init_cache_context(cfg, side="gt")
    target = np.zeros((1, 2, 3, 3), dtype=np.float32)
    result = fov_masks(ctx, "A/1/0", target, seg_model=_FakeSegModel())

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
    ctx = init_cache_context(cfg, side="gt")
    target = np.zeros((1, 2, 3, 3), dtype=np.float32)
    with pytest.raises(StaleCacheError, match="organelle_masks"):
        fov_masks(ctx, "A/1/0", target, seg_model=_FakeSegModel())


def test_fov_pred_masks_require_complete_raises_on_miss(tmp_path: Path, monkeypatch) -> None:
    """require_complete_cache=true turns a prediction-mask miss into StaleCacheError."""
    import dynacell.evaluation.segmentation as segmentation

    monkeypatch.setattr(segmentation, "segment", _seg_fn_factory(1))
    cfg = _make_config(
        **{
            "io.pred_cache_dir": str(tmp_path),
            "io.require_complete_cache": True,
        }
    )
    ctx = init_cache_context(cfg, side="pred")
    prediction = np.zeros((1, 2, 3, 3), dtype=np.float32)
    with pytest.raises(StaleCacheError, match="pred_organelle_masks"):
        fov_masks(ctx, "A/1/0", prediction, seg_model=_FakeSegModel())


def test_fov_gt_masks_no_cache_always_computes(tmp_path: Path, monkeypatch) -> None:
    """With caching disabled (gt_cache_dir=null), masks are always recomputed."""
    import dynacell.evaluation.segmentation as segmentation

    monkeypatch.setattr(segmentation, "segment", _seg_fn_factory(1))
    ctx = init_cache_context(_make_config(), side="gt")
    target = np.zeros((1, 2, 3, 3), dtype=np.float32)
    masks = fov_masks(ctx, "A/1/0", target, seg_model=_FakeSegModel())
    assert masks.all()


def test_flush_manifest_persists_entries(tmp_path: Path, monkeypatch) -> None:
    """flush_manifest writes accumulated artifact entries to manifest.yaml."""
    import dynacell.evaluation.segmentation as segmentation

    monkeypatch.setattr(segmentation, "segment", _seg_fn_factory(1))
    cfg = _make_config(**{"io.gt_cache_dir": str(tmp_path)})
    ctx = init_cache_context(cfg, side="gt")
    fov_masks(ctx, "A/1/0", np.zeros((1, 2, 3, 3), dtype=np.float32), seg_model=_FakeSegModel())
    flush_manifest(ctx)

    reloaded = load_manifest(cache_paths(tmp_path))
    er_entry = reloaded["artifacts"]["organelle_masks"]["er"]
    assert er_entry["target_name"] == "er"
    assert "A/1/0" in er_entry["positions"]
    assert "built_at" in er_entry


def test_fov_pred_masks_writes_manifest_source(tmp_path: Path, monkeypatch) -> None:
    """Prediction mask writes are marked as prediction-side artifacts in the manifest."""
    import dynacell.evaluation.segmentation as segmentation

    monkeypatch.setattr(segmentation, "segment", _seg_fn_factory(1))
    cfg = _make_config(**{"io.pred_cache_dir": str(tmp_path)})
    ctx = init_cache_context(cfg, side="pred")
    fov_masks(ctx, "A/1/0", np.zeros((1, 2, 3, 3), dtype=np.float32), seg_model=_FakeSegModel())
    flush_manifest(ctx)

    reloaded = load_manifest(cache_paths(tmp_path))
    er_entry = reloaded["artifacts"]["organelle_masks"]["er"]
    assert er_entry["source"] == "prediction"
    assert "A/1/0" in er_entry["positions"]


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
    ctx = init_cache_context(cfg, side="gt", dinov3_model_name="facebook/test-dinov3")

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

    results = fov_deep_features(ctx, pos_name, target, cell_seg, ExplodingExtractor(), "dinov3")
    assert len(results) == 2
    np.testing.assert_array_equal(results[0], precomputed)
    np.testing.assert_array_equal(results[1], precomputed + 1)


def test_fov_pred_deep_features_dinov3_cache_hit(tmp_path: Path) -> None:
    """Pre-populated prediction DINOv3 cache is returned without calling the extractor."""
    cfg = _make_config(
        **{
            "io.pred_cache_dir": str(tmp_path),
            "compute_feature_metrics": True,
            "feature_extractor": {"dinov3": {"pretrained_model_name": "facebook/test-dinov3"}},
        }
    )
    ctx = init_cache_context(cfg, side="pred", dinov3_model_name="facebook/test-dinov3")

    pos_name = "A/1/0"
    paths = cache_paths(tmp_path)
    precomputed = np.arange(6, dtype=np.float32).reshape(3, 2)
    for t in (0, 1):
        write_features(paths, "dinov3", pos_name, t, precomputed + t, model_name="facebook/test-dinov3")

    class ExplodingExtractor:
        def extract_features(self, img):
            raise AssertionError("extractor should not be called on pred-cache hit")

    prediction = np.zeros((2, 1, 4, 4), dtype=np.float32)
    cell_seg = np.zeros((2, 1, 4, 4), dtype=np.int32)

    results = fov_deep_features(ctx, pos_name, prediction, cell_seg, ExplodingExtractor(), "dinov3")
    assert len(results) == 2
    np.testing.assert_array_equal(results[0], precomputed)
    np.testing.assert_array_equal(results[1], precomputed + 1)


def test_fov_gt_cp_features_writes_on_miss(tmp_path: Path, monkeypatch) -> None:
    """CP feature miss computes via cp_regionprops and writes per timepoint."""

    def fake_cp(target, cell_seg, spacing, use_gpu=True):
        del cell_seg, spacing, use_gpu
        return np.full((2, 3), float(target.sum()), dtype=np.float32)

    # Patch the globals of fov_cp_features itself — robust against sys.modules
    # churn from other tests (e.g. test_lazy_init.py) that pop dynacell modules.
    monkeypatch.setitem(fov_cp_features.__globals__, "cp_regionprops", fake_cp)

    cfg = _make_config(**{"io.gt_cache_dir": str(tmp_path)})
    ctx = init_cache_context(cfg, side="gt")
    target = np.stack([np.full((1, 2, 2), 1.0), np.full((1, 2, 2), 2.0)])
    cell_seg = np.ones_like(target, dtype=np.int32)

    results = fov_cp_features(ctx, "A/1/0", target, cell_seg)
    assert len(results) == 2
    flush_manifest(ctx)
    paths = cache_paths(tmp_path)
    for t in (0, 1):
        np.testing.assert_array_equal(
            read_features(paths, "cp", "A/1/0", t),
            results[t],
        )


def test_fov_pred_cp_features_writes_on_miss(tmp_path: Path, monkeypatch) -> None:
    """Prediction CP feature miss computes via cp_regionprops (side-agnostic) and writes per timepoint."""

    def fake_cp(prediction, cell_seg, spacing, use_gpu=True):
        del cell_seg, spacing, use_gpu
        return np.full((2, 3), float(prediction.sum()), dtype=np.float32)

    monkeypatch.setitem(fov_cp_features.__globals__, "cp_regionprops", fake_cp)

    cfg = _make_config(**{"io.pred_cache_dir": str(tmp_path)})
    ctx = init_cache_context(cfg, side="pred")
    prediction = np.stack([np.full((1, 2, 2), 1.0), np.full((1, 2, 2), 2.0)])
    cell_seg = np.ones_like(prediction, dtype=np.int32)

    results = fov_cp_features(ctx, "A/1/0", prediction, cell_seg)
    assert len(results) == 2
    flush_manifest(ctx)
    paths = cache_paths(tmp_path)
    for t in (0, 1):
        np.testing.assert_array_equal(
            read_features(paths, "cp", "A/1/0", t),
            results[t],
        )
    manifest = load_manifest(paths)
    assert manifest["artifacts"]["cp_features"]["source"] == "prediction"


def _write_tiny_hcs_plate(
    path: Path,
    positions: list[tuple[str, str, str]],
    *,
    n_t: int = 3,
    channel: str = "target",
) -> None:
    """Write a tiny ``(T, 1, 2, 8, 8)`` plate for batched precompute tests.

    Image content is constant 0.5 — the deep-feature stub doesn't care
    about pixel values, only the per-(pos, t) cell counts.
    """
    from iohub.ngff import open_ome_zarr

    data = np.full((n_t, 1, 2, 8, 8), 0.5, dtype=np.float32)
    with open_ome_zarr(path, mode="w", layout="hcs", channel_names=[channel], version="0.5") as plate:
        for row, col, fov in positions:
            pos = plate.create_position(row, col, fov)
            pos.create_image("0", data)


def _write_tiny_seg_plate(
    path: Path,
    positions: list[tuple[str, str, str]],
    *,
    n_t: int = 3,
    n_cells: int = 2,
) -> None:
    """Write a tiny ``(T, 1, 2, 8, 8)`` cell-segmentation plate with N labeled cells."""
    from iohub.ngff import open_ome_zarr

    seg = np.zeros((n_t, 1, 2, 8, 8), dtype=np.int32)
    if n_cells >= 1:
        seg[:, 0, :, 0:3, 0:3] = 1
    if n_cells >= 2:
        seg[:, 0, :, 5:8, 5:8] = 2
    if n_cells >= 3:
        seg[:, 0, :, 0:3, 5:8] = 3
    if n_cells >= 4:
        seg[:, 0, :, 5:8, 0:3] = 4
    with open_ome_zarr(path, mode="w", layout="hcs", channel_names=["cell_segmentation"], version="0.5") as plate:
        for row, col, fov in positions:
            pos = plate.create_position(row, col, fov)
            pos.create_image("0", seg)


class _BatchedConstantExtractor:
    """Stub extractor that honors the ``extract_features_batch`` contract.

    Returns ones of shape ``(len(images), feature_dim)`` so equality
    checks across paths are bit-exact. Counts batch calls and records
    every batch size so tests can assert skipped slots.
    """

    def __init__(self, feature_dim: int = 16) -> None:
        self.feature_dim = feature_dim
        self.batch_call_count = 0
        self.batch_sizes: list[int] = []

    def extract_features_batch(self, images):
        import torch

        self.batch_call_count += 1
        self.batch_sizes.append(len(images))
        return torch.from_numpy(np.ones((len(images), self.feature_dim), dtype=np.float32))

    def extract_features(self, img):
        import torch

        return torch.from_numpy(np.ones((self.feature_dim,), dtype=np.float32))


def _open_precompute_inputs(tmp_path: Path, *, n_t: int = 3, n_cells: int = 2):
    """Open tiny GT + seg plates and return the test fixtures the precompute helper consumes."""
    from iohub.ngff import open_ome_zarr

    gt_path = tmp_path / "gt.zarr"
    seg_path = tmp_path / "seg.zarr"
    pos_keys = [("A", "1", "0"), ("A", "1", "1")]
    _write_tiny_hcs_plate(gt_path, pos_keys, n_t=n_t)
    _write_tiny_seg_plate(seg_path, pos_keys, n_t=n_t, n_cells=n_cells)
    gt_plate = open_ome_zarr(gt_path, mode="r")
    seg_plate = open_ome_zarr(seg_path, mode="r")
    return gt_plate, seg_plate, gt_path, seg_path


def test_precompute_deep_features_writes_all_slots(tmp_path: Path) -> None:
    """precompute_deep_features writes every (pos, t) slot to the deep-feature cache."""
    gt_plate, seg_plate, gt_path, seg_path = _open_precompute_inputs(tmp_path)
    cache_dir = tmp_path / "cache"
    try:
        cfg = _make_config(
            **{
                "io.gt_path": str(gt_path),
                "io.cell_segmentation_path": str(seg_path),
                "io.gt_cache_dir": str(cache_dir),
            }
        )
        ctx = init_cache_context(cfg, side="gt", dinov3_model_name="facebook/test-dinov3")

        extractor = _BatchedConstantExtractor(feature_dim=16)
        precompute_deep_features(
            sides={"gt": ctx},
            side_positions={"gt": list(gt_plate.positions())},
            side_channel_names={"gt": "target"},
            seg_positions=list(seg_plate.positions()),
            extractors={"dinov3": extractor},
        )

        paths = cache_paths(cache_dir)
        for pos_name in ("A/1/0", "A/1/1"):
            for t in range(3):
                feats = read_features(paths, "dinov3", pos_name, t, model_name="facebook/test-dinov3")
                assert feats is not None, f"missing slot {pos_name}/t{t}"
                assert feats.shape == (2, 16), f"unexpected shape at {pos_name}/t{t}: {feats.shape}"
        # 2 positions × 3 timepoints × 2 cells = 12 crops, plus a single
        # batched call (well below threshold=256).
        assert sum(extractor.batch_sizes) == 12
    finally:
        gt_plate.close()
        seg_plate.close()


def test_precompute_deep_features_matches_per_fov(tmp_path: Path) -> None:
    """Cache produced by precompute_deep_features matches per-FOV fov_deep_features."""
    gt_plate, seg_plate, gt_path, seg_path = _open_precompute_inputs(tmp_path)
    cache_a = tmp_path / "cache_a"
    cache_b = tmp_path / "cache_b"
    try:
        cfg_a = _make_config(
            **{
                "io.gt_path": str(gt_path),
                "io.cell_segmentation_path": str(seg_path),
                "io.gt_cache_dir": str(cache_a),
            }
        )
        cfg_b = _make_config(
            **{
                "io.gt_path": str(gt_path),
                "io.cell_segmentation_path": str(seg_path),
                "io.gt_cache_dir": str(cache_b),
            }
        )
        ctx_a = init_cache_context(cfg_a, side="gt", dinov3_model_name="facebook/test-dinov3")
        ctx_b = init_cache_context(cfg_b, side="gt", dinov3_model_name="facebook/test-dinov3")

        precompute_deep_features(
            sides={"gt": ctx_a},
            side_positions={"gt": list(gt_plate.positions())},
            side_channel_names={"gt": "target"},
            seg_positions=list(seg_plate.positions()),
            extractors={"dinov3": _BatchedConstantExtractor(feature_dim=16)},
        )

        # Per-FOV path into cache B for the same data + a fresh stub instance.
        per_fov_extractor = _BatchedConstantExtractor(feature_dim=16)
        for (pos_name_gt, pos_gt), (_, pos_seg) in zip(list(gt_plate.positions()), list(seg_plate.positions())):
            target = np.asarray(pos_gt.data[:, pos_gt.get_channel_index("target")])
            cell_seg = np.asarray(pos_seg.data[:, 0])
            fov_deep_features(ctx_b, pos_name_gt, target, cell_seg, per_fov_extractor, "dinov3")
            flush_manifest(ctx_b)

        paths_a = cache_paths(cache_a)
        paths_b = cache_paths(cache_b)
        for pos_name in ("A/1/0", "A/1/1"):
            for t in range(3):
                feats_a = read_features(paths_a, "dinov3", pos_name, t, model_name="facebook/test-dinov3")
                feats_b = read_features(paths_b, "dinov3", pos_name, t, model_name="facebook/test-dinov3")
                np.testing.assert_array_equal(feats_a, feats_b)
    finally:
        gt_plate.close()
        seg_plate.close()


def test_precompute_deep_features_skips_cached_slots(tmp_path: Path) -> None:
    """Pre-seeded (pos, t) slots are skipped — extractor only sees remaining cells."""
    gt_plate, seg_plate, gt_path, seg_path = _open_precompute_inputs(tmp_path)
    cache_dir = tmp_path / "cache"
    try:
        cfg = _make_config(
            **{
                "io.gt_path": str(gt_path),
                "io.cell_segmentation_path": str(seg_path),
                "io.gt_cache_dir": str(cache_dir),
            }
        )
        ctx = init_cache_context(cfg, side="gt", dinov3_model_name="facebook/test-dinov3")

        # Pre-seed (A/1/0, t=1) with the same shape (2 cells, 16 dims).
        paths = cache_paths(cache_dir)
        write_features(
            paths,
            "dinov3",
            "A/1/0",
            1,
            np.full((2, 16), 7.0, dtype=np.float32),
            model_name="facebook/test-dinov3",
        )

        extractor = _BatchedConstantExtractor(feature_dim=16)
        precompute_deep_features(
            sides={"gt": ctx},
            side_positions={"gt": list(gt_plate.positions())},
            side_channel_names={"gt": "target"},
            seg_positions=list(seg_plate.positions()),
            extractors={"dinov3": extractor},
        )

        # Total cells across 2 pos × 3 t × 2 cells = 12; pre-seeded slot
        # carried 2 cells, so extractor must have seen exactly 10.
        assert sum(extractor.batch_sizes) == 10
        # Pre-seeded slot stays untouched (constant 7.0).
        np.testing.assert_array_equal(
            read_features(paths, "dinov3", "A/1/0", 1, model_name="facebook/test-dinov3"),
            np.full((2, 16), 7.0, dtype=np.float32),
        )
    finally:
        gt_plate.close()
        seg_plate.close()


def test_preprocess_version_missing_in_manifest_is_lenient(tmp_path: Path) -> None:
    """A pre-version-tracking manifest entry must NOT auto-invalidate.

    The bootstrap transition is the operator's responsibility (set the
    matching ``force_recompute.<side>_<kind>`` flag explicitly). Missing
    cached version is treated as 'no constraint'.
    """
    from dynacell.evaluation.cache import save_manifest

    cache_dir = tmp_path
    paths = cache_paths(cache_dir)
    save_manifest(
        paths,
        {
            "cache_schema_version": 1,
            "gt": None,
            "pred": {"plate_path": "/tmp/pred.zarr", "channel_name": "prediction"},
            "cell_segmentation": {"plate_path": "/tmp/seg.zarr"},
            "artifacts": {
                "celldino_features": {
                    "abc123def456": {
                        "path": "features/celldino/abc123def456.zarr",
                        "weights_sha256_12": "abc123def456",
                        "patch_size": 4,
                        "source": "prediction",
                        # NOTE: no preprocess_version field — pre-tracking entry
                    }
                }
            },
        },
    )

    ctx = init_cache_context(
        _make_config(**{"io.pred_cache_dir": str(cache_dir)}),
        side="pred",
        celldino_weights_path=None,
        celldino_preprocess_version="self_normalize_v1",
    )
    # Without celldino_weights_sha12 we can't even reach the entry; provide it.
    # Re-init with the right sha so the entry lookup hits.
    ctx = init_cache_context(
        _make_config(**{"io.pred_cache_dir": str(cache_dir)}),
        side="pred",
        celldino_weights_path=None,
        celldino_preprocess_version="self_normalize_v1",
    )
    # Manually set the sha to simulate a loaded extractor with a matching sha
    # (in production this comes from ckpt_sha256_12(weights_path)):
    ctx.celldino_weights_sha12 = "abc123def456"
    # Re-run the validator manually with the populated sha:
    from dynacell.evaluation.pipeline_cache import (
        _auto_invalidate_on_preprocess_version_mismatch,
    )

    _auto_invalidate_on_preprocess_version_mismatch(ctx)
    assert ctx.force["pred_celldino"] is False, "missing preprocess_version must not invalidate"


def test_preprocess_version_mismatch_auto_invalidates(tmp_path: Path) -> None:
    """A known cached version different from current auto-sets force_recompute.

    This is the auto-invalidation case: the cache stored
    ``preprocess_version: self_normalize_v0`` (hypothetical prior), the
    current run advertises ``self_normalize_v1``. The validator must set
    ``force["pred_celldino"] = True`` and emit a warning.
    """
    from dynacell.evaluation.cache import save_manifest

    cache_dir = tmp_path
    paths = cache_paths(cache_dir)
    save_manifest(
        paths,
        {
            "cache_schema_version": 1,
            "gt": None,
            "pred": {"plate_path": "/tmp/pred.zarr", "channel_name": "prediction"},
            "cell_segmentation": {"plate_path": "/tmp/seg.zarr"},
            "artifacts": {
                "celldino_features": {
                    "abc123def456": {
                        "path": "features/celldino/abc123def456.zarr",
                        "weights_sha256_12": "abc123def456",
                        "patch_size": 4,
                        "source": "prediction",
                        "preprocess_version": "self_normalize_v0",
                    }
                }
            },
        },
    )

    ctx = init_cache_context(
        _make_config(**{"io.pred_cache_dir": str(cache_dir)}),
        side="pred",
        celldino_preprocess_version="self_normalize_v1",
    )
    ctx.celldino_weights_sha12 = "abc123def456"
    import warnings as _warnings

    from dynacell.evaluation.pipeline_cache import (
        _auto_invalidate_on_preprocess_version_mismatch,
    )

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _auto_invalidate_on_preprocess_version_mismatch(ctx)
    assert ctx.force["pred_celldino"] is True
    assert any("preprocess_version mismatch" in str(w.message) for w in caught), (
        f"expected mismatch warning; got {[str(w.message) for w in caught]}"
    )


def test_preprocess_version_match_is_noop(tmp_path: Path) -> None:
    """A cached version equal to current must NOT trigger invalidation."""
    from dynacell.evaluation.cache import save_manifest

    cache_dir = tmp_path
    paths = cache_paths(cache_dir)
    save_manifest(
        paths,
        {
            "cache_schema_version": 1,
            "gt": None,
            "pred": {"plate_path": "/tmp/pred.zarr", "channel_name": "prediction"},
            "cell_segmentation": {"plate_path": "/tmp/seg.zarr"},
            "artifacts": {
                "celldino_features": {
                    "abc123def456": {
                        "path": "features/celldino/abc123def456.zarr",
                        "weights_sha256_12": "abc123def456",
                        "patch_size": 4,
                        "source": "prediction",
                        "preprocess_version": "self_normalize_v1",
                    }
                }
            },
        },
    )

    ctx = init_cache_context(
        _make_config(**{"io.pred_cache_dir": str(cache_dir)}),
        side="pred",
        celldino_preprocess_version="self_normalize_v1",
    )
    ctx.celldino_weights_sha12 = "abc123def456"
    from dynacell.evaluation.pipeline_cache import (
        _auto_invalidate_on_preprocess_version_mismatch,
    )

    _auto_invalidate_on_preprocess_version_mismatch(ctx)
    assert ctx.force["pred_celldino"] is False
