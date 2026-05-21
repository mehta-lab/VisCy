"""Worker-side and aggregator-side smoke tests for the FOV parallelism wiring.

These tests cover the cross-process boundary in two pieces:

1. ``FovResult`` survives ``pickle`` round-trip with realistic
   shapes (the ``concurrent.futures.ProcessPoolExecutor`` future-result
   transport uses pickle).
2. ``_aggregate_fov_result`` correctly extends each backbone's six lists
   in the parent-side ``parent_lists: dict[str, _BackboneLists]`` from a
   synthetic FovResult, preserving the existing ``if size > 0: append``
   semantics that the worker internally maintains.

End-to-end serial/process parity on a real eval (with iohub fixtures and
prebuilt mask caches) is a follow-up CPU integration test — see the plan
``.claude/plans/eval-parallelism.md`` C5 section.
"""

from __future__ import annotations

import importlib
import pickle

import numpy as np


def _live_pipeline_module():
    """Return the *currently cached* dynacell.evaluation.pipeline module.

    Other tests in the suite (notably ``test_evaluation_pipeline.py``)
    use ``monkeypatch.setitem(sys.modules, ...)`` to swap the pipeline
    module with a stubbed re-import. After teardown, ``sys.modules`` is
    restored but any class references resolved at our module's import
    time still point at the *stubbed* re-import — pickle then fails with
    ``"it's not the same object as dynacell.evaluation.pipeline.FovResult"``.
    Resolving lazily inside each test avoids the stale binding.
    """
    return importlib.import_module("dynacell.evaluation.pipeline")


def _make_synthetic_result(
    pos_name: str = "A/1/0",
    t_count: int = 2,
    d: int = 4,
    h: int = 16,
    w: int = 16,
    cp_dim: int = 8,
    deep_dim: int = 768,
    cells_per_t: int = 3,
):
    """Build a FovResult with realistic shapes for round-trip testing."""
    pipeline = _live_pipeline_module()
    FovResult = pipeline.FovResult
    _BackboneLists = pipeline._BackboneLists
    row, col, fov = pos_name.split("/")
    per_t_pixel = [{"FOV": pos_name, "Timepoint": t, "PCC": 0.9 + 0.01 * t, "MicroMS3IM": 0.85} for t in range(t_count)]
    per_t_mask = [{"FOV": pos_name, "Timepoint": t, "DICE": 0.7 + 0.01 * t} for t in range(t_count)]
    per_t_feat = [{"FOV": pos_name, "Timepoint": t, "CP_cos": 0.8, "DINOv3_cos": 0.95} for t in range(t_count)]
    seg_array = np.zeros((t_count, 2, d, h, w), dtype=bool)
    seg_array[:, 0, :, : h // 2, :] = True  # pred channel
    seg_array[:, 1, :, : h // 2, : w // 2] = True  # gt channel

    cp = _BackboneLists()
    dinov3 = _BackboneLists()
    dynaclr = _BackboneLists()
    celldino = _BackboneLists()
    for t in range(t_count):
        fov_arr = np.full(cells_per_t, pos_name)
        t_arr = np.full(cells_per_t, t, dtype=np.int32)
        cp.pred_feats.append(np.full((cells_per_t, cp_dim), float(t), dtype=np.float32))
        cp.gt_feats.append(np.full((cells_per_t, cp_dim), float(t) + 0.5, dtype=np.float32))
        cp.pred_fovs.append(fov_arr)
        cp.gt_fovs.append(fov_arr)
        cp.pred_ts.append(t_arr)
        cp.gt_ts.append(t_arr)
        for bl in (dinov3, dynaclr, celldino):
            bl.pred_feats.append(np.full((cells_per_t, deep_dim), float(t), dtype=np.float32))
            bl.gt_feats.append(np.full((cells_per_t, deep_dim), float(t) + 0.5, dtype=np.float32))
            bl.pred_fovs.append(fov_arr)
            bl.gt_fovs.append(fov_arr)
            bl.pred_ts.append(t_arr)
            bl.gt_ts.append(t_arr)

    return FovResult(
        pos_name=pos_name,
        row=row,
        col=col,
        fov=fov,
        per_t_pixel_rows=per_t_pixel,
        per_t_mask_rows=per_t_mask,
        per_t_feature_rows=per_t_feat,
        seg_array=seg_array,
        cp=cp,
        dinov3=dinov3,
        dynaclr=dynaclr,
        celldino=celldino,
        timings=[(pos_name, None, "mask_gt", 0.05), (pos_name, 0, "pixel_metrics", 0.02)],
    )


def test_fov_result_pickle_round_trip_preserves_arrays():
    result = _make_synthetic_result()
    restored = pickle.loads(pickle.dumps(result))
    assert restored.pos_name == result.pos_name
    assert restored.row == "A"
    assert restored.col == "1"
    assert restored.fov == "0"
    assert restored.seg_array.shape == result.seg_array.shape
    assert restored.seg_array.dtype == np.bool_
    assert np.array_equal(restored.seg_array, result.seg_array)
    for backbone_attr in ("cp", "dinov3", "dynaclr", "celldino"):
        original = getattr(result, backbone_attr)
        restored_bb = getattr(restored, backbone_attr)
        for list_name in ("pred_feats", "gt_feats", "pred_fovs", "gt_fovs", "pred_ts", "gt_ts"):
            for a, b in zip(getattr(original, list_name), getattr(restored_bb, list_name)):
                assert np.array_equal(a, b)
    assert restored.per_t_pixel_rows == result.per_t_pixel_rows
    assert restored.timings == result.timings


def test_fov_result_pickle_handles_empty_backbones():
    """An empty _BackboneLists round-trips correctly (no len-zero -> None coercion)."""
    result = _make_synthetic_result()
    # Wipe one backbone to simulate "feature_metrics disabled" / "zero cells".
    result.celldino.pred_feats.clear()
    result.celldino.gt_feats.clear()
    result.celldino.pred_fovs.clear()
    result.celldino.gt_fovs.clear()
    result.celldino.pred_ts.clear()
    result.celldino.gt_ts.clear()
    restored = pickle.loads(pickle.dumps(result))
    assert restored.celldino.pred_feats == []
    assert restored.celldino.gt_feats == []


def test_aggregate_fov_result_extends_backbone_lists():
    """Aggregator must extend each backbone's six lists with worker contributions."""
    pipeline = _live_pipeline_module()
    _aggregate_fov_result = pipeline._aggregate_fov_result
    _BackboneLists = pipeline._BackboneLists
    # Mock segmentation_results plate handle (only create_position is exercised).
    written = {}

    class _FakeSegPos:
        def create_image(self, name, data):
            written[(row_, col_, fov_, name)] = np.asarray(data)

    class _FakeSegPlate:
        def create_position(self, row, col, fov):
            nonlocal row_, col_, fov_
            row_, col_, fov_ = row, col, fov
            return _FakeSegPos()

    row_ = col_ = fov_ = None

    result = _make_synthetic_result()

    all_pix: list[dict] = []
    all_mask: list[dict] = []
    all_feat: list[dict] = []
    parent_lists = {name: _BackboneLists() for name in pipeline._BACKBONE_KEYS}

    _aggregate_fov_result(
        result,
        _FakeSegPlate(),
        all_pix,
        all_mask,
        all_feat,
        parent_lists,
        extend_worker_timings=True,
    )

    assert len(all_pix) == 2
    assert len(all_mask) == 2
    assert len(all_feat) == 2
    assert (row_, col_, fov_) == ("A", "1", "0")
    assert written[("A", "1", "0", "0")].shape == result.seg_array.shape
    # Stronger than the old test: assert every backbone's six lists landed
    # lockstep. Catches a future regression where _extend_backbone drifts
    # (e.g. someone adds a field to _BackboneLists and forgets a list).
    for name in pipeline._BACKBONE_KEYS:
        bb = parent_lists[name]
        assert len(bb.pred_feats) == 2
        assert len(bb.gt_feats) == 2
        assert len(bb.pred_fovs) == 2
        assert len(bb.gt_fovs) == 2
        assert len(bb.pred_ts) == 2
        assert len(bb.gt_ts) == 2
