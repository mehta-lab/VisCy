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
import inspect
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
    morphem = _BackboneLists()
    for t in range(t_count):
        fov_arr = np.full(cells_per_t, pos_name)
        t_arr = np.full(cells_per_t, t, dtype=np.int32)
        cp.pred_feats.append(np.full((cells_per_t, cp_dim), float(t), dtype=np.float32))
        cp.gt_feats.append(np.full((cells_per_t, cp_dim), float(t) + 0.5, dtype=np.float32))
        cp.pred_fovs.append(fov_arr)
        cp.gt_fovs.append(fov_arr)
        cp.pred_ts.append(t_arr)
        cp.gt_ts.append(t_arr)
        for bl in (dinov3, dynaclr, celldino, morphem):
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
        morphem=morphem,
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
    for backbone_attr in ("cp", "dinov3", "dynaclr", "celldino", "morphem"):
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
    # All six lists per backbone must land lockstep — catches a regression
    # where _extend_backbone drifts (e.g. a new field added to _BackboneLists
    # but the helper forgets to append to it).
    for name in pipeline._BACKBONE_KEYS:
        bb = parent_lists[name]
        assert len(bb.pred_feats) == 2
        assert len(bb.gt_feats) == 2
        assert len(bb.pred_fovs) == 2
        assert len(bb.gt_fovs) == 2
        assert len(bb.pred_ts) == 2
        assert len(bb.gt_ts) == 2


def test_worker_run_fov_hands_the_cp_space_to_process_one_fov(tmp_path, monkeypatch):
    """``_worker_run_fov`` forwards the parent's CP space, and the space survives the pickle hop.

    Under ``executor=process`` the parent ships the verified CP space with every
    submission; the worker must score with that object, not reload or drop it.
    """
    from dynacell.evaluation.cp_reference import DatasetFit, fit_cp_reference, load_cp_reference, write_cp_reference

    from ._eval_fixtures import build_eval_config, make_hcs_plate

    pipeline = _live_pipeline_module()
    names = tuple(f"f{i}" for i in range(5))
    fit = DatasetFit(
        dataset="ds",
        cells=np.random.default_rng(0).standard_normal((40, 5)),
        record={"positions": ["A/1/0"], "gt_cache_dir": None, "cp_cache_built_at": None},
        in_mask_fit=True,
    )
    write_cp_reference(
        fit_cp_reference([fit], target_name="er", feature_names=names, cp_identity={}, lite={}), tmp_path / "er.json"
    )
    cp_space = pickle.loads(pickle.dumps(load_cp_reference(tmp_path / "er.json", target_name="er").for_dataset("ds")))

    make_hcs_plate(tmp_path / "pred.zarr", "prediction", seed=0, n_positions=1)
    make_hcs_plate(tmp_path / "gt.zarr", "target", seed=1, n_positions=1)
    config = build_eval_config(
        tmp_path / "pred.zarr",
        tmp_path / "gt.zarr",
        tmp_path / "g",
        tmp_path / "p",
        tmp_path,
        executor="process",
        fov_workers=1,
    )
    seen = {}
    signature = inspect.signature(pipeline._process_one_fov)

    def _capture(*args, **kwargs):
        seen["cp_space"] = signature.bind(*args, **kwargs).arguments["cp_space"]
        return "result"

    monkeypatch.setattr(pipeline, "_worker_setup", lambda cfg: None)
    monkeypatch.setattr(pipeline, "_process_one_fov", _capture)
    monkeypatch.setattr(pipeline, "flush_manifest", lambda ctx: None)
    monkeypatch.setitem(pipeline._WORKER_STATE, "cache_ctx", None)
    monkeypatch.setitem(pipeline._WORKER_STATE, "pred_cache_ctx", None)
    for key in ("seg_model", "dinov3", "dynaclr", "celldino", "morphem"):
        monkeypatch.setitem(pipeline._WORKER_STATE, key, None)

    assert pipeline._worker_run_fov(config, "A/1/0", 0, None, cp_space) == "result"
    assert seen["cp_space"] is cp_space
    np.testing.assert_array_equal(seen["cp_space"].mean, cp_space.mean)
