"""Prediction identity of the eval caches: a re-predict into the same ``io.pred_path``.

A re-predict writes new voxels (and, since PR #497, new writer markers) under the same
path, which the cache identity (path + channel + cell segmentation) cannot see. Every
cached prediction-side position therefore records the source it was built from (the
channel's marker and the mtime of its first chunk), and is reused only while the store
still holds that source. These tests drive the real cache builders,
``init_cache_context``, ``flush_manifest`` and the ``evaluate_model`` /
``evaluate_predictions_grouped`` save paths on tiny OME-Zarr v2 and v3 plates, and
re-predict the store the way the writer (or an older writer) does.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from iohub.ngff import open_ome_zarr
from omegaconf import OmegaConf

from dynacell.evaluation.cache import (
    StaleCacheError,
    cache_paths,
    load_manifest,
    prediction_sources,
    prediction_sources_sha256_12,
    save_manifest,
)
from dynacell.evaluation.metrics import active_cp_feature_names
from dynacell.evaluation.pipeline_cache import (
    flush_manifest,
    fov_cp_features,
    fov_deep_features,
    fov_masks,
    fov_nucleus_instances,
    init_cache_context,
)
from dynacell.evaluation.provenance import PROVENANCE_FILENAME
from viscy_utils.prediction_metadata import (
    PREDICTION_COMPLETE_KEY,
    completion_marker,
    mark_complete,
    mark_started,
    prediction_run,
    tzyx_shape,
)

from ._eval_fixtures import build_eval_config, live_pipeline_module

_CHANNEL = "prediction"
_OTHER = "other"
_POSITIONS = ("A/1/0", "A/1/1")
_T, _D, _H, _W = 2, 3, 8, 8
_FAMILIES = (
    "organelle_masks",
    "instance_masks",
    "cp_features",
    "dinov3_features",
    "dynaclr_features",
    "celldino_features",
    "morphem_features",
)
_EVERYTHING = {family: set(_POSITIONS) for family in _FAMILIES}
_DAY = 86400.0
# Every store starts dated here, so a later rewrite is strictly newer at any clock tick.
_WRITTEN_AT = datetime.fromisoformat("2026-07-01T00:00:00+00:00").timestamp()
_BUILT_AT = "2026-07-22T07:24:33+00:00"
_BUILT_AT_S = datetime.fromisoformat(_BUILT_AT).timestamp()
# CP stubs return the real column count: fov_cp_features rounds columns it finds by name.
_CP_WIDTH = len(active_cp_feature_names(False))
_STORE_VERSIONS = pytest.mark.parametrize("version", ["0.5", "0.4"], ids=["zarr-v3", "zarr-v2"])


def _run(settings: str) -> dict:
    """A predict-run identity; ``settings`` stands in for what differs between two predicts."""
    return prediction_run(
        array_key="0", z_window_size=1, z_reduction="blend", checkpoint_path=None, settings_sha256_12=settings
    )


def _write_prediction(
    path: Path, settings: str | None, *, channels: tuple[str, ...] = (_CHANNEL,), version: str = "0.5"
) -> None:
    """Write a prediction store dated ``_WRITTEN_AT``; ``settings=None`` leaves it unmarked, like a pre-marker writer.

    ``version="0.4"`` writes zarr v2 (``.zattrs``/``.zgroup``), as the older production
    predictions are. Values are not the fill value, so every chunk is stored.
    """
    with open_ome_zarr(path, mode="w", layout="hcs", channel_names=list(channels), version=version) as plate:
        for name in _POSITIONS:
            position = plate.create_position(*name.split("/"))
            position.create_image("0", np.full((_T, len(channels), _D, _H, _W), 0.5, dtype=np.float32))
            if settings is not None:
                mark_complete(position, list(channels), completion_marker(tzyx_shape(position["0"]), _run(settings)))
    _date_store(path, attributes=_WRITTEN_AT, chunks=_WRITTEN_AT)


def _metadata_mtimes(path: Path) -> dict[Path, float]:
    return {
        f: f.stat().st_mtime for f in path.rglob("*") if f.is_file() and (f.name == "zarr.json" or f.name[0] == ".")
    }


def _repredict(
    path: Path,
    settings: str,
    *,
    channel: str = _CHANNEL,
    positions: tuple[str, ...] = _POSITIONS,
    mark: bool = True,
) -> dict[Path, float]:
    """Re-predict *channel* of *positions* in place: mark started, rewrite its voxels, mark complete.

    ``mark=False`` rewrites the voxels only, as a code-only fix under an unchanged marker,
    or a pre-marker writer's ``--overwrite``, does. Returns the metadata files' mtimes
    from before the write.
    """
    before = _metadata_mtimes(path)
    run = _run(settings)
    with open_ome_zarr(path, mode="r+") as plate:
        for name in positions:
            position = plate[name]
            if mark:
                mark_started(position, [channel], run)
            position["0"][:, position.get_channel_index(channel)] = 1.0
            if mark:
                mark_complete(position, [channel], completion_marker(tzyx_shape(position["0"]), run))
    return before


def _date_store(path: Path, *, attributes: float, chunks: float) -> None:
    """Date each position's attribute file (v3 ``zarr.json``, v2 ``.zattrs``) and its chunk files."""
    for name in _POSITIONS:
        position = path / name
        for file in position.rglob("*"):
            if not file.is_file():
                continue
            is_metadata = file.name == "zarr.json" or file.name.startswith(".")
            mtime = attributes if is_metadata else chunks
            os.utime(file, (mtime, mtime))


def _config(tmp_path: Path, *, cache: str = "pred_cache", channel: str = _CHANNEL, **overrides):
    """Eval config over ``tmp_path/pred.zarr`` with both caches enabled and recompute allowed."""
    config = build_eval_config(
        tmp_path / "pred.zarr",
        tmp_path / "gt.zarr",
        tmp_path / "gt_cache",
        tmp_path / cache,
        tmp_path / "out",
        executor="serial",
        fov_workers=1,
    )
    config.io.require_complete_cache = False
    config.io.pred_channel_name = channel
    for key, value in overrides.items():
        OmegaConf.update(config, key, value, merge=True)
    return config


def _family_config(tmp_path: Path, **overrides):
    """Config under which a run reads every prediction-side artifact family, instances included."""
    return _config(
        tmp_path,
        **{
            "target_name": "nucleus",
            "compute_instance_ap": True,
            "segmentation": {
                "backend": "cellpose",
                "dimension": "2d",
                "cellpose": {"target_voxel_um": 0.58, "cellprob_threshold": 0.0, "min_obj_size": 30},
            },
            **overrides,
        },
    )


def _open_pred_ctx(config, tmp_path: Path):
    """Open the prediction-side context with every deep-feature family configured."""
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    for name in ("dynaclr.ckpt", "celldino.pth"):
        if not (weights / name).exists():
            (weights / name).write_bytes(name.encode())
    return init_cache_context(
        config,
        side="pred",
        dinov3_model_name="facebook/test-dinov3",
        dynaclr_ckpt_path=str(weights / "dynaclr.ckpt"),
        dynaclr_encoder_cfg={"name": "test"},
        celldino_weights_path=str(weights / "celldino.pth"),
        morphem_model_name="test/morphem",
    )


def _open_quietly(open_ctx):
    """Call *open_ctx*, failing on any stale-position warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ctx = open_ctx()
    assert not [str(w.message) for w in caught if "built from another prediction" in str(w.message)]
    return ctx


def _stub_segmenters(monkeypatch) -> None:
    monkeypatch.setattr("dynacell.evaluation.segmentation.segment", lambda img, *a, **k: np.ones(img.shape, dtype=bool))
    monkeypatch.setattr(
        "dynacell.evaluation.segmentation_cellpose.segment_nucleus_instances",
        lambda img, spacing, model, **k: np.ones(img.shape, dtype=np.uint16),
    )
    monkeypatch.setitem(
        fov_cp_features.__globals__, "cp_regionprops", lambda *a, **k: np.zeros((1, _CP_WIDTH), dtype=np.float32)
    )


def _flush_written(ctx) -> dict[str, set[str]]:
    """Return ``{family: positions}`` this context wrote since its last flush, then flush."""
    written = {keys[0]: set(positions) for keys, positions in ctx._written_sources.items()}
    flush_manifest(ctx)
    return written


def _build_every_family(ctx, monkeypatch, positions: tuple[str, ...] = _POSITIONS) -> dict[str, set[str]]:
    """Load-or-build masks, instance labels, CP and all four deep-feature families; return what was written.

    The cell segmentation is empty, so the deep writers record zero-cell slots without
    running an extractor.
    """
    _stub_segmenters(monkeypatch)
    image = np.zeros((_T, _D, _H, _W), dtype=np.float32)
    cell_seg = np.zeros((_T, _D, _H, _W), dtype=np.int32)
    for name in positions:
        fov_masks(ctx, name, image, seg_model=None)
        fov_nucleus_instances(ctx, name, image[:, 0], None)
        fov_cp_features(ctx, name, image, cell_seg)
        for kind in ("dinov3", "dynaclr", "celldino", "morphem"):
            fov_deep_features(ctx, name, image, cell_seg, None, kind)
    return _flush_written(ctx)


def _masks(ctx, monkeypatch, positions: tuple[str, ...] = _POSITIONS) -> dict[str, set[str]]:
    """Load-or-build organelle masks for *positions*; return what was written."""
    _stub_segmenters(monkeypatch)
    for name in positions:
        fov_masks(ctx, name, np.zeros((_T, _D, _H, _W), dtype=np.float32), seg_model=None)
    return _flush_written(ctx)


def _entries(manifest: dict) -> list[dict]:
    """Every artifact entry of a manifest (``cp_features`` is a leaf, the other families are keyed)."""
    artifacts = manifest["artifacts"]
    return ([artifacts["cp_features"]] if "cp_features" in artifacts else []) + [
        entry for family, keyed in artifacts.items() if family != "cp_features" for entry in keyed.values()
    ]


def _make_legacy(cache_dir: Path, built_at: str | None) -> None:
    """Rewrite the manifest as a pre-source run left it: no ``sources``, the given ``built_at``."""
    paths = cache_paths(cache_dir)
    manifest = load_manifest(paths)
    for entry in _entries(manifest):
        entry.pop("sources")
        entry.pop("built_at")
        if built_at is not None:
            entry["built_at"] = built_at
    save_manifest(paths, manifest)


# -- prediction-side artifact cache ---------------------------------------------------


def test_unchanged_prediction_reuses_every_family(tmp_path: Path, monkeypatch) -> None:
    """Every written position records its source, and a rerun on the same store rewrites nothing."""
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1")
    config = _family_config(tmp_path)
    assert _build_every_family(_open_pred_ctx(config, tmp_path), monkeypatch) == _EVERYTHING

    current = prediction_sources(pred, _CHANNEL)
    for entry in _entries(load_manifest(cache_paths(tmp_path / "pred_cache"))):
        assert entry["sources"] == current
    assert _build_every_family(_open_quietly(lambda: _open_pred_ctx(config, tmp_path)), monkeypatch) == {}


def test_repredict_rebuilds_every_family(tmp_path: Path, monkeypatch) -> None:
    """A re-predict with new markers makes every position of every family a miss; the rebuild records the new source."""
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1")
    config = _family_config(tmp_path)
    _build_every_family(_open_pred_ctx(config, tmp_path), monkeypatch)

    _repredict(pred, "run-2")
    with pytest.warns(UserWarning, match="built from another prediction"):
        ctx = _open_pred_ctx(config, tmp_path)
    assert _build_every_family(ctx, monkeypatch) == _EVERYTHING
    assert _build_every_family(_open_quietly(lambda: _open_pred_ctx(config, tmp_path)), monkeypatch) == {}


def test_interrupted_rebuild_resumes_on_the_unvisited_position(tmp_path: Path, monkeypatch) -> None:
    """A rebuild killed after one of two FOVs leaves the other a miss on the rerun, not a stale hit."""
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1")
    config = _config(tmp_path)
    _masks(init_cache_context(config, side="pred"), monkeypatch)
    _repredict(pred, "run-2")

    with pytest.warns(UserWarning, match="built from another prediction"):
        assert _masks(init_cache_context(config, side="pred"), monkeypatch, ("A/1/0",)) == {
            "organelle_masks": {"A/1/0"}
        }
    with pytest.warns(UserWarning, match=r"1 cached position\(s\)"):
        assert _masks(init_cache_context(config, side="pred"), monkeypatch) == {"organelle_masks": {"A/1/1"}}


@_STORE_VERSIONS
def test_code_only_repredict_is_caught(tmp_path: Path, monkeypatch, version: str) -> None:
    """Rewritten chunks under an unchanged marker (a code-only fix, or a pre-marker writer) are a miss."""
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1", version=version)
    config = _family_config(tmp_path)
    _build_every_family(_open_pred_ctx(config, tmp_path), monkeypatch)

    before = _repredict(pred, "run-1", mark=False)
    assert _metadata_mtimes(pred) == before  # the rewrite touched chunks only
    with pytest.warns(UserWarning, match="built from another prediction"):
        assert _build_every_family(_open_pred_ctx(config, tmp_path), monkeypatch) == _EVERYTHING


@pytest.mark.parametrize(
    ("built_at", "chunks_at", "rebuilt"),
    [
        pytest.param(_BUILT_AT, _BUILT_AT_S - _DAY, set(), id="chunks-older-upgraded"),
        pytest.param(_BUILT_AT, _BUILT_AT_S + _DAY, set(_POSITIONS), id="chunks-newer-rebuilt"),
        pytest.param(None, _BUILT_AT_S - _DAY, set(_POSITIONS), id="no-built-at-rebuilt"),
    ],
)
@_STORE_VERSIONS
def test_legacy_entry_is_upgraded_per_position(
    tmp_path: Path, monkeypatch, built_at: str | None, chunks_at: float, rebuilt: set, version: str
) -> None:
    """A pre-source entry keeps a position iff its chunk is no newer than ``built_at``, recording it without recompute.

    Metadata newer than ``built_at`` does not count: only chunks carry the prediction.
    """
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1", version=version)
    config = _family_config(tmp_path)
    _build_every_family(_open_pred_ctx(config, tmp_path), monkeypatch)
    _make_legacy(tmp_path / "pred_cache", built_at)
    _date_store(pred, attributes=_BUILT_AT_S + 2 * _DAY, chunks=chunks_at)

    assert _build_every_family(_open_pred_ctx(config, tmp_path), monkeypatch) == (
        {family: rebuilt for family in _FAMILIES} if rebuilt else {}
    )
    current = prediction_sources(pred, _CHANNEL)
    for entry in _entries(load_manifest(cache_paths(tmp_path / "pred_cache"))):
        assert entry["sources"] == current


def test_interrupted_legacy_rebuild_keeps_the_rest_stale(tmp_path: Path, monkeypatch) -> None:
    """Rebuilding one FOV of a stale pre-source entry moves its ``built_at`` past the store; the other FOV stays a miss.

    The upgrade is decided once, against the ``built_at`` the entry was loaded with.
    """
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1")
    config = _config(tmp_path)
    _masks(init_cache_context(config, side="pred"), monkeypatch)
    _make_legacy(tmp_path / "pred_cache", _BUILT_AT)
    _date_store(pred, attributes=_WRITTEN_AT, chunks=_BUILT_AT_S + _DAY)  # re-predicted after the cache was built

    with pytest.warns(UserWarning, match="built from another prediction"):
        assert _masks(init_cache_context(config, side="pred"), monkeypatch, ("A/1/0",)) == {
            "organelle_masks": {"A/1/0"}
        }
    with pytest.warns(UserWarning, match=r"1 cached position\(s\)"):
        assert _masks(init_cache_context(config, side="pred"), monkeypatch) == {"organelle_masks": {"A/1/1"}}


def _blank(path: Path, timepoints: slice) -> None:
    """Write fill values into *timepoints* of every position; zarr then deletes those chunks."""
    with open_ome_zarr(path, mode="r+") as plate:
        for _, position in plate.positions():
            position["0"][timepoints] = 0.0


def _chunks(path: Path) -> list[Path]:
    return sorted(f for f in path.rglob("*") if f.is_file() and f.name != "zarr.json" and not f.name.startswith("."))


@pytest.mark.parametrize(
    ("chunks_at", "rebuilt"),
    [(_BUILT_AT_S - _DAY, set()), (_BUILT_AT_S + _DAY, set(_POSITIONS))],
    ids=["t1-older-upgraded", "t1-newer-rebuilt"],
)
@_STORE_VERSIONS
def test_legacy_entry_is_dated_by_the_first_stored_chunk(
    tmp_path: Path, monkeypatch, chunks_at: float, rebuilt: set, version: str
) -> None:
    """A channel blank at ``t=0`` stores no chunk there, so the position is dated by its ``t=1`` chunk."""
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1", version=version)
    _blank(pred, slice(0, 1))
    config = _config(tmp_path)
    _masks(init_cache_context(config, side="pred"), monkeypatch)
    _make_legacy(tmp_path / "pred_cache", _BUILT_AT)
    _date_store(pred, attributes=_WRITTEN_AT, chunks=chunks_at)

    t1 = [c for c in _chunks(pred / "A/1/0") if c.relative_to(pred / "A/1/0/0").parts[-5:][0] == "1"]
    assert len(_chunks(pred / "A/1/0")) == len(t1) > 0  # only t=1 is stored
    assert prediction_sources(pred, _CHANNEL)["A/1/0"]["written_ns"] == t1[0].stat().st_mtime_ns
    assert _masks(init_cache_context(config, side="pred"), monkeypatch) == (
        {"organelle_masks": rebuilt} if rebuilt else {}
    )


def test_legacy_entry_over_a_blank_channel_is_rebuilt_once(tmp_path: Path, monkeypatch) -> None:
    """A position with no stored chunk cannot be dated: rebuilt once, then its recorded ``None`` is an exact hit."""
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1")
    _blank(pred, slice(None))
    assert _chunks(pred) == []
    config = _config(tmp_path)
    _masks(init_cache_context(config, side="pred"), monkeypatch)
    _make_legacy(tmp_path / "pred_cache", _BUILT_AT)

    assert _masks(init_cache_context(config, side="pred"), monkeypatch) == {"organelle_masks": set(_POSITIONS)}
    assert _masks(_open_quietly(lambda: init_cache_context(config, side="pred")), monkeypatch) == {}


def test_a_position_added_after_init_is_recorded_and_reused(tmp_path: Path, monkeypatch) -> None:
    """A position a running predict finishes after the context opened is read on demand, recorded, and hits next run."""
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1")
    config = _config(tmp_path)
    ctx = init_cache_context(config, side="pred")
    with open_ome_zarr(pred, mode="r+") as plate:
        position = plate.create_position("A", "1", "2")
        position.create_image("0", np.full((_T, 1, _D, _H, _W), 0.5, dtype=np.float32))
        mark_complete(position, [_CHANNEL], completion_marker(tzyx_shape(position["0"]), _run("run-1")))

    everything = (*_POSITIONS, "A/1/2")
    assert _masks(ctx, monkeypatch, everything) == {"organelle_masks": set(everything)}
    assert _masks(_open_quietly(lambda: init_cache_context(config, side="pred")), monkeypatch, everything) == {}


def test_multichannel_repredict_invalidates_only_its_channel(tmp_path: Path, monkeypatch) -> None:
    """Rewriting channel 1 of a two-channel store misses a channel-1 cache and leaves a channel-0 cache hit."""
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1", channels=(_CHANNEL, _OTHER))
    channel0 = _config(tmp_path, cache="cache_channel0")
    channel1 = _config(tmp_path, cache="cache_channel1", channel=_OTHER)
    for config in (channel0, channel1):
        _masks(init_cache_context(config, side="pred"), monkeypatch)

    _repredict(pred, "run-1", channel=_OTHER, mark=False)  # voxels only, so no marker can give it away
    assert _masks(_open_quietly(lambda: init_cache_context(channel0, side="pred")), monkeypatch) == {}
    with pytest.warns(UserWarning, match="built from another prediction"):
        assert _masks(init_cache_context(channel1, side="pred"), monkeypatch) == {"organelle_masks": set(_POSITIONS)}


@pytest.mark.parametrize("legacy", [False, True], ids=["recorded", "legacy"])
def test_metadata_writes_do_not_invalidate(tmp_path: Path, monkeypatch, legacy: bool) -> None:
    """Another channel's markers or a focus estimate written into the store leave this channel's cache a hit."""
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1", channels=(_CHANNEL, _OTHER))
    config = _config(tmp_path)
    _masks(init_cache_context(config, side="pred"), monkeypatch)
    if legacy:
        _make_legacy(tmp_path / "pred_cache", _BUILT_AT)

    with open_ome_zarr(pred, mode="r+") as plate:
        for _, position in plate.positions():
            mark_started(position, [_OTHER], _run("run-2"))
            position.zattrs["focus_slice"] = {_CHANNEL: {"dataset_statistics": {"z_focus_mean": 1}}}
    assert _masks(_open_quietly(lambda: init_cache_context(config, side="pred")), monkeypatch) == {}


def test_require_complete_raises_on_the_stale_position_only(tmp_path: Path, monkeypatch) -> None:
    """A cache-only run opens, serves the current position, and refuses the re-predicted one as a miss."""
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1")
    _masks(init_cache_context(_config(tmp_path), side="pred"), monkeypatch)
    _repredict(pred, "run-2", positions=("A/1/1",))

    with pytest.warns(UserWarning, match="built from another prediction"):
        ctx = init_cache_context(_config(tmp_path, **{"io.require_complete_cache": True}), side="pred")
    monkeypatch.setattr("dynacell.evaluation.segmentation.segment", pytest.fail)
    fov_masks(ctx, "A/1/0", np.zeros((_T, _D, _H, _W), dtype=np.float32), seg_model=None)
    with pytest.raises(StaleCacheError, match="cache miss at A/1/1 and io.require_complete_cache=true"):
        fov_masks(ctx, "A/1/1", np.zeros((_T, _D, _H, _W), dtype=np.float32), seg_model=None)


def test_limit_positions_rebuilds_only_the_stale_position(tmp_path: Path, monkeypatch) -> None:
    """A partial walk needs no store-wide refusal: it rebuilds its stale FOV and leaves the rest recorded."""
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1")
    _masks(init_cache_context(_config(tmp_path), side="pred"), monkeypatch)
    _repredict(pred, "run-2", positions=("A/1/1",))

    with pytest.warns(UserWarning, match="built from another prediction"):
        ctx = init_cache_context(_config(tmp_path, limit_positions=1), side="pred")
    assert _masks(ctx, monkeypatch) == {"organelle_masks": {"A/1/1"}}
    assert _masks(_open_quietly(lambda: init_cache_context(_config(tmp_path), side="pred")), monkeypatch) == {}


def test_excluded_walk_records_sources_for_the_fovs_it_walked(tmp_path: Path, monkeypatch) -> None:
    """``io.exclude_fov_names`` rebuilds and records its FOVs; the skipped stale FOV stays a miss for the next walk."""
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1")
    _masks(init_cache_context(_config(tmp_path), side="pred"), monkeypatch)
    _repredict(pred, "run-2")

    excluded = _config(tmp_path, **{"io.exclude_fov_names": ["A/1/1"]})
    with pytest.warns(UserWarning, match="built from another prediction"):
        assert _masks(init_cache_context(excluded, side="pred"), monkeypatch, ("A/1/0",)) == {
            "organelle_masks": {"A/1/0"}
        }
    with pytest.warns(UserWarning, match=r"1 cached position\(s\)"):
        assert _masks(init_cache_context(_config(tmp_path), side="pred"), monkeypatch) == {"organelle_masks": {"A/1/1"}}


def test_concurrent_writers_keep_each_others_sources(tmp_path: Path, monkeypatch) -> None:
    """Two processes rebuilding different FOVs of one cache both keep their sources, whichever flushes last.

    Each loaded the other's FOV as stale; the last flush must not write that stale view
    back over the other's fresh source.
    """
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1")
    config = _config(tmp_path)
    _stub_segmenters(monkeypatch)
    image, cell_seg = np.zeros((_T, _D, _H, _W), dtype=np.float32), np.zeros((_T, _D, _H, _W), dtype=np.int32)
    first = init_cache_context(config, side="pred")
    for name in _POSITIONS:
        fov_masks(first, name, image, seg_model=None)
        fov_cp_features(first, name, image, cell_seg)
    flush_manifest(first)
    _repredict(pred, "run-2")

    with pytest.warns(UserWarning, match="built from another prediction"):
        a, b = init_cache_context(config, side="pred"), init_cache_context(config, side="pred")
    for ctx, name in ((a, "A/1/0"), (b, "A/1/1")):
        fov_masks(ctx, name, image, seg_model=None)
        fov_cp_features(ctx, name, image, cell_seg)
    flush_manifest(a)
    flush_manifest(b)

    current = prediction_sources(pred, _CHANNEL)
    artifacts = load_manifest(cache_paths(tmp_path / "pred_cache"))["artifacts"]
    assert artifacts["organelle_masks"]["er"]["sources"] == current
    assert artifacts["cp_features"]["sources"] == current
    assert artifacts["cp_features"]["positions"] == list(_POSITIONS)


def test_gt_cache_records_no_sources(tmp_path: Path, monkeypatch) -> None:
    """The GT side reads no prediction source and records none, whatever its store's markers do."""
    gt = tmp_path / "gt.zarr"
    _write_prediction(gt, "run-1", channels=("target",))
    config = _config(tmp_path)
    ctx = init_cache_context(config, side="gt")
    assert ctx.prediction_sources is None
    _masks(ctx, monkeypatch)
    _repredict(gt, "run-2", channel="target")

    assert "sources" not in load_manifest(cache_paths(tmp_path / "gt_cache"))["artifacts"]["organelle_masks"]["er"]
    assert _masks(_open_quietly(lambda: init_cache_context(config, side="gt")), monkeypatch) == {}


@_STORE_VERSIONS
def test_sources_track_marker_and_chunk_not_provenance(tmp_path: Path, version: str) -> None:
    """A moved checkpoint keeps the source; a new run's marker or a rewritten chunk changes it."""
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1", version=version)
    first = prediction_sources(pred, _CHANNEL)
    assert set(first) == set(_POSITIONS)

    with open_ome_zarr(pred, mode="r+") as plate:
        position = plate["A/1/0"]
        moved = {**position.zattrs[PREDICTION_COMPLETE_KEY][_CHANNEL], "checkpoint_path": "/moved.ckpt"}
        mark_complete(position, [_CHANNEL], moved)
    assert prediction_sources(pred, _CHANNEL) == first

    with open_ome_zarr(pred, mode="r+") as plate:
        mark_started(plate["A/1/0"], [_CHANNEL], _run("run-2"))
    started = prediction_sources(pred, _CHANNEL)
    assert started["A/1/0"]["marker"] != first["A/1/0"]["marker"]
    assert started["A/1/0"]["written_ns"] == first["A/1/0"]["written_ns"]
    assert started["A/1/1"] == first["A/1/1"]

    _repredict(pred, "run-2", positions=("A/1/1",), mark=False)
    assert prediction_sources(pred, _CHANNEL)["A/1/1"]["written_ns"] > first["A/1/1"]["written_ns"]


@pytest.mark.parametrize("layout", ["zarr-v3", "zarr-v3-sharded", "zarr-v2"])
def test_source_chunk_is_found_in_every_layout(tmp_path: Path, layout: str) -> None:
    """The channel's ``t=0`` chunk is found through the array's own key encoding, past a fill-value first chunk."""
    pred = tmp_path / "pred.zarr"
    data = np.full((_T, 2, _D, _H, _W), 0.5, dtype=np.float32)
    data[:, :, 0] = 0.0  # z-chunk 0 is the fill value, so it is never stored
    version = "0.4" if layout.startswith("zarr-v2") else "0.5"
    extra = {"shards_ratio": (1, 1, _D, 1, 1)} if layout == "zarr-v3-sharded" else {}
    with open_ome_zarr(pred, mode="w", layout="hcs", channel_names=[_CHANNEL, _OTHER], version=version) as plate:
        for name in _POSITIONS:
            plate.create_position(*name.split("/")).create_image("0", data, chunks=(1, 1, 1, _H, _W), **extra)
    chunks = sorted(
        f for f in (pred / "A/1/0/0").rglob("*") if f.is_file() and f.name != "zarr.json" and f.name[0] != "."
    )
    for i, chunk in enumerate(chunks):  # every stored chunk gets its own date
        os.utime(chunk, (_WRITTEN_AT + i, _WRITTEN_AT + i))

    written = prediction_sources(pred, _OTHER)["A/1/0"]["written_ns"]
    (match,) = [chunk for chunk in chunks if chunk.stat().st_mtime_ns == written]
    key = str(match.relative_to(pred / "A/1/0/0")).removeprefix("c/")
    assert key.split("/")[:2] == ["0", "1"], key  # t=0, channel 1


@_STORE_VERSIONS
def test_a_store_without_chunks_has_no_written_time(tmp_path: Path, version: str) -> None:
    """An unwritten predict (fill values only, no chunk stored) has no written time rather than raising."""
    pred = tmp_path / "pred.zarr"
    with open_ome_zarr(pred, mode="w", layout="hcs", channel_names=[_CHANNEL], version=version) as plate:
        for name in _POSITIONS:
            plate.create_position(*name.split("/")).create_image("0", np.zeros((_T, 1, _D, _H, _W), dtype=np.float32))
    assert {source["written_ns"] for source in prediction_sources(pred, _CHANNEL).values()} == {None}


# -- final-metrics cache ---------------------------------------------------------------

_PIXEL = [{"FOV": "A/1/0", "Timepoint": 0, "SI_PSNR": 1.0, "SI_SSIM": 1.0, "SI_NRMSE": 1.0}]
_MASK = [{"FOV": "A/1/0", "Timepoint": 0, "Dice": 0.5}]


def _evaluate_and_save(pipeline, config, monkeypatch, *, during_scoring=None) -> None:
    """Run the real ``evaluate_model`` save path over stubbed scoring rows."""

    def _fake_evaluate_predictions(cfg, *, cp_space):
        if during_scoring is not None:
            during_scoring()
        return _PIXEL, _MASK, []

    monkeypatch.setattr(pipeline, "check_cubic_pin", lambda: None)
    monkeypatch.setattr(pipeline, "apply_dataset_ref", lambda cfg: None)
    monkeypatch.setattr(pipeline, "evaluate_predictions", _fake_evaluate_predictions)
    getattr(pipeline.evaluate_model, "__wrapped__", pipeline.evaluate_model)(config)


def _stamp(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "out" / PROVENANCE_FILENAME).read_text())


@pytest.mark.parametrize("mark", [True, False], ids=["new-markers", "code-only"])
def test_repredict_invalidates_the_final_metrics_cache(tmp_path: Path, monkeypatch, mark: bool) -> None:
    """Saved metrics are reused on the same store and refused after a re-predict, marked or not."""
    pipeline = live_pipeline_module()
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1")
    config = _config(tmp_path)
    _evaluate_and_save(pipeline, config, monkeypatch)

    assert _stamp(tmp_path)["prediction_sources_sha256_12"] == prediction_sources_sha256_12(
        prediction_sources(pred, _CHANNEL)
    )
    assert pipeline._final_metrics_cache_valid(config)
    _repredict(pred, "run-2" if mark else "run-1", mark=mark)
    assert not pipeline._final_metrics_cache_valid(config)


def test_final_metrics_stamp_the_prediction_scored(tmp_path: Path, monkeypatch) -> None:
    """A re-predict landing mid-scoring leaves the pre-scoring stamp, so the rows are not reused."""
    pipeline = live_pipeline_module()
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1")
    scored = prediction_sources_sha256_12(prediction_sources(pred, _CHANNEL))
    config = _config(tmp_path)
    _evaluate_and_save(pipeline, config, monkeypatch, during_scoring=lambda: _repredict(pred, "run-2"))

    assert _stamp(tmp_path)["prediction_sources_sha256_12"] == scored
    assert not pipeline._final_metrics_cache_valid(config)


def test_grouped_final_metrics_stamp_the_prediction_scored(tmp_path: Path, monkeypatch) -> None:
    """The grouped driver also takes the sources digest before scoring each condition."""
    pipeline = live_pipeline_module()
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1")
    scored = prediction_sources_sha256_12(prediction_sources(pred, _CHANNEL))
    base = _config(tmp_path)
    grouped = OmegaConf.merge(
        base,
        {
            "conditions": [{"name": "only", "save": {"save_dir": str(tmp_path / "out")}}],
            "cross_condition_probe": {"enabled": False},
        },
    )

    def _fake_evaluate_predictions(cfg, *, models, cp_space):
        _repredict(pred, "run-2")
        return _PIXEL, _MASK, []

    monkeypatch.setattr(pipeline, "apply_dataset_ref", lambda cfg: None)
    monkeypatch.setattr(pipeline, "load_eval_models", lambda cfg: object())
    monkeypatch.setattr(pipeline, "evaluate_predictions", _fake_evaluate_predictions)
    pipeline.evaluate_predictions_grouped(grouped)

    assert _stamp(tmp_path)["prediction_sources_sha256_12"] == scored
    assert not pipeline._final_metrics_cache_valid(base)


@pytest.mark.parametrize(("chunks_offset", "reusable"), [(-_DAY, True), (_DAY, False)], ids=["older", "newer"])
@_STORE_VERSIONS
def test_legacy_final_metrics_are_dated_by_chunks(
    tmp_path: Path, monkeypatch, chunks_offset: float, reusable: bool, version: str
) -> None:
    """A sidecar written before the digest is reused iff no chunk is newer than it; newer metadata does not count."""
    pipeline = live_pipeline_module()
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1", version=version)
    config = _config(tmp_path)
    _evaluate_and_save(pipeline, config, monkeypatch)

    sidecar = tmp_path / "out" / PROVENANCE_FILENAME
    payload = json.loads(sidecar.read_text())
    del payload["prediction_sources_sha256_12"]
    sidecar.write_text(json.dumps(payload))
    saved_at = sidecar.stat().st_mtime - 10 * _DAY
    os.utime(sidecar, (saved_at, saved_at))
    _date_store(pred, attributes=saved_at + _DAY, chunks=saved_at + chunks_offset)
    assert pipeline._final_metrics_cache_valid(config) is reusable


def test_legacy_final_metrics_over_a_blank_store_are_recomputed_once(tmp_path: Path, monkeypatch) -> None:
    """A pre-digest sidecar over a store with no stored chunk cannot be dated, so it is refused; the re-save is kept."""
    pipeline = live_pipeline_module()
    pred = tmp_path / "pred.zarr"
    _write_prediction(pred, "run-1")
    _blank(pred, slice(None))
    config = _config(tmp_path)
    _evaluate_and_save(pipeline, config, monkeypatch)
    sidecar = tmp_path / "out" / PROVENANCE_FILENAME
    payload = json.loads(sidecar.read_text())
    del payload["prediction_sources_sha256_12"]
    sidecar.write_text(json.dumps(payload))

    assert not pipeline._final_metrics_cache_valid(config)
    _evaluate_and_save(pipeline, config, monkeypatch)
    assert pipeline._final_metrics_cache_valid(config)
