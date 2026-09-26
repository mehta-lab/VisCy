"""CP feature names travel with the CP cache, so the reference masks by name, not by position."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dynacell.evaluation.cache import StaleCacheError, cache_paths, load_manifest, save_manifest
from dynacell.evaluation.metrics import CP_FEATURE_NAMES_BY_VERSION, CP_FEATURE_VERSION, active_cp_feature_names
from dynacell.evaluation.pipeline_cache import (
    cached_cp_feature_names,
    check_cp_cache_feature_names,
    fov_cp_features,
    init_cache_context,
)

from ._eval_fixtures import build_eval_config

_NAMES_GLCM = active_cp_feature_names(True)
_NAMES_BASE = active_cp_feature_names(False)


def test_frozen_table_matches_the_live_column_order() -> None:
    """Reordering the live CP tuples without a CP_FEATURE_VERSION bump fails here, not silently in old caches."""
    assert CP_FEATURE_NAMES_BY_VERSION[CP_FEATURE_VERSION] == _NAMES_GLCM


def test_names_are_read_back_from_old_and_new_entries() -> None:
    """New entries carry their names; old ones are inferred exactly from the recorded recipe."""
    assert cached_cp_feature_names({"cp_feature_names": ["b", "a"]}) == ("b", "a")
    old = {"cp_feature_version": CP_FEATURE_VERSION, "cp_glcm_enabled": True}
    assert cached_cp_feature_names(old) == _NAMES_GLCM
    assert cached_cp_feature_names({**old, "cp_glcm_enabled": False}) == _NAMES_BASE
    with pytest.raises(StaleCacheError, match="unknown recipe 'v0'"):
        cached_cp_feature_names({"cp_feature_version": "v0", "cp_glcm_enabled": True})


def _config(tmp_path: Path):
    config = build_eval_config(
        tmp_path / "pred.zarr",
        tmp_path / "gt.zarr",
        tmp_path / "gt_cache",
        tmp_path / "pred_cache",
        tmp_path,
        executor="serial",
        fov_workers=1,
    )
    config.io.require_complete_cache = False
    config.feature_metrics.cp = {
        "norm": {"p_lo": 1.0, "p_hi": 99.0},
        "glcm": {"enabled": False},
        "reference_path": None,
    }
    return config


def test_the_cache_writer_records_names_and_a_reorder_is_refused(tmp_path: Path) -> None:
    """``fov_cp_features`` stamps the column names; a cache whose names are reordered is refused by name."""
    config = _config(tmp_path)
    ctx = init_cache_context(config, side="gt")
    image = np.random.default_rng(0).uniform(size=(1, 11, 16, 16)).astype(np.float32)
    seg = np.zeros((1, 11, 16, 16), dtype=np.int32)
    seg[0, 2:8, 2:8, 2:8] = 1
    fov_cp_features(ctx, "A/1/0", image, seg)
    assert ctx.manifest["artifacts"]["cp_features"]["cp_feature_names"] == list(_NAMES_BASE)
    check_cp_cache_feature_names(ctx, _NAMES_BASE)

    swapped = list(_NAMES_BASE)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(StaleCacheError, match="Masking by position would misalign them"):
        check_cp_cache_feature_names(ctx, tuple(swapped))


def test_a_forced_recompute_skips_the_name_check(tmp_path: Path) -> None:
    """With force_recompute.<side>_cp the cached columns are about to be rewritten, so they are not checked."""
    config = _config(tmp_path)
    paths = cache_paths(tmp_path / "gt_cache")
    manifest = load_manifest(paths)
    manifest["artifacts"]["cp_features"] = {"cp_feature_names": ["x"], "cp_feature_version": "v0"}
    save_manifest(paths, manifest)
    config.force_recompute.gt_cp = True
    ctx = init_cache_context(config, side="gt")
    check_cp_cache_feature_names(ctx, _NAMES_BASE)
