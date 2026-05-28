"""Unit tests for the evaluation cache module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("iohub")
pytest.importorskip("omegaconf")

from dynacell.evaluation.cache import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    StaleCacheError,
    cache_paths,
    check_cache_identity,
    ckpt_sha256_12,
    diff_artifact_params,
    encoder_config_sha256_12,
    load_manifest,
    read_features,
    read_mask,
    save_manifest,
    seed_cache_identity,
    write_features,
    write_mask,
)


def test_cache_paths_layout(tmp_path: Path) -> None:
    """CachePaths maps to the documented on-disk layout."""
    paths = cache_paths(tmp_path)
    assert paths.root == tmp_path
    assert paths.manifest == tmp_path / "manifest.yaml"
    assert paths.masks_dir == tmp_path / "organelle_masks"
    assert paths.features_dir == tmp_path / "features"
    assert paths.mask_plate("er") == tmp_path / "organelle_masks" / "er.zarr"
    assert paths.cp_features() == tmp_path / "features" / "cp.zarr"
    assert paths.dinov3_features("facebook/dinov3-vitl16") == (
        tmp_path / "features" / "dinov3" / "facebook__dinov3-vitl16.zarr"
    )
    assert paths.dynaclr_features("abcdef012345") == (tmp_path / "features" / "dynaclr" / "abcdef012345.zarr")


def test_load_manifest_missing_returns_skeleton(tmp_path: Path) -> None:
    """A missing manifest file returns a valid empty skeleton."""
    paths = cache_paths(tmp_path)
    manifest = load_manifest(paths)
    assert manifest["cache_schema_version"] == CACHE_SCHEMA_VERSION
    assert manifest["artifacts"] == {}
    assert manifest["gt"] is None
    assert manifest["pred"] is None
    assert manifest["cell_segmentation"] is None


def test_save_and_load_manifest_roundtrip(tmp_path: Path) -> None:
    """Manifest written and reloaded preserves nested structure."""
    paths = cache_paths(tmp_path)
    manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "gt": {"plate_path": "/data/gt.zarr", "channel_name": "target"},
        "pred": None,
        "cell_segmentation": {"plate_path": "/data/seg.zarr"},
        "artifacts": {
            "organelle_masks": {"er": {"path": "organelle_masks/er.zarr", "target_name": "er"}},
            "cp_features": {"path": "features/cp.zarr", "spacing": [0.29, 0.108, 0.108]},
        },
    }
    save_manifest(paths, manifest)
    loaded = load_manifest(paths)
    assert loaded == manifest


def test_check_cache_identity_version_mismatch(tmp_path: Path) -> None:
    """Wrong cache_schema_version raises with a clear message."""
    manifest = {"cache_schema_version": CACHE_SCHEMA_VERSION + 99, "gt": None, "cell_segmentation": None}
    with pytest.raises(StaleCacheError, match="schema version mismatch"):
        check_cache_identity(
            manifest,
            source="gt",
            plate_path="/x.zarr",
            channel_name="target",
            cell_segmentation_path=None,
        )


def test_check_cache_identity_gt_path_mismatch() -> None:
    """A different gt_path against an existing gt entry raises."""
    manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "gt": {"plate_path": "/old.zarr", "channel_name": "target"},
        "cell_segmentation": None,
    }
    with pytest.raises(StaleCacheError, match="gt.plate_path mismatch"):
        check_cache_identity(
            manifest,
            source="gt",
            plate_path="/new.zarr",
            channel_name="target",
            cell_segmentation_path=None,
        )


def test_check_cache_identity_channel_name_mismatch() -> None:
    """A different gt_channel_name raises — prevents silent mis-serving."""
    manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "gt": {"plate_path": "/g.zarr", "channel_name": "target"},
        "cell_segmentation": None,
    }
    with pytest.raises(StaleCacheError, match="gt.channel_name mismatch"):
        check_cache_identity(
            manifest,
            source="gt",
            plate_path="/g.zarr",
            channel_name="fluorescence",
            cell_segmentation_path=None,
        )


def test_check_cache_identity_cell_seg_mismatch() -> None:
    """Different cell_segmentation_path raises when both sides are set."""
    manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "gt": None,
        "cell_segmentation": {"plate_path": "/seg_v1.zarr"},
    }
    with pytest.raises(StaleCacheError, match="cell_segmentation.plate_path mismatch"):
        check_cache_identity(
            manifest,
            source="gt",
            plate_path="/g.zarr",
            channel_name="target",
            cell_segmentation_path="/seg_v2.zarr",
        )


def test_check_cache_identity_pred_path_mismatch() -> None:
    """A different pred_path against an existing pred entry raises."""
    manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "gt": None,
        "pred": {"plate_path": "/old_pred.zarr", "channel_name": "prediction"},
        "cell_segmentation": None,
    }
    with pytest.raises(StaleCacheError, match="pred.plate_path mismatch"):
        check_cache_identity(
            manifest,
            source="pred",
            plate_path="/new_pred.zarr",
            channel_name="prediction",
            cell_segmentation_path=None,
        )


def test_check_cache_identity_pred_channel_mismatch() -> None:
    """A different pred_channel_name raises — prevents silent mis-serving."""
    manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "gt": None,
        "pred": {"plate_path": "/p.zarr", "channel_name": "prediction"},
        "cell_segmentation": None,
    }
    with pytest.raises(StaleCacheError, match="pred.channel_name mismatch"):
        check_cache_identity(
            manifest,
            source="pred",
            plate_path="/p.zarr",
            channel_name="other_prediction",
            cell_segmentation_path=None,
        )


def test_check_cache_identity_empty_manifest_is_noop() -> None:
    """Empty manifest (fresh cache) passes identity checks silently."""
    manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "gt": None,
        "pred": None,
        "cell_segmentation": None,
    }
    check_cache_identity(
        manifest,
        source="gt",
        plate_path="/g.zarr",
        channel_name="target",
        cell_segmentation_path="/seg.zarr",
    )


def test_seed_cache_identity_populates_empty() -> None:
    """seed_cache_identity fills missing gt / cell_segmentation entries."""
    manifest: dict = {"cache_schema_version": CACHE_SCHEMA_VERSION, "gt": None, "pred": None, "cell_segmentation": None}
    seed_cache_identity(
        manifest,
        source="gt",
        plate_path="/g.zarr",
        channel_name="target",
        cell_segmentation_path="/seg.zarr",
    )
    assert manifest["gt"] == {"plate_path": "/g.zarr", "channel_name": "target"}
    assert manifest["cell_segmentation"] == {"plate_path": "/seg.zarr"}


def test_seed_cache_identity_populates_prediction() -> None:
    """seed_cache_identity fills missing prediction source entries."""
    manifest: dict = {"cache_schema_version": CACHE_SCHEMA_VERSION, "gt": None, "pred": None, "cell_segmentation": None}
    seed_cache_identity(
        manifest,
        source="pred",
        plate_path="/p.zarr",
        channel_name="prediction",
        cell_segmentation_path="/seg.zarr",
    )
    assert manifest["pred"] == {"plate_path": "/p.zarr", "channel_name": "prediction"}
    assert manifest["cell_segmentation"] == {"plate_path": "/seg.zarr"}


def test_seed_cache_identity_preserves_existing() -> None:
    """seed_cache_identity does not overwrite already-set entries."""
    manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "gt": {"plate_path": "/orig.zarr", "channel_name": "target"},
        "pred": {"plate_path": "/orig_pred.zarr", "channel_name": "prediction"},
        "cell_segmentation": {"plate_path": "/orig_seg.zarr"},
    }
    seed_cache_identity(
        manifest,
        source="gt",
        plate_path="/new.zarr",
        channel_name="target",
        cell_segmentation_path="/new_seg.zarr",
    )
    seed_cache_identity(
        manifest,
        source="pred",
        plate_path="/new_pred.zarr",
        channel_name="other_prediction",
    )
    assert manifest["gt"]["plate_path"] == "/orig.zarr"
    assert manifest["pred"]["plate_path"] == "/orig_pred.zarr"
    assert manifest["cell_segmentation"]["plate_path"] == "/orig_seg.zarr"


def test_diff_artifact_params_none_entry_returns_empty() -> None:
    """No manifest entry means no comparison to do; diff returns empty list."""
    assert diff_artifact_params(None, {"spacing": [1.0, 1.0, 1.0]}) == []


def test_diff_artifact_params_numeric_allclose_returns_empty() -> None:
    """Near-identical floats pass the numeric comparison via np.allclose."""
    entry = {"spacing": [0.29, 0.108, 0.108]}
    assert (
        diff_artifact_params(
            entry,
            {"spacing": [0.29, 0.10800000000001, 0.108]},
            numeric_keys=("spacing",),
        )
        == []
    )


def test_diff_artifact_params_numeric_mismatch_lists_key() -> None:
    """Materially different spacing values surface as a mismatch tuple."""
    entry = {"spacing": [0.29, 0.108, 0.108]}
    mismatches = diff_artifact_params(
        entry,
        {"spacing": [0.3, 0.108, 0.108]},
        numeric_keys=("spacing",),
    )
    assert mismatches == [("spacing", [0.29, 0.108, 0.108], [0.3, 0.108, 0.108])]


def test_diff_artifact_params_scalar_mismatch_lists_key() -> None:
    """Non-numeric scalar mismatches surface with the param name."""
    entry = {"patch_size": 256, "model_name": "foo"}
    mismatches = diff_artifact_params(entry, {"patch_size": 128, "model_name": "foo"})
    assert mismatches == [("patch_size", 256, 128)]


def test_diff_artifact_params_numeric_length_mismatch_lists_key() -> None:
    """A malformed numeric cached value (wrong length) surfaces as a mismatch.

    np.allclose raises ValueError on incompatible broadcast shapes; the
    helper must catch that so the caller can soft-invalidate rather than
    crash inside init_cache_context.
    """
    entry = {"spacing": [0.29, 0.108]}
    mismatches = diff_artifact_params(
        entry,
        {"spacing": [0.29, 0.108, 0.108]},
        numeric_keys=("spacing",),
    )
    assert mismatches == [("spacing", [0.29, 0.108], [0.29, 0.108, 0.108])]


def test_diff_artifact_params_numeric_nonconvertible_lists_key() -> None:
    """A numeric cached value that isn't array-castable surfaces as a mismatch.

    np.asarray(..., dtype=float) raises TypeError on non-numeric input
    (e.g. a hand-edited manifest with a string sentinel); the helper must
    catch that so a malformed manifest does not bypass soft-invalidation.
    """
    entry = {"spacing": "unknown"}
    mismatches = diff_artifact_params(
        entry,
        {"spacing": [0.29, 0.108, 0.108]},
        numeric_keys=("spacing",),
    )
    assert mismatches == [("spacing", "unknown", [0.29, 0.108, 0.108])]


def test_diff_artifact_params_non_dict_entry_lists_all_keys() -> None:
    """A non-mapping manifest entry surfaces every current key as a mismatch.

    Hand-edited or partially-written manifest YAML can put a string,
    list, or scalar where an artifact dict is expected; `entry.get(key)`
    would raise AttributeError and crash init_cache_context. The helper
    must treat the malformed entry as "no usable cached params" so the
    caller can soft-invalidate.
    """
    mismatches = diff_artifact_params(
        "corrupted-string-instead-of-dict",  # type: ignore[arg-type]
        {"spacing": [0.29, 0.108, 0.108], "patch_size": 4},
        numeric_keys=("spacing",),
    )
    assert mismatches == [
        ("spacing", "corrupted-string-instead-of-dict", [0.29, 0.108, 0.108]),
        ("patch_size", "corrupted-string-instead-of-dict", 4),
    ]


def test_write_and_read_mask_roundtrip(tmp_path: Path) -> None:
    """Masks written for one position are readable back as a bool array."""
    paths = cache_paths(tmp_path)
    rng = np.random.default_rng(0)
    masks = (rng.random((3, 4, 8, 8)) > 0.5).astype(bool)  # (T, D, H, W)
    write_mask(paths, "er", "A/1/0", masks)

    loaded = read_mask(paths, "er", "A/1/0")
    assert loaded is not None
    assert loaded.dtype == bool
    assert loaded.shape == masks.shape
    np.testing.assert_array_equal(loaded, masks)


def test_read_mask_missing_plate_returns_none(tmp_path: Path) -> None:
    """Reading a mask from a non-existent plate returns None (not an error)."""
    paths = cache_paths(tmp_path)
    assert read_mask(paths, "er", "A/1/0") is None


def test_read_mask_missing_position_returns_none(tmp_path: Path) -> None:
    """A position absent from an existing plate returns None."""
    paths = cache_paths(tmp_path)
    masks = np.zeros((2, 3, 4, 4), dtype=bool)
    write_mask(paths, "er", "A/1/0", masks)
    assert read_mask(paths, "er", "A/2/0") is None


def test_write_mask_multiple_positions_same_plate(tmp_path: Path) -> None:
    """Appending a second position to an existing plate preserves the first."""
    paths = cache_paths(tmp_path)
    m0 = np.ones((1, 2, 3, 3), dtype=bool)
    m1 = np.zeros((1, 2, 3, 3), dtype=bool)
    write_mask(paths, "er", "A/1/0", m0)
    write_mask(paths, "er", "A/1/1", m1)

    np.testing.assert_array_equal(read_mask(paths, "er", "A/1/0"), m0)
    np.testing.assert_array_equal(read_mask(paths, "er", "A/1/1"), m1)


def test_write_mask_recovers_from_malformed_position(tmp_path: Path) -> None:
    """``write_mask`` rewrites a position whose inner array metadata is missing.

    Simulates the failure mode where a prior eval crashed mid-position-write,
    leaving the position group on disk with the chunk file present but no
    inner ``0/zarr.json``. The next ``write_mask`` call must clean up the
    orphan and produce a position that round-trips cleanly through
    ``read_mask``, without touching unrelated wells.
    """
    paths = cache_paths(tmp_path)
    m_keep = np.ones((1, 2, 3, 3), dtype=bool)
    write_mask(paths, "er", "1/100/01", m_keep)
    write_mask(paths, "er", "1/200/02", m_keep)
    write_mask(paths, "er", "1/300/03", m_keep)
    plate_path = paths.mask_plate("er")
    broken_pos_dir = plate_path / "1" / "300" / "03"
    inner_meta = broken_pos_dir / "0" / "zarr.json"
    stale_chunk = broken_pos_dir / "0" / "c" / "0" / "0" / "0" / "0" / "0"
    assert inner_meta.exists() and stale_chunk.exists()
    inner_meta.unlink()

    m_new = np.zeros((1, 2, 3, 3), dtype=bool)
    write_mask(paths, "er", "1/300/03", m_new)
    np.testing.assert_array_equal(read_mask(paths, "er", "1/300/03"), m_new)
    np.testing.assert_array_equal(read_mask(paths, "er", "1/100/01"), m_keep)
    np.testing.assert_array_equal(read_mask(paths, "er", "1/200/02"), m_keep)


@pytest.mark.parametrize(
    ("kind", "extras"),
    [
        ("cp", {}),
        ("dinov3", {"model_name": "facebook/dinov3-vitl16"}),
        ("dynaclr", {"ckpt_sha12": "abcdef012345"}),
    ],
)
def test_write_and_read_features_roundtrip(tmp_path: Path, kind: str, extras: dict) -> None:
    """Feature arrays round-trip per (position, timepoint) key."""
    paths = cache_paths(tmp_path)
    feats = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    write_features(paths, kind, "A/1/0", 0, feats, **extras)

    loaded = read_features(paths, kind, "A/1/0", 0, **extras)
    assert loaded is not None
    np.testing.assert_array_equal(loaded, feats)


def test_read_features_missing_returns_none(tmp_path: Path) -> None:
    """Unwritten (position, timepoint) reads back as None."""
    paths = cache_paths(tmp_path)
    feats = np.zeros((2, 4), dtype=np.float32)
    write_features(paths, "cp", "A/1/0", 0, feats)

    assert read_features(paths, "cp", "A/1/0", 1) is None  # same pos, different t
    assert read_features(paths, "cp", "A/1/1", 0) is None  # different pos
    assert read_features(paths, "cp", "A/1/0", 0) is not None  # sanity


def test_write_features_empty_cells(tmp_path: Path) -> None:
    """Zero-cell timepoint is stored as an empty array and distinguishable from missing."""
    paths = cache_paths(tmp_path)
    empty = np.zeros((0, 8), dtype=np.float32)
    write_features(paths, "cp", "A/1/0", 5, empty)

    loaded = read_features(paths, "cp", "A/1/0", 5)
    assert loaded is not None
    assert loaded.shape == (0, 8)


def test_write_features_overwrites_existing(tmp_path: Path) -> None:
    """Re-writing the same key replaces the previous value."""
    paths = cache_paths(tmp_path)
    write_features(paths, "cp", "A/1/0", 0, np.ones((2, 3), dtype=np.float32))
    write_features(paths, "cp", "A/1/0", 0, np.full((4, 3), 7.0, dtype=np.float32))

    loaded = read_features(paths, "cp", "A/1/0", 0)
    assert loaded is not None
    np.testing.assert_array_equal(loaded, np.full((4, 3), 7.0, dtype=np.float32))


def test_write_features_invalid_kind_raises(tmp_path: Path) -> None:
    """Unknown feature kind is rejected."""
    paths = cache_paths(tmp_path)
    with pytest.raises(ValueError, match="Unknown feature kind"):
        write_features(paths, "bogus", "A/1/0", 0, np.zeros((1, 1)))


def test_write_features_dinov3_requires_model_name(tmp_path: Path) -> None:
    """DINOv3 cache key needs a model name."""
    paths = cache_paths(tmp_path)
    with pytest.raises(ValueError, match="model_name is required"):
        write_features(paths, "dinov3", "A/1/0", 0, np.zeros((1, 1)))


def test_write_features_dynaclr_requires_ckpt_sha(tmp_path: Path) -> None:
    """DynaCLR cache key needs a checkpoint hash."""
    paths = cache_paths(tmp_path)
    with pytest.raises(ValueError, match="ckpt_sha12 is required"):
        write_features(paths, "dynaclr", "A/1/0", 0, np.zeros((1, 1)))


def test_write_features_rejects_wrong_ndim(tmp_path: Path) -> None:
    """Features must be 2-D (n_cells, feature_dim)."""
    paths = cache_paths(tmp_path)
    with pytest.raises(ValueError, match="must be 2-D"):
        write_features(paths, "cp", "A/1/0", 0, np.zeros((3,)))


def test_write_mask_rejects_wrong_ndim(tmp_path: Path) -> None:
    """Masks must be 4-D (T, D, H, W)."""
    paths = cache_paths(tmp_path)
    with pytest.raises(ValueError, match="must be 4-D"):
        write_mask(paths, "er", "A/1/0", np.zeros((2, 3, 4), dtype=bool))


def test_ckpt_sha256_12(tmp_path: Path) -> None:
    """Returns the first 12 hex chars of sha256; differs for different content."""
    file_a = tmp_path / "a.ckpt"
    file_b = tmp_path / "b.ckpt"
    file_a.write_bytes(b"model-weights-a")
    file_b.write_bytes(b"model-weights-b")

    h_a = ckpt_sha256_12(file_a)
    h_b = ckpt_sha256_12(file_b)
    assert len(h_a) == 12
    assert len(h_b) == 12
    assert h_a != h_b
    assert ckpt_sha256_12(file_a) == h_a  # deterministic


def test_ckpt_sha256_12_writes_and_reuses_sidecar(tmp_path: Path, monkeypatch) -> None:
    """First call writes ``<ckpt>.sha256``; second call skips the hash."""
    import hashlib as _hashlib

    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    h1 = ckpt_sha256_12(ckpt)
    sidecar = tmp_path / "last.ckpt.sha256"
    assert sidecar.exists()
    written = sidecar.read_text().strip()
    assert written[:12] == h1
    assert len(written) == 64

    calls = {"n": 0}
    real_sha256 = _hashlib.sha256

    def tracking_sha256(*args, **kwargs):
        calls["n"] += 1
        return real_sha256(*args, **kwargs)

    monkeypatch.setattr("dynacell.evaluation.cache.hashlib.sha256", tracking_sha256)
    h2 = ckpt_sha256_12(ckpt)
    assert h2 == h1
    assert calls["n"] == 0


def test_ckpt_sha256_12_recomputes_when_sidecar_older(tmp_path: Path) -> None:
    """Newer ckpt mtime invalidates the sidecar and forces a recompute."""
    import os

    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights-v1")
    h1 = ckpt_sha256_12(ckpt)

    ckpt.write_bytes(b"weights-v2")
    sidecar = tmp_path / "last.ckpt.sha256"
    old = sidecar.stat().st_mtime
    os.utime(ckpt, (old + 10, old + 10))

    h2 = ckpt_sha256_12(ckpt)
    assert h2 != h1
    assert sidecar.read_text().strip()[:12] == h2


def test_ckpt_sha256_12_ignores_corrupt_sidecar(tmp_path: Path) -> None:
    """Non-hex sidecar is treated as missing and recomputed."""
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    sidecar = tmp_path / "last.ckpt.sha256"
    sidecar.write_text("not-a-hex-digest\n")
    # Match ckpt mtime so the mtime check passes and we exercise the hex guard.
    import os

    st = ckpt.stat()
    os.utime(sidecar, (st.st_mtime, st.st_mtime))

    h = ckpt_sha256_12(ckpt)
    assert all(c in "0123456789abcdef" for c in h)
    assert len(h) == 12
    assert sidecar.read_text().strip()[:12] == h


def test_ckpt_sha256_12_read_only_dir(tmp_path: Path) -> None:
    """Read-only parent dir does not raise; digest still returned."""
    import os

    ckpt_dir = tmp_path / "frozen"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "last.ckpt"
    ckpt.write_bytes(b"weights")
    os.chmod(ckpt_dir, 0o555)
    try:
        h = ckpt_sha256_12(ckpt)
        assert len(h) == 12
        assert not (ckpt_dir / "last.ckpt.sha256").exists()
    finally:
        os.chmod(ckpt_dir, 0o755)


def test_encoder_config_sha256_12_key_order_invariant() -> None:
    """Dict key ordering does not change the hash — sorted JSON serialization."""
    cfg_a = {"z_window_size": 15, "num_blocks": 6}
    cfg_b = {"num_blocks": 6, "z_window_size": 15}
    assert encoder_config_sha256_12(cfg_a) == encoder_config_sha256_12(cfg_b)


def test_encoder_config_sha256_12_differs_on_value_change() -> None:
    """Different values produce different hashes."""
    cfg_a = {"patch_size": 256}
    cfg_b = {"patch_size": 128}
    assert encoder_config_sha256_12(cfg_a) != encoder_config_sha256_12(cfg_b)
