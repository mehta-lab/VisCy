"""Tests for the best-checkpoint baking codemod (bake_best_ckpt)."""

from __future__ import annotations

from pathlib import Path

import bake_best_ckpt as bb

from dynacell.evaluation import paths

MODELS = str(paths.MODELS_ROOT)


def _fixed(path):
    """A resolver stub that ignores the dir and returns a fixed Path (or None)."""
    return lambda _d: path


def _boom(_d):
    raise AssertionError("resolver must not run for preserved values")


def test_ckpt_dir_for_standard_and_alias():
    assert bb._ckpt_dir_for(f"{MODELS}/a549/mito/fnet3d_paper/checkpoints/epoch=248-step=1.ckpt") == Path(
        f"{MODELS}/a549/mito/fnet3d_paper/checkpoints"
    )
    # bare best-epoch alias at the model run-dir root -> the alias's own dir
    assert bb._ckpt_dir_for(f"{MODELS}/ipsc/er/fcmae_vscyto3d_pretrained/best_ep123_val0.40979.ckpt") == Path(
        f"{MODELS}/ipsc/er/fcmae_vscyto3d_pretrained"
    )


def test_rebake_stale_filename():
    """A dangling baked epoch is rewritten to the resolver's best (the retrained case)."""
    old = f"{MODELS}/a549/mito/fnet3d_paper/checkpoints/epoch=248-step=999.ckpt"
    new = f"{MODELS}/a549/mito/fnet3d_paper/checkpoints/epoch=170-step=1.ckpt"
    text = f"predict:\n  model:\n    init_args:\n      ckpt_path: {old}\n"
    new_text, changes = bb.bake_leaf(text, _fixed(Path(new)))
    assert f"ckpt_path: {new}\n" in new_text
    assert changes == [(old, new, "rebaked")]


def test_already_best_is_noop():
    val = f"{MODELS}/a549/er/fnet3d_paper/checkpoints/epoch=350-step=1.ckpt"
    text = f"model:\n  init_args:\n    ckpt_path: {val}\n"
    new_text, changes = bb.bake_leaf(text, _fixed(Path(val)))
    assert new_text == text
    assert changes == [(val, val, "already_best")]


def test_replace_me_and_external_preserved():
    published = "/hpc/projects/comp.micro/virtual_staining/datasets/public/VS_models/VSCyto3D/epoch=83.ckpt"
    for val, cat in (("REPLACE_ME_WITH_PRODUCTION_CHECKPOINT_PATH", "replace_me"), (published, "external")):
        text = f"model:\n  init_args:\n    ckpt_path: {val}\n"
        new_text, changes = bb.bake_leaf(text, _boom)
        assert new_text == text
        assert changes == [(val, val, cat)]


def test_canonical_ckpt_dir_for_leaf():
    rel = Path("er/pix2pix3d_unetvit/joint_ipsc_confocal_a549_mantis/predict__a549_mantis_mock.yml")
    assert bb.canonical_ckpt_dir_for_leaf(rel) == Path(f"{MODELS}/joint/er/pix2pix3d_unetvit/checkpoints")
    # organelle/train alias also resolves (sec61b -> er, a549_mantis -> a549)
    rel2 = Path("mito/fnet3d_paper/a549_mantis/predict__a549_mantis_mock.yml")
    assert bb.canonical_ckpt_dir_for_leaf(rel2) == Path(f"{MODELS}/a549/mito/fnet3d_paper/checkpoints")
    # a train term that does not normalize -> None (no grammar fallback)
    assert bb.canonical_ckpt_dir_for_leaf(Path("er/x/_no_train_randinit/predict__a549_mantis_mock.yml")) is None


def test_repoint_when_baked_dir_stale():
    """A baked dir the resolver can't resolve falls back to the canonical dir (path miss)."""
    stale = f"{MODELS}/joint_ipsc_confocal_a549_mantis/sec61b/pix2pix3d_unetvit/checkpoints/last.ckpt"
    canonical = Path(f"{MODELS}/joint/er/pix2pix3d_unetvit/checkpoints")
    best = canonical / "epoch=26-step=118422.ckpt"

    def resolver(d):
        return best if d == canonical else None

    text = f"model:\n  init_args:\n    ckpt_path: {stale}\n"
    new_text, changes = bb.bake_leaf(text, resolver, canonical)
    assert f"ckpt_path: {best}\n" in new_text
    assert changes == [(stale, str(best), "repointed")]


def test_kept_current_vs_unresolvable(tmp_path, monkeypatch):
    """resolver -> None: a curated alias that exists is kept; a missing file is flagged."""
    monkeypatch.setattr(bb, "_MODELS_ROOT_PREFIX", str(tmp_path) + "/")
    existing = tmp_path / "m" / "best_ep1_val0.1.ckpt"
    existing.parent.mkdir(parents=True)
    existing.write_text("c")
    assert bb.resolve_leaf_ckpt(str(existing), _fixed(None)) == (None, "kept_current")
    missing = str(tmp_path / "m" / "gone.ckpt")
    assert bb.resolve_leaf_ckpt(missing, _fixed(None)) == (None, "unresolvable")
