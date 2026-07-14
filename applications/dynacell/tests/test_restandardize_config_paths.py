"""Tests for the predict/train config-leaf path codemod (restandardize_config_paths)."""

from __future__ import annotations

import os

import restandardize_config_paths as rc

from dynacell.evaluation import paths

MODELS = str(paths.MODELS_ROOT)
CELLDIFF = str(paths.MODELS_ROOT.parent / "cell_diff_vs_viscy")
DATA = str(paths.DATA_ROOT)

# A migrated a549 ER checkpoint (recipe-suffixed run dir) -> canonical raw a549 home.
_OLD_CKPT_DIR = f"{MODELS}/a549_mantis/sec61b/fcmae_vscyto3d_pretrained_ws8500"
_NEW_CKPT_DIR = f"{MODELS}/a549/er/fcmae_vscyto3d_pretrained"
_CKPT_MAP = {_OLD_CKPT_DIR: _NEW_CKPT_DIR}


def test_rewrite_predict_leaf_deconv_ck_vs_pred_split():
    """ER/mito A549 predict: ckpt -> raw a549 home, prediction -> a549__deconv home."""
    text = (
        "predict:\n"
        "  model:\n"
        "    init_args:\n"
        f"      ckpt_path: {_OLD_CKPT_DIR}/checkpoints/epoch=132-step=22876.ckpt\n"
        "  trainer:\n"
        "    callbacks:\n"
        "      - init_args:\n"
        f"          output_store: {DATA}/a549/predictions/sec61b_fcmae_vscyto3d_pretrained_a549trained_mock.zarr\n"
        "  launcher:\n"
        f"    run_root: {DATA}/a549/predictions\n"
    )
    new_text, changes = rc.rewrite_leaf(text, _CKPT_MAP)
    assert f"ckpt_path: {_NEW_CKPT_DIR}/checkpoints/epoch=132-step=22876.ckpt" in new_text
    assert f"output_store: {DATA}/er/fcmae_vscyto3d_pretrained/a549__deconv/a549__mock/prediction.zarr" in new_text
    # predict run_root -> canonical prediction leaf dir (output_store parent)
    assert f"run_root: {DATA}/er/fcmae_vscyto3d_pretrained/a549__deconv/a549__mock" in new_text
    assert len(changes) == 3


def test_rewrite_train_leaf_fields():
    """Train: save_dir/dirpath/run_root -> canonical model dir; init ckpt + data_path preserved."""
    text = (
        "model:\n"
        "  init_args:\n"
        "    ckpt_path: /hpc/projects/virtual_staining/models/mehta-lab/VSCyto3D/fcmae.ckpt\n"
        "data:\n"
        "  init_args:\n"
        f"    data_path: {DATA}/a549/mantis_v1/train/SEC61B_all.zarr\n"
        "trainer:\n"
        "  logger:\n"
        "    init_args:\n"
        f"      save_dir: {_OLD_CKPT_DIR}\n"
        "  callbacks:\n"
        "    - init_args:\n"
        f"        dirpath: {_OLD_CKPT_DIR}/checkpoints\n"
        "launcher:\n"
        f"  run_root: {_OLD_CKPT_DIR}\n"
    )
    new_text, changes = rc.rewrite_leaf(text, _CKPT_MAP)
    assert f"save_dir: {_NEW_CKPT_DIR}\n" in new_text
    assert f"dirpath: {_NEW_CKPT_DIR}/checkpoints\n" in new_text
    assert f"run_root: {_NEW_CKPT_DIR}\n" in new_text
    # Preserved: pretrained init + training-input data_path.
    assert "ckpt_path: /hpc/projects/virtual_staining/models/mehta-lab/VSCyto3D/fcmae.ckpt" in new_text
    assert f"data_path: {DATA}/a549/mantis_v1/train/SEC61B_all.zarr" in new_text
    assert len(changes) == 3


def test_rewrite_ckpt_run_root_alias_no_checkpoints_subdir():
    """A best-epoch alias directly at the run-dir root (no ``checkpoints/`` subdir) is
    repointed to the canonical model dir + alias filename, not silently preserved.

    Regression: several legacy iPSC runs store the best ckpt as a bare alias at the run
    root; ``value.find('/checkpoints')`` returns -1, and the old code let ``model_dir``
    swallow the filename so the map lookup missed and the migrated-away path survived.
    """
    old_dir = f"{MODELS}/ipsc/sec61b/fcmae_vscyto3d_pretrained_ws8500"
    new_dir = f"{MODELS}/ipsc/er/fcmae_vscyto3d_pretrained"
    ckpt_map = {old_dir: new_dir}
    text = f"model:\n  init_args:\n    ckpt_path: {old_dir}/best_ep123_val0.40979.ckpt\n"
    new_text, changes = rc.rewrite_leaf(text, ckpt_map)
    assert f"ckpt_path: {new_dir}/best_ep123_val0.40979.ckpt" in new_text
    assert len(changes) == 1


def test_preserve_unmigrated_and_external_ckpt():
    """ckpt_path not in the map (iPSC-trained / published baseline / REPLACE_ME) -> untouched."""
    ipsc = f"{MODELS}/ipsc/nucl/fcmae_vscyto3d_pretrained/checkpoints/epoch=89-step=28080.ckpt"
    published = "/hpc/projects/comp.micro/virtual_staining/datasets/public/VS_models/VSCyto3D/epoch=83.ckpt"
    for val in (ipsc, published, "REPLACE_ME_WITH_PRODUCTION_CHECKPOINT_PATH"):
        text = f"model:\n  init_args:\n    ckpt_path: {val}\n"
        new_text, changes = rc.rewrite_leaf(text, _CKPT_MAP)
        assert changes == []
        assert new_text == text


def test_prediction_skip_left_untouched():
    """An ablation output_store (normalize_legacy -> None) is preserved, not blanked."""
    val = f"{DATA}/a549/predictions/nucl_vscyto3d_randinit_mock.zarr"
    text = f"trainer:\n  callbacks:\n    - init_args:\n        output_store: {val}\n"
    new_text, changes = rc.rewrite_leaf(text, _CKPT_MAP)
    assert changes == []
    assert new_text == text


def test_cross_root_celldiff_joint():
    """celldiff_r2 joint ER: cross-root ckpt -> dynacell/joint/er; prediction -> joint__legacy_deconvgt."""
    old_ckpt = f"{CELLDIFF}/joint_ipsc_confocal_a549_mantis/sec61b/celldiff_r2"
    ckpt_map = {old_ckpt: f"{MODELS}/joint/er/celldiff_r2"}
    text = (
        "model:\n  init_args:\n"
        f"    ckpt_path: {old_ckpt}/checkpoints/last.ckpt\n"
        "trainer:\n  callbacks:\n    - init_args:\n"
        f"        output_store: {DATA}/a549/joint_predictions/sec61b_celldiff_r2_denv.zarr\n"
    )
    new_text, _ = rc.rewrite_leaf(text, ckpt_map)
    assert f"ckpt_path: {MODELS}/joint/er/celldiff_r2/checkpoints/last.ckpt" in new_text
    assert f"output_store: {DATA}/er/celldiff_r2/joint__legacy_deconvgt/a549__denv/prediction.zarr" in new_text


def test_build_ckpt_map_on_fixture(tmp_path):
    """build_ckpt_map returns old-dir -> dest for both move winner and dedup_legacy siblings."""
    models_root = tmp_path / "models" / "dynacell"
    (models_root).mkdir(parents=True)

    def mk(root_name, term, org, run):
        base = models_root if root_name == "dynacell" else models_root.parent / root_name
        d = base / term / org / run / "checkpoints"
        d.mkdir(parents=True)
        (d / "last.ckpt").write_text("c")
        return d.parent

    # Both names are canonical pix2pix keeps (clean + Run D), so they hardlink-dedup
    # rather than being pruned as pre-D experiments.
    a = mk("dynacell", "joint_ipsc_confocal_a549_mantis", "memb", "pix2pix3d_unetvit")
    b = mk(
        "cell_diff_vs_viscy",
        "joint_ipsc_confocal_a549_mantis",
        "memb",
        "pix2pix3d_unetvit_modernized_lambdaL1_10_lecam_40ep",
    )
    # hardlink b's ckpt to a's so the two dedup to one dest
    os.remove(b / "checkpoints" / "last.ckpt")
    os.link(a / "checkpoints" / "last.ckpt", b / "checkpoints" / "last.ckpt")

    cmap = rc.build_ckpt_map(models_root)
    dest = f"{models_root}/joint/membrane/pix2pix3d_unetvit"
    assert cmap[str(a)] == dest
    assert cmap[str(b)] == dest  # dedup_legacy sibling maps to the same dest


def test_load_ckpt_map_from_manifest(tmp_path):
    """After migration the live tree is gone; the frozen manifest supplies the map.

    Only checkpoint move/dedup_legacy rows contribute; predictions/evals/skips do not.
    """
    old_dedup = f"{CELLDIFF}/joint_ipsc_confocal_a549_mantis/memb/pix2pix3d_unetvit_modernized_lambdaL1_10_lecam_40ep"
    new_dedup = f"{MODELS}/joint/membrane/pix2pix3d_unetvit"
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "kind,status,src,dest,reason\n"
        f"checkpoint,move,{_OLD_CKPT_DIR},{_NEW_CKPT_DIR},raw a549 home\n"
        f"checkpoint,dedup_legacy,{old_dedup},{new_dedup},cross-root dup\n"
        f"checkpoint,skip,{MODELS}/ipsc/nucl/celldiff/checkpoints,,superseded R1\n"
        "prediction,move,/data/a/x.zarr,/data/b/prediction.zarr,pred\n"
        "eval,move,/data/a/eval_x,/data/b/leaf,eval\n"
    )
    cmap = rc.load_ckpt_map_from_manifest(manifest)
    assert cmap == {_OLD_CKPT_DIR: _NEW_CKPT_DIR, old_dedup: new_dedup}
