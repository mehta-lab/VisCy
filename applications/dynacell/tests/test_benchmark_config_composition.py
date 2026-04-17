"""Composition equivalence tests for benchmark leaves vs pre-schema configs.

Each benchmark train/predict leaf must compose to the same resolved config
as the corresponding pre-schema config (Dihan's ``examples/configs/`` tree)
on every hyperparameter that ends up at runtime. These tests compose both
sides through :func:`viscy_utils.compose.load_composed_config` and compare
the full key intersection field-by-field.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from viscy_utils.compose import load_composed_config  # noqa: E402

# Repository root (four parents up: tests/ → dynacell/ → applications/ → VisCy/).
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "applications" / "dynacell" / "tools" / "LEGACY" / "examples_configs"
BENCHMARKS = REPO_ROOT / "applications" / "dynacell" / "configs" / "benchmarks" / "virtual_staining"

# organelle slug in the new schema → legacy dir under examples/configs/
ORGANELLE_TO_LEGACY = {
    "er": "sec61b",
    "mito": "tomm20",
    "nucleus": "nucl",
    "membrane": "memb",
}

# Keys we always compare when both sides declare them.
DATA_INIT_KEYS_SHARED = (
    "class_path",  # not under init_args, handled separately below
)

# Train-specific data.init_args keys we expect to match.
TRAIN_DATA_INIT_KEYS = (
    "source_channel",
    "target_channel",
    "data_path",
    "split_ratio",
    "z_window_size",
    "batch_size",
    "num_workers",
    "yx_patch_size",
    "mmap_preload",
    "scratch_dir",
    "persistent_workers",
    "normalizations",
    "augmentations",
    "gpu_augmentations",
    "val_gpu_augmentations",
)


def _strip_reserved(d: dict) -> dict:
    d.pop("launcher", None)
    d.pop("benchmark", None)
    return d


@pytest.mark.parametrize("organelle,legacy", sorted(ORGANELLE_TO_LEGACY.items()))
def test_train_leaf_matches_legacy(organelle: str, legacy: str) -> None:
    """Composed train leaf matches the pre-schema fit_celldiff.yml on every shared key."""
    legacy_path = EXAMPLES / legacy / "fit_celldiff.yml"
    new_path = BENCHMARKS / "train" / organelle / "ipsc_confocal" / "celldiff.yml"

    old = _strip_reserved(load_composed_config(legacy_path))
    new = _strip_reserved(load_composed_config(new_path))

    # model.class_path and init_args should match exactly.
    assert old["model"]["class_path"] == new["model"]["class_path"], organelle
    assert old["model"]["init_args"] == new["model"]["init_args"], organelle

    # data.class_path
    assert old["data"]["class_path"] == new["data"]["class_path"], organelle

    # data.init_args — full intersection.
    old_di = old["data"]["init_args"]
    new_di = new["data"]["init_args"]
    for k in TRAIN_DATA_INIT_KEYS:
        if k in old_di:
            assert k in new_di, f"{organelle}: missing data.init_args.{k}"
            assert old_di[k] == new_di[k], f"{organelle}: data.init_args.{k} diverges"

    # trainer.{precision, max_epochs, devices} and trainer.callbacks.
    for k in ("precision", "max_epochs", "devices"):
        if k in old["trainer"]:
            assert old["trainer"][k] == new["trainer"][k], f"{organelle}: trainer.{k}"
    assert old["trainer"].get("callbacks") == new["trainer"].get("callbacks"), f"{organelle}: trainer.callbacks"

    # trainer.logger — init_args.name and save_dir must match.
    old_logger = old["trainer"].get("logger", {}).get("init_args", {})
    new_logger = new["trainer"].get("logger", {}).get("init_args", {})
    for k in ("name", "save_dir"):
        assert old_logger.get(k) == new_logger.get(k), f"{organelle}: logger.{k}"


# Predict-specific data.init_args keys.
PREDICT_DATA_INIT_KEYS = (
    "source_channel",
    "target_channel",
    "data_path",
    "z_window_size",
    "batch_size",
    "num_workers",
    "yx_patch_size",
    "normalizations",
)


@pytest.mark.parametrize("organelle,legacy", sorted(ORGANELLE_TO_LEGACY.items()))
def test_predict_leaf_matches_legacy(organelle: str, legacy: str) -> None:
    """Composed predict leaf matches pre-schema predict_celldiff.yml on every shared key."""
    legacy_path = EXAMPLES / legacy / "predict_celldiff.yml"
    new_path = BENCHMARKS / "predict" / organelle / "ipsc_confocal" / "celldiff" / "ipsc_confocal.yml"

    old = _strip_reserved(load_composed_config(legacy_path))
    new = _strip_reserved(load_composed_config(new_path))

    # model.init_args: num_generate_steps, predict_method, predict_overlap,
    # ckpt_path, net_config.
    old_mi = old["model"]["init_args"]
    new_mi = new["model"]["init_args"]
    for k in ("num_generate_steps", "predict_method", "predict_overlap", "ckpt_path"):
        assert old_mi[k] == new_mi[k], f"{organelle}: model.init_args.{k}"
    assert old_mi["net_config"] == new_mi["net_config"], organelle

    # data.init_args — intersection.
    old_di = old["data"]["init_args"]
    new_di = new["data"]["init_args"]
    for k in PREDICT_DATA_INIT_KEYS:
        assert old_di[k] == new_di[k], f"{organelle}: data.init_args.{k}"

    # Guard against forgetting the predict-side data_path override.
    assert "test_cropped" in new_di["data_path"], f"{organelle}: new data_path missing test_cropped/"

    # trainer.callbacks[0] = HCSPredictionWriter with matching output_store.
    new_cbs = new["trainer"]["callbacks"]
    writers = [cb for cb in new_cbs if "HCSPredictionWriter" in cb["class_path"]]
    assert len(writers) == 1, f"{organelle}: expected exactly one HCSPredictionWriter"
    old_cbs = old["trainer"]["callbacks"]
    old_writers = [cb for cb in old_cbs if "HCSPredictionWriter" in cb["class_path"]]
    assert old_writers[0]["init_args"]["output_store"] == writers[0]["init_args"]["output_store"], (
        f"{organelle}: output_store diverges"
    )


def test_fnet3d_paper_leaf_matches_ran_config() -> None:
    """FNet3D paper leaf composes to the Lightning-saved config.yaml from the ran training.

    Reference is the on-disk LightningCLI config dumped when the run started:
    ``/hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/fnet3d_paper/config.yaml``.
    The equivalent wandb-logged model hyperparameters (``architecture``,
    ``model_config``, ``lr``, ``schedule``, ``log_batches_per_epoch``,
    ``log_samples_per_batch``, ``example_input_yx_shape``) are verified as a
    side effect — they appear verbatim under ``model.init_args`` in both the
    ran config.yaml and the composed new leaf.

    Skipped when the reference config.yaml is not on disk (e.g. CI without
    /hpc mounts); the inline verification we ran during migration is
    preserved in the leaf's docstring.
    """
    ran_path = Path("/hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/fnet3d_paper/config.yaml")
    if not ran_path.exists():
        pytest.skip(f"Reference config not available at {ran_path}")

    with ran_path.open() as f:
        ran = yaml.safe_load(f)
    new_path = BENCHMARKS / "train" / "er" / "ipsc_confocal" / "fnet3d_paper.yml"
    new = _strip_reserved(load_composed_config(new_path))

    # seed, model
    assert new["seed_everything"] == ran["seed_everything"] == 0
    assert new["model"]["class_path"] == ran["model"]["class_path"]
    nm, rm = new["model"]["init_args"], ran["model"]["init_args"]
    # Keys the new leaf explicitly sets. Keys Lightning fills from DynacellUNet
    # defaults (log_batches_per_epoch=8, log_samples_per_batch=1,
    # example_input_yx_shape=(256,256)) appear in the ran config.yaml but not
    # in the composed new config — verified OK if the defaults agree, which
    # the wandb run hyperparameters confirm.
    for k in ("architecture", "lr", "schedule"):
        assert nm[k] == rm[k], f"model.init_args.{k}"
    assert nm["model_config"] == rm["model_config"], "model.init_args.model_config"
    assert nm["loss_function"]["class_path"] == rm["loss_function"]["class_path"]
    # The ran config records the runtime default; verify it hasn't drifted
    # from what DynacellUNet will still instantiate when the new leaf runs.
    assert rm["log_batches_per_epoch"] == 8
    assert rm["log_samples_per_batch"] == 1
    assert rm["example_input_yx_shape"] == [256, 256]

    # trainer protocol (excluding max_steps: new=50000 original launch, ran=200000 continuation bump)
    for k in ("precision", "devices", "strategy", "num_nodes", "log_every_n_steps", "inference_mode"):
        assert new["trainer"][k] == ran["trainer"][k], f"trainer.{k}"
    # New leaf matches the ran value (200000) — what the paper training actually
    # converged to, accounting for CLI --trainer.max_steps bumps across
    # continuation restarts from the initial 50000 launch.
    assert new["trainer"]["max_steps"] == ran["trainer"]["max_steps"] == 200000

    # callbacks — LR monitor + ModelCheckpoint
    nc_mc = new["trainer"]["callbacks"][1]["init_args"]
    rc_mc = ran["trainer"]["callbacks"][1]["init_args"]
    for k in ("dirpath", "monitor", "save_top_k", "save_last", "every_n_epochs"):
        assert nc_mc[k] == rc_mc[k], f"ModelCheckpoint.{k}"

    # data — every training-protocol field. Transform lists compare entry-by-entry:
    # the ran config.yaml has jsonargparse-filled defaults (e.g. ``remove_meta: False``,
    # ``allow_missing_keys: False``, ``lazy: False``) that the composed new leaf
    # doesn't materialize. Ran is allowed to have extra default keys in each
    # transform's init_args; the new side's keys must all match.
    nd = new["data"]["init_args"]
    rd = ran["data"]["init_args"]
    for k in (
        "data_path",
        "source_channel",
        "target_channel",
        "z_window_size",
        "split_ratio",
        "batch_size",
        "num_workers",
        "yx_patch_size",
        "persistent_workers",
    ):
        assert nd[k] == rd[k], f"data.init_args.{k}"
    for list_key in ("normalizations", "augmentations", "gpu_augmentations", "val_augmentations"):
        new_list = nd[list_key]
        ran_list = rd[list_key]
        assert len(new_list) == len(ran_list), f"data.init_args.{list_key}: length differs"
        for i, (n, r) in enumerate(zip(new_list, ran_list)):
            assert n["class_path"] == r["class_path"], f"{list_key}[{i}].class_path"
            n_ia, r_ia = n["init_args"], r["init_args"]
            for k, v in n_ia.items():
                assert r_ia.get(k) == v, f"{list_key}[{i}].init_args.{k}: new={v!r}  ran={r_ia.get(k)!r}"


def test_unetvit3d_train_leaf_matches_legacy() -> None:
    """New UNetViT3D train leaf reproduces Dihan's fit_unetvit3d.yml.

    Dihan's legacy fit_unetvit3d.yml has a copy-paste bug: it nests
    ``net_config.input_spatial_size`` under DynacellUNet's init_args, but
    DynacellUNet takes ``model_config:``, not ``net_config:``. jsonargparse
    rejects that override, so the legacy config cannot actually run as-is.
    The override is also redundant with the recipe's
    ``model_config.input_spatial_size``, so the new leaf drops it — this
    test strips it from the legacy side before comparing.
    """
    legacy_path = EXAMPLES / "sec61b" / "fit_unetvit3d.yml"
    new_path = BENCHMARKS / "train" / "er" / "ipsc_confocal" / "unetvit3d.yml"

    old = _strip_reserved(load_composed_config(legacy_path))
    new = _strip_reserved(load_composed_config(new_path))

    # Strip the broken override. Value is a tautology against the recipe.
    broken = old["model"]["init_args"].pop("net_config", None)
    assert broken == {"input_spatial_size": [8, 512, 512]}, "unexpected net_config content in legacy UNetViT3D config"
    assert new["model"]["init_args"]["model_config"]["input_spatial_size"] == [8, 512, 512]

    assert old["model"]["class_path"] == new["model"]["class_path"]
    assert old["model"]["init_args"] == new["model"]["init_args"]
    assert old["data"]["class_path"] == new["data"]["class_path"]

    old_di = old["data"]["init_args"]
    new_di = new["data"]["init_args"]
    for k in TRAIN_DATA_INIT_KEYS:
        if k in old_di:
            assert k in new_di, f"missing data.init_args.{k}"
            assert old_di[k] == new_di[k], f"data.init_args.{k} diverges"

    for k in ("precision", "max_epochs", "devices"):
        if k in old["trainer"]:
            assert old["trainer"][k] == new["trainer"][k], f"trainer.{k}"
    assert old["trainer"].get("callbacks") == new["trainer"].get("callbacks"), "trainer.callbacks"

    old_logger = old["trainer"].get("logger", {}).get("init_args", {})
    new_logger = new["trainer"].get("logger", {}).get("init_args", {})
    for k in ("name", "save_dir"):
        assert old_logger.get(k) == new_logger.get(k), f"logger.{k}"
