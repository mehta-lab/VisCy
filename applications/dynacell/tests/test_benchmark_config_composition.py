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
    "preload",
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
