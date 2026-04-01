"""Training integration tests for dynacell models.

Validates forward+backward pass using ``Trainer(fast_dev_run=True)``.
"""

import importlib
from pathlib import Path

import pytest
from iohub.ngff import open_ome_zarr
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from dynacell.engine import DynacellUNet
from viscy_data.hcs import HCSDataModule
from viscy_utils.callbacks.prediction_writer import HCSPredictionWriter
from viscy_utils.compose import load_composed_config
from viscy_utils.losses import SpotlightLoss
from viscy_utils.meta_utils import generate_fg_masks

# Small model configs for tests (not production sizes).
VIT_TEST_CONFIG = {
    "input_spatial_size": [8, 32, 32],
    "in_channels": 1,
    "out_channels": 1,
    "dims": [32, 64, 128],
    "num_res_block": [2, 2],
    "hidden_size": 64,
    "num_heads": 4,
    "dim_head": 16,
    "num_hidden_layers": 1,
    "patch_size": 4,
}

FNET_TEST_CONFIG = {
    "in_channels": 1,
    "out_channels": 1,
    "depth": 1,
    "mult_chan": 8,
    "in_stack_depth": 4,
}


# ---- Synthetic tests (CPU) ----


def test_unetvit3d_fast_dev_run(tmp_path, _SyntheticDataModule):
    """DynacellUNet + UNetViT3D trains for 1 batch."""
    seed_everything(42)
    module = DynacellUNet(
        architecture="UNetViT3D",
        model_config=VIT_TEST_CONFIG,
        log_batches_per_epoch=1,
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=_SyntheticDataModule(depth=8, height=32, width=32))
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def test_fnet3d_fast_dev_run(tmp_path, _SyntheticDataModule):
    """DynacellUNet + FNet3D trains for 1 batch."""
    seed_everything(42)
    module = DynacellUNet(
        architecture="FNet3D",
        model_config=FNET_TEST_CONFIG,
        log_batches_per_epoch=1,
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=_SyntheticDataModule(depth=4, height=16, width=16))
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


# ---- Real OME-Zarr tests (CPU) ----


def test_unetvit3d_real_datamodule_fast_dev_run(tmp_path, tiny_hcs_zarr):
    """DynacellUNet + UNetViT3D + real HCSDataModule for 1 batch."""
    seed_everything(42)
    module = DynacellUNet(
        architecture="UNetViT3D",
        model_config=VIT_TEST_CONFIG,
        log_batches_per_epoch=1,
    )
    datamodule = HCSDataModule(
        data_path=str(tiny_hcs_zarr),
        source_channel="Phase3D",
        target_channel="Fluorescence",
        z_window_size=8,
        batch_size=2,
        num_workers=0,
        split_ratio=0.5,
        yx_patch_size=(32, 32),
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=datamodule)
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def test_fnet3d_real_datamodule_fast_dev_run(tmp_path, tiny_hcs_zarr):
    """DynacellUNet + FNet3D + real HCSDataModule for 1 batch."""
    seed_everything(42)
    module = DynacellUNet(
        architecture="FNet3D",
        model_config=FNET_TEST_CONFIG,
        log_batches_per_epoch=1,
    )
    datamodule = HCSDataModule(
        data_path=str(tiny_hcs_zarr),
        source_channel="Phase3D",
        target_channel="Fluorescence",
        z_window_size=4,
        batch_size=2,
        num_workers=0,
        split_ratio=0.5,
        yx_patch_size=(16, 16),
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=datamodule)
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def test_spotlight_with_fg_mask_fast_dev_run(tmp_path, tiny_hcs_zarr):
    """DynacellUNet + FNet3D + SpotlightLoss with fg_mask trains."""
    generate_fg_masks(tiny_hcs_zarr, channel_names=["Fluorescence"])
    seed_everything(42)
    module = DynacellUNet(
        architecture="FNet3D",
        model_config=FNET_TEST_CONFIG,
        loss_function=SpotlightLoss(lambda_mse=0.5, sigmoid_k=-0.95),
        log_batches_per_epoch=1,
    )
    datamodule = HCSDataModule(
        data_path=str(tiny_hcs_zarr),
        source_channel="Phase3D",
        target_channel="Fluorescence",
        z_window_size=4,
        batch_size=2,
        num_workers=0,
        split_ratio=0.5,
        yx_patch_size=(16, 16),
        fg_mask_key="fg_mask",
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=datamodule)
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


# ---- Predict integration tests (CPU) ----


def test_fnet3d_predict_integration(tmp_path, tiny_hcs_zarr):
    """DynacellUNet + FNet3D runs predict and writes predictions to OME-Zarr."""
    seed_everything(42)
    module = DynacellUNet(architecture="FNet3D", model_config=FNET_TEST_CONFIG)
    datamodule = HCSDataModule(
        data_path=str(tiny_hcs_zarr),
        source_channel="Phase3D",
        target_channel="Fluorescence",
        z_window_size=4,
        batch_size=2,
        num_workers=0,
        yx_patch_size=(16, 16),
    )
    output_store = str(tmp_path / "predict_out.zarr")
    writer = HCSPredictionWriter(output_store=output_store)
    trainer = Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[writer],
    )
    trainer.predict(module, datamodule=datamodule, return_predictions=False)
    with open_ome_zarr(output_store, mode="r") as plate:
        positions = list(plate.positions())
    assert len(positions) == 4
    for _, pos in positions:
        assert "Fluorescence_prediction" in pos.channel_names


def test_unetvit3d_predict_integration(tmp_path, tiny_hcs_zarr):
    """DynacellUNet + UNetViT3D runs predict with spatial-matching tiles."""
    seed_everything(42)
    module = DynacellUNet(architecture="UNetViT3D", model_config=VIT_TEST_CONFIG)
    datamodule = HCSDataModule(
        data_path=str(tiny_hcs_zarr),
        source_channel="Phase3D",
        target_channel="Fluorescence",
        z_window_size=8,
        batch_size=2,
        num_workers=0,
        yx_patch_size=(32, 32),
    )
    output_store = str(tmp_path / "predict_out.zarr")
    writer = HCSPredictionWriter(output_store=output_store)
    trainer = Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[writer],
    )
    trainer.predict(module, datamodule=datamodule, return_predictions=False)
    with open_ome_zarr(output_store, mode="r") as plate:
        positions = list(plate.positions())
    assert len(positions) == 4
    for _, pos in positions:
        assert "Fluorescence_prediction" in pos.channel_names


# ---- Config validation tests ----


def _extract_class_paths(obj):
    """Recursively extract all class_path values from a nested dict/list."""
    paths = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "class_path" and isinstance(value, str):
                paths.append(value)
            else:
                paths.extend(_extract_class_paths(value))
    elif isinstance(obj, list):
        for item in obj:
            paths.extend(_extract_class_paths(item))
    return paths


def _resolve_class_path(class_path: str):
    """Resolve a dotted class_path to the actual class object."""
    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _discover_leaf_configs():
    """Discover leaf configs (skip recipes/ directory)."""
    configs_dir = Path(__file__).resolve().parents[1] / "examples" / "configs"
    leaf_configs = []
    for yml in sorted(configs_dir.rglob("*.yml")):
        if "recipes" not in yml.parts:
            leaf_configs.append(yml)
    return leaf_configs


@pytest.mark.parametrize(
    "config_path",
    _discover_leaf_configs(),
    ids=lambda p: str(p.relative_to(p.parents[2])),
)
def test_config_class_paths_resolve(config_path):
    """All class_path entries in composed configs resolve to importable classes."""
    assert config_path.exists()
    composed = load_composed_config(config_path)
    class_paths = _extract_class_paths(composed)
    assert len(class_paths) > 0, f"No class_path entries in {config_path.name}"
    for cp in class_paths:
        cls = _resolve_class_path(cp)
        assert cls is not None, f"Failed to resolve: {cp}"
