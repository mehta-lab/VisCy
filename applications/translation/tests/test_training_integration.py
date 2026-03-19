"""Training integration tests for translation models.

Validates that the forward+backward pass works for translation modules
using ``fast_dev_run=True`` (1 batch of train + val). Follows the DynaCLR
``test_training_integration.py`` pattern.

Synthetic tests use lightweight random data and always run on CPU.
Real integration tests exercise the full data-to-model pipeline with a
tiny HCS OME-Zarr fixture.
"""

import importlib
from pathlib import Path

import pytest
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from viscy_translation.engine import FcmaeUNet, MaskedMSELoss, VSUNet
from viscy_utils.losses import MixedLoss

from .conftest import (
    MIXED_LOSS_H,
    MIXED_LOSS_W,
    SYNTH_D,
    SYNTH_H,
    SYNTH_W,
    SyntheticHCSDataModule,
    make_synthetic_combined_datamodule,
)

# ---------------------------------------------------------------------------
# Synthetic tests (CPU, always run)
# ---------------------------------------------------------------------------


def test_vsunet_fast_dev_run(tmp_path):
    """VSUNet + UNeXt2 + MSELoss trains for 1 batch."""
    seed_everything(42)
    module = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": 1, "out_channels": 1, "in_stack_depth": SYNTH_D},
        log_batches_per_epoch=1,
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=SyntheticHCSDataModule())
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def test_vsunet_mixed_loss_fast_dev_run(tmp_path):
    """VSUNet + UNeXt2 + MixedLoss (L1 + MS-DSSIM) trains for 1 batch."""
    seed_everything(42)
    module = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": 1, "out_channels": 1, "in_stack_depth": SYNTH_D},
        loss_function=MixedLoss(l1_alpha=0.5, ms_dssim_alpha=0.5),
        log_batches_per_epoch=1,
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    # 192x192 spatial needed: MS-SSIM kernel 11x11, 5 scales → spatial/16 >= 11.
    trainer.fit(
        module,
        datamodule=SyntheticHCSDataModule(height=MIXED_LOSS_H, width=MIXED_LOSS_W),
    )
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def test_fcmae_pretrain_fast_dev_run(tmp_path):
    """FcmaeUNet FCMAE pretraining (MaskedMSELoss) trains for 1 batch."""
    seed_everything(42)
    module = FcmaeUNet(
        model_config={"in_channels": 1, "out_channels": 1, "in_stack_depth": SYNTH_D},
        loss_function=MaskedMSELoss(),
        fit_mask_ratio=0.5,
        log_batches_per_epoch=1,
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=make_synthetic_combined_datamodule())
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def test_fcmae_finetune_fast_dev_run(tmp_path):
    """FcmaeUNet supervised fine-tuning (MSELoss) trains for 1 batch."""
    seed_everything(42)
    module = FcmaeUNet(
        model_config={
            "in_channels": 1,
            "out_channels": 1,
            "in_stack_depth": SYNTH_D,
            "pretraining": False,
        },
        log_batches_per_epoch=1,
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=make_synthetic_combined_datamodule())
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


# ---------------------------------------------------------------------------
# Real integration tests (CPU, tiny HCS OME-Zarr)
# ---------------------------------------------------------------------------


def test_vsunet_real_datamodule_fast_dev_run(tmp_path, tiny_hcs_zarr):
    """VSUNet + real HCSDataModule end-to-end training for 1 batch."""
    from viscy_data.hcs import HCSDataModule

    seed_everything(42)
    module = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": 1, "out_channels": 1, "in_stack_depth": SYNTH_D},
        log_batches_per_epoch=1,
    )
    datamodule = HCSDataModule(
        data_path=str(tiny_hcs_zarr),
        source_channel="Phase3D",
        target_channel="Fluorescence",
        z_window_size=SYNTH_D,
        batch_size=2,
        num_workers=0,
        yx_patch_size=(SYNTH_H, SYNTH_W),
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


def test_fcmae_real_datamodule_fast_dev_run(tmp_path, tiny_hcs_zarr):
    """FcmaeUNet + real CachedOmeZarrDataModule + CombinedDataModule for 1 batch."""
    from monai.transforms import Decollated

    from viscy_data.combined import CombinedDataModule
    from viscy_data.gpu_aug import CachedOmeZarrDataModule
    from viscy_transforms import StackChannelsd

    seed_everything(42)
    channels = ["Phase3D", "Fluorescence"]
    stack = StackChannelsd({"source": ["Phase3D"], "target": ["Fluorescence"]})
    dm = CachedOmeZarrDataModule(
        data_path=tiny_hcs_zarr,
        channels=channels,
        batch_size=2,
        num_workers=0,
        split_ratio=0.5,
        train_cpu_transforms=[],
        val_cpu_transforms=[],
        train_gpu_transforms=[Decollated(keys=channels), stack],
        val_gpu_transforms=[Decollated(keys=channels), stack],
        pin_memory=False,
    )
    combined = CombinedDataModule([dm])
    module = FcmaeUNet(
        model_config={
            "in_channels": 1,
            "out_channels": 1,
            "in_stack_depth": SYNTH_D,
            "pretraining": False,
        },
        log_batches_per_epoch=1,
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=combined)
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


def _extract_class_paths(obj):
    """Recursively extract all class_path values from a parsed YAML dict."""
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


@pytest.mark.parametrize("config_name", ["fit.yml", "predict.yml"])
def test_config_class_paths_resolve(config_name):
    """All class_path entries in example configs resolve to importable classes."""
    configs_dir = Path(__file__).parents[1] / "examples" / "configs"
    config_path = configs_dir / config_name
    assert config_path.exists(), f"Config file not found: {config_path}"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    class_paths = _extract_class_paths(config)
    assert len(class_paths) > 0, f"No class_path entries found in {config_name}"

    for cp in class_paths:
        cls = _resolve_class_path(cp)
        assert cls is not None, f"Failed to resolve class_path: {cp}"
