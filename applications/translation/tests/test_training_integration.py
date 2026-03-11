"""Training integration tests for translation models.

Validates that the forward+backward pass works for translation modules
using ``fast_dev_run=True`` (1 batch of train + val). Follows the DynaCLR
``test_training_integration.py`` pattern.

Synthetic tests use lightweight random data and always run on CPU.
Real integration tests exercise the full data-to-model pipeline with a
tiny HCS OME-Zarr fixture.
"""

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
