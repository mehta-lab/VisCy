"""Training integration tests for dynacell models.

Validates forward+backward pass using ``Trainer(fast_dev_run=True)``.
"""

import importlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from iohub.ngff import open_ome_zarr
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from dynacell.engine import DynacellFlowMatching, DynacellGAN, DynacellUNet
from viscy_data.hcs import HCSDataModule
from viscy_utils.callbacks.prediction_writer import HCSPredictionWriter
from viscy_utils.compose import load_composed_config
from viscy_utils.losses import MixedLoss, SegAuxDice, SpotlightLoss
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

UNEXT2_TEST_CONFIG = {
    "in_channels": 1,
    "out_channels": 1,
    "in_stack_depth": 5,
    "backbone": "convnextv2_tiny",
    "stem_kernel_size": [5, 4, 4],
    "decoder_mode": "pixelshuffle",
    "head_expansion_ratio": 4,
    "head_pool": True,
}

CELLDIFF_TEST_NET_CONFIG = {
    "input_spatial_size": [8, 32, 32],
    "in_channels": 1,
    "dims": [8, 16],
    "num_res_block": [1],
    "hidden_size": 32,
    "num_heads": 2,
    "dim_head": 16,
    "num_hidden_layers": 1,
    "patch_size": 4,
}

CELLDIFF_TEST_TRANSPORT_CONFIG = {"path_type": "Linear", "prediction": "velocity"}

# Discriminator for GAN tests, sized against VIT_TEST_CONFIG's 32x32 YX.
# num_scales=1 (single-scale, no downsampled second pass) is required here:
# MultiScalePatchGAN3D's second scale operates on a (1, 2, 2)-avg-pooled
# 16x16 input, which collapses to 1x1x1 by its fourth conv and raises in
# InstanceNorm3d. num_scales=2 needs >=64x64 (see test_engine.py's
# GAN_DISC_TEST_CONFIG), which tiny_hcs_zarr's 32x32 images don't provide.
GAN_DISC_TEST_CONFIG = {
    "in_channels": 2,
    "base_channels": 8,
    "num_scales": 1,
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


def test_unext2_fast_dev_run(tmp_path, _SyntheticDataModule):
    """DynacellUNet + UNeXt2 trains for 1 batch (YX=64 required)."""
    seed_everything(42)
    module = DynacellUNet(
        architecture="UNeXt2",
        model_config=UNEXT2_TEST_CONFIG,
        log_batches_per_epoch=1,
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=_SyntheticDataModule(depth=5, height=64, width=64))
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
        yx_patch_size=(32, 32),
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
    # The fused scalar cannot say whether the Dice term contributes anything,
    # which is what the lambda_mse choice rests on. Mirrors the GAN path's
    # ``loss/g_recon_*_train`` assertions below.
    assert "loss/masked_mse_train" in trainer.callback_metrics
    assert "loss/dice_train" in trainer.callback_metrics
    assert torch.isfinite(trainer.callback_metrics["loss/masked_mse_train"])
    assert torch.isfinite(trainer.callback_metrics["loss/dice_train"])


def test_dynacell_unet_logged_components_reconstruct_the_fused_loss():
    """The logged terms are the ones actually summed, not a re-derivation.

    Re-deriving masked_mse/dice in the engine would let them drift from the
    objective being optimized while still looking plausible on a dashboard.
    Pinning ``lambda*mse + (1-lambda)*dice == total`` catches that fork.
    """
    seed_everything(0)
    lambda_mse = 0.5
    module = DynacellUNet(
        architecture="FNet3D",
        model_config=FNET_TEST_CONFIG,
        loss_function=SpotlightLoss(lambda_mse=lambda_mse, sigmoid_k=-0.95),
    )
    pred = torch.randn(2, 1, 4, 32, 32)
    target = torch.randn(2, 1, 4, 32, 32)
    mask = torch.zeros(2, 1, 4, 32, 32)
    mask[..., :16, :] = 1.0
    total, components = module._compute_loss(pred, target, {"fg_mask": mask})
    assert set(components) == {"masked_mse", "dice"}
    expected = lambda_mse * components["masked_mse"] + (1 - lambda_mse) * components["dice"]
    torch.testing.assert_close(total, expected)


def test_dynacell_unet_plain_criterion_reports_no_components():
    """A criterion without ``return_components`` still returns the pair, empty."""
    module = DynacellUNet(
        architecture="FNet3D",
        model_config=FNET_TEST_CONFIG,
        loss_function=torch.nn.MSELoss(),
    )
    pred = torch.randn(2, 1, 4, 32, 32)
    target = torch.randn(2, 1, 4, 32, 32)
    loss, components = module._compute_loss(pred, target, {})
    assert components == {}
    torch.testing.assert_close(loss, F.mse_loss(pred, target))


# ---- DynacellGAN + swappable recon_loss (CPU) ----


def test_dynacell_gan_spotlight_recon_fast_dev_run(tmp_path, tiny_hcs_zarr):
    """DynacellGAN + UNetViT3D + SpotlightLoss recon_loss trains for 1 batch.

    Builds the store the same way as ``test_spotlight_with_fg_mask_fast_dev_run``
    (real ``generate_fg_masks`` output read through ``HCSDataModule``'s
    ``fg_mask_key``), so the mask reaches ``DynacellGAN._compute_recon`` through
    the real data path rather than a synthetic batch.
    """
    generate_fg_masks(tiny_hcs_zarr, channel_names=["Fluorescence"])
    seed_everything(42)
    module = DynacellGAN(
        architecture="UNetViT3D",
        generator_config=VIT_TEST_CONFIG,
        discriminator_config=GAN_DISC_TEST_CONFIG,
        recon_loss=SpotlightLoss(lambda_mse=0.5, sigmoid_k=-0.95),
        log_batches_per_epoch=0,
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
        fg_mask_key="fg_mask",
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=datamodule)
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"
    assert "loss/g_recon_masked_mse_train" in trainer.callback_metrics
    assert "loss/g_recon_dice_train" in trainer.callback_metrics
    assert torch.isfinite(trainer.callback_metrics["loss/g_recon_masked_mse_train"])
    assert torch.isfinite(trainer.callback_metrics["loss/g_recon_dice_train"])


def test_dynacell_gan_default_recon_matches_l1():
    """No recon_loss configured computes exactly ``F.l1_loss``, not just an
    ``isinstance`` check on the default."""
    seed_everything(0)
    module = DynacellGAN(
        architecture="UNetViT3D",
        generator_config=VIT_TEST_CONFIG,
        discriminator_config=GAN_DISC_TEST_CONFIG,
    )
    pred = torch.randn(2, 1, 8, 32, 32)
    target = torch.randn(2, 1, 8, 32, 32)
    recon, components = module._compute_recon(pred, target, {})
    assert components == {}
    assert torch.equal(recon, F.l1_loss(pred, target))


def test_dynacell_gan_default_recon_rejects_fg_mask():
    """Default L1 recon_loss + a batch carrying fg_mask raises TypeError
    naming the loss class, matching DynacellUNet's ``_compute_loss`` contract."""
    module = DynacellGAN(
        architecture="UNetViT3D",
        generator_config=VIT_TEST_CONFIG,
        discriminator_config=GAN_DISC_TEST_CONFIG,
    )
    pred = torch.randn(2, 1, 8, 32, 32)
    target = torch.randn(2, 1, 8, 32, 32)
    batch = {"fg_mask": torch.ones(2, 1, 8, 32, 32)}
    with pytest.raises(TypeError, match="L1Loss"):
        module._compute_recon(pred, target, batch)


def test_dynacell_gan_ckpt_excludes_recon_loss_from_hparams(tmp_path, _SyntheticDataModule):
    """SpotlightLoss must not leak into the checkpoint's ``hyper_parameters``.

    An "it reloads fine" assertion alone would pass even if ``ignore=["recon_loss"]``
    were dropped from ``save_hyperparameters``: SpotlightLoss holds only plain
    floats, so it round-trips through pickle regardless. This asserts directly
    against the saved hparams dict, and separately confirms the checkpoint loads
    under ``torch.load(..., weights_only=True)`` -- what
    ``dynacell.engine._ckpt_state_dict`` uses -- which raises if any class (a
    pickled loss module) made it into the file.
    """
    seed_everything(0)
    module = DynacellGAN(
        architecture="UNetViT3D",
        generator_config=VIT_TEST_CONFIG,
        discriminator_config=GAN_DISC_TEST_CONFIG,
        recon_loss=SpotlightLoss(lambda_mse=0.5, sigmoid_k=-0.95),
        log_batches_per_epoch=0,
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=_SyntheticDataModule(depth=8, height=32, width=32))
    ckpt_path = tmp_path / "gan.ckpt"
    trainer.save_checkpoint(ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    assert "recon_loss" not in checkpoint["hyper_parameters"]

    checkpoint_weights_only = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert "recon_loss" not in checkpoint_weights_only["hyper_parameters"]


# ---- SegAuxDice auxiliary term (Spotlight v2, CPU) ----

FNET2D_TEST_CONFIG = {**FNET_TEST_CONFIG, "in_stack_depth": 1, "downsample_z": False}

FCMAE2D_TEST_CONFIG = {
    "in_channels": 1,
    "out_channels": 1,
    "encoder_blocks": [1, 1, 1, 1],
    "dims": [8, 16, 32, 64],
    "decoder_conv_blocks": 1,
    "stem_kernel_size": [1, 2, 2],
    "in_stack_depth": 1,
    "pretraining": False,
}

FCMAE3D_TEST_CONFIG = {**FCMAE2D_TEST_CONFIG, "stem_kernel_size": [5, 4, 4], "in_stack_depth": 5}

# (architecture, model_config, z_window_size) per DynacellUNet family in the v2 grid.
SEG_AUX_UNET_CASES = {
    "fnet3d": ("FNet3D", FNET_TEST_CONFIG, 4),
    "fnet2d": ("FNet3D", FNET2D_TEST_CONFIG, 1),
    "fcmae2d": ("fcmae", FCMAE2D_TEST_CONFIG, 1),
    "fcmae3d": ("fcmae", FCMAE3D_TEST_CONFIG, 5),
}


def _masked_datamodule(zarr_path: Path, z_window_size: int, fg_mask: bool = True) -> HCSDataModule:
    """Real HCSDataModule over ``tiny_hcs_zarr``, optionally reading ``generate_fg_masks`` output."""
    return HCSDataModule(
        data_path=str(zarr_path),
        source_channel="Phase3D",
        target_channel="Fluorescence",
        z_window_size=z_window_size,
        batch_size=2,
        num_workers=0,
        split_ratio=0.5,
        yx_patch_size=(32, 32),
        fg_mask_key="fg_mask" if fg_mask else None,
    )


def _cpu_trainer(tmp_path: Path | None = None, **kwargs) -> Trainer:
    return Trainer(
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path) if tmp_path is not None else False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **kwargs,
    )


def _assert_finite_metrics(metrics: dict, keys: list[str]) -> None:
    for key in keys:
        assert key in metrics, f"{key} not logged; got {sorted(metrics)}"
        assert torch.isfinite(metrics[key]), f"{key}={metrics[key]}"


@pytest.mark.parametrize("case", list(SEG_AUX_UNET_CASES))
def test_unet_seg_aux_fast_dev_run(tmp_path, tiny_hcs_zarr, case):
    """DynacellUNet + SegAuxDice trains on real fg_mask output and logs each term."""
    architecture, model_config, z_window_size = SEG_AUX_UNET_CASES[case]
    generate_fg_masks(tiny_hcs_zarr, channel_names=["Fluorescence"])
    seed_everything(42)
    module = DynacellUNet(
        architecture=architecture,
        model_config=model_config,
        loss_function=MixedLoss(l1_alpha=0.5, l2_alpha=0.5, ms_dssim_alpha=0.0),
        seg_aux=SegAuxDice(c=0.1),
        seg_aux_weight=0.5,
        log_batches_per_epoch=1,
    )
    trainer = _cpu_trainer(tmp_path, fast_dev_run=True)
    trainer.fit(module, datamodule=_masked_datamodule(tiny_hcs_zarr, z_window_size))
    assert trainer.state.status == "finished"
    metrics = trainer.callback_metrics
    _assert_finite_metrics(
        metrics,
        ["loss/train", "loss/base_train", "loss/dice_train", "loss/validate", "loss/validate_dice"],
    )
    assert metrics["loss/dice_n_valid_train"] > 0
    assert 0.0 < metrics["loss/dice_train"] < 1.0


def test_unet_seg_aux_none_is_bit_identical_to_the_base_path(tiny_hcs_zarr):
    """seg_aux=None leaves both the plain-MSE and the v1 SpotlightLoss paths
    computing exactly what they compute without the new arguments."""
    seed_everything(0)
    batch = {"source": torch.randn(2, 1, 4, 16, 16), "target": torch.randn(2, 1, 4, 16, 16)}
    mask = torch.zeros(2, 1, 4, 16, 16)
    mask[..., :8, :] = 1.0

    mse_module = DynacellUNet(architecture="FNet3D", model_config=FNET_TEST_CONFIG, seg_aux=None)
    pred = mse_module(batch["source"])
    assert torch.equal(mse_module.training_step(batch, 99), F.mse_loss(pred, batch["target"]))

    spotlight = SpotlightLoss(lambda_mse=0.5, sigmoid_k=-0.95)
    v1_module = DynacellUNet(architecture="FNet3D", model_config=FNET_TEST_CONFIG, loss_function=spotlight)
    v1_module.load_state_dict(mse_module.state_dict())
    v1_batch = {**batch, "fg_mask": mask}
    assert torch.equal(v1_module.training_step(v1_batch, 99), spotlight(pred, batch["target"], fg_mask=mask))


def test_unet_seg_aux_weight_scales_only_the_aux_term():
    """``w=0`` reproduces the baseline loss bit for bit; ``w>0`` adds exactly
    ``w * SegAuxDice``, and the base loss never receives the mask."""
    seed_everything(0)
    target = torch.randn(2, 1, 4, 16, 16)
    mask = (target > 0.5).float()
    batch = {"source": torch.randn(2, 1, 4, 16, 16), "target": target, "fg_mask": mask}
    baseline = DynacellUNet(architecture="FNet3D", model_config=FNET_TEST_CONFIG)
    pred = baseline(batch["source"])
    base = F.mse_loss(pred, target)
    dice = SegAuxDice()(pred, target, mask)
    for weight in (0.0, 0.7):
        module = DynacellUNet(
            architecture="FNet3D", model_config=FNET_TEST_CONFIG, seg_aux=SegAuxDice(), seg_aux_weight=weight
        )
        module.load_state_dict(baseline.state_dict())
        loss = module.training_step(batch, 99)
        if weight == 0.0:
            assert torch.equal(loss, base)
        else:
            torch.testing.assert_close(loss, base + weight * dice)


def test_seg_aux_rejects_mask_consuming_base_loss_and_orphan_weight():
    with pytest.raises(ValueError, match="SpotlightLoss accepts fg_mask"):
        DynacellUNet(
            architecture="FNet3D",
            model_config=FNET_TEST_CONFIG,
            loss_function=SpotlightLoss(),
            seg_aux=SegAuxDice(),
            seg_aux_weight=1.0,
        )
    with pytest.raises(ValueError, match="no effect without seg_aux"):
        DynacellUNet(architecture="FNet3D", model_config=FNET_TEST_CONFIG, seg_aux_weight=1.0)
    with pytest.raises(ValueError, match="no effect without seg_aux"):
        DynacellFlowMatching(net_config=CELLDIFF_TEST_NET_CONFIG, seg_aux_weight=1.0)


def test_seg_aux_requires_fg_mask_in_batch():
    module = DynacellUNet(
        architecture="FNet3D", model_config=FNET_TEST_CONFIG, seg_aux=SegAuxDice(), seg_aux_weight=1.0
    )
    batch = {"source": torch.randn(2, 1, 4, 16, 16), "target": torch.randn(2, 1, 4, 16, 16)}
    with pytest.raises(KeyError, match="fg_mask_key"):
        module.training_step(batch, 0)


def test_unet_logged_validate_excludes_dice(tiny_hcs_zarr):
    """``loss/validate`` of a seg_aux arm equals the baseline's on identical
    weights, so ``ModelCheckpoint(monitor="loss/validate")`` keeps its meaning."""
    generate_fg_masks(tiny_hcs_zarr, channel_names=["Fluorescence"])
    seed_everything(0)
    baseline = DynacellUNet(architecture="FNet3D", model_config=FNET_TEST_CONFIG, log_batches_per_epoch=0)
    arm = DynacellUNet(
        architecture="FNet3D",
        model_config=FNET_TEST_CONFIG,
        seg_aux=SegAuxDice(),
        seg_aux_weight=5.0,
        log_batches_per_epoch=0,
    )
    arm.load_state_dict(baseline.state_dict())
    # Same seed: HCSDataModule's train/val FOV split is random.
    seed_everything(7)
    base_metrics = _cpu_trainer().validate(baseline, datamodule=_masked_datamodule(tiny_hcs_zarr, 4, fg_mask=False))[0]
    seed_everything(7)
    arm_metrics = _cpu_trainer().validate(arm, datamodule=_masked_datamodule(tiny_hcs_zarr, 4))[0]
    assert "loss/validate_dice" not in base_metrics
    assert arm_metrics["loss/validate_dice"] > 0.0
    assert arm_metrics["loss/validate"] == pytest.approx(base_metrics["loss/validate"], rel=1e-6)


def test_gan_seg_aux_fast_dev_run(tmp_path, tiny_hcs_zarr):
    """DynacellGAN + SegAuxDice (with EMA) trains on real fg_mask output and
    logs the Dice apart from the reconstruction monitors."""
    generate_fg_masks(tiny_hcs_zarr, channel_names=["Fluorescence"])
    seed_everything(42)
    module = DynacellGAN(
        architecture="UNetViT3D",
        generator_config=VIT_TEST_CONFIG,
        discriminator_config=GAN_DISC_TEST_CONFIG,
        ema_kimg=10.0,
        seg_aux=SegAuxDice(),
        seg_aux_weight=0.5,
        log_batches_per_epoch=0,
    )
    trainer = _cpu_trainer(tmp_path, fast_dev_run=True)
    trainer.fit(module, datamodule=_masked_datamodule(tiny_hcs_zarr, 8))
    assert trainer.state.status == "finished"
    _assert_finite_metrics(
        trainer.callback_metrics,
        [
            "loss/g_train",
            "loss/g_l1_train",
            "loss/g_dice_train",
            "loss/validate",
            "loss/validate_ema",
            "loss/validate_dice",
            "loss/validate_ema_dice",
        ],
    )
    assert trainer.callback_metrics["loss/g_dice_n_valid_train"] > 0


def test_gan_seg_aux_recon_ignores_mask_and_validate_stays_l1(tiny_hcs_zarr):
    """With seg_aux, ``_compute_recon`` is plain L1 even when the batch carries
    fg_mask (no TypeError), and ``loss/validate`` matches the baseline GAN."""
    generate_fg_masks(tiny_hcs_zarr, channel_names=["Fluorescence"])
    seed_everything(0)
    kwargs = {
        "architecture": "UNetViT3D",
        "generator_config": VIT_TEST_CONFIG,
        "discriminator_config": GAN_DISC_TEST_CONFIG,
        "log_batches_per_epoch": 0,
    }
    baseline = DynacellGAN(**kwargs)
    arm = DynacellGAN(**kwargs, seg_aux=SegAuxDice(), seg_aux_weight=5.0)
    arm.load_state_dict(baseline.state_dict())

    pred = torch.randn(2, 1, 8, 32, 32)
    target = torch.randn(2, 1, 8, 32, 32)
    recon, components = arm._compute_recon(pred, target, {"fg_mask": torch.ones_like(target)})
    assert components == {}
    assert torch.equal(recon, F.l1_loss(pred, target))

    # Same seed: HCSDataModule's train/val FOV split is random.
    seed_everything(7)
    base_metrics = _cpu_trainer().validate(baseline, datamodule=_masked_datamodule(tiny_hcs_zarr, 8, fg_mask=False))[0]
    seed_everything(7)
    arm_metrics = _cpu_trainer().validate(arm, datamodule=_masked_datamodule(tiny_hcs_zarr, 8))[0]
    assert arm_metrics["loss/validate_dice"] > 0.0
    assert arm_metrics["loss/validate"] == pytest.approx(base_metrics["loss/validate"], rel=1e-6)


def test_flow_matching_seg_aux_fast_dev_run(tmp_path, tiny_hcs_zarr):
    """DynacellFlowMatching + SegAuxDice trains and validates on real fg_mask output."""
    generate_fg_masks(tiny_hcs_zarr, channel_names=["Fluorescence"])
    seed_everything(42)
    module = DynacellFlowMatching(
        net_config=CELLDIFF_TEST_NET_CONFIG,
        transport_config=CELLDIFF_TEST_TRANSPORT_CONFIG,
        compute_validation_loss=True,
        num_log_steps=2,
        log_batches_per_epoch=1,
        seg_aux=SegAuxDice(),
        seg_aux_weight=0.5,
        seg_aux_t0=1.0,  # ungated, so the single fast_dev_run batch always contributes
    )
    trainer = _cpu_trainer(tmp_path, fast_dev_run=True)
    trainer.fit(module, datamodule=_masked_datamodule(tiny_hcs_zarr, 8))
    assert trainer.state.status == "finished"
    metrics = trainer.callback_metrics
    _assert_finite_metrics(
        metrics,
        ["loss/train", "loss/base_train", "loss/dice_train", "loss/validate", "loss/validate_dice"],
    )
    assert metrics["loss/dice_n_valid_train"] > 0
    assert metrics["loss/dice_n_gated_train"] == 2


def test_celldiff_seg_aux_velocity_loss_unchanged_and_gate():
    """The velocity loss is the same draw with or without fg_mask, the aux term
    uses x1_hat = x_t + (1 - t) v_hat, and samples below the gate add nothing."""
    seed_everything(0)
    module = DynacellFlowMatching(
        net_config=CELLDIFF_TEST_NET_CONFIG,
        transport_config=CELLDIFF_TEST_TRANSPORT_CONFIG,
        seg_aux=SegAuxDice(),
        seg_aux_weight=1.0,
        seg_aux_t0=0.7,
    )
    model = module.model
    phase = torch.randn(4, 1, 8, 32, 32)
    target = torch.randn(4, 1, 8, 32, 32)
    mask = (target > 0.5).float()

    torch.manual_seed(1)
    plain = model(phase, target)
    torch.manual_seed(1)
    velocity, aux = model(phase, target, mask)
    assert torch.equal(plain, velocity)

    # Reproduce the draw to check x1_hat and the gate against the definition.
    torch.manual_seed(1)
    t, x0, x1 = model.transport.sample(target)
    t, xt, _ = model.transport.path_sampler.plan(t, x0, x1)
    with torch.no_grad():
        v_hat = model.net(xt, phase, t)
    x1_hat = xt + (1 - t).view(-1, 1, 1, 1, 1) * v_hat
    dice, valid = SegAuxDice().per_channel(x1_hat, target, mask)
    gate = (t >= 0.3).unsqueeze(1)
    expected = (dice * valid * gate).sum() / valid.sum()
    torch.testing.assert_close(aux["dice"], expected)
    assert aux["n_gated"] == gate.sum()

    # t0 so small that no draw passes: the term is a connected zero.
    model.seg_aux_t0 = 1e-6
    torch.manual_seed(1)
    _, aux = model(phase, target, mask)
    assert aux["dice"].item() == 0.0
    assert aux["dice"].requires_grad


def test_celldiff_seg_aux_rejects_non_linear_velocity():
    with pytest.raises(ValueError, match="Linear"):
        DynacellFlowMatching(
            net_config=CELLDIFF_TEST_NET_CONFIG,
            transport_config={"path_type": "GVP", "prediction": "velocity"},
            seg_aux=SegAuxDice(),
            seg_aux_weight=1.0,
        )


def test_flow_matching_logged_validate_excludes_dice(tiny_hcs_zarr):
    """``loss/validate`` stays velocity-only: same weights and seed as the
    baseline give the same value, with the Dice logged apart."""
    generate_fg_masks(tiny_hcs_zarr, channel_names=["Fluorescence"])
    kwargs = {
        "net_config": CELLDIFF_TEST_NET_CONFIG,
        "transport_config": CELLDIFF_TEST_TRANSPORT_CONFIG,
        "compute_validation_loss": True,
    }
    seed_everything(0)
    baseline = DynacellFlowMatching(**kwargs)
    arm = DynacellFlowMatching(**kwargs, seg_aux=SegAuxDice(), seg_aux_weight=5.0, seg_aux_t0=1.0)
    arm.load_state_dict(baseline.state_dict())
    seed_everything(7)
    base_metrics = _cpu_trainer().validate(baseline, datamodule=_masked_datamodule(tiny_hcs_zarr, 8))[0]
    seed_everything(7)
    arm_metrics = _cpu_trainer().validate(arm, datamodule=_masked_datamodule(tiny_hcs_zarr, 8))[0]
    assert "loss/validate_dice" not in base_metrics
    assert arm_metrics["loss/validate_dice"] > 0.0
    assert arm_metrics["loss/validate"] == pytest.approx(base_metrics["loss/validate"], rel=1e-6)


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


def test_unext2_predict_integration(tmp_path, tiny_hcs_zarr):
    """DynacellUNet + UNeXt2 runs predict and writes predictions to OME-Zarr."""
    seed_everything(42)
    module = DynacellUNet(architecture="UNeXt2", model_config=UNEXT2_TEST_CONFIG)
    datamodule = HCSDataModule(
        data_path=str(tiny_hcs_zarr),
        source_channel="Phase3D",
        target_channel="Fluorescence",
        z_window_size=5,
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


# ---- Flow-matching integration tests (CPU) ----


def test_celldiff_fm_warmup_cosine_fast_dev_run(tmp_path, _SyntheticDataModule):
    """DynacellFlowMatching + WarmupCosine trains for 1 batch."""
    seed_everything(42)
    module = DynacellFlowMatching(
        net_config=CELLDIFF_TEST_NET_CONFIG,
        transport_config=CELLDIFF_TEST_TRANSPORT_CONFIG,
        lr=1e-4,
        schedule="WarmupCosine",
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


def test_celldiff_fm_constant_schedule_fast_dev_run(tmp_path, _SyntheticDataModule):
    """DynacellFlowMatching + Constant schedule trains for 1 batch."""
    seed_everything(42)
    module = DynacellFlowMatching(
        net_config=CELLDIFF_TEST_NET_CONFIG,
        transport_config=CELLDIFF_TEST_TRANSPORT_CONFIG,
        lr=1e-4,
        schedule="Constant",
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


def test_celldiff_fm_validation_loss_keeps_generation(tmp_path, _SyntheticDataModule, monkeypatch):
    """Validation loss can be enabled without disabling validation sample generation."""
    seed_everything(42)
    module = DynacellFlowMatching(
        net_config=CELLDIFF_TEST_NET_CONFIG,
        transport_config=CELLDIFF_TEST_TRANSPORT_CONFIG,
        lr=1e-4,
        schedule="Constant",
        log_batches_per_epoch=1,
        log_samples_per_batch=1,
        num_log_steps=2,
        compute_validation_loss=True,
    )
    generate_calls: list[tuple[tuple[int, ...], int]] = []

    def fake_generate(phase, num_steps=100):
        generate_calls.append((tuple(phase.shape), num_steps))
        return phase.new_zeros(phase.shape)

    monkeypatch.setattr(module.model, "generate", fake_generate)

    trainer = Trainer(
        accelerator="cpu",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=_SyntheticDataModule(depth=8, height=32, width=32))
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"
    assert "loss/validate" in trainer.callback_metrics
    assert torch.isfinite(trainer.callback_metrics["loss/validate"])
    assert generate_calls == [((1, 1, 8, 32, 32), 2)]


def test_celldiff_fm_predict_integration(tmp_path, tiny_hcs_zarr):
    """DynacellFlowMatching runs predict and writes predictions to OME-Zarr."""
    seed_everything(42)
    module = DynacellFlowMatching(
        net_config=CELLDIFF_TEST_NET_CONFIG,
        transport_config=CELLDIFF_TEST_TRANSPORT_CONFIG,
        num_generate_steps=2,
        predict_method="generate",
    )
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
    configs_dir = Path(__file__).resolve().parents[1] / "configs" / "examples"
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
