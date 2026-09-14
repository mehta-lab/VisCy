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
