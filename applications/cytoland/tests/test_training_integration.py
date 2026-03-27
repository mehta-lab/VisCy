"""Training integration tests for cytoland models.

Validates that the forward+backward pass works for cytoland modules
using ``fast_dev_run=True`` (1 batch of train + val). Follows the DynaCLR
``test_training_integration.py`` pattern.

Synthetic tests use lightweight random data and always run on CPU.
Real integration tests exercise the full data-to-model pipeline with a
tiny HCS OME-Zarr fixture.
"""

import importlib
from pathlib import Path

import pytest
import torch
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from cytoland.engine import FcmaeUNet, MaskedMSELoss, VSUNet
from viscy_data.combined import CombinedDataModule
from viscy_data.gpu_aug import CachedOmeZarrDataModule
from viscy_data.hcs import HCSDataModule
from viscy_transforms import BatchedStackChannelsd
from viscy_utils.compose import load_composed_config
from viscy_utils.losses import MixedLoss, SpotlightLoss
from viscy_utils.meta_utils import generate_fg_masks

# ---------------------------------------------------------------------------
# Synthetic tests (CPU, always run)
# ---------------------------------------------------------------------------


def test_vsunet_fast_dev_run(tmp_path, _SyntheticHCSDataModule, synth_dims):
    """VSUNet + UNeXt2 + MSELoss trains for 1 batch."""
    seed_everything(42)
    module = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": 1, "out_channels": 1, "in_stack_depth": synth_dims["d"]},
        log_batches_per_epoch=1,
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=_SyntheticHCSDataModule())
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def test_vsunet_mixed_loss_fast_dev_run(tmp_path, _SyntheticHCSDataModule, synth_dims):
    """VSUNet + UNeXt2 + MixedLoss (L1 + MS-DSSIM) trains for 1 batch."""
    seed_everything(42)
    module = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": 1, "out_channels": 1, "in_stack_depth": synth_dims["d"]},
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
        datamodule=_SyntheticHCSDataModule(height=synth_dims["mixed_loss_h"], width=synth_dims["mixed_loss_w"]),
    )
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def test_fnet3d_fast_dev_run(tmp_path, _SyntheticHCSDataModule):
    """VSUNet + FNet3D + MSELoss trains for 1 batch."""
    seed_everything(42)
    module = VSUNet(
        architecture="FNet3D",
        model_config={
            "in_channels": 1,
            "out_channels": 1,
            "depth": 1,
            "mult_chan": 8,
            "in_stack_depth": 4,
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
    trainer.fit(module, datamodule=_SyntheticHCSDataModule(depth=4))
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def test_spotlight_with_fg_mask_fast_dev_run(tmp_path, tiny_hcs_zarr):
    """VSUNet + FNet3D + SpotlightLoss with precomputed fg_mask trains for 1 batch."""

    # Fixture already has otsu_threshold in norm_meta; just generate masks
    generate_fg_masks(tiny_hcs_zarr, channel_names=["Fluorescence"])

    seed_everything(42)
    module = VSUNet(
        architecture="FNet3D",
        model_config={
            "in_channels": 1,
            "out_channels": 1,
            "depth": 1,
            "mult_chan": 8,
            "in_stack_depth": 4,
        },
        loss_function=SpotlightLoss(lambda_mse=0.5, sigmoid_k=-0.95),
        log_batches_per_epoch=1,
    )
    datamodule = HCSDataModule(
        data_path=tiny_hcs_zarr,
        source_channel="Phase3D",
        target_channel="Fluorescence",
        z_window_size=4,
        batch_size=2,
        num_workers=0,
        yx_patch_size=(32, 32),
        fg_mask_key="fg_mask",
        split_ratio=0.5,
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


def test_spotlight_fast_dev_run(tmp_path, _SyntheticHCSDataModule):
    """VSUNet + FNet3D + SpotlightLoss trains for 1 batch."""

    seed_everything(42)
    module = VSUNet(
        architecture="FNet3D",
        model_config={
            "in_channels": 1,
            "out_channels": 1,
            "depth": 1,
            "mult_chan": 8,
            "in_stack_depth": 4,
        },
        loss_function=SpotlightLoss(lambda_mse=0.5, sigmoid_k=-0.95),
        log_batches_per_epoch=1,
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=_SyntheticHCSDataModule(depth=4))
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def test_fcmae_pretrain_fast_dev_run(tmp_path, _make_synthetic_combined_datamodule, synth_dims):
    """FcmaeUNet FCMAE pretraining (MaskedMSELoss) trains for 1 batch."""
    seed_everything(42)
    module = FcmaeUNet(
        model_config={"in_channels": 1, "out_channels": 1, "in_stack_depth": synth_dims["d"]},
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
    trainer.fit(module, datamodule=_make_synthetic_combined_datamodule())
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def test_fcmae_finetune_fast_dev_run(tmp_path, _make_synthetic_combined_datamodule, synth_dims):
    """FcmaeUNet supervised fine-tuning (MSELoss) trains for 1 batch."""
    seed_everything(42)
    module = FcmaeUNet(
        model_config={
            "in_channels": 1,
            "out_channels": 1,
            "in_stack_depth": synth_dims["d"],
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
    trainer.fit(module, datamodule=_make_synthetic_combined_datamodule())
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def test_fcmae_encoder_only_load(tmp_path, synth_dims):
    """FcmaeUNet encoder_only=True loads only encoder weights from a checkpoint."""
    seed_everything(42)
    pretrain_model = FcmaeUNet(
        model_config={"in_channels": 1, "out_channels": 1, "in_stack_depth": synth_dims["d"]},
        loss_function=MaskedMSELoss(),
        fit_mask_ratio=0.5,
    )
    ckpt_path = str(tmp_path / "pretrained.ckpt")
    torch.save({"state_dict": pretrain_model.state_dict()}, ckpt_path)

    # Load encoder-only into a model with different out_channels
    finetune_model = FcmaeUNet(
        model_config={
            "in_channels": 1,
            "out_channels": 2,
            "in_stack_depth": synth_dims["d"],
            "pretraining": False,
        },
        encoder_only=True,
        ckpt_path=ckpt_path,
    )

    # Verify encoder weights match
    for key in pretrain_model.model.encoder.state_dict():
        assert torch.equal(
            pretrain_model.model.encoder.state_dict()[key],
            finetune_model.model.encoder.state_dict()[key],
        ), f"Encoder weight mismatch for key: {key}"

    # Verify forward pass with new out_channels
    x = torch.randn(2, 1, synth_dims["d"], synth_dims["fcmae_h"], synth_dims["fcmae_w"])
    finetune_model.eval()
    with torch.no_grad():
        out = finetune_model(x)
    assert out.shape[1] == 2, f"Expected out_channels=2, got {out.shape[1]}"


def test_fcmae_encoder_only_requires_ckpt():
    """FcmaeUNet encoder_only=True without ckpt_path raises ValueError."""
    with pytest.raises(ValueError, match="encoder_only=True requires ckpt_path"):
        FcmaeUNet(encoder_only=True)


def test_fcmae_finetune_encoder_only_fast_dev_run(tmp_path, _make_synthetic_combined_datamodule, synth_dims):
    """FcmaeUNet fine-tuning with encoder_only=True trains for 1 batch."""
    seed_everything(42)
    pretrain_model = FcmaeUNet(
        model_config={"in_channels": 1, "out_channels": 1, "in_stack_depth": synth_dims["d"]},
        loss_function=MaskedMSELoss(),
        fit_mask_ratio=0.5,
    )
    ckpt_path = str(tmp_path / "pretrained.ckpt")
    torch.save({"state_dict": pretrain_model.state_dict()}, ckpt_path)

    finetune_model = FcmaeUNet(
        model_config={
            "in_channels": 1,
            "out_channels": 1,
            "in_stack_depth": synth_dims["d"],
            "pretraining": False,
        },
        encoder_only=True,
        ckpt_path=ckpt_path,
        log_batches_per_epoch=1,
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(finetune_model, datamodule=_make_synthetic_combined_datamodule())
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


# ---------------------------------------------------------------------------
# Real integration tests (CPU, tiny HCS OME-Zarr)
# ---------------------------------------------------------------------------


def test_vsunet_real_datamodule_fast_dev_run(tmp_path, tiny_hcs_zarr, synth_dims):
    """VSUNet + real HCSDataModule end-to-end training for 1 batch."""

    seed_everything(42)
    module = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": 1, "out_channels": 1, "in_stack_depth": synth_dims["d"]},
        log_batches_per_epoch=1,
    )
    datamodule = HCSDataModule(
        data_path=str(tiny_hcs_zarr),
        source_channel="Phase3D",
        target_channel="Fluorescence",
        z_window_size=synth_dims["d"],
        batch_size=2,
        num_workers=0,
        yx_patch_size=(synth_dims["h"], synth_dims["w"]),
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
    """VSUNet + FNet3D + real HCSDataModule end-to-end training for 1 batch."""

    seed_everything(42)
    module = VSUNet(
        architecture="FNet3D",
        model_config={
            "in_channels": 1,
            "out_channels": 1,
            "depth": 1,
            "mult_chan": 8,
            "in_stack_depth": 4,
        },
        log_batches_per_epoch=1,
    )
    datamodule = HCSDataModule(
        data_path=str(tiny_hcs_zarr),
        source_channel="Phase3D",
        target_channel="Fluorescence",
        z_window_size=4,
        batch_size=2,
        num_workers=0,
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


def test_fcmae_real_datamodule_fast_dev_run(tmp_path, tiny_hcs_zarr, synth_dims):
    """FcmaeUNet + real CachedOmeZarrDataModule + CombinedDataModule for 1 batch."""

    seed_everything(42)
    stack = BatchedStackChannelsd({"source": ["Phase3D"], "target": ["Fluorescence"]})
    dm = CachedOmeZarrDataModule(
        data_path=tiny_hcs_zarr,
        channels=["Phase3D", "Fluorescence"],
        batch_size=2,
        num_workers=0,
        split_ratio=0.5,
        train_cpu_transforms=[],
        val_cpu_transforms=[],
        train_gpu_transforms=[stack],
        val_gpu_transforms=[stack],
        pin_memory=False,
    )
    combined = CombinedDataModule([dm])
    module = FcmaeUNet(
        model_config={
            "in_channels": 1,
            "out_channels": 1,
            "in_stack_depth": synth_dims["d"],
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


def _discover_leaf_configs():
    """Discover all leaf configs (skip recipes/ directory)."""
    configs_dir = Path(__file__).parents[1] / "examples" / "configs"
    leaf_configs = []
    for yml in sorted(configs_dir.rglob("*.yml")):
        if "recipes" not in yml.parts:
            leaf_configs.append(yml)
    return leaf_configs


@pytest.mark.parametrize("config_path", _discover_leaf_configs(), ids=lambda p: str(p.relative_to(p.parents[1])))
def test_config_class_paths_resolve(config_path):
    """All class_path entries in composed example configs resolve to importable classes."""

    assert config_path.exists(), f"Config file not found: {config_path}"
    composed = load_composed_config(config_path)
    class_paths = _extract_class_paths(composed)
    assert len(class_paths) > 0, f"No class_path entries found in {config_path.name}"

    for cp in class_paths:
        cls = _resolve_class_path(cp)
        assert cls is not None, f"Failed to resolve class_path: {cp}"


def test_compose_passthrough_without_base(tmp_path):
    """Config without base: key is returned unchanged."""
    config = {"model": {"class_path": "torch.nn.Identity"}, "data": {"batch_size": 4}}
    config_path = tmp_path / "plain.yml"
    config_path.write_text(yaml.dump(config))
    result = load_composed_config(config_path)
    assert result == config


def test_compose_spotlight_overrides_normalizations():
    """Spotlight mode replaces data recipe's default normalizations."""
    configs_dir = Path(__file__).parents[1] / "examples" / "configs"
    cfg = load_composed_config(configs_dir / "vscyto3d" / "train_spotlight.yml")
    # Spotlight must set Otsu normalization
    norm = cfg["data"]["init_args"]["normalizations"][0]
    assert norm["init_args"]["subtrahend"] == "otsu_threshold"
    # Spotlight must set fg_mask_key
    assert cfg["data"]["init_args"]["fg_mask_key"] == "fg_mask"
    # Spotlight must set loss
    assert cfg["model"]["init_args"]["loss_function"]["class_path"] == "viscy_utils.losses.SpotlightLoss"


def test_cli_compose_with_long_flag(tmp_path):
    """CLI composes leaf config when --config is used."""
    import sys

    from viscy_utils.cli import _maybe_compose_config

    # Create a minimal base recipe
    base_dir = tmp_path / "recipes"
    base_dir.mkdir()
    base_path = base_dir / "base.yml"
    base_path.write_text(yaml.dump({"trainer": {"accelerator": "cpu"}}))
    # Create a leaf config referencing the base
    leaf_path = tmp_path / "leaf.yml"
    leaf_path.write_text(yaml.dump({"base": ["recipes/base.yml"], "seed_everything": 42}))
    # Simulate CLI args
    original_argv = sys.argv[:]
    sys.argv = ["fit", "--config", str(leaf_path)]
    try:
        _maybe_compose_config()
        # sys.argv should now point to a composed temp file
        composed_path = sys.argv[2]
        assert composed_path != str(leaf_path)
        with open(composed_path) as f:
            composed = yaml.safe_load(f)
        assert composed["trainer"]["accelerator"] == "cpu"
        assert composed["seed_everything"] == 42
        assert "base" not in composed
    finally:
        sys.argv = original_argv


def test_cli_compose_with_short_flag(tmp_path):
    """CLI composes leaf config when -c is used."""
    import sys

    from viscy_utils.cli import _maybe_compose_config

    base_path = tmp_path / "base.yml"
    base_path.write_text(yaml.dump({"model": {"lr": 0.001}}))
    leaf_path = tmp_path / "leaf.yml"
    leaf_path.write_text(yaml.dump({"base": ["base.yml"], "model": {"name": "test"}}))
    original_argv = sys.argv[:]
    sys.argv = ["fit", "-c", str(leaf_path)]
    try:
        _maybe_compose_config()
        with open(sys.argv[2]) as f:
            composed = yaml.safe_load(f)
        assert composed["model"]["lr"] == 0.001
        assert composed["model"]["name"] == "test"
    finally:
        sys.argv = original_argv


def test_cli_passthrough_without_base(tmp_path):
    """CLI passes config unchanged when no base: key."""
    import sys

    from viscy_utils.cli import _maybe_compose_config

    config_path = tmp_path / "plain.yml"
    config_path.write_text(yaml.dump({"trainer": {"devices": 1}}))
    original_argv = sys.argv[:]
    sys.argv = ["fit", "--config", str(config_path)]
    try:
        _maybe_compose_config()
        # sys.argv should be unchanged — no temp file created
        assert sys.argv[2] == str(config_path)
    finally:
        sys.argv = original_argv
