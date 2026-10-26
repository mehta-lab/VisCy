"""SegConsistencyLoss wired into DynacellUNet (FNet3D, UNeXt2-3D) and DynacellGAN (pix2pix2d geometry).

The engine tests use a tiny stand-in segmenter; ``test_cellpose_dense.py`` covers the
Cellpose-SAM one.
"""

from pathlib import Path

import pytest
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, nn

from dynacell.engine import DynacellGAN, DynacellUNet
from viscy_data.hcs import HCSDataModule
from viscy_utils.losses import SegAuxDice, SegConsistencyLoss

FNET_CONFIG = {"in_channels": 1, "out_channels": 1, "depth": 1, "mult_chan": 8, "in_stack_depth": 4}
FCMAE3D_CONFIG = {
    "in_channels": 1,
    "out_channels": 1,
    "encoder_blocks": [1, 1, 1, 1],
    "dims": [8, 16, 32, 64],
    "decoder_conv_blocks": 1,
    "stem_kernel_size": [5, 4, 4],
    "in_stack_depth": 5,
    "pretraining": False,
}
# pix2pix2d geometry: a Z=1 UNetViT generator and a Z-preserving discriminator.
VIT2D_CONFIG = {
    "input_spatial_size": [1, 32, 32],
    "in_channels": 1,
    "out_channels": 1,
    "dims": [32, 64, 128],
    "num_res_block": [2, 2],
    "hidden_size": 64,
    "num_heads": 4,
    "dim_head": 16,
    "num_hidden_layers": 1,
    "patch_size": [1, 4, 4],
}
DISC2D_CONFIG = {"in_channels": 2, "base_channels": 8, "num_scales": 1, "preserve_z": True}
# (architecture, model_config, z_window_size)
UNET_CASES = {"fnet3d": ("FNet3D", FNET_CONFIG, 4), "unext2_3d": ("fcmae", FCMAE3D_CONFIG, 5)}


class TinySegmenter(nn.Module):
    """Two conv layers to flow (2 ch) + cell-probability logit (1 ch), per-image min-max normalized."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.Tanh(), nn.Conv2d(8, 3, 3, padding=1))

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        with torch.no_grad():
            lo = x.amin((1, 2, 3), keepdim=True)
            hi = x.amax((1, 2, 3), keepdim=True)
        y = self.net((x - lo) / (hi - lo + 1e-6))
        return {"flow": y[:, :2], "cellprob_logit": y[:, 2:3]}


def _segcons() -> SegConsistencyLoss:
    return SegConsistencyLoss(TinySegmenter())


def _datamodule(zarr_path: Path, z_window_size: int) -> HCSDataModule:
    return HCSDataModule(
        data_path=str(zarr_path),
        source_channel="Phase3D",
        target_channel="Fluorescence",
        z_window_size=z_window_size,
        batch_size=2,
        num_workers=0,
        split_ratio=0.5,
        yx_patch_size=(32, 32),
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


def _assert_finite(metrics: dict, keys: list[str]) -> None:
    for key in keys:
        assert key in metrics, f"{key} not logged; got {sorted(metrics)}"
        assert torch.isfinite(metrics[key]), f"{key}={metrics[key]}"


@pytest.mark.parametrize("case", list(UNET_CASES))
def test_unet_seg_consistency_fast_dev_run(tmp_path, tiny_hcs_zarr, case):
    architecture, model_config, z_window_size = UNET_CASES[case]
    seed_everything(42)
    module = DynacellUNet(
        architecture=architecture,
        model_config=model_config,
        seg_consistency=_segcons(),
        seg_consistency_weight=0.5,
        log_batches_per_epoch=0,
    )
    trainer = _cpu_trainer(tmp_path, fast_dev_run=True)
    trainer.fit(module, datamodule=_datamodule(tiny_hcs_zarr, z_window_size))
    assert trainer.state.status == "finished"
    metrics = trainer.callback_metrics
    keys = ["loss/train", "loss/base_train", "loss/segcons_train", "loss/segcons_flow_train", "loss/segcons_prob_train"]
    _assert_finite(metrics, [*keys, "loss/validate", "loss/validate_segcons"])
    assert metrics["loss/segcons_train"] > 0
    expected = metrics["loss/base_train"] + 0.5 * metrics["loss/segcons_train"]
    assert float(metrics["loss/train"]) == pytest.approx(float(expected), rel=1e-5)


def test_unet_training_step_gradient_includes_the_term():
    """The step's backbone gradient is the base gradient plus weight x the term's own gradient."""
    seed_everything(0)
    batch = {"source": torch.randn(2, 1, 4, 32, 32), "target": torch.randn(2, 1, 4, 32, 32)}
    term = _segcons()
    baseline = DynacellUNet(architecture="FNet3D", model_config=FNET_CONFIG, log_batches_per_epoch=0)
    arm = DynacellUNet(
        architecture="FNet3D",
        model_config=FNET_CONFIG,
        seg_consistency=term,
        seg_consistency_weight=2.0,
        log_batches_per_epoch=0,
    )
    arm.load_state_dict(baseline.state_dict())

    def grad(module: DynacellUNet, fn) -> Tensor:
        params = list(module.model.parameters())
        return torch.cat([g.flatten() for g in torch.autograd.grad(fn(), params)])

    for module in (baseline, arm):
        module.log = lambda *args, **kw: None
    g_arm = grad(arm, lambda: arm.training_step(batch, 0))
    g_base = grad(baseline, lambda: baseline.training_step(batch, 0))
    g_term = grad(arm, lambda: term(arm(batch["source"]), batch["target"]))
    assert g_term.abs().sum() > 0
    torch.testing.assert_close(g_arm, g_base + 2.0 * g_term)


def test_unet_validate_excludes_the_term(tiny_hcs_zarr):
    """``loss/validate`` matches the baseline's on identical weights, so ``--ckpt best`` keeps its meaning."""
    seed_everything(0)
    baseline = DynacellUNet(architecture="FNet3D", model_config=FNET_CONFIG, log_batches_per_epoch=0)
    arm = DynacellUNet(
        architecture="FNet3D",
        model_config=FNET_CONFIG,
        seg_consistency=_segcons(),
        seg_consistency_weight=5.0,
        log_batches_per_epoch=0,
    )
    arm.load_state_dict(baseline.state_dict())
    seed_everything(7)
    base_metrics = _cpu_trainer().validate(baseline, datamodule=_datamodule(tiny_hcs_zarr, 4))[0]
    seed_everything(7)
    arm_metrics = _cpu_trainer().validate(arm, datamodule=_datamodule(tiny_hcs_zarr, 4))[0]
    assert "loss/validate_segcons" not in base_metrics
    assert arm_metrics["loss/validate_segcons"] > 0.0
    assert arm_metrics["loss/validate"] == pytest.approx(base_metrics["loss/validate"], rel=1e-6)


def test_gan_seg_consistency_fast_dev_run(tmp_path, tiny_hcs_zarr):
    """pix2pix2d geometry (Z=1) with EMA: the term trains and is logged apart from the recon monitors."""
    seed_everything(42)
    module = DynacellGAN(
        architecture="UNetViT3D",
        generator_config=VIT2D_CONFIG,
        discriminator_config=DISC2D_CONFIG,
        ema_kimg=10.0,
        seg_consistency=_segcons(),
        seg_consistency_weight=0.5,
        log_batches_per_epoch=0,
    )
    trainer = _cpu_trainer(tmp_path, fast_dev_run=True)
    trainer.fit(module, datamodule=_datamodule(tiny_hcs_zarr, 1))
    assert trainer.state.status == "finished"
    _assert_finite(
        trainer.callback_metrics,
        [
            "loss/g_train",
            "loss/g_l1_train",
            "loss/g_segcons_train",
            "loss/g_segcons_flow_train",
            "loss/g_segcons_prob_train",
            "loss/validate",
            "loss/validate_ema",
            "loss/validate_segcons",
            "loss/validate_ema_segcons",
        ],
    )
    assert trainer.callback_metrics["loss/g_segcons_train"] > 0


@pytest.mark.parametrize("family", ["unet", "gan"])
def test_weight_is_required_with_the_term_and_seg_aux_is_rejected_with_it(family):
    if family == "unet":
        kwargs = {"architecture": "FNet3D", "model_config": FNET_CONFIG}
        cls = DynacellUNet
    else:
        kwargs = {"generator_config": VIT2D_CONFIG, "discriminator_config": DISC2D_CONFIG}
        cls = DynacellGAN
    with pytest.raises(ValueError, match="without seg_consistency_weight"):
        cls(**kwargs, seg_consistency=_segcons(), seg_consistency_weight=None)
    with pytest.raises(ValueError, match="no effect without seg_consistency"):
        cls(**kwargs, seg_consistency_weight=1.0)
    for weight in (0.0, -1.0):
        with pytest.raises(ValueError, match="> 0"):
            cls(**kwargs, seg_consistency=_segcons(), seg_consistency_weight=weight)
    with pytest.raises(ValueError, match="cannot be combined with seg_aux"):
        cls(**kwargs, seg_consistency=_segcons(), seg_consistency_weight=1.0, seg_aux=SegAuxDice(), seg_aux_weight=1.0)


@pytest.mark.parametrize("family", ["unet", "gan"])
def test_fork_loads_a_baseline_checkpoint_strictly_and_saves_no_segmenter(tmp_path, family):
    """A fine-tune arm starts from a baseline checkpoint via ``ckpt_path`` and writes the same keys."""
    if family == "unet":
        kwargs = {"architecture": "FNet3D", "model_config": FNET_CONFIG}
        cls = DynacellUNet
    else:
        kwargs = {"generator_config": VIT2D_CONFIG, "discriminator_config": DISC2D_CONFIG, "ema_kimg": 10.0}
        cls = DynacellGAN
    seed_everything(0)
    baseline = cls(**kwargs)
    ckpt = tmp_path / "baseline.ckpt"
    torch.save({"state_dict": baseline.state_dict()}, ckpt)
    arm = cls(**kwargs, ckpt_path=str(ckpt), seg_consistency=_segcons(), seg_consistency_weight=1.0)
    assert arm.state_dict().keys() == baseline.state_dict().keys()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(arm.state_dict()[key], value)
    assert "seg_consistency" not in arm.hparams
    assert arm.hparams["seg_consistency_weight"] == 1.0
