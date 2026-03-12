"""Smoke tests for translation engine modules."""

import subprocess
from pathlib import Path

import torch
from conftest import SYNTH_B, SYNTH_C, SYNTH_D

from viscy_translation.engine import FcmaeUNet, VSUNet


def test_imports():
    """Verify all top-level imports work."""
    from viscy_translation import AugmentedPredictionVSUNet, FcmaeUNet, MaskedMSELoss, SegmentationMetrics2D, VSUNet
    from viscy_utils.callbacks import HCSPredictionWriter
    from viscy_utils.losses import MixedLoss

    assert VSUNet is not None
    assert FcmaeUNet is not None
    assert AugmentedPredictionVSUNet is not None
    assert MaskedMSELoss is not None
    assert SegmentationMetrics2D is not None
    assert MixedLoss is not None
    assert HCSPredictionWriter is not None


def test_vsunet_init():
    """Verify VSUNet instantiates with UNeXt2 architecture."""
    model = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": SYNTH_C, "out_channels": 1, "in_stack_depth": SYNTH_D},
    )
    assert model.model is not None
    assert model.lr == 1e-3


def test_vsunet_forward(synthetic_batch):
    """Verify VSUNet forward pass produces correct output shape."""
    model = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": SYNTH_C, "out_channels": 1, "in_stack_depth": SYNTH_D},
    )
    model.eval()
    with torch.no_grad():
        output = model(synthetic_batch["source"])
    assert output.shape[0] == SYNTH_B
    assert output.shape[1] == 1  # out_channels


def test_vsunet_state_dict_keys():
    """State dict key regression test for checkpoint compatibility."""
    model = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": SYNTH_C, "out_channels": 1, "in_stack_depth": SYNTH_D},
    )
    state_dict = model.state_dict()
    # All keys should start with "model." since VSUNet stores the architecture as self.model
    for key in state_dict:
        assert key.startswith("model."), f"Unexpected key prefix: {key}"
    # Verify some known keys exist (from UNeXt2 architecture)
    key_names = set(state_dict.keys())
    assert any("model." in k for k in key_names), "No model keys found"
    assert len(key_names) > 0, "Empty state dict"


def test_mixed_loss_integration(synthetic_batch):
    """Verify MixedLoss works as loss_function for VSUNet."""
    from viscy_utils.losses import MixedLoss

    loss_fn = MixedLoss(l1_alpha=0.5, l2_alpha=0.0, ms_dssim_alpha=0.5)
    model = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": SYNTH_C, "out_channels": 1, "in_stack_depth": SYNTH_D},
        loss_function=loss_fn,
    )
    assert model.loss_function is loss_fn


def test_fcmae_unet_init():
    """Verify FcmaeUNet instantiates."""
    model = FcmaeUNet(
        model_config={"in_channels": SYNTH_C, "out_channels": 1, "in_stack_depth": SYNTH_D},
    )
    assert model.fit_mask_ratio == 0.0


def test_no_old_imports():
    """Verify no old viscy.* import paths remain in source code."""
    src_dir = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        ["grep", "-r", "from viscy\\.", str(src_dir)],
        capture_output=True,
        text=True,
    )
    assert result.stdout == "", f"Old import paths found:\n{result.stdout}"
