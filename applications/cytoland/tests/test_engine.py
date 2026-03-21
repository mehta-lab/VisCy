"""Smoke tests for cytoland engine modules."""

import subprocess
from pathlib import Path

import pytest
import torch

from cytoland.engine import AugmentedPredictionVSUNet, FcmaeUNet, VSUNet

SYNTH_B = 2
SYNTH_C = 1
SYNTH_D = 5
FCMAE_H = 128
FCMAE_W = 128


def test_imports():
    """Verify all top-level imports work."""
    from cytoland import AugmentedPredictionVSUNet, FcmaeUNet, MaskedMSELoss, SegmentationMetrics2D, VSUNet
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


def test_augmented_prediction_optional_transforms():
    """Verify AugmentedPredictionVSUNet works without specifying transforms."""
    model = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": SYNTH_C, "out_channels": 1, "in_stack_depth": SYNTH_D},
    )
    vs = AugmentedPredictionVSUNet(model=model.model)
    vs.eval()
    x = torch.randn(SYNTH_B, SYNTH_C, SYNTH_D, 64, 64)
    with torch.inference_mode():
        output = vs._predict_with_tta(x)
    assert output.shape[0] == SYNTH_B
    assert output.shape[1] == 1


def test_predict_sliding_windows_output_shape():
    """Verify predict_sliding_windows produces correct output shape."""
    z_window = SYNTH_D
    out_channels = 2
    depth = 12

    model = VSUNet(
        architecture="fcmae",
        model_config={
            "in_channels": SYNTH_C,
            "out_channels": out_channels,
            "encoder_blocks": [2, 2, 2, 2],
            "dims": [4, 8, 16, 32],
            "decoder_conv_blocks": 2,
            "stem_kernel_size": [z_window, 4, 4],
            "in_stack_depth": z_window,
            "pretraining": False,
        },
    )
    vs = AugmentedPredictionVSUNet(model=model.model)
    vs.eval()
    x = torch.randn(1, SYNTH_C, depth, FCMAE_H, FCMAE_W)
    with torch.inference_mode():
        output = vs.predict_sliding_windows(x, out_channel=out_channels, step=1)
    expected = (1, out_channels, depth, FCMAE_H, FCMAE_W)
    assert output.shape == expected, f"Expected {expected}, got {output.shape}"


def test_predict_sliding_windows_invalid_input():
    """Verify predict_sliding_windows rejects non-5D input."""
    model = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": 1, "out_channels": 1, "in_stack_depth": SYNTH_D},
    )
    vs = AugmentedPredictionVSUNet(model=model.model)
    with pytest.raises(ValueError, match="5 dimensions"):
        vs.predict_sliding_windows(torch.randn(1, SYNTH_D, 64, 64))


def test_predict_sliding_windows_missing_out_stack_depth():
    """Verify predict_sliding_windows rejects model without out_stack_depth."""
    model = torch.nn.Linear(10, 10)
    model.num_blocks = 1  # satisfy DivisiblePad
    vs = AugmentedPredictionVSUNet(model=model)
    with pytest.raises(ValueError, match="out_stack_depth"):
        vs.predict_sliding_windows(torch.randn(1, 1, 10, 4, 4))
