"""Smoke tests for cytoland engine modules."""

import subprocess
from pathlib import Path

import pytest
import torch

from cytoland.engine import AugmentedPredictionVSUNet, FcmaeUNet, VSUNet


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


def test_vsunet_init(synth_dims):
    """Verify VSUNet instantiates with UNeXt2 architecture."""
    model = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": synth_dims["c"], "out_channels": 1, "in_stack_depth": synth_dims["d"]},
    )
    assert model.model is not None
    assert model.lr == 1e-3


def test_vsunet_forward(synthetic_batch, synth_dims):
    """Verify VSUNet forward pass produces correct output shape."""
    model = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": synth_dims["c"], "out_channels": 1, "in_stack_depth": synth_dims["d"]},
    )
    model.eval()
    with torch.no_grad():
        output = model(synthetic_batch["source"])
    assert output.shape[0] == synth_dims["b"]
    assert output.shape[1] == 1  # out_channels


def test_vsunet_state_dict_keys(synth_dims):
    """State dict key regression test for checkpoint compatibility."""
    model = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": synth_dims["c"], "out_channels": 1, "in_stack_depth": synth_dims["d"]},
    )
    state_dict = model.state_dict()
    for key in state_dict:
        assert key.startswith("model."), f"Unexpected key prefix: {key}"
    key_names = set(state_dict.keys())
    assert any("model." in k for k in key_names), "No model keys found"
    assert len(key_names) > 0, "Empty state dict"


def test_fnet3d_init(synth_dims):
    """Verify VSUNet instantiates with FNet3D architecture."""
    model = VSUNet(
        architecture="FNet3D",
        model_config={
            "in_channels": synth_dims["c"],
            "out_channels": 1,
            "depth": 1,
            "mult_chan": 8,
            "in_stack_depth": synth_dims["d"],
        },
    )
    assert model.model is not None


def test_fnet3d_forward(synthetic_batch, synth_dims):
    """Verify FNet3D forward pass produces correct output shape."""
    model = VSUNet(
        architecture="FNet3D",
        model_config={
            "in_channels": synth_dims["c"],
            "out_channels": 1,
            "depth": 1,
            "mult_chan": 8,
            "in_stack_depth": synth_dims["d"],
        },
    )
    model.eval()
    with torch.no_grad():
        output = model(synthetic_batch["source"])
    assert output.shape[0] == synth_dims["b"]
    assert output.shape[1] == 1


def test_fnet3d_predict_start(synth_dims):
    """Verify FNet3D on_predict_start sets up DivisiblePad."""
    model = VSUNet(
        architecture="FNet3D",
        model_config={
            "in_channels": synth_dims["c"],
            "out_channels": 1,
            "depth": 1,
            "mult_chan": 8,
            "in_stack_depth": synth_dims["d"],
        },
    )
    model.on_predict_start()
    assert model.predict_pad is not None


def test_mixed_loss_integration(synthetic_batch, synth_dims):
    """Verify MixedLoss works as loss_function for VSUNet."""
    from viscy_utils.losses import MixedLoss

    loss_fn = MixedLoss(l1_alpha=0.5, l2_alpha=0.0, ms_dssim_alpha=0.5)
    model = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": synth_dims["c"], "out_channels": 1, "in_stack_depth": synth_dims["d"]},
        loss_function=loss_fn,
    )
    assert model.loss_function is loss_fn


def test_fcmae_unet_init(synth_dims):
    """Verify FcmaeUNet instantiates."""
    model = FcmaeUNet(
        model_config={"in_channels": synth_dims["c"], "out_channels": 1, "in_stack_depth": synth_dims["d"]},
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


def test_augmented_prediction_optional_transforms(synth_dims):
    """Verify AugmentedPredictionVSUNet works without specifying transforms."""
    model = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": synth_dims["c"], "out_channels": 1, "in_stack_depth": synth_dims["d"]},
    )
    vs = AugmentedPredictionVSUNet(model=model.model)
    vs.eval()
    x = torch.randn(synth_dims["b"], synth_dims["c"], synth_dims["d"], 64, 64)
    with torch.inference_mode():
        output = vs._predict_with_tta(x)
    assert output.shape[0] == synth_dims["b"]
    assert output.shape[1] == 1


def test_predict_sliding_windows_output_shape(synth_dims):
    """Verify predict_sliding_windows produces correct output shape."""
    z_window = synth_dims["d"]
    out_channels = 2
    depth = 12

    model = VSUNet(
        architecture="fcmae",
        model_config={
            "in_channels": synth_dims["c"],
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
    x = torch.randn(1, synth_dims["c"], depth, synth_dims["fcmae_h"], synth_dims["fcmae_w"])
    with torch.inference_mode():
        output = vs.predict_sliding_windows(x, out_channel=out_channels, step=1)
    expected = (1, out_channels, depth, synth_dims["fcmae_h"], synth_dims["fcmae_w"])
    assert output.shape == expected, f"Expected {expected}, got {output.shape}"


def test_predict_sliding_windows_invalid_input(synth_dims):
    """Verify predict_sliding_windows rejects non-5D input."""
    model = VSUNet(
        architecture="UNeXt2",
        model_config={"in_channels": 1, "out_channels": 1, "in_stack_depth": synth_dims["d"]},
    )
    vs = AugmentedPredictionVSUNet(model=model.model)
    with pytest.raises(ValueError, match="5 dimensions"):
        vs.predict_sliding_windows(torch.randn(1, synth_dims["d"], 64, 64))


def test_predict_sliding_windows_missing_out_stack_depth():
    """Verify predict_sliding_windows rejects model without out_stack_depth."""
    model = torch.nn.Linear(10, 10)
    model.num_blocks = 1  # satisfy DivisiblePad
    vs = AugmentedPredictionVSUNet(model=model)
    with pytest.raises(ValueError, match="out_stack_depth"):
        vs.predict_sliding_windows(torch.randn(1, 1, 10, 4, 4))
