"""Smoke tests for dynacell engine."""

import pytest
import torch

from dynacell.engine import DynacellUNet

# Small model configs for tests (not production sizes).
VIT_CONFIG = {
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

FNET_CONFIG = {
    "in_channels": 1,
    "out_channels": 1,
    "depth": 1,
    "mult_chan": 8,
    "in_stack_depth": 4,
}


def test_unetvit3d_init():
    """DynacellUNet instantiates with UNetViT3D architecture."""
    model = DynacellUNet(architecture="UNetViT3D", model_config=VIT_CONFIG)
    assert model.model is not None
    assert model.lr == 1e-3


def test_unetvit3d_forward(synth_vit_batch):
    """UNetViT3D forward pass produces correct output shape."""
    model = DynacellUNet(architecture="UNetViT3D", model_config=VIT_CONFIG)
    model.eval()
    with torch.no_grad():
        output = model(synth_vit_batch["source"])
    assert output.shape == synth_vit_batch["source"].shape


def test_unetvit3d_rejects_wrong_spatial():
    """UNetViT3D raises ValueError on mismatched spatial dims."""
    model = DynacellUNet(architecture="UNetViT3D", model_config=VIT_CONFIG)
    model.eval()
    wrong_input = torch.randn(1, 1, 8, 64, 64)
    with pytest.raises(ValueError, match="spatial size"):
        model(wrong_input)


def test_unetvit3d_example_input_array():
    """UNetViT3D example_input_array matches input_spatial_size."""
    model = DynacellUNet(architecture="UNetViT3D", model_config=VIT_CONFIG)
    assert model.example_input_array.shape == (1, 1, 8, 32, 32)


def test_fnet3d_init():
    """DynacellUNet instantiates with FNet3D architecture."""
    model = DynacellUNet(architecture="FNet3D", model_config=FNET_CONFIG)
    assert model.model is not None


def test_fnet3d_forward(synth_fnet_batch):
    """FNet3D forward produces correct output shape."""
    model = DynacellUNet(architecture="FNet3D", model_config=FNET_CONFIG)
    model.eval()
    with torch.no_grad():
        output = model(synth_fnet_batch["source"])
    assert output.shape == synth_fnet_batch["source"].shape


def test_fnet3d_example_input_array():
    """FNet3D example_input_array uses in_stack_depth."""
    model = DynacellUNet(
        architecture="FNet3D",
        model_config=FNET_CONFIG,
        example_input_yx_shape=(64, 64),
    )
    assert model.example_input_array.shape == (1, 1, 4, 64, 64)


def test_state_dict_keys():
    """State dict keys are prefixed with 'model.'."""
    model = DynacellUNet(architecture="FNet3D", model_config=FNET_CONFIG)
    for key in model.state_dict():
        assert key.startswith("model."), f"Unexpected key prefix: {key}"


def test_invalid_architecture():
    """Invalid architecture raises ValueError."""
    with pytest.raises(ValueError, match="not in"):
        DynacellUNet(architecture="NonExistent")


def test_predict_step_raises():
    """predict_step raises NotImplementedError."""
    model = DynacellUNet(architecture="FNet3D", model_config=FNET_CONFIG)
    with pytest.raises(NotImplementedError, match="not supported"):
        model.predict_step({"source": torch.randn(1, 1, 4, 16, 16)}, batch_idx=0)
