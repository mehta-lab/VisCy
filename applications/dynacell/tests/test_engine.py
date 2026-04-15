"""Smoke tests for dynacell engine."""

import pytest
import torch
from monai.data import MetaTensor

from dynacell.engine import DynacellFlowMatching, DynacellUNet

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


def test_unetvit3d_init():
    """DynacellUNet instantiates with UNetViT3D architecture."""
    model = DynacellUNet(architecture="UNetViT3D", model_config=VIT_TEST_CONFIG)
    assert model.model is not None
    assert model.lr == 1e-3


def test_unetvit3d_forward(synth_vit_batch):
    """UNetViT3D forward pass produces correct output shape."""
    model = DynacellUNet(architecture="UNetViT3D", model_config=VIT_TEST_CONFIG)
    model.eval()
    with torch.no_grad():
        output = model(synth_vit_batch["source"])
    assert output.shape == synth_vit_batch["source"].shape


def test_unetvit3d_rejects_wrong_spatial():
    """UNetViT3D raises ValueError on mismatched spatial dims."""
    model = DynacellUNet(architecture="UNetViT3D", model_config=VIT_TEST_CONFIG)
    model.eval()
    wrong_input = torch.randn(1, 1, 8, 64, 64)
    with pytest.raises(ValueError, match="spatial size"):
        model(wrong_input)


def test_unetvit3d_example_input_array():
    """UNetViT3D example_input_array matches input_spatial_size."""
    model = DynacellUNet(architecture="UNetViT3D", model_config=VIT_TEST_CONFIG)
    assert model.example_input_array.shape == (1, 1, 8, 32, 32)


def test_fnet3d_init():
    """DynacellUNet instantiates with FNet3D architecture."""
    model = DynacellUNet(architecture="FNet3D", model_config=FNET_TEST_CONFIG)
    assert model.model is not None


def test_fnet3d_forward(synth_fnet_batch):
    """FNet3D forward produces correct output shape."""
    model = DynacellUNet(architecture="FNet3D", model_config=FNET_TEST_CONFIG)
    model.eval()
    with torch.no_grad():
        output = model(synth_fnet_batch["source"])
    assert output.shape == synth_fnet_batch["source"].shape


def test_fnet3d_example_input_array():
    """FNet3D example_input_array uses in_stack_depth."""
    model = DynacellUNet(
        architecture="FNet3D",
        model_config=FNET_TEST_CONFIG,
        example_input_yx_shape=(64, 64),
    )
    assert model.example_input_array.shape == (1, 1, 4, 64, 64)


def test_state_dict_keys():
    """State dict keys are prefixed with 'model.'."""
    model = DynacellUNet(architecture="FNet3D", model_config=FNET_TEST_CONFIG)
    for key in model.state_dict():
        assert key.startswith("model."), f"Unexpected key prefix: {key}"


def test_invalid_architecture():
    """Invalid architecture raises ValueError."""
    with pytest.raises(ValueError, match="not in"):
        DynacellUNet(architecture="NonExistent")


def test_fnet3d_predict_step(synth_fnet_batch):
    """FNet3D predict_step returns the input spatial shape."""
    model = DynacellUNet(architecture="FNet3D", model_config=FNET_TEST_CONFIG)
    model.eval()
    model.on_predict_start()
    batch = {**synth_fnet_batch, "source": MetaTensor(synth_fnet_batch["source"])}
    with torch.no_grad():
        prediction = model.predict_step(batch, batch_idx=0)
    assert prediction.shape == synth_fnet_batch["source"].shape


def test_fnet3d_predict_step_pads_odd_spatial():
    """FNet3D predict_step crops back to original shape for odd spatial inputs."""
    model = DynacellUNet(architecture="FNet3D", model_config=FNET_TEST_CONFIG)
    model.eval()
    model.on_predict_start()
    source = MetaTensor(torch.randn(1, 1, 4, 17, 17))
    with torch.no_grad():
        prediction = model.predict_step({"source": source}, batch_idx=0)
    assert prediction.shape == source.shape


def test_unext2_init():
    """DynacellUNet instantiates with UNeXt2 architecture."""
    model = DynacellUNet(architecture="UNeXt2", model_config=UNEXT2_TEST_CONFIG)
    assert model.model is not None
    assert model.lr == 1e-3


def test_unext2_forward(synth_unext2_batch):
    """UNeXt2 forward pass produces correct output shape."""
    model = DynacellUNet(architecture="UNeXt2", model_config=UNEXT2_TEST_CONFIG)
    model.eval()
    with torch.no_grad():
        output = model(synth_unext2_batch["source"])
    assert output.shape == synth_unext2_batch["source"].shape


def test_unext2_predict_step(synth_unext2_batch):
    """UNeXt2 predict_step returns the input spatial shape."""
    model = DynacellUNet(architecture="UNeXt2", model_config=UNEXT2_TEST_CONFIG)
    model.eval()
    model.on_predict_start()
    batch = {**synth_unext2_batch, "source": MetaTensor(synth_unext2_batch["source"])}
    with torch.no_grad():
        prediction = model.predict_step(batch, batch_idx=0)
    assert prediction.shape == synth_unext2_batch["source"].shape


def test_unetvit3d_predict_step(synth_vit_batch):
    """UNetViT3D predict_step returns the input spatial shape."""
    model = DynacellUNet(architecture="UNetViT3D", model_config=VIT_TEST_CONFIG)
    model.eval()
    model.on_predict_start()
    batch = {**synth_vit_batch, "source": MetaTensor(synth_vit_batch["source"])}
    with torch.no_grad():
        prediction = model.predict_step(batch, batch_idx=0)
    assert prediction.shape == synth_vit_batch["source"].shape


# ---- DynacellFlowMatching tests ----


def test_flow_matching_instantiation():
    """DynacellFlowMatching instantiates with test configs."""
    model = DynacellFlowMatching(
        net_config=CELLDIFF_TEST_NET_CONFIG,
        transport_config=CELLDIFF_TEST_TRANSPORT_CONFIG,
    )
    assert model.model is not None
    assert model.lr == 1e-4


def test_flow_matching_forward_loss(synth_celldiff_batch):
    """Flow-matching forward returns a finite scalar loss."""
    model = DynacellFlowMatching(
        net_config=CELLDIFF_TEST_NET_CONFIG,
        transport_config=CELLDIFF_TEST_TRANSPORT_CONFIG,
    )
    model.train()
    loss = model.model(synth_celldiff_batch["source"], synth_celldiff_batch["target"])
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_flow_matching_generate_shape(synth_celldiff_batch):
    """model.generate() returns correct shape."""
    model = DynacellFlowMatching(
        net_config=CELLDIFF_TEST_NET_CONFIG,
        transport_config=CELLDIFF_TEST_TRANSPORT_CONFIG,
    )
    model.eval()
    phase = synth_celldiff_batch["source"]
    with torch.no_grad():
        generated = model.model.generate(phase, num_steps=2)
    assert generated.shape == phase.shape


def test_flow_matching_validation_step_records_loss_when_enabled(synth_celldiff_batch):
    """Validation step can record a scalar loss without changing batch capture."""
    model = DynacellFlowMatching(
        net_config=CELLDIFF_TEST_NET_CONFIG,
        transport_config=CELLDIFF_TEST_TRANSPORT_CONFIG,
        compute_validation_loss=True,
    )
    model.log = lambda *args, **kwargs: None
    model.eval()
    model.validation_step(synth_celldiff_batch, batch_idx=0)
    assert model._val_log_batch is not None
    assert len(model._validation_losses) == 1
    assert len(model._validation_losses[0]) == 1
    loss, batch_size = model._validation_losses[0][0]
    assert torch.isfinite(loss)
    assert batch_size == synth_celldiff_batch["source"].shape[0]


def test_flow_matching_predict_step_pad_crop(synth_celldiff_batch):
    """Flow-matching predict_step pads small input and crops back."""
    model = DynacellFlowMatching(
        net_config=CELLDIFF_TEST_NET_CONFIG,
        transport_config=CELLDIFF_TEST_TRANSPORT_CONFIG,
        num_generate_steps=2,
        predict_method="generate",
    )
    model.eval()
    # Use smaller spatial dims to exercise padding.
    small_source = torch.randn(1, 1, 6, 24, 24)
    batch = {"source": small_source}
    with torch.no_grad():
        prediction = model.predict_step(batch, batch_idx=0)
    assert prediction.shape == small_source.shape
