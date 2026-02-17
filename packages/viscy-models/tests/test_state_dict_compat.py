"""State dict key compatibility tests ensuring migrated models preserve
checkpoint-loadable structure.

Each test verifies:
1. Total parameter count matches expected (catches added/removed parameters)
2. Top-level prefixes match expected (catches structural renames)
3. Sentinel keys from different model parts are present (catches internal renames)

This guards requirement COMPAT-01: state dict keys must match original
monolithic counterparts exactly for backward-compatible checkpoint loading.
"""

import pytest

from viscy_models import (
    BetaVae25D,
    BetaVaeMonai,
    ContrastiveEncoder,
    FullyConvolutionalMAE,
    ResNet3dEncoder,
    UNeXt2,
    Unet2d,
    Unet25d,
)


def _get_prefixes(state_dict: dict) -> set[str]:
    """Extract top-level prefixes from state dict keys."""
    return {k.split(".")[0] for k in state_dict}


class TestUNeXt2StateDictCompat:
    """State dict compatibility for UNeXt2."""

    def test_parameter_count(self):
        model = UNeXt2(backbone="convnextv2_atto")
        assert len(model.state_dict()) == 213

    def test_top_level_prefixes(self):
        model = UNeXt2(backbone="convnextv2_atto")
        prefixes = _get_prefixes(model.state_dict())
        assert prefixes == {"decoder", "encoder_stages", "head", "stem"}

    def test_sentinel_keys(self):
        model = UNeXt2(backbone="convnextv2_atto")
        keys = set(model.state_dict().keys())
        sentinels = [
            "stem.conv.weight",
            "stem.conv.bias",
            "encoder_stages.stages_1.blocks.1.mlp.fc2.bias",
            "decoder.decoder_stages.0.conv.blocks.0.conv_dw.weight",
            "decoder.decoder_stages.0.conv.blocks.0.mlp.fc1.bias",
            "decoder.decoder_stages.2.conv.blocks.0.mlp.grn.bias",
            "head.conv.1.weight",
        ]
        for key in sentinels:
            assert key in keys, f"Missing sentinel key: {key}"


class TestFullyConvolutionalMAEStateDictCompat:
    """State dict compatibility for FullyConvolutionalMAE."""

    def test_parameter_count(self):
        model = FullyConvolutionalMAE(in_channels=1, out_channels=1)
        assert len(model.state_dict()) == 222

    def test_top_level_prefixes(self):
        model = FullyConvolutionalMAE(in_channels=1, out_channels=1)
        prefixes = _get_prefixes(model.state_dict())
        assert prefixes == {"decoder", "encoder"}

    def test_sentinel_keys(self):
        model = FullyConvolutionalMAE(in_channels=1, out_channels=1)
        keys = set(model.state_dict().keys())
        sentinels = [
            "encoder.stem.conv3d.weight",
            "encoder.stem.norm.bias",
            "encoder.stages.0.blocks.1.mlp.fc1.weight",
            "encoder.stages.2.blocks.1.layernorm.weight",
            "decoder.decoder_stages.0.conv.blocks.0.conv_dw.bias",
            "decoder.decoder_stages.0.conv.blocks.0.mlp.fc1.bias",
        ]
        for key in sentinels:
            assert key in keys, f"Missing sentinel key: {key}"


class TestContrastiveEncoderStateDictCompat:
    """State dict compatibility for ContrastiveEncoder."""

    def test_parameter_count(self):
        model = ContrastiveEncoder(
            backbone="convnext_tiny",
            in_channels=1,
            in_stack_depth=5,
            stem_kernel_size=(5, 4, 4),
        )
        assert len(model.state_dict()) == 194

    def test_top_level_prefixes(self):
        model = ContrastiveEncoder(
            backbone="convnext_tiny",
            in_channels=1,
            in_stack_depth=5,
            stem_kernel_size=(5, 4, 4),
        )
        prefixes = _get_prefixes(model.state_dict())
        assert prefixes == {"encoder", "projection", "stem"}

    def test_sentinel_keys(self):
        model = ContrastiveEncoder(
            backbone="convnext_tiny",
            in_channels=1,
            in_stack_depth=5,
            stem_kernel_size=(5, 4, 4),
        )
        keys = set(model.state_dict().keys())
        sentinels = [
            "stem.conv.weight",
            "stem.conv.bias",
            "encoder.head.norm.bias",
            "encoder.stages.0.blocks.0.conv_dw.bias",
            "encoder.stages.2.blocks.4.gamma",
            "projection.4.weight",
        ]
        for key in sentinels:
            assert key in keys, f"Missing sentinel key: {key}"


class TestResNet3dEncoderStateDictCompat:
    """State dict compatibility for ResNet3dEncoder."""

    def test_parameter_count(self):
        model = ResNet3dEncoder(backbone="resnet10", in_channels=1)
        assert len(model.state_dict()) == 86

    def test_top_level_prefixes(self):
        model = ResNet3dEncoder(backbone="resnet10", in_channels=1)
        prefixes = _get_prefixes(model.state_dict())
        assert prefixes == {"encoder", "projection"}

    def test_sentinel_keys(self):
        model = ResNet3dEncoder(backbone="resnet10", in_channels=1)
        keys = set(model.state_dict().keys())
        sentinels = [
            "encoder.bn1.bias",
            "encoder.bn1.running_mean",
            "encoder.layer2.0.bn1.weight",
            "encoder.layer3.0.bn2.running_var",
            "projection.4.weight",
            "projection.4.running_mean",
        ]
        for key in sentinels:
            assert key in keys, f"Missing sentinel key: {key}"


class TestBetaVae25DStateDictCompat:
    """State dict compatibility for BetaVae25D."""

    def test_parameter_count(self):
        model = BetaVae25D(
            in_channels=1,
            in_stack_depth=5,
            input_spatial_size=(64, 64),
        )
        assert len(model.state_dict()) == 470

    def test_top_level_prefixes(self):
        model = BetaVae25D(
            in_channels=1,
            in_stack_depth=5,
            input_spatial_size=(64, 64),
        )
        prefixes = _get_prefixes(model.state_dict())
        assert prefixes == {"decoder", "encoder"}

    def test_sentinel_keys(self):
        model = BetaVae25D(
            in_channels=1,
            in_stack_depth=5,
            input_spatial_size=(64, 64),
        )
        keys = set(model.state_dict().keys())
        sentinels = [
            "encoder.stem.conv.weight",
            "encoder.stem.conv.bias",
            "encoder.encoder.layer2.1.bn1.running_mean",
            "encoder.fc_mu.weight",
            "decoder.decoder_stages.0.conv.0.conv.unit0.adn.N.bias",
            "decoder.decoder_stages.3.conv.0.residual.bias",
        ]
        for key in sentinels:
            assert key in keys, f"Missing sentinel key: {key}"


class TestBetaVaeMonaiStateDictCompat:
    """State dict compatibility for BetaVaeMonai."""

    def test_parameter_count(self):
        model = BetaVaeMonai(
            spatial_dims=2,
            in_shape=(1, 64, 64),
            out_channels=1,
            latent_size=128,
            channels=(16, 32, 64),
            strides=(2, 2, 2),
        )
        assert len(model.state_dict()) == 23

    def test_top_level_prefixes(self):
        model = BetaVaeMonai(
            spatial_dims=2,
            in_shape=(1, 64, 64),
            out_channels=1,
            latent_size=128,
            channels=(16, 32, 64),
            strides=(2, 2, 2),
        )
        prefixes = _get_prefixes(model.state_dict())
        assert prefixes == {"model"}

    def test_sentinel_keys(self):
        model = BetaVaeMonai(
            spatial_dims=2,
            in_shape=(1, 64, 64),
            out_channels=1,
            latent_size=128,
            channels=(16, 32, 64),
            strides=(2, 2, 2),
        )
        keys = set(model.state_dict().keys())
        sentinels = [
            "model.encode.encode_0.conv.bias",
            "model.encode.encode_0.conv.weight",
            "model.decode.decode_0.conv.conv.bias",
            "model.decode.decode_2.conv.conv.bias",
            "model.mu.weight",
            "model.mu.bias",
            "model.logvar.weight",
        ]
        for key in sentinels:
            assert key in keys, f"Missing sentinel key: {key}"


class TestUnet2dStateDictCompat:
    """State dict compatibility for Unet2d."""

    def test_parameter_count(self):
        model = Unet2d(in_channels=1, out_channels=1)
        assert len(model.state_dict()) == 148

    def test_top_level_prefixes(self):
        model = Unet2d(in_channels=1, out_channels=1)
        prefixes = _get_prefixes(model.state_dict())
        expected = {
            "bottom_transition_block",
            "down_conv_block_0",
            "down_conv_block_1",
            "down_conv_block_2",
            "down_conv_block_3",
            "terminal_block",
            "up_conv_block_0",
            "up_conv_block_1",
            "up_conv_block_2",
            "up_conv_block_3",
        }
        assert prefixes == expected

    def test_sentinel_keys(self):
        model = Unet2d(in_channels=1, out_channels=1)
        keys = set(model.state_dict().keys())
        sentinels = [
            "bottom_transition_block.Conv2d_0.weight",
            "down_conv_block_0.batch_norm_0.weight",
            "down_conv_block_3.batch_norm_1.running_mean",
            "up_conv_block_0.batch_norm_0.weight",
            "up_conv_block_3.resid_conv.weight",
            "terminal_block.Conv2d_0.bias",
        ]
        for key in sentinels:
            assert key in keys, f"Missing sentinel key: {key}"


class TestUnet25dStateDictCompat:
    """State dict compatibility for Unet25d."""

    def test_parameter_count(self):
        model = Unet25d(in_channels=1, out_channels=1)
        assert len(model.state_dict()) == 147

    def test_top_level_prefixes(self):
        model = Unet25d(in_channels=1, out_channels=1)
        prefixes = _get_prefixes(model.state_dict())
        expected = {
            "bottom_transition_block",
            "down_conv_block_0",
            "down_conv_block_1",
            "down_conv_block_2",
            "down_conv_block_3",
            "skip_conv_layer_0",
            "skip_conv_layer_1",
            "skip_conv_layer_2",
            "skip_conv_layer_3",
            "terminal_block",
            "up_conv_block_0",
            "up_conv_block_1",
            "up_conv_block_2",
            "up_conv_block_3",
        }
        assert prefixes == expected

    def test_sentinel_keys(self):
        model = Unet25d(in_channels=1, out_channels=1)
        keys = set(model.state_dict().keys())
        sentinels = [
            "bottom_transition_block.weight",
            "down_conv_block_0.Conv3d_0.bias",
            "down_conv_block_2.Conv3d_1.weight",
            "skip_conv_layer_3.weight",
            "up_conv_block_3.resid_conv.bias",
            "terminal_block.Conv3d_0.bias",
        ]
        for key in sentinels:
            assert key in keys, f"Missing sentinel key: {key}"
