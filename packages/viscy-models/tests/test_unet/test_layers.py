"""Tests for ConvBlock2D and ConvBlock3D layer implementations."""

import torch

from viscy_models.unet._layers import ConvBlock2D, ConvBlock3D


class TestConvBlock2D:
    """Tests for the 2D convolutional block."""

    def test_conv_block_2d_default_forward(self):
        """Verify default ConvBlock2D produces correct output shape."""
        model = ConvBlock2D(16, 32)
        x = torch.randn(1, 16, 64, 64)
        out = model(x)
        assert out.shape == (1, 32, 64, 64)

    def test_conv_block_2d_state_dict_keys(self):
        """Verify state dict uses add_module naming pattern."""
        model = ConvBlock2D(16, 32)
        sd = model.state_dict()
        keys = list(sd.keys())
        assert any("Conv2d_0" in k for k in keys), f"Missing Conv2d_0 in {keys}"
        assert any("batch_norm_0" in k for k in keys), (
            f"Missing batch_norm_0 in {keys}"
        )

    def test_conv_block_2d_residual_true(self):
        """Verify forward pass works with residual=True."""
        model = ConvBlock2D(16, 32, residual=True)
        x = torch.randn(1, 16, 64, 64)
        out = model(x)
        assert out.shape == (1, 32, 64, 64)

    def test_conv_block_2d_residual_false(self):
        """Verify forward pass works with residual=False."""
        model = ConvBlock2D(16, 32, residual=False)
        x = torch.randn(1, 16, 64, 64)
        out = model(x)
        assert out.shape == (1, 32, 64, 64)

    def test_conv_block_2d_filter_steps_linear(self):
        """Verify forward pass with linear filter steps."""
        model = ConvBlock2D(16, 64, filter_steps="linear")
        x = torch.randn(1, 16, 32, 32)
        out = model(x)
        assert out.shape == (1, 64, 32, 32)

    def test_conv_block_2d_instance_norm(self):
        """Verify instance norm variant registers correct named modules."""
        model = ConvBlock2D(16, 32, norm="instance")
        named_modules = dict(model.named_modules())
        assert "instance_norm_0" in named_modules, (
            f"Missing instance_norm_0 in {list(named_modules.keys())}"
        )


class TestConvBlock3D:
    """Tests for the 3D convolutional block."""

    def test_conv_block_3d_default_forward(self):
        """Verify default ConvBlock3D produces correct output shape."""
        model = ConvBlock3D(8, 16)
        x = torch.randn(1, 8, 5, 32, 32)
        out = model(x)
        assert out.shape == (1, 16, 5, 32, 32)

    def test_conv_block_3d_state_dict_keys(self):
        """Verify state dict uses add_module naming pattern."""
        model = ConvBlock3D(8, 16)
        sd = model.state_dict()
        keys = list(sd.keys())
        assert any("Conv3d_0" in k for k in keys), f"Missing Conv3d_0 in {keys}"
        assert any("batch_norm_0" in k for k in keys), (
            f"Missing batch_norm_0 in {keys}"
        )

    def test_conv_block_3d_dropout_registered(self):
        """Verify ConvBlock3D registers dropout modules (unlike ConvBlock2D)."""
        model = ConvBlock3D(8, 16, dropout=0.5)
        named_modules = dict(model.named_modules())
        assert "dropout_0" in named_modules, (
            f"Missing dropout_0 in {list(named_modules.keys())}"
        )

    def test_conv_block_3d_layer_order_cna(self):
        """Verify forward pass with cna layer order."""
        model = ConvBlock3D(8, 16, layer_order="cna")
        x = torch.randn(1, 8, 5, 16, 16)
        out = model(x)
        assert out.shape == (1, 16, 5, 16, 16)
