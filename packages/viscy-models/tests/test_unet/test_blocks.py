"""Tests for generalized 3D convolutional building blocks."""

import pytest
import torch

from viscy_models.unet.blocks import (
    Block,
    ConvBottleneck3D,
    ResnetBlock,
    TimestepEmbedder,
)

# ── Block ───────────────────────────────────────────────────────────────────


class TestBlock:
    """Tests for the configurable Conv3d + Norm + Activation block."""

    def test_forward_default(self):
        """Default GroupNorm + SiLU."""
        block = Block(16, 32)
        x = torch.randn(2, 16, 4, 8, 8)
        y = block(x)
        assert y.shape == (2, 32, 4, 8, 8)

    def test_forward_batch_relu(self):
        """BatchNorm + ReLU (FNet-style)."""
        block = Block(16, 32, norm="batch", activation="relu")
        x = torch.randn(2, 16, 4, 8, 8)
        y = block(x)
        assert y.shape == (2, 32, 4, 8, 8)

    def test_forward_with_scale_shift(self):
        """Affine conditioning via scale_shift."""
        block = Block(16, 32)
        x = torch.randn(2, 16, 4, 8, 8)
        scale = torch.randn(2, 32, 1, 1, 1)
        shift = torch.randn(2, 32, 1, 1, 1)
        y = block(x, scale_shift=(scale, shift))
        assert y.shape == (2, 32, 4, 8, 8)

    def test_invalid_norm_raises(self):
        with pytest.raises(ValueError, match="Unknown norm type"):
            Block(16, 32, norm="instance")

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation type"):
            Block(16, 32, activation="gelu")


# ── ResnetBlock ─────────────────────────────────────────────────────────────


class TestResnetBlock:
    """Tests for the configurable residual / double-conv block."""

    def test_residual_same_channels(self):
        """Residual block with same in/out channels (identity skip)."""
        block = ResnetBlock(32, 32)
        x = torch.randn(2, 32, 4, 8, 8)
        y = block(x)
        assert y.shape == (2, 32, 4, 8, 8)

    def test_residual_channel_change(self):
        """Residual block with different in/out channels (1x1 conv skip)."""
        block = ResnetBlock(32, 64)
        x = torch.randn(2, 32, 4, 8, 8)
        y = block(x)
        assert y.shape == (2, 64, 4, 8, 8)

    def test_no_residual(self):
        """Non-residual block (FNet-style double conv)."""
        block = ResnetBlock(32, 64, residual=False)
        x = torch.randn(2, 32, 4, 8, 8)
        y = block(x)
        assert y.shape == (2, 64, 4, 8, 8)
        assert block.res_conv is None

    def test_batch_relu_no_residual(self):
        """FNet configuration: BatchNorm + ReLU, no residual."""
        block = ResnetBlock(16, 16, residual=False, norm="batch", activation="relu")
        x = torch.randn(2, 16, 4, 8, 8)
        y = block(x)
        assert y.shape == (2, 16, 4, 8, 8)

    def test_time_conditioning(self):
        """Timestep-conditioned block."""
        block = ResnetBlock(32, 32, time_emb_dim=64)
        x = torch.randn(2, 32, 4, 8, 8)
        t = torch.randn(2, 64)
        y = block(x, time_emb=t)
        assert y.shape == (2, 32, 4, 8, 8)

    def test_time_conditioning_none(self):
        """Block with time_emb_dim but no time_emb passed — no error."""
        block = ResnetBlock(32, 32, time_emb_dim=64)
        x = torch.randn(2, 32, 4, 8, 8)
        y = block(x)
        assert y.shape == (2, 32, 4, 8, 8)

    def test_no_time_conditioning(self):
        """Block without time support, time_emb=None — no error."""
        block = ResnetBlock(32, 32)
        x = torch.randn(2, 32, 4, 8, 8)
        y = block(x, time_emb=None)
        assert y.shape == (2, 32, 4, 8, 8)


# ── TimestepEmbedder ────────────────────────────────────────────────────────


class TestTimestepEmbedder:
    """Tests for sinusoidal timestep embedding."""

    def test_output_shape(self):
        embedder = TimestepEmbedder(hidden_size=128)
        t = torch.rand(4)
        out = embedder(t)
        assert out.shape == (4, 128)

    def test_different_timesteps_give_different_embeddings(self):
        embedder = TimestepEmbedder(hidden_size=64)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = embedder(t)
        # All three embeddings should be different
        assert not torch.allclose(out[0], out[1])
        assert not torch.allclose(out[1], out[2])


# ── ConvBottleneck3D ────────────────────────────────────────────────────────


class TestConvBottleneck3D:
    """Tests for the convolutional bottleneck module."""

    def test_forward_default(self):
        """Default GroupNorm + SiLU + residual."""
        bn = ConvBottleneck3D(64)
        x = torch.randn(2, 64, 4, 8, 8)
        y = bn(x)
        assert y.shape == (2, 64, 4, 8, 8)

    def test_forward_fnet_config(self):
        """FNet configuration: BatchNorm + ReLU, no residual."""
        bn = ConvBottleneck3D(32, residual=False, norm="batch", activation="relu")
        x = torch.randn(2, 32, 4, 8, 8)
        y = bn(x)
        assert y.shape == (2, 32, 4, 8, 8)

    def test_forward_time_embeds_none(self):
        """Bottleneck with time_embeds=None exercises no-conditioning path."""
        bn = ConvBottleneck3D(32)
        x = torch.randn(2, 32, 4, 8, 8)
        y = bn(x, time_embeds=None)
        assert y.shape == (2, 32, 4, 8, 8)

    def test_forward_with_time_embeds(self):
        """Bottleneck with time conditioning."""
        bn = ConvBottleneck3D(32, time_emb_dim=64)
        x = torch.randn(2, 32, 4, 8, 8)
        t = torch.randn(2, 64)
        y = bn(x, time_embeds=t)
        assert y.shape == (2, 32, 4, 8, 8)
