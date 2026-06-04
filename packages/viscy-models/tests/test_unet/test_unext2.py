"""Forward-pass tests for UNeXt2 model covering multiple configurations."""

import pytest
import torch

from viscy_models.unet import UNeXt2


def test_unext2_default_forward(device):
    """Default UNeXt2: 1ch in, 1ch out, depth=5, convnextv2_tiny, spatial=128x128."""
    model = UNeXt2().to(device)
    x = torch.randn(1, 1, 5, 128, 128, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 5, 128, 128)


def test_unext2_small_backbone(device):
    """Small backbone: convnextv2_atto, 1ch in, 1ch out, depth=5, spatial=128x128."""
    model = UNeXt2(backbone="convnextv2_atto").to(device)
    x = torch.randn(1, 1, 5, 128, 128, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 5, 128, 128)


def test_unext2_multichannel(device):
    """Multi-channel: convnextv2_atto, 3ch in, 2ch out, depth=5, spatial=64x64."""
    model = UNeXt2(in_channels=3, out_channels=2, backbone="convnextv2_atto").to(device)
    x = torch.randn(1, 3, 5, 64, 64, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2, 5, 64, 64)


def test_unext2_different_stack_depths(device):
    """Different in/out stack depths: convnextv2_atto, in=5, out=3, spatial=64x64."""
    model = UNeXt2(in_stack_depth=5, out_stack_depth=3, backbone="convnextv2_atto").to(device)
    x = torch.randn(1, 1, 5, 64, 64, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 3, 64, 64)


@pytest.mark.xfail(
    reason="Deconv decoder path has channel mismatch bug in original code "
    "(ResidualUnit expects in_channels but receives upsample+skip concat). "
    "Never exercised in production -- UNeXt2 defaults to pixelshuffle.",
    strict=True,
)
def test_unext2_deconv_decoder(device):
    """Deconv decoder mode: convnextv2_atto, spatial=64x64."""
    model = UNeXt2(backbone="convnextv2_atto", decoder_mode="deconv").to(device)
    x = torch.randn(1, 1, 5, 64, 64, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 5, 64, 64)


def test_unext2_stem_validation():
    """Raises ValueError when stack depth not divisible by stem kernel depth."""
    with pytest.raises(ValueError, match="not divisible"):
        UNeXt2(in_stack_depth=7, stem_kernel_size=(5, 4, 4))
