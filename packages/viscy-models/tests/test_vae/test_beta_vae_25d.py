"""Forward-pass tests for BetaVae25D model covering multiple backbones."""

from types import SimpleNamespace

import torch

from viscy_models.vae import BetaVae25D


def test_beta_vae_25d_resnet50(device):
    """ResNet50 backbone: 2ch in/out, depth=16, latent=256, spatial=128x128."""
    model = BetaVae25D(
        backbone="resnet50",
        in_channels=2,
        in_stack_depth=16,
        out_stack_depth=16,
        latent_dim=256,
        input_spatial_size=(128, 128),
        stem_kernel_size=(2, 4, 4),
        stem_stride=(2, 4, 4),
        decoder_stages=3,
    ).to(device)
    model.eval()
    x = torch.randn(2, 2, 16, 128, 128, device=device)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SimpleNamespace)
    # ResNet50 with stem_stride=(2,4,4) + 3 decoder stages recovers 64x64
    # (stem reduces 128->32, backbone 4x reduction to 4, decoder 3 stages
    #  of 2x = 8x to 32, head 2x shuffle to 64)
    assert out.recon_x.shape == (2, 2, 16, 64, 64)
    assert out.mean.shape == (2, 256)
    assert out.logvar.shape == (2, 256)
    assert out.z.shape == (2, 256)


def test_beta_vae_25d_convnext(device):
    """ConvNeXt-tiny backbone: 1ch in/out, depth=15, latent=256, spatial=128x128."""
    model = BetaVae25D(
        backbone="convnext_tiny",
        in_channels=1,
        in_stack_depth=15,
        out_stack_depth=15,
        latent_dim=256,
        input_spatial_size=(128, 128),
        stem_kernel_size=(5, 4, 4),
        stem_stride=(5, 4, 4),
        decoder_stages=3,
    ).to(device)
    model.eval()
    x = torch.randn(2, 1, 15, 128, 128, device=device)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SimpleNamespace)
    assert out.recon_x.shape == (2, 1, 15, 128, 128)
    assert out.mean.shape == (2, 256)
    assert out.logvar.shape == (2, 256)
    assert out.z.shape == (2, 256)
