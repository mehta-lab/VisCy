"""Forward-pass tests for BetaVaeMonai model covering 2D and 3D."""

from types import SimpleNamespace

import torch

from viscy_models.vae import BetaVaeMonai


def test_beta_vae_monai_2d(device):
    """2D VAE: 1ch, 64x64, latent=128, channels=(32,64), strides=(2,2)."""
    model = BetaVaeMonai(
        spatial_dims=2,
        in_shape=(1, 64, 64),
        out_channels=1,
        latent_size=128,
        channels=(32, 64),
        strides=(2, 2),
    ).to(device)
    model.eval()
    x = torch.randn(2, 1, 64, 64, device=device)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SimpleNamespace)
    assert out.recon_x.shape == (2, 1, 64, 64)
    assert out.mean.shape == (2, 128)
    assert out.logvar.shape == (2, 128)
    assert out.z.shape == (2, 128)


def test_beta_vae_monai_3d(device):
    """3D VAE: 1ch, 32x32x32, latent=64, channels=(16,32), strides=(2,2)."""
    model = BetaVaeMonai(
        spatial_dims=3,
        in_shape=(1, 32, 32, 32),
        out_channels=1,
        latent_size=64,
        channels=(16, 32),
        strides=(2, 2),
    ).to(device)
    model.eval()
    x = torch.randn(2, 1, 32, 32, 32, device=device)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SimpleNamespace)
    assert out.recon_x.shape == (2, 1, 32, 32, 32)
    assert out.mean.shape == (2, 64)
    assert out.logvar.shape == (2, 64)
    assert out.z.shape == (2, 64)
