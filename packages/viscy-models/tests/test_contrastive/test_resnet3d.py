"""Forward-pass tests for ResNet3dEncoder model."""

import torch

from viscy_models.contrastive import ResNet3dEncoder


def test_resnet3d_encoder_resnet18(device):
    """ResNet-18 backbone: 1ch in, embedding=512, projection=128."""
    model = ResNet3dEncoder(
        backbone="resnet18",
        in_channels=1,
        embedding_dim=512,
        projection_dim=128,
        pretrained=False,
    ).to(device)
    model.eval()
    x = torch.randn(2, 1, 16, 16, 16, device=device)
    with torch.no_grad():
        embedding, projection = model(x)
    assert embedding.shape == (2, 512)
    assert projection.shape == (2, 128)


def test_resnet3d_encoder_resnet10(device):
    """ResNet-10 backbone: 2ch in, embedding=512, projection=64."""
    model = ResNet3dEncoder(
        backbone="resnet10",
        in_channels=2,
        embedding_dim=512,
        projection_dim=64,
        pretrained=False,
    ).to(device)
    model.eval()
    x = torch.randn(2, 2, 16, 16, 16, device=device)
    with torch.no_grad():
        embedding, projection = model(x)
    assert embedding.shape == (2, 512)
    assert projection.shape == (2, 64)
