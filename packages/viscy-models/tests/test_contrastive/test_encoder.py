"""Forward-pass tests for ContrastiveEncoder model."""

import torch

from viscy_models.contrastive import ContrastiveEncoder


def test_contrastive_encoder_convnext_tiny(device):
    """ConvNeXt-tiny backbone: 2ch in, 15 Z-slices, embedding=768, projection=128."""
    model = ContrastiveEncoder(
        backbone="convnext_tiny",
        in_channels=2,
        in_stack_depth=15,
        embedding_dim=768,
        projection_dim=128,
        pretrained=False,
    ).to(device)
    model.eval()
    x = torch.randn(2, 2, 15, 64, 64, device=device)
    with torch.no_grad():
        embedding, projection = model(x)
    assert embedding.shape == (2, 768)
    assert projection.shape == (2, 128)


def test_contrastive_encoder_resnet50(device):
    """ResNet50 backbone (exercises bug fix): 1ch in, 10 Z-slices, embedding=2048."""
    model = ContrastiveEncoder(
        backbone="resnet50",
        in_channels=1,
        in_stack_depth=10,
        stem_kernel_size=(5, 4, 4),
        stem_stride=(5, 4, 4),
        embedding_dim=2048,
        projection_dim=128,
        pretrained=False,
    ).to(device)
    model.eval()
    x = torch.randn(2, 1, 10, 64, 64, device=device)
    with torch.no_grad():
        embedding, projection = model(x)
    assert embedding.shape == (2, 2048)
    assert projection.shape == (2, 128)


def test_contrastive_encoder_custom_stem(device):
    """ConvNeXt-tiny with custom stem: kernel=(3,2,2), stride=(3,2,2), depth=9."""
    model = ContrastiveEncoder(
        backbone="convnext_tiny",
        in_channels=1,
        in_stack_depth=9,
        stem_kernel_size=(3, 2, 2),
        stem_stride=(3, 2, 2),
        embedding_dim=768,
        projection_dim=64,
        pretrained=False,
    ).to(device)
    model.eval()
    x = torch.randn(2, 1, 9, 64, 64, device=device)
    with torch.no_grad():
        embedding, projection = model(x)
    assert embedding.shape == (2, 768)
    assert projection.shape == (2, 64)
