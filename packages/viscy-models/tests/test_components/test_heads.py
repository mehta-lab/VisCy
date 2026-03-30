"""Tests for head modules in viscy_models.components.heads."""

import torch

from viscy_models.components.heads import (
    ClassificationHead,
    PixelToVoxelHead,
    PixelToVoxelShuffleHead,
    UnsqueezeHead,
)

IN_DIMS = 64
NUM_CLASSES = 10


def test_classification_head_output_shape(device):
    """ClassificationHead forward produces logits of shape (B, num_classes)."""
    head = ClassificationHead("gene_ko", "gene_ko", in_dims=IN_DIMS, hidden_dims=32, num_classes=NUM_CLASSES).to(device)
    x = torch.randn(4, IN_DIMS, device=device)
    logits = head(x)
    assert logits.shape == (4, NUM_CLASSES)


def test_classification_head_compute_loss(device):
    """compute_loss returns a scalar cross-entropy loss."""
    head = ClassificationHead("gene_ko", "gene_ko", in_dims=IN_DIMS, hidden_dims=32, num_classes=NUM_CLASSES).to(device)
    logits = torch.randn(4, NUM_CLASSES, device=device)
    y = torch.randint(0, NUM_CLASSES, (4,), device=device)
    loss = head.compute_loss(logits, y)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_classification_head_log_metrics(device):
    """log_metrics calls log_fn with loss, top1, and top-k accuracy keys."""
    head = ClassificationHead(
        "gene_ko", "gene_ko", in_dims=IN_DIMS, hidden_dims=32, num_classes=NUM_CLASSES, top_k=5
    ).to(device)
    logits = torch.randn(4, NUM_CLASSES, device=device)
    y = torch.randint(0, NUM_CLASSES, (4,), device=device)
    loss = head.compute_loss(logits, y)

    logged = {}
    head.log_metrics({"loss": loss, "logits": logits, "y": y}, lambda k, v: logged.update({k: v}), "train")

    assert "loss/aux/gene_ko/train" in logged
    assert "metrics/acc_top1/gene_ko/train" in logged
    assert "metrics/acc_top5/gene_ko/train" in logged


def test_classification_head_loss_weight():
    """loss_weight attribute is stored correctly."""
    head = ClassificationHead(
        "gene_ko", "gene_ko", in_dims=IN_DIMS, hidden_dims=32, num_classes=NUM_CLASSES, loss_weight=0.3
    )
    assert head.loss_weight == 0.3
    assert head.head_name == "gene_ko"
    assert head.batch_key == "gene_ko"


def test_pixel_to_voxel_head_output_shape(device):
    """PixelToVoxelHead produces 5D output with correct channels and depth.

    Uses params matching UNeXt2's actual usage:
    in_channels = (out_stack_depth + 2) * out_channels * 2^2 * expansion_ratio
               = (5 + 2) * 2 * 4 * 4 = 224.
    After pixelshuffle 2x: 224//4=56 channels at 128x128.
    Reshape to 3D: 56//7=8 channels at depth=7.
    After 3D conv (padding=(0,1,1)): depth=5, then pixelshuffle 2x -> 256x256.
    Output: (1, 2, 5, 256, 256).
    """
    head = PixelToVoxelHead(
        in_channels=224,
        out_channels=2,
        out_stack_depth=5,
        expansion_ratio=4,
        pool=False,
    ).to(device)
    x = torch.randn(1, 224, 64, 64, device=device)
    out = head(x)
    assert out.ndim == 5
    assert out.shape[0] == 1
    assert out.shape[1] == 2
    assert out.shape[2] == 5
    assert out.shape == (1, 2, 5, 256, 256)


def test_unsqueeze_head(device):
    """UnsqueezeHead adds a depth=1 dimension at position 2."""
    head = UnsqueezeHead().to(device)
    x = torch.randn(2, 16, 32, 32, device=device)
    out = head(x)
    assert out.shape == (2, 16, 1, 32, 32)


def test_pixel_to_voxel_shuffle_head_output_shape(device):
    """PixelToVoxelShuffleHead upsamples 2D to 3D with pixel shuffle.

    Uses params matching FCMAE's actual usage:
    in_channels = out_channels * out_stack_depth * xy_scaling^2 = 2 * 5 * 4^2 = 160.
    UpSample pixelshuffle 4x: out_channels=5*2=10.
    Need in_channels/scale^2 = 160/16 = 10 = target out. Correct.
    Reshape: (1, 2, 5, 64, 64).
    """
    head = PixelToVoxelShuffleHead(
        in_channels=160,
        out_channels=2,
        out_stack_depth=5,
        xy_scaling=4,
        pool=False,
    ).to(device)
    x = torch.randn(1, 160, 16, 16, device=device)
    out = head(x)
    assert out.shape == (1, 2, 5, 64, 64)
