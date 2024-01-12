import torch

from viscy.unet.networks.fcmae import (
    MaskedConvNeXtV2Block,
    MaskedConvNeXtV2Stage,
    MaskedGlobalResponseNorm,
)


def test_masked_grn() -> None:
    x = torch.rand(2, 3, 4, 5)
    grn = MaskedGlobalResponseNorm(3, channels_last=False)
    grn.gamma.data = torch.ones_like(grn.gamma.data)
    mask = torch.ones((1, 1, 4, 5), dtype=torch.bool)
    mask[:, :, 2:, 2:] = False
    normalized = grn(x)
    assert not torch.allclose(normalized, x)
    assert torch.allclose(grn(x, mask)[:, :, 2:, 2:], grn(x[:, :, 2:, 2:]))
    grn = MaskedGlobalResponseNorm(5, channels_last=True)
    grn.gamma.data = torch.ones_like(grn.gamma.data)
    mask = torch.ones((1, 3, 4, 1), dtype=torch.bool)
    mask[:, 1:, 2:, :] = False
    assert torch.allclose(grn(x, mask)[:, 1:, 2:, :], grn(x[:, 1:, 2:, :]))


def test_masked_convnextv2_block() -> None:
    x = torch.rand(2, 3, 4, 5)
    mask = x[0, 0] > 0.5
    block = MaskedConvNeXtV2Block(3, 3 * 2)
    assert len(block(x).unique()) == x.numel() * 2
    block = MaskedConvNeXtV2Block(3, 3)
    masked_out = block(x, mask)
    assert len(masked_out[:, :, mask].unique()) == x.shape[1]


def test_masked_convnextv2_stage() -> None:
    x = torch.rand(2, 3, 16, 16)
    mask = torch.rand(4, 4) > 0.5
    stage = MaskedConvNeXtV2Stage(3, 3, kernel_size=7, stride=2, num_blocks=2)
    out = stage(x)
    assert out.shape == (2, 3, 8, 8)
    masked_out = stage(x, mask)
    assert not torch.allclose(masked_out, out)
