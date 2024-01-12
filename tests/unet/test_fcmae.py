import torch

from viscy.unet.networks.fcmae import (
    MaskedAdaptiveProjection,
    MaskedConvNeXtV2Block,
    MaskedConvNeXtV2Stage,
    MaskedGlobalResponseNorm,
    MaskedMultiscaleEncoder,
    upsample_mask,
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


def test_adaptive_projection() -> None:
    proj = MaskedAdaptiveProjection(
        3, 12, kernel_size_2d=4, kernel_depth=5, in_stack_depth=5
    )
    assert proj(torch.rand(2, 3, 5, 8, 8)).shape == (2, 12, 2, 2)
    assert proj(torch.rand(2, 3, 1, 12, 16)).shape == (2, 12, 3, 4)
    proj = MaskedAdaptiveProjection(
        3, 12, kernel_size_2d=(2, 4), kernel_depth=5, in_stack_depth=15
    )
    assert proj(torch.rand(2, 3, 15, 6, 8)).shape == (2, 12, 3, 2)


def test_masked_multiscale_encoder() -> None:
    xy_size = 64
    dims = [12, 24, 48, 96]
    x = torch.rand(2, 3, 5, xy_size, xy_size)
    encoder = MaskedMultiscaleEncoder(3, dims=dims)
    auto_masked_features, mask = encoder(x, mask_ratio=0.5)
    target_shape = list(x.shape)
    target_shape.pop(1)
    pre_masked_features = encoder(x * ~upsample_mask(mask, target_shape).unsqueeze(1))
    assert len(auto_masked_features) == len(pre_masked_features) == 4
    for i, (dim, afeat, pfeat) in enumerate(
        zip(dims, auto_masked_features, pre_masked_features)
    ):
        assert afeat.shape[0] == x.shape[0]
        assert afeat.shape[1] == dim
        stride = 2 * 2 ** (i + 1)
        assert afeat.shape[2] == afeat.shape[3] == xy_size // stride
        assert torch.allclose(afeat, pfeat, rtol=5e-2, atol=5e-2), (
            i,
            (afeat - pfeat).abs().max(),
        )
