import torch

from viscy.unet.networks.fcmae import (
    FullyConvolutionalMAE,
    MaskedAdaptiveProjection,
    MaskedConvNeXtV2Block,
    MaskedConvNeXtV2Stage,
    MaskedMultiscaleEncoder,
    PixelToVoxelShuffleHead,
    generate_mask,
    masked_patchify,
    masked_unpatchify,
    upsample_mask,
)


def test_generate_mask():
    w = 64
    s = 16
    m = 0.75
    mask = generate_mask((2, 3, w, w), stride=s, mask_ratio=m, device="cpu")
    assert mask.shape == (2, 1, w // s, w // s)
    assert mask.dtype == torch.bool
    ratio = mask.sum((2, 3)) / mask.numel() * mask.shape[0]
    assert torch.allclose(ratio, torch.ones_like(ratio) * m)


def test_masked_patchify():
    b, c, h, w = 2, 3, 4, 8
    x = torch.rand(b, c, h, w)
    mask_ratio = 0.75
    mask = generate_mask(x.shape, stride=2, mask_ratio=mask_ratio, device=x.device)
    mask = upsample_mask(mask, x.shape)
    feat = masked_patchify(x, ~mask)
    assert feat.shape == (b, int(h * w * (1 - mask_ratio)), c)


def test_unmasked_patchify_roundtrip():
    x = torch.rand(2, 3, 4, 8)
    y = masked_unpatchify(masked_patchify(x), out_shape=x.shape)
    assert torch.allclose(x, y)


def test_masked_patchify_roundtrip():
    x = torch.rand(2, 3, 4, 8)
    mask = generate_mask(x.shape, stride=2, mask_ratio=0.5, device=x.device)
    mask = upsample_mask(mask, x.shape)
    y = masked_unpatchify(masked_patchify(x, ~mask), out_shape=x.shape, unmasked=~mask)
    assert torch.all((y == 0) ^ (x == y))
    assert torch.all((y == 0)[:, 0:1] == mask)


def test_masked_convnextv2_block() -> None:
    x = torch.rand(2, 3, 4, 5)
    mask = generate_mask(x.shape, stride=1, mask_ratio=0.5, device=x.device)
    block = MaskedConvNeXtV2Block(3, 3 * 2)
    unmasked_out = block(x)
    assert len(unmasked_out.unique()) == x.numel() * 2
    all_unmasked = torch.ones_like(mask)
    empty_masked_out = block(x, all_unmasked)
    assert torch.allclose(unmasked_out, empty_masked_out)
    block = MaskedConvNeXtV2Block(3, 3)
    masked_out = block(x, mask)
    assert len(masked_out.unique()) == mask.sum() * x.shape[1] + 1


def test_masked_convnextv2_stage():
    x = torch.rand(2, 3, 16, 16)
    mask = generate_mask(x.shape, stride=4, mask_ratio=0.5, device=x.device)
    stage = MaskedConvNeXtV2Stage(3, 3, kernel_size=7, stride=2, num_blocks=2)
    out = stage(x)
    assert out.shape == (2, 3, 8, 8)
    masked_out = stage(x, mask)
    assert not torch.allclose(masked_out, out)


def test_adaptive_projection():
    proj = MaskedAdaptiveProjection(
        3, 12, kernel_size_2d=4, kernel_depth=5, in_stack_depth=5
    )
    assert proj(torch.rand(2, 3, 5, 8, 8)).shape == (2, 12, 2, 2)
    assert proj(torch.rand(2, 3, 1, 12, 16)).shape == (2, 12, 3, 4)
    mask = generate_mask((1, 3, 5, 8, 8), stride=4, mask_ratio=0.6, device="cpu")
    masked_out = proj(torch.rand(1, 3, 5, 16, 16), mask)
    assert masked_out.shape == (1, 12, 4, 4)
    proj = MaskedAdaptiveProjection(
        3, 12, kernel_size_2d=(2, 4), kernel_depth=5, in_stack_depth=15
    )
    assert proj(torch.rand(2, 3, 15, 6, 8)).shape == (2, 12, 3, 2)


def test_masked_multiscale_encoder():
    xy_size = 64
    dims = [12, 24, 48, 96]
    x = torch.rand(2, 3, 5, xy_size, xy_size)
    encoder = MaskedMultiscaleEncoder(3, dims=dims)
    auto_masked_features, _ = encoder(x, mask_ratio=0.5)
    target_shape = list(x.shape)
    target_shape.pop(1)
    assert len(auto_masked_features) == 4
    for i, (dim, afeat) in enumerate(zip(dims, auto_masked_features)):
        assert afeat.shape[0] == x.shape[0]
        assert afeat.shape[1] == dim
        stride = 2 * 2 ** (i + 1)
        assert afeat.shape[2] == afeat.shape[3] == xy_size // stride


def test_pixel_to_voxel_shuffle_head():
    head = PixelToVoxelShuffleHead(240, 3, out_stack_depth=5, xy_scaling=4)
    x = torch.rand(2, 240, 16, 16)
    y = head(x)
    assert y.shape == (2, 3, 5, 64, 64)


def test_fcmae():
    x = torch.rand(2, 3, 5, 128, 128)
    model = FullyConvolutionalMAE(3, 3)
    y, m = model(x)
    assert y.shape == x.shape
    assert m is None
    y, m = model(x, mask_ratio=0.6)
    assert y.shape == x.shape
    assert m.shape == (2, 1, 128, 128)


def test_fcmae_head_conv():
    x = torch.rand(2, 3, 5, 128, 128)
    model = FullyConvolutionalMAE(
        3, 3, head_conv=True, head_conv_expansion_ratio=4, head_conv_pool=True
    )
    y, m = model(x)
    assert y.shape == x.shape
    assert m is None
    y, m = model(x, mask_ratio=0.6)
    assert y.shape == x.shape
    assert m.shape == (2, 1, 128, 128)
