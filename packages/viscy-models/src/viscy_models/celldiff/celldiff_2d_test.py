"""CELLDiffNet at Z=1 with a per-axis patch (the CellDiff-2D configuration)."""

import pytest
import torch

from viscy_models.celldiff.celldiff_net import CELLDiffNet
from viscy_models.celldiff.modules.patch_embed_3d import normalize_patch_size

#: Small enough to build and run on CPU in a test; same topology as the recipe.
_NET_KWARGS = dict(
    in_channels=1,
    dims=[8, 16, 32, 32],
    num_res_block=[2, 2, 2],
    hidden_size=64,
    num_heads=4,
    dim_head=16,
    num_hidden_layers=1,
)


def test_normalize_patch_size_expands_int_and_validates() -> None:
    """An int stays cubic; a triple passes through; anything else raises."""
    assert normalize_patch_size(4) == (4, 4, 4)
    assert normalize_patch_size([1, 4, 4]) == (1, 4, 4)
    assert normalize_patch_size((2, 3, 4)) == (2, 3, 4)
    for bad in (0, -1, [1, 4], [0, 4, 4], [1, 4, 4, 4]):
        with pytest.raises(ValueError):
            normalize_patch_size(bad)


@pytest.mark.parametrize(
    ("spatial", "patch"),
    [([1, 64, 64], [1, 4, 4]), ([1, 32, 32], [1, 2, 2]), ([8, 64, 64], 4)],
)
def test_celldiff_net_roundtrips_spatial_shape(spatial: list[int], patch: int | list[int]) -> None:
    """Z=1 with a (1, p, p) patch runs, and the cubic 3D path is unchanged."""
    net = CELLDiffNet(input_spatial_size=spatial, patch_size=patch, **_NET_KWARGS)
    x = torch.randn(2, 1, *spatial)
    out = net(x, cond=torch.randn(2, 1, *spatial), t=torch.rand(2))
    assert out.shape == (2, 1, *spatial)
    assert torch.isfinite(out).all()


def test_celldiff_net_z1_backward_reaches_every_parameter() -> None:
    """A frozen/disconnected patch path would silently starve gradients at Z=1."""
    spatial = [1, 64, 64]
    net = CELLDiffNet(input_spatial_size=spatial, patch_size=[1, 4, 4], **_NET_KWARGS)
    out = net(torch.randn(1, 1, *spatial), cond=torch.randn(1, 1, *spatial), t=torch.rand(1))
    out.square().mean().backward()
    starved = [
        name
        for name, p in net.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
    ]
    assert not starved, f"parameters with missing/non-finite grads: {starved[:5]}"


def test_cubic_patch_is_rejected_at_z1() -> None:
    """The failure that motivated per-axis patches must stay a loud error."""
    with pytest.raises(ValueError, match="not divisible by its patch extent"):
        CELLDiffNet(input_spatial_size=[1, 64, 64], patch_size=4, **_NET_KWARGS)


def test_z1_patch_grid_drops_the_z_axis() -> None:
    """The token grid must be (1, H/2**n/p, W/2**n/p) — no phantom Z tokens."""
    net = CELLDiffNet(input_spatial_size=[1, 64, 64], patch_size=[1, 4, 4], **_NET_KWARGS)
    # 3 downsamples of stride (1, 2, 2): 64 -> 8, then patch 4 -> 2
    assert net.bottleneck.latent_grid_size == [1, 2, 2]
    assert net.bottleneck.img_pos_embed.shape[1] == 4
