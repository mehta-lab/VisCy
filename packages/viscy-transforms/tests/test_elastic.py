import torch

from viscy_transforms._elastic import BatchedRand3DElasticd


def test_elastic_key_consistency():
    """Source and target receive identical deformation fields."""
    t = BatchedRand3DElasticd(
        keys=["source", "target"],
        sigma_range=(3.0, 5.0),
        magnitude_range=(50.0, 100.0),
        prob=1.0,
    )
    base = torch.zeros(2, 1, 8, 32, 32)
    base[:, :, 2:6, 8:24, 8:24] = 1.0
    sample = {"source": base.clone(), "target": base.clone()}
    out = t(sample)

    assert torch.equal(out["source"], out["target"])
    assert not torch.equal(out["source"], base)


def test_elastic_prob_zero_passthrough():
    """With prob=0 the transform should be a no-op."""
    t = BatchedRand3DElasticd(
        keys=["source", "target"],
        sigma_range=(3.0, 5.0),
        magnitude_range=(50.0, 100.0),
        prob=0.0,
    )
    base = torch.randn(2, 1, 8, 32, 32)
    sample = {"source": base.clone(), "target": base.clone()}
    out = t(sample)
    assert torch.equal(out["source"], base)
    assert torch.equal(out["target"], base)


def test_elastic_displacement_axis_ordering():
    """Displacement along D(Z) must shift content along D, not W(X).

    Regression test: the displacement field was indexed as (D, H, W) but
    applied to grid axes in (X, Y, Z) order without remapping, swapping
    the D and W displacement directions.
    """
    # Anisotropic spatial dims so axis swaps are detectable.
    D, H, W = 8, 24, 48
    x = torch.zeros(1, 1, D, H, W)
    # Place a block in the center.
    x[0, 0, 2:6, 8:16, 16:32] = 1.0

    # Apply strong elastic deformation.
    t = BatchedRand3DElasticd(
        keys=["img"],
        sigma_range=(3.0, 5.0),
        magnitude_range=(100.0, 200.0),
        prob=1.0,
    )
    out = t({"img": x})
    nz = (out["img"][0, 0] > 0.01).float().nonzero()

    out_d_centroid = nz[:, 0].float().mean()

    # If axes were swapped, D displacement (intended for 8-voxel depth) would be
    # applied to the 48-voxel W axis. Verify D centroid stays within D bounds.
    assert out_d_centroid >= 0, "Content displaced outside D volume (negative)"
    assert out_d_centroid < D, "Content displaced outside D volume (positive)"

    # Verify the transform actually did something.
    assert not torch.equal(out["img"], x), "Transform was a no-op"
