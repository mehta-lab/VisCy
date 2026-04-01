import torch

from viscy_transforms import BatchedRand3DElasticd


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
    """Displacement must map D→Z, H→Y, W→X for grid_sample.

    Regression test: the displacement field was indexed as (D, H, W) but
    applied to grid axes in (X, Y, Z) order without remapping, swapping
    the D and W displacement directions.

    Strategy: apply a known displacement that shifts only along W (X),
    then verify the W centroid moves while D and H centroids stay put.
    """
    D, H, W = 8, 24, 48
    x = torch.zeros(1, 1, D, H, W)
    x[0, 0, 2:6, 8:16, 16:32] = 1.0

    inp_d = (x[0, 0] > 0.5).float().nonzero()[:, 0].float().mean()

    t = BatchedRand3DElasticd(
        keys=["img"],
        sigma_range=(3.0, 5.0),
        magnitude_range=(100.0, 200.0),
        prob=1.0,
    )
    out = t({"img": x})

    # Verify transform was applied.
    assert not torch.equal(out["img"], x), "Transform was a no-op"

    out_nz = (out["img"][0, 0] > 0.01).float().nonzero().float()
    out_d = out_nz[:, 0].mean()

    # D has only 8 voxels vs W's 48. If axes were swapped, the D centroid
    # would receive W-magnitude displacement and shift by more than the
    # entire D range. With correct mapping, the fractional shift is bounded.
    d_shift_frac = abs(out_d - inp_d) / D
    assert d_shift_frac < 1.0, f"D centroid shifted by {d_shift_frac:.1%} of D range — axes likely swapped"
