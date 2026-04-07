import torch

from viscy_transforms import BatchedRandAffined


def test_affine_key_consistency():
    """Source and target receive identical spatial transforms."""
    t = BatchedRandAffined(
        keys=["source", "target"],
        prob=1.0,
        rotate_range=[3.14, 0, 0],
        scale_range=[0.8, 1.2],
    )
    base = torch.zeros(2, 1, 8, 32, 32)
    base[:, :, 2:6, 8:24, 8:24] = 1.0
    sample = {"source": base.clone(), "target": base.clone()}

    torch.manual_seed(0)
    out = t(sample)

    assert torch.equal(out["source"], out["target"])
    # Verify the transform actually changed the data.
    assert not torch.equal(out["source"], base)


def test_affine_isotropic_scale():
    """With isotropic_scale=True, all axes get the same scale factor."""
    t = BatchedRandAffined(
        keys=["source", "target"],
        prob=1.0,
        scale_range=[0.5, 1.5],
        isotropic_scale=True,
    )
    x = torch.randn(8, 1, 8, 32, 32)
    # Verify params directly: all three axes must have the same scale per sample.
    params = t.random_affine.forward_parameters(x.shape)
    BatchedRandAffined._make_scale_isotropic(params)
    scale = params["scale"]
    assert torch.equal(scale[:, 0], scale[:, 1])
    assert torch.equal(scale[:, 0], scale[:, 2])

    # Also verify end-to-end key consistency.
    base = torch.zeros(4, 1, 8, 24, 48)
    base[:, :, 2:6, 8:16, 16:32] = 1.0
    sample = {"source": base.clone(), "target": base.clone()}
    torch.manual_seed(0)
    out = t(sample)
    assert torch.equal(out["source"], out["target"])
    assert not torch.equal(out["source"], base)


def test_affine_anisotropic_scale_default():
    """Without isotropic_scale, flat scale_range samples independently per axis."""
    t = BatchedRandAffined(
        keys=["img"],
        prob=1.0,
        scale_range=[0.5, 1.5],
    )
    assert not t._isotropic_scale
    x = torch.randn(8, 1, 8, 32, 32)
    params = t.random_affine.forward_parameters(x.shape)
    scale = params["scale"]
    # With 8 samples, it's extremely unlikely all axes match for every sample.
    any_differ = not all(torch.equal(scale[i, 0:1].expand(3), scale[i]) for i in range(8))
    assert any_differ, "Per-axis scale should differ across axes by default"


def test_affine_prob_zero_passthrough():
    """With prob=0 the transform should be a no-op."""
    t = BatchedRandAffined(
        keys=["source", "target"],
        prob=0.0,
        rotate_range=[3.14, 0, 0],
        scale_range=[0.5, 1.5],
    )
    base = torch.randn(2, 1, 8, 32, 32)
    sample = {"source": base.clone(), "target": base.clone()}
    out = t(sample)
    assert torch.allclose(out["source"], base, atol=1e-6)
    assert torch.allclose(out["target"], base, atol=1e-6)


def test_affine_per_sample_variation():
    """Different samples in the batch should get different transforms."""
    t = BatchedRandAffined(
        keys=["img"],
        prob=1.0,
        rotate_range=[3.14, 0, 0],
        scale_range=[0.5, 1.5],
    )
    # Use identical data for all batch items so any difference is from the transform.
    single = torch.randn(1, 1, 8, 32, 32)
    batch = single.expand(4, -1, -1, -1, -1).clone()
    sample = {"img": batch}

    torch.manual_seed(0)
    out = t(sample)

    # At least two samples should differ (probabilistically guaranteed with seed 0).
    all_same = all(torch.equal(out["img"][0], out["img"][i]) for i in range(1, 4))
    assert not all_same, "All batch samples got the same transform"


def test_affine_scale_range_not_inverted():
    """scale_range=(min, max) must not be axis-inverted.

    Regression test: ``_invert_per_axis`` previously reversed the
    (min, max) tuple as if it were per-axis (Z, Y, X) values.
    """
    t = BatchedRandAffined(
        keys=["img"],
        prob=1.0,
        scale_range=[0.5, 1.5],
    )
    # Generate params many times and check all scale values are in [0.5, 1.5].
    x = torch.randn(8, 1, 8, 32, 32)
    for _ in range(10):
        params = t.random_affine.forward_parameters(x.shape)
        assert params["scale"].min() >= 0.5 - 0.01
        assert params["scale"].max() <= 1.5 + 0.01


def test_affine_per_axis_scale():
    """Per-axis scale_range with different Z vs YX ranges.

    scale_range=[[z_min, z_max], [y_min, y_max], [x_min, x_max]] should
    generate independent per-axis scale factors within each range.
    """
    t = BatchedRandAffined(
        keys=["source", "target"],
        prob=1.0,
        scale_range=[[0.9, 1.1], [0.5, 1.5], [0.5, 1.5]],
    )
    x = torch.randn(16, 1, 8, 32, 32)
    params = t.random_affine.forward_parameters(x.shape)
    scale = params["scale"]  # (B, 3) in Kornia XYZ order

    # Kornia axis 2 = Z (was first in ZYX config, reversed to last).
    z_scale = scale[:, 2]
    assert z_scale.min() >= 0.9 - 0.01
    assert z_scale.max() <= 1.1 + 0.01

    # Kornia axes 0,1 = X,Y — wider range.
    xy_scale = scale[:, :2]
    assert xy_scale.min() >= 0.5 - 0.01
    assert xy_scale.max() <= 1.5 + 0.01


def test_affine_per_axis_scale_key_consistency():
    """Per-axis scale should still share params across keys."""
    t = BatchedRandAffined(
        keys=["source", "target"],
        prob=1.0,
        rotate_range=[3.14, 0, 0],
        scale_range=[[0.9, 1.1], [0.5, 1.5], [0.5, 1.5]],
    )
    base = torch.zeros(2, 1, 8, 32, 32)
    base[:, :, 2:6, 8:24, 8:24] = 1.0
    sample = {"source": base.clone(), "target": base.clone()}

    torch.manual_seed(0)
    out = t(sample)

    assert torch.equal(out["source"], out["target"])
    assert not torch.equal(out["source"], base)


def test_affine_rotation_axis_zyx():
    """rotate_range=[angle, 0, 0] should rotate in the YX plane (around Z).

    Verifies MONAI ZYX → Kornia XYZ axis conversion is correct.
    """
    # Rotation only around Z (the first element in MONAI ZYX order).
    t = BatchedRandAffined(
        keys=["img"],
        prob=1.0,
        rotate_range=[3.14, 0, 0],
    )
    # Asymmetric pattern: a block in the top-left of the YX plane.
    x = torch.zeros(1, 1, 8, 32, 32)
    x[0, 0, :, 0:16, 0:8] = 1.0

    torch.manual_seed(0)
    out = t({"img": x})

    # Rotation around Z changes the YX distribution but not the Z extent.
    # The Z centroid (dim 2) should stay near the middle.
    inp_z_centroid = (x[0, 0] > 0.5).float().nonzero()[:, 0].float().mean()
    out_z_centroid = (out["img"][0, 0] > 0.01).float().nonzero()[:, 0].float().mean()
    assert abs(inp_z_centroid - out_z_centroid) < 2.0, "Z centroid shifted — rotation around wrong axis"

    # The YX distribution should change.
    inp_y_centroid = (x[0, 0] > 0.5).float().nonzero()[:, 1].float().mean()
    out_y_centroid = (out["img"][0, 0] > 0.01).float().nonzero()[:, 1].float().mean()
    assert abs(inp_y_centroid - out_y_centroid) > 1.0, "YX unchanged — rotation not applied"
