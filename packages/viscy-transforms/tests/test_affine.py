import math

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


def test_compute_scale_floor_known_angles():
    """_compute_scale_floor returns correct values for known geometries."""
    B = 4
    # Angles: 0°, 45°, 90°, 180° around Z (Kornia XYZ order: col 2 = Z).
    angles_deg = torch.tensor([[0, 0, 0.0], [0, 0, 45.0], [0, 0, 90.0], [0, 0, 180.0]])
    # CellDiff-like: source 624×624, crop 512×512, Z: source 13, crop 8.
    input_shape = torch.Size([B, 1, 13, 624, 624])
    safe_crop = (8, 512, 512)

    s_floor = BatchedRandAffined._compute_scale_floor(angles_deg, input_shape, safe_crop)
    assert s_floor.shape == (B, 3)

    R = 624 / 512  # 1.21875
    # θ=0°: k=1, s_min = 1/R
    assert math.isclose(s_floor[0, 0].item(), 1 / R, rel_tol=1e-5)
    # θ=45°: k=√2, s_min = √2/R
    assert math.isclose(s_floor[1, 0].item(), math.sqrt(2) / R, rel_tol=1e-5)
    # θ=90°: k=1 (for square crop), s_min = 1/R
    assert math.isclose(s_floor[2, 0].item(), 1 / R, rel_tol=1e-5)
    # θ=180°: k=1, s_min = 1/R
    assert math.isclose(s_floor[3, 0].item(), 1 / R, rel_tol=1e-5)

    # Z axis: s_min_z = 8 / 13
    for i in range(B):
        assert math.isclose(s_floor[i, 2].item(), 8 / 13, rel_tol=1e-5)


def test_safe_crop_size_clamps_infeasible_scale():
    """Infeasible scale+rotation combos get clamped; safe combos pass through."""
    # CellDiff geometry: 624→512, full Z rotation.
    t = BatchedRandAffined(
        keys=["source", "target"],
        prob=1.0,
        rotate_range=[3.14, 0, 0],
        scale_range=[[0.7, 1.3], [0.5, 1.5], [0.5, 1.5]],
        safe_crop_size=[8, 512, 512],
    )
    x = torch.randn(16, 1, 13, 624, 624)
    params = t.random_affine.forward_parameters(x.shape)

    # Record original scale.
    orig_scale = params["scale"].clone()

    # Compute floor and apply clamping (replicate __call__ logic).
    s_floor = BatchedRandAffined._compute_scale_floor(params["angles"], x.shape, (8, 512, 512))
    clamped_scale = torch.max(orig_scale, s_floor)

    # Every clamped value should be ≥ the floor.
    assert (clamped_scale >= s_floor - 1e-6).all()
    # Samples that were already above the floor should be unchanged.
    above_mask = orig_scale >= s_floor
    assert torch.allclose(clamped_scale[above_mask], orig_scale[above_mask])
    # Samples that were below should be raised to exactly the floor.
    below_mask = orig_scale < s_floor
    if below_mask.any():
        assert torch.allclose(clamped_scale[below_mask], s_floor[below_mask])


def test_safe_crop_size_eliminates_zero_corners():
    """With safe_crop_size, no output pixel should sample outside the source."""
    # Use a non-zero constant input so any zero pixel indicates out-of-bounds.
    t = BatchedRandAffined(
        keys=["img"],
        prob=1.0,
        rotate_range=[3.14, 0, 0],
        scale_range=[0.5, 1.5],
        safe_crop_size=[8, 32, 32],
        padding_mode="zeros",
    )
    # Fill with 1.0 — after affine, any 0.0 pixel means out-of-bounds sampling.
    x = torch.ones(4, 1, 10, 48, 48)

    # Run multiple seeds to cover various rotation angles.
    for seed in range(20):
        torch.manual_seed(seed)
        out = t({"img": x})
        # Center-crop to the safe region (the guarantee).
        d, h, w = 8, 32, 32
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        crop = out["img"][
            :,
            :,
            (D - d) // 2 : (D + d) // 2,
            (H - h) // 2 : (H + h) // 2,
            (W - w) // 2 : (W + w) // 2,
        ]
        assert (crop > 0).all(), f"Seed {seed}: zero pixels found in safe crop region — coverage guarantee violated"


def test_safe_crop_size_preserves_key_consistency():
    """safe_crop_size should not break source/target consistency."""
    t = BatchedRandAffined(
        keys=["source", "target"],
        prob=1.0,
        rotate_range=[3.14, 0, 0],
        scale_range=[0.5, 1.5],
        safe_crop_size=[8, 32, 32],
    )
    base = torch.ones(2, 1, 10, 48, 48)
    base[:, :, 2:8, 10:38, 10:38] = 2.0
    sample = {"source": base.clone(), "target": base.clone()}

    torch.manual_seed(42)
    out = t(sample)
    assert torch.equal(out["source"], out["target"])
