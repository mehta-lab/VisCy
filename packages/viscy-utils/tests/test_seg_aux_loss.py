"""Unit tests for SegAuxDice."""

import numpy as np
import pytest
import torch
from scipy import ndimage

from viscy_utils.losses import SegAuxDice
from viscy_utils.losses.seg_aux import _masked_median, _quantile_sorted
from viscy_utils.losses.seg_aux_maps import box_max, sauna_signed_map, sauna_weight_map, soft_skeleton, squared_edt


def _blob_batch(shape: tuple[int, ...], seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Target = bright box on a noisy background; mask = the box."""
    g = torch.Generator().manual_seed(seed)
    target = 0.1 * torch.randn(shape, generator=g)
    mask = torch.zeros(shape)
    mask[..., shape[-2] // 4 : shape[-2] // 2, shape[-1] // 4 : shape[-1] // 2] = 1.0
    target = target + 2.0 * mask
    return target, mask


@pytest.mark.parametrize("shape", [(2, 1, 1, 32, 32), (2, 2, 4, 16, 16)], ids=["2d", "3d"])
def test_perfect_prediction_beats_perturbed(shape):
    target, mask = _blob_batch(shape)
    loss_fn = SegAuxDice()
    perfect = loss_fn(target.clone(), target, mask)
    g = torch.Generator().manual_seed(1)
    perturbed = loss_fn(target + 0.5 * torch.randn(shape, generator=g), target, mask)
    assert perfect.ndim == 0
    assert perfect < perturbed
    blurred_out = loss_fn(torch.zeros(shape), target, mask)
    assert perfect < blurred_out


def test_gradient_nonzero_for_foreground_far_below_tau():
    """A foreground voxel predicted several knee widths below tau still gets a
    gradient pushing it up; v1's clamped sigmoid gave exactly zero there."""
    target, mask = _blob_batch((1, 1, 1, 32, 32))
    flat = target.flatten()
    tau = torch.quantile(flat, 1 - mask.mean())
    fg = mask.flatten() > 0.5
    s = 0.1 * (flat[fg].median() - flat[~fg].median())
    pred = target.clone()
    fg_idx = (0, 0, 0, 10, 10)
    assert mask[fg_idx] == 1
    pred[fg_idx] = tau - 8 * s
    pred.requires_grad_(True)
    SegAuxDice(c=0.1)(pred, target, mask).backward()
    assert pred.grad[fg_idx] < 0  # increasing pred there lowers the loss


@pytest.mark.parametrize("a,b", [(3.0, -5.0), (0.01, 1.0), (1.0, 0.7)])
def test_affine_invariance(a, b):
    target, mask = _blob_batch((2, 1, 4, 16, 16))
    pred = target + 0.3 * torch.randn(target.shape, generator=torch.Generator().manual_seed(2))
    loss_fn = SegAuxDice(c=0.1)
    ref = loss_fn(pred, target, mask)
    moved = loss_fn(a * pred + b, a * target + b, mask)
    torch.testing.assert_close(moved, ref, rtol=1e-4, atol=1e-5)


def test_masked_median_matches_torch_on_selection():
    g = torch.Generator().manual_seed(7)
    values = torch.randn(4, 101, generator=g)
    keep = torch.rand(4, 101, generator=g) > 0.6
    got = _masked_median(values, keep)
    want = torch.stack([torch.quantile(v[k], 0.5) for v, k in zip(values, keep)])
    torch.testing.assert_close(got, want)


def test_mask_darker_than_background_is_excluded():
    target, mask = _blob_batch((1, 1, 1, 32, 32))
    loss, comps = SegAuxDice()(target.clone(), -target, mask, return_components=True)
    assert comps["n_valid"] == 0


def test_empty_and_full_masks_are_excluded():
    target, mask = _blob_batch((3, 1, 1, 32, 32))
    mask[1] = 0.0
    mask[2] = 1.0
    pred = target + 0.3 * torch.randn(target.shape, generator=torch.Generator().manual_seed(3))
    loss_fn = SegAuxDice()
    loss, comps = loss_fn(pred, target, mask, return_components=True)
    only_valid = loss_fn(pred[:1], target[:1], mask[:1])
    assert comps["n_valid"] == 1
    torch.testing.assert_close(loss, only_valid)
    torch.testing.assert_close(comps["dice"], loss)


def test_all_invalid_is_zero_with_graph():
    target = torch.randn(2, 1, 1, 16, 16)
    pred = torch.randn(2, 1, 1, 16, 16, requires_grad=True)
    loss, comps = SegAuxDice()(pred, target, torch.zeros_like(target), return_components=True)
    assert loss.item() == 0.0
    assert comps["n_valid"] == 0
    assert loss.requires_grad
    loss.backward()
    assert torch.equal(pred.grad, torch.zeros_like(pred))


def test_constant_target_is_excluded():
    target = torch.ones(1, 1, 1, 16, 16)
    mask = torch.zeros_like(target)
    mask[..., :8, :] = 1.0
    loss, comps = SegAuxDice()(torch.randn_like(target), target, mask, return_components=True)
    assert comps["n_valid"] == 0
    assert loss.item() == 0.0


def test_fractional_mask_is_binarized():
    target, mask = _blob_batch((1, 1, 1, 32, 32))
    pred = target + 0.2
    loss_fn = SegAuxDice()
    torch.testing.assert_close(loss_fn(pred, target, mask * 0.8), loss_fn(pred, target, mask))


def test_quantile_matches_torch_quantile():
    rows = torch.randn(4, 1001)
    q = torch.tensor([0.0, 0.25, 0.731, 1.0])
    got = _quantile_sorted(rows.sort(-1).values, q)
    expected = torch.stack([torch.quantile(rows[i], q[i]) for i in range(4)])
    torch.testing.assert_close(got, expected)


def test_quantile_beyond_torch_quantile_size_limit():
    """torch.quantile raises above 2**24 elements; the sort-based path does not."""
    n = 2**24 + 16
    row = torch.arange(n, dtype=torch.float32).unsqueeze(0)
    got = _quantile_sorted(row, torch.tensor([0.5]))
    torch.testing.assert_close(got, torch.tensor([(n - 1) / 2]))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_autocast_computes_in_float32(dtype):
    target, mask = _blob_batch((2, 1, 4, 16, 16))
    pred = (target + 0.3).requires_grad_(True)
    ref = SegAuxDice()(pred, target, mask)
    with torch.autocast(device_type="cpu", dtype=dtype):
        low = SegAuxDice()(pred.to(dtype), target.to(dtype), mask.to(dtype))
    assert low.dtype == torch.float32
    assert torch.isfinite(low)
    torch.testing.assert_close(low, ref, rtol=0.05, atol=0.02)
    low.backward()
    assert torch.isfinite(pred.grad).all()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="share a shape"):
        SegAuxDice()(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 5))


def _mask_disagrees_with_target(shape: tuple[int, ...] = (2, 1, 1, 32, 32)) -> tuple[torch.Tensor, torch.Tensor]:
    """Box target plus a bright blob outside the mask: the mask is not a threshold of the target."""
    target, mask = _blob_batch(shape)
    target[..., -6:, -6:] += 2.0
    return target, mask


def test_target_label_is_minimised_at_pred_equals_target():
    """label='target' is ~0 with zero gradient at pred == target, where label='mask' has a floor."""
    target, mask = _mask_disagrees_with_target()
    floor = SegAuxDice(label="mask")(target.clone(), target, mask)
    pred = target.clone().requires_grad_(True)
    loss = SegAuxDice(label="target")(pred, target, mask)
    loss.backward()
    assert floor > 0.05
    assert loss < 1e-4
    assert pred.grad.abs().max() < 1e-4
    g = torch.Generator().manual_seed(1)
    perturbed = target + 0.5 * torch.randn(target.shape, generator=g)
    assert SegAuxDice(label="target")(perturbed, target, mask) > 10 * loss


def test_target_label_keeps_the_mask_validity_rule():
    """The mask still decides which patches count: an empty mask is excluded under either label."""
    target, mask = _blob_batch((2, 1, 1, 16, 16))
    mask[0] = 0.0
    for label in ("mask", "target"):
        _, valid = SegAuxDice(label=label).per_channel(target, target, mask)
        assert valid.tolist() == [[False], [True]]


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="label"):
        SegAuxDice(label="soft")


# ---- Defaults are bit-identical to the pre-extension implementation ----

# per_channel(...).dice on _reference_inputs(), computed before weighting/topology existed.
_PRE_EXTENSION_DICE = {
    "mask": ["0x1.71c02p-3", "0x1.73a7cp-3", "0x1.37d6bp-3", "0x1.66fff8p-3", "0x1.530968p-3", "0x1.0p+0"],
    "target": ["0x1.9c865p-4", "0x1.b3d3ap-4", "0x1.57231p-4", "0x1.a811fp-4", "0x1.80088p-4", "0x1.740c6p-5"],
}


def _reference_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(123)
    shape = (3, 2, 3, 12, 12)
    target = torch.randn(shape, generator=g)
    mask = (target + 0.3 * torch.randn(shape, generator=g) > 0.5).float()
    mask[2, 1] = 0.0
    pred = target + 0.4 * torch.randn(shape, generator=g)
    return pred, target, mask


@pytest.mark.parametrize("label", ["mask", "target"])
def test_defaults_are_bit_identical_to_pre_extension(label):
    pred, target, mask = _reference_inputs()
    loss_fn = SegAuxDice(label=label, weighting="none", topology="none")
    dice, valid = loss_fn.per_channel(pred, target, mask)
    assert dice.dtype == torch.float32
    assert dice.flatten().tolist() == [float.fromhex(h) for h in _PRE_EXTENSION_DICE[label]]
    assert valid.flatten().tolist() == [True] * 5 + [False]


# ---- Exact torch EDT ----


def _random_mask(shape: tuple[int, ...], fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape) < fraction


@pytest.mark.parametrize(
    "shape,spacing",
    [
        ((40, 37), (1.0, 1.0)),
        ((40, 37), (0.3, 1.7)),
        ((9, 21, 18), (1.0, 1.0, 1.0)),
        ((9, 21, 18), (0.29, 0.108, 0.108)),
        ((1, 30, 25), (0.29, 0.108, 0.108)),
    ],
    ids=["2d-iso", "2d-aniso", "3d-iso", "3d-aniso", "3d-z1"],
)
@pytest.mark.parametrize("fraction", [0.02, 0.5])
def test_squared_edt_matches_scipy(shape, spacing, fraction):
    seeds = np.stack([_random_mask(shape, fraction, seed) for seed in range(3)])
    seeds[:, (0,) * len(shape)] = True  # at least one seed per row
    got = squared_edt(torch.from_numpy(seeds), spacing).sqrt().numpy()
    want = np.stack([ndimage.distance_transform_edt(~s, sampling=spacing) for s in seeds])
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("max_distance", [0.5, 2.0, 7.5])
def test_squared_edt_capped_is_exact_below_the_cap(max_distance):
    shape, spacing = (9, 41, 38), (0.29, 0.108, 0.108)
    seeds = np.stack([_random_mask(shape, 0.01, seed) for seed in range(2)])
    seeds[:, 0, 0, 0] = True
    got = squared_edt(torch.from_numpy(seeds), spacing, max_distance=max_distance).sqrt().numpy()
    want = np.stack([ndimage.distance_transform_edt(~s, sampling=spacing) for s in seeds])
    np.testing.assert_allclose(got, np.minimum(want, max_distance), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("shape", [(2, 1, 5, 20, 20), (2, 1, 1, 20, 20), (2, 1, 24, 20)], ids=["3d", "3d-z1", "2d"])
def test_soft_skeleton_matches_official_values_and_gradients(shape):
    """Shifted-slice morphology + checkpointing reproduce jocpae/clDice soft_skel."""
    img = torch.rand(shape, generator=torch.Generator().manual_seed(8))
    weights = torch.rand(img.shape, generator=torch.Generator().manual_seed(9))
    a = img.clone().requires_grad_(True)
    (soft_skeleton(a, 4) * weights).sum().backward()
    b = img.clone().requires_grad_(True)
    official = _official_soft_skel(b, 4)
    (official * weights).sum().backward()
    with torch.no_grad():
        torch.testing.assert_close(soft_skeleton(img, 4), official)
    torch.testing.assert_close(a.grad, b.grad)


def _official_soft_skel(img: torch.Tensor, iters: int) -> torch.Tensor:
    """jocpae/clDice soft_skeleton.py, verbatim for 4D and 5D input."""
    F = torch.nn.functional

    def soft_erode(img):
        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(img):
        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(img):
        return soft_dilate(soft_erode(img))

    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


@pytest.mark.parametrize("windows", [(1, 3, 5), (3, 1, 7), (5, 9, 31), (9, 3, 1)])
def test_box_max_matches_scipy_maximum_filter(windows):
    x = torch.rand(2, 4, 13, 11, generator=torch.Generator().manual_seed(3))
    want = np.stack([ndimage.maximum_filter(r, size=windows, mode="constant", cval=-np.inf) for r in x.numpy()])
    np.testing.assert_array_equal(box_max(x, windows).numpy(), want)


def test_squared_edt_without_seeds_is_inf():
    d2 = squared_edt(torch.zeros(2, 4, 5, dtype=torch.bool), (1.0, 1.0))
    assert torch.isinf(d2).all()


# ---- SAUNA map ----


def _sauna_reference(gt: np.ndarray) -> np.ndarray:
    """SAUNA's combined "h" map, line by line from Oulu-IMEDS/SAUNA
    ``generate_uncertainty_masks.py``, with scipy's exact EDT in place of
    ``cv2.distanceTransform(..., DIST_L2, DIST_MASK_5)``. ``gt`` is uint8 0/255."""

    def compute_distance_transform(image):
        return ndimage.distance_transform_edt(image).astype(np.float32)

    def do_max_pooling(img_dist):
        kernel_size = int(np.ceil(img_dist.max()))
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return maxpool(torch.tensor(img_dist).unsqueeze(0).unsqueeze(0))[0, 0].numpy()

    # extract_boundary_uncertainty_map
    img_fg_dist = compute_distance_transform(gt)
    fg_max = img_fg_dist.max()
    img_bg_dist = -compute_distance_transform(255 - gt)
    img_bg_dist[img_bg_dist <= -fg_max] = -fg_max
    gt_b = (img_fg_dist + img_bg_dist) / (fg_max + 1e-6)
    # extract_thickness_uncertainty_map, target_c_label="h"
    img_dist = compute_distance_transform(gt)
    thickness_max = img_dist.max()
    img_thick_pos = (gt > 0) * do_max_pooling(img_dist) / (thickness_max + 1e-6)
    img_bg_dist = compute_distance_transform(255 - gt)
    img_bg_dist[img_bg_dist >= fg_max] = fg_max
    img_thick_neg = (gt <= 0) * do_max_pooling(img_bg_dist) / (thickness_max + 1e-6)
    img_thick_neg = np.clip(img_thick_neg, a_min=0.0, a_max=1.0)
    gt_t = np.where(gt > 0, img_thick_pos, img_thick_neg)
    # extract_combined_uncertainty_map, target_c_label="h"
    gt_c = gt_b.copy()
    gt_t_abs = np.abs(gt_t)
    fg = (gt_t_abs > 0) & (gt_b > 0)
    gt_c[fg] = gt_b[fg] + (1.0 - gt_t[fg])
    bg = (gt_t_abs > 0) & (gt_b < 0)
    gt_c[bg] = gt_b[bg] - (1.0 - gt_t[bg])
    return np.clip(gt_c, a_min=-1.0, a_max=1.0)


def _disk(canvas: np.ndarray, cy: int, cx: int, r: float) -> None:
    yy, xx = np.ogrid[: canvas.shape[0], : canvas.shape[1]]
    canvas[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = True


def _synthetic_shapes() -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Fat disk, 2-px line, two disks across a 2-px gap and a 2-px slit; probe voxels."""
    m = np.zeros((320, 320), dtype=bool)
    _disk(m, 50, 50, 20)  # rows/cols 30..70
    m[120:122, 20:230] = True  # 2-px-wide line
    _disk(m, 200, 100, 20)  # cols 80..120
    _disk(m, 200, 143, 20)  # cols 123..163: gap = cols 121, 122
    m[250:291, 20:100] = True  # slit = cols 100, 101 between two 41-px-tall slabs
    m[250:291, 102:180] = True
    probes = {
        "fat_disk_boundary_fg": (50, 70),
        "fat_disk_boundary_bg": (50, 71),
        "disk_center": (50, 50),
        "thin_line": (120, 120),
        "gap_center": (200, 121),
        "slit_center": (270, 100),
        "deep_background": (310, 310),
    }
    return m, probes


def _blob_masks(n: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, size, size), dtype=bool)
    for m in masks:
        for _ in range(4):
            _disk(m, *rng.integers(0, size, 2), rng.uniform(2, 9))
        row = rng.integers(0, size - 1)
        m[row : row + 1, :] = True  # 1-px line
    return masks


def test_sauna_map_matches_sauna_reference():
    synthetic, _ = _synthetic_shapes()
    masks = [synthetic, *_blob_masks(4, 64, seed=0)]
    for m in masks:
        want = _sauna_reference(m.astype(np.uint8) * 255)
        got = sauna_signed_map(torch.from_numpy(m)[None], (1.0, 1.0))[0].numpy()
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_sauna_map_on_synthetic_shapes(capsys):
    m, probes = _synthetic_shapes()
    y = sauna_signed_map(torch.from_numpy(m)[None], (1.0, 1.0))[0]
    values = {name: y[idx].item() for name, idx in probes.items()}
    with capsys.disabled():
        print("\nSAUNA y~ on synthetic shapes:", {k: round(v, 4) for k, v in values.items()})
    assert values["disk_center"] == pytest.approx(1.0, abs=1e-4)
    assert values["deep_background"] == -1.0
    assert values["thin_line"] > 0.9
    assert values["fat_disk_boundary_fg"] > 0 > values["fat_disk_boundary_bg"]
    assert values["gap_center"] < 0 and values["slit_center"] < 0
    # Thick-structure boundaries are the uncertain, down-weighted region: the
    # ceil(fg_max)-voxel window reaches ~fg_max / 2 inward, so |y~| ~ 0.5 there.
    assert 0.35 < abs(values["fat_disk_boundary_fg"]) < 0.65
    assert 0.35 < abs(values["fat_disk_boundary_bg"]) < 0.65
    assert values["gap_center"] < -0.75 and values["slit_center"] < -0.75


def test_sauna_map_z1_equals_2d_and_degenerate_rows_weigh_one():
    masks = torch.from_numpy(_blob_masks(2, 48, seed=1))
    flat = sauna_signed_map(masks, (1.0, 1.0))
    z1 = sauna_signed_map(masks[:, None], (0.29, 1.0, 1.0))[:, 0]
    torch.testing.assert_close(z1, flat)
    degenerate = torch.stack([torch.zeros(8, 8, dtype=torch.bool), torch.ones(8, 8, dtype=torch.bool)])
    torch.testing.assert_close(sauna_weight_map(degenerate, (1.0, 1.0)), torch.ones(2, 8, 8))


def test_sauna_weighting_changes_dice_and_stays_finite():
    target, mask = _blob_batch((3, 1, 4, 32, 32))
    mask[1] = 0.0  # invalid patch: weights 1, entry finite and excluded
    pred = (target + 0.5 * torch.randn(target.shape, generator=torch.Generator().manual_seed(4))).requires_grad_(True)
    plain = SegAuxDice()
    sauna = SegAuxDice(weighting="sauna", spacing=(0.29, 0.108, 0.108))
    d_plain, v_plain = plain.per_channel(pred, target, mask)
    d_sauna, v_sauna = sauna.per_channel(pred, target, mask)
    assert torch.equal(v_plain, v_sauna)
    assert torch.isfinite(d_sauna).all()
    assert not torch.allclose(d_plain[v_plain], d_sauna[v_sauna])
    loss, comps = sauna(pred, target, mask, return_components=True)
    assert comps["n_valid"] == 2
    loss.backward()
    assert torch.isfinite(pred.grad).all()


# ---- clDice ----


def _line_mask(shape: tuple[int, ...] = (1, 1, 1, 48, 48)) -> torch.Tensor:
    mask = torch.zeros(shape)
    mask[..., 23:25, 4:44] = 1.0  # 2-px-wide line
    return mask


def _crisp(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Target = mask + noise; a prediction that is a high-contrast copy of a mask."""
    noise = 0.05 * torch.randn(mask.shape, generator=torch.Generator().manual_seed(5))
    return mask + noise, 3.0 * mask - 1.0


def _skeleton_precision_sensitivity(loss_fn: SegAuxDice, pred, target, mask) -> tuple[float, float]:
    """Tprec/Tsens on the loss's own soft prediction, via the module's soft_skeleton."""
    flat_t = target.flatten()
    fg = mask.flatten() > 0.5
    tau = torch.quantile(flat_t, 1 - mask.mean())
    s = loss_fn.c * (flat_t[fg].median() - flat_t[~fg].median())
    p = torch.sigmoid((pred - tau) / s)[:, 0]
    r = mask[:, 0]
    sp = soft_skeleton(p, loss_fn.cldice_iters)
    sr = soft_skeleton(r, loss_fn.cldice_iters)
    return ((sp * r).sum() / sp.sum()).item(), ((sr * p).sum() / sr.sum()).item()


def _line_and_disk_mask() -> torch.Tensor:
    mask = _line_mask()
    _disk(mask[0, 0, 0].numpy(), 10, 10, 6)  # writes through the numpy view
    return mask


def test_cldice_near_one_for_crisp_copy_of_mask():
    mask = _line_and_disk_mask()
    target, pred = _crisp(mask)
    pure = SegAuxDice(label="mask", topology="cldice", cldice_alpha=1.0)
    one_minus_cl, valid = pure.per_channel(pred, target, mask)
    assert valid.all()
    assert one_minus_cl.item() < 0.02


def test_cldice_target_label_prefers_pred_equal_target():
    """With label='target' the reference is soft, so clDice < 1 even at pred == target,
    but a broken prediction still scores worse."""
    mask = _line_and_disk_mask()
    target, _ = _crisp(mask)
    pure = SegAuxDice(label="target", topology="cldice", cldice_alpha=1.0)
    broken = target.clone()
    broken[..., 18:30] = target.min()
    assert pure(broken, target, mask) > pure(target.clone(), target, mask)


def test_cldice_bridge_lowers_precision_and_break_lowers_sensitivity():
    loss_fn = SegAuxDice(topology="cldice", cldice_alpha=1.0)
    two = torch.zeros(1, 1, 1, 48, 64)
    _disk(two[0, 0, 0].numpy(), 24, 18, 8)
    _disk(two[0, 0, 0].numpy(), 24, 45, 8)  # disjoint, gap at cols 27..36
    target, perfect = _crisp(two)
    bridged = perfect.clone()
    bridged[..., 22:27, 18:46] = 2.0
    prec_ok, sens_ok = _skeleton_precision_sensitivity(loss_fn, perfect, target, two)
    prec_br, sens_br = _skeleton_precision_sensitivity(loss_fn, bridged, target, two)
    assert prec_br < prec_ok - 0.1
    assert loss_fn(bridged, target, two) > loss_fn(perfect, target, two)

    line = _line_mask()
    target, perfect = _crisp(line)
    broken = perfect.clone()
    broken[..., 18:30] = -1.0  # 12-px break
    prec_ok, sens_ok = _skeleton_precision_sensitivity(loss_fn, perfect, target, line)
    prec_bk, sens_bk = _skeleton_precision_sensitivity(loss_fn, broken, target, line)
    assert sens_bk < sens_ok - 0.1
    assert prec_bk == pytest.approx(prec_ok, abs=0.05)
    assert loss_fn(broken, target, line) > loss_fn(perfect, target, line)


@pytest.mark.parametrize("weighting", ["none", "sauna"])
def test_cldice_gradients_finite_and_invalid_excluded(weighting):
    target, mask = _blob_batch((3, 1, 4, 32, 32))
    mask[1] = 0.0
    mask[2] = 1.0
    pred = (target + 0.3 * torch.randn(target.shape, generator=torch.Generator().manual_seed(6))).requires_grad_(True)
    spacing = (0.29, 0.108, 0.108) if weighting == "sauna" else None
    loss_fn = SegAuxDice(topology="cldice", cldice_alpha=0.5, weighting=weighting, spacing=spacing)
    loss, comps = loss_fn(pred, target, mask, return_components=True)
    assert comps["n_valid"] == 1
    torch.testing.assert_close(loss, loss_fn(pred[:1], target[:1], mask[:1]))
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_cldice_alpha_zero_equals_dice():
    pred, target, mask = _reference_inputs()
    plain, _ = SegAuxDice().per_channel(pred, target, mask)
    mixed, _ = SegAuxDice(topology="cldice", cldice_alpha=0.0).per_channel(pred, target, mask)
    torch.testing.assert_close(mixed, plain)


@pytest.mark.parametrize("label", ["mask", "target"])
def test_extensions_run_under_bf16_autocast(label):
    target, mask = _blob_batch((2, 1, 4, 16, 16))
    pred = (target + 0.3).requires_grad_(True)
    loss_fn = SegAuxDice(label=label, weighting="sauna", spacing=(0.29, 0.108, 0.108), topology="cldice")
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = loss_fn(pred.to(torch.bfloat16), target.to(torch.bfloat16), mask.to(torch.bfloat16))
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"weighting": "boundary"}, "weighting"),
        ({"weighting": "sauna"}, "needs spacing"),
        ({"weighting": "sauna", "spacing": (1.0, -1.0)}, "positive"),
        ({"spacing": (1.0, 1.0, 1.0)}, "no effect"),
        ({"topology": "skeleton"}, "topology"),
        ({"topology": "cldice", "cldice_alpha": 1.5}, "cldice_alpha"),
        ({"topology": "cldice", "cldice_iters": 0}, "cldice_iters"),
    ],
)
def test_invalid_extension_args_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SegAuxDice(**kwargs)


def test_spacing_length_must_match_spatial_dims():
    target, mask = _blob_batch((1, 1, 4, 16, 16))
    with pytest.raises(ValueError, match="spatial dims"):
        SegAuxDice(weighting="sauna", spacing=(1.0, 1.0)).per_channel(target, target, mask)
