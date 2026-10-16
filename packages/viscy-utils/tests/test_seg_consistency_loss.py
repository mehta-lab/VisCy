"""Tests for SegConsistencyLoss on a tiny stand-in segmenter (the Cellpose-SAM one is in dynacell)."""

import pytest
import torch
from torch import Tensor, nn

from viscy_utils.losses import SegConsistencyLoss
from viscy_utils.losses.seg_consistency import _OTSU_BINS, _binary_kl, _otsu_thresholds, nucleus_area_planes


class TinySegmenter(nn.Module):
    """Two conv layers to flow (2 ch) + cell-probability logit (1 ch), per-image min-max normalized."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.Tanh(), nn.Conv2d(8, 3, 3, padding=1))

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        with torch.no_grad():
            lo = x.amin((1, 2, 3), keepdim=True)
            hi = x.amax((1, 2, 3), keepdim=True)
        y = self.net((x - lo) / (hi - lo + 1e-6))
        return {"flow": y[:, :2], "cellprob_logit": y[:, 2:3]}


def _nuclei(shape: tuple[int, ...], seed: int = 0) -> Tensor:
    """Bright blobs on noise, peaked in z around a random plane per sample."""
    g = torch.Generator().manual_seed(seed)
    b, z, y, x = shape[0], shape[2], shape[3], shape[4]
    yy, xx = torch.meshgrid(torch.arange(y), torch.arange(x), indexing="ij")
    out = 0.05 * torch.randn(shape, generator=g)
    for i in range(b):
        zc = float(torch.randint(z // 4, max(z // 4 + 1, 3 * z // 4), (1,), generator=g))
        for _ in range(4):
            cy, cx = (torch.rand(2, generator=g) * torch.tensor([y, x])).tolist()
            disk = ((yy - cy) ** 2 + (xx - cx) ** 2 < 36).float()
            prof = torch.exp(-(((torch.arange(z) - zc) / 2.0) ** 2))
            out[i, 0] += prof[:, None, None] * disk
    return out


@pytest.mark.parametrize("shape", [(2, 1, 8, 32, 32), (2, 1, 1, 32, 32)], ids=["3d", "z1"])
def test_zero_at_pred_equals_target_and_positive_otherwise(shape):
    target = _nuclei(shape)
    loss_fn = SegConsistencyLoss(TinySegmenter())
    assert float(loss_fn(target.clone(), target)) == pytest.approx(0.0, abs=1e-6)
    g = torch.Generator().manual_seed(1)
    assert float(loss_fn(target + 0.3 * torch.randn(shape, generator=g), target)) > 1e-4


def test_invariant_to_positive_affine_units_of_both_inputs():
    """The segmenter normalizes per image, so a z-scored pred vs a raw target loses nothing."""
    target = _nuclei((2, 1, 6, 32, 32))
    loss_fn = SegConsistencyLoss(TinySegmenter())
    assert float(loss_fn(3.0 * target + 7.0, target)) == pytest.approx(0.0, abs=1e-5)


def test_components_match_their_definition():
    target = _nuclei((2, 1, 6, 32, 32))
    pred = target + 0.2 * torch.randn_like(target)
    seg = TinySegmenter()
    loss_fn = SegConsistencyLoss(seg, crop_size=None)
    loss, comps = loss_fn(pred, target, return_components=True)
    pred2d, target2d = loss_fn.project(pred, target)
    out, ref = seg(pred2d), seg(target2d)
    d_flow = ((out["flow"] - ref["flow"]) ** 2).mean()
    d_prob = _binary_kl(out["cellprob_logit"], ref["cellprob_logit"]).mean()
    torch.testing.assert_close(comps["segcons_flow"], d_flow)
    torch.testing.assert_close(comps["segcons_prob"], d_prob)
    torch.testing.assert_close(loss, d_flow + d_prob)
    assert comps["segcons"] is loss


def test_gradient_reaches_pred_only_and_matches_finite_difference():
    target = _nuclei((1, 1, 6, 16, 16)).double()
    seg = TinySegmenter().double()
    loss_fn = SegConsistencyLoss(seg)
    pred = (target + 0.2 * torch.randn_like(target)).requires_grad_(True)
    loss_fn(pred, target).backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    assert all(p.grad is None for p in seg.parameters())
    # Only the 3 focus-slab planes feed the segmenter.
    plane = int(nucleus_area_planes(target[:, 0])[0])
    touched = (pred.grad[0, 0].abs().sum((1, 2)) > 0).nonzero().flatten().tolist()
    assert set(touched) <= {plane - 1, plane, plane + 1}
    # Directional central difference. The segmenter's min/max statistics are stop-gradient,
    # so perturb away from the projected image's extreme pixels, which leaves them fixed.
    v = torch.zeros_like(pred)
    v[0, 0, plane] = torch.randn(16, 16, dtype=torch.float64)
    pred2d, _ = loss_fn.project(pred.detach(), target)
    for flat_idx in (pred2d.argmax(), pred2d.argmin()):
        y, x = divmod(int(flat_idx), 16)
        v[0, 0, :, y, x] = 0.0
    eps = 1e-6
    with torch.no_grad():
        num = (loss_fn(pred + eps * v, target) - loss_fn(pred - eps * v, target)) / (2 * eps)
    ana = (pred.grad * v).sum()
    assert float(num) == pytest.approx(float(ana), rel=1e-3)


def test_segmenter_is_frozen_unregistered_and_stays_in_eval():
    seg = TinySegmenter()
    loss_fn = SegConsistencyLoss(seg)
    parent = nn.Sequential(nn.Linear(2, 2))
    parent.loss = loss_fn
    assert all(not k.startswith("loss") for k in parent.state_dict())
    assert list(loss_fn.parameters()) == []
    parent.train()
    assert not seg.training
    assert not any(p.requires_grad for p in seg.parameters())


def test_crop_size_shapes_what_the_segmenter_sees():
    target = _nuclei((4, 1, 6, 40, 40))
    loss_fn = SegConsistencyLoss(TinySegmenter(), crop_size=24)
    pred2d, target2d = loss_fn.project(target, target)
    assert pred2d.shape == target2d.shape == (4, 1, 24, 24)


def test_rejects_bad_arguments_and_shapes():
    with pytest.raises(ValueError, match="slab_halfwidth"):
        SegConsistencyLoss(TinySegmenter(), slab_halfwidth=-1)
    with pytest.raises(ValueError, match="crop_size"):
        SegConsistencyLoss(TinySegmenter(), crop_size=0)
    loss_fn = SegConsistencyLoss(TinySegmenter())
    with pytest.raises(ValueError, match="share a shape"):
        loss_fn(torch.zeros(1, 1, 4, 8, 8), torch.zeros(1, 1, 4, 8, 9))
    with pytest.raises(ValueError, match="expected"):
        loss_fn(torch.zeros(1, 2, 4, 8, 8), torch.zeros(1, 2, 4, 8, 8))
    with pytest.raises(ValueError, match="expected"):
        loss_fn(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))


def test_nucleus_area_planes_fallbacks():
    assert nucleus_area_planes(torch.zeros(2, 7, 8, 8)).tolist() == [3, 3]
    assert nucleus_area_planes(torch.zeros(1, 1, 8, 8)).tolist() == [0]


def _otsu_thresholds_bincount(flat: Tensor) -> Tensor:
    """The first (bincount-histogram) implementation of ``_otsu_thresholds``, kept as the reference."""
    r = flat.shape[0]
    lo = flat.amin(1, keepdim=True)
    width = ((flat.amax(1, keepdim=True) - lo) / _OTSU_BINS).clamp_min(torch.finfo(flat.dtype).tiny)
    idx = ((flat - lo) / width).floor().long().clamp(0, _OTSU_BINS - 1)
    idx += torch.arange(r, device=flat.device).unsqueeze(1) * _OTSU_BINS
    hist = torch.bincount(idx.reshape(-1), minlength=r * _OTSU_BINS).reshape(r, _OTSU_BINS).double()
    centers = lo.double() + width.double() * (torch.arange(_OTSU_BINS, device=flat.device) + 0.5)
    w1 = hist.cumsum(1)
    w2 = hist.flip(1).cumsum(1).flip(1)
    m1 = (hist * centers).cumsum(1) / w1
    m2 = (hist * centers).flip(1).cumsum(1).flip(1) / w2
    var = w1[:, :-1] * w2[:, 1:] * (m1[:, :-1] - m2[:, 1:]) ** 2
    return centers.gather(1, var.argmax(1, keepdim=True)).squeeze(1).to(flat.dtype)


def test_otsu_thresholds_equal_the_bincount_implementation():
    vols = [_nuclei((3, 1, z, 24, 24), seed=z).reshape(3, -1) for z in (1, 5, 16)]
    vols.append(torch.rand(2, 1000) ** 3 * 50 - 7)
    for flat in vols:
        assert torch.equal(_otsu_thresholds(flat), _otsu_thresholds_bincount(flat))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host-sync detection needs CUDA")
def test_focus_planes_make_no_host_sync():
    vol = _nuclei((2, 1, 8, 32, 32))[:, 0].cuda()
    torch.cuda.set_sync_debug_mode("error")
    try:
        nucleus_area_planes(vol)
    finally:
        torch.cuda.set_sync_debug_mode("default")
