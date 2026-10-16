"""Segmenter-consistency loss for virtual staining (ControlNet++ style).

A frozen, differentiable segmenter ``S`` is run on the prediction and on the target,
and their dense, pre-threshold outputs are matched:

``L = mean_px ( mean_c (flow_pred - flow_target)**2 + KL(q_target || q_pred) )``

- ``flow`` are the segmenter's two flow channels and ``q = sigmoid(cellprob_logit)``.
- The probability term is the binary KL, i.e. BCE-with-logits against the target's
  soft probability minus that probability's entropy: the gradient is BCE's, and the
  loss is exactly 0 at ``pred == target``. The loss equals the G4 gate's
  ``D_total = D_flow + D_prob``.
- ``S(target)`` is computed under ``no_grad`` on the same augmented crop, every step;
  ``S(pred)`` carries the gradient into the prediction. ``S`` itself gets none.

The segmenter sees 2-D images. A crop ``(B, 1, Z, Y, X)`` is reduced to the
``2 * slab_halfwidth + 1``-plane max projection around its nuclear-area focus plane:
the plane of largest Otsu foreground fraction of the target crop, with the outer 15%
of planes excluded and the crop center as the fallback, as
``dynacell.evaluation.focus.nucleus_area_plane`` picks the plane the eval segments
(slab halfwidth 1 there). The same planes are projected from the prediction. One
segmenter pass per sample is the cheapest geometry the G4 bench measured, and it is
the geometry the eval scores. ``Z = 1`` crops (2-D models) give their single plane.
The projection is then center-cropped to ``crop_size`` (one 256-px Cellpose tile).

The segmenter is held outside the module tree, so its weights never enter a
``state_dict`` (checkpoints stay the backbone's, and ``ckpt_path`` loads of baseline
checkpoints stay strict) and the module's ``train()`` cannot switch it out of eval
mode. It follows the prediction's device on first use. It runs with autocast
disabled, in its own dtype, as stock inference does.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["SegConsistencyLoss", "nucleus_area_planes"]

_GUARD_FRAC = 0.15
_MIN_FOREGROUND = 0.01
_OTSU_BINS = 256


def _otsu_thresholds(flat: Tensor) -> Tensor:
    """Per-row Otsu threshold, as ``skimage.filters.threshold_otsu`` (256 bins over each row's range).

    Parameters
    ----------
    flat : Tensor
        Values, shape ``(R, N)``. A constant row gets a meaningless threshold.

    Returns
    -------
    Tensor
        Threshold per row, shape ``(R,)``: the center of the bin maximizing the
        between-class variance.
    """
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


@torch.no_grad()
def nucleus_area_planes(vol: Tensor) -> Tensor:
    """Return each volume's plane of largest nuclear foreground area, without a host sync.

    Mirrors ``dynacell.evaluation.focus.nucleus_area_plane``: Otsu over the whole
    volume, per-plane foreground fraction, argmax inside a band that drops the outer
    15% of planes; a constant volume, or one whose band peak is below 1% foreground,
    falls back to the center plane.

    Parameters
    ----------
    vol : Tensor
        Volumes, shape ``(B, Z, Y, X)``.

    Returns
    -------
    Tensor
        Plane index per volume, shape ``(B,)``, int64.
    """
    b, z = vol.shape[:2]
    center = torch.full((b,), z // 2, device=vol.device, dtype=torch.long)
    flat = vol.reshape(b, -1).float()
    # A constant row has no threshold; its NaN/arbitrary one is discarded below.
    constant = flat.amax(1) <= flat.amin(1)
    thr = _otsu_thresholds(flat)
    frac = (flat > thr.unsqueeze(1)).reshape(b, z, -1).float().mean(2)
    lo = int(round(_GUARD_FRAC * z))
    hi = max(lo + 1, int(round((1.0 - _GUARD_FRAC) * z)))
    band = frac[:, lo:hi]
    peak, arg = band.max(1)
    use_center = constant | (peak < _MIN_FOREGROUND)
    return torch.where(use_center, center, lo + arg)


def _slab_mip(vol: Tensor, planes: Tensor, halfwidth: int) -> Tensor:
    """Max-project ``(B, Z, Y, X)`` onto the planes ``[p - h, p + h]`` clipped to the stack.

    Clipping repeats an edge plane, which leaves the maximum unchanged, so this equals
    ``dynacell.evaluation.focus.slab_mip``. Returns ``(B, Y, X)``.
    """
    z = vol.shape[1]
    offsets = torch.arange(-halfwidth, halfwidth + 1, device=vol.device)
    idx = (planes.unsqueeze(1) + offsets).clamp(0, z - 1)
    idx = idx[:, :, None, None].expand(-1, -1, *vol.shape[2:])
    return vol.gather(1, idx).amax(1)


def _center_crop(img: Tensor, size: int | None) -> Tensor:
    """Center-crop ``(..., Y, X)`` to at most ``size`` per side (no-op when ``None``)."""
    if size is None:
        return img
    y, x = img.shape[-2:]
    oy, ox = max(0, (y - size) // 2), max(0, (x - size) // 2)
    return img[..., oy : oy + min(y, size), ox : ox + min(x, size)]


def _binary_kl(p_logit: Tensor, q_logit: Tensor) -> Tensor:
    """Per-element binary KL(q || p) for ``q = sigmoid(q_logit)``; 0 iff the logits match."""
    q = torch.sigmoid(q_logit)
    bce = F.binary_cross_entropy_with_logits(p_logit, q, reduction="none")
    ent = F.binary_cross_entropy_with_logits(q_logit, q, reduction="none")
    return bce - ent


class SegConsistencyLoss(nn.Module):
    """Match a frozen segmenter's flows and cell probability on prediction and target.

    Parameters
    ----------
    segmenter : nn.Module
        Frozen differentiable segmenter mapping ``(N, 1, Y, X)`` images in any
        positive-affine units to ``{"flow": (N, 2, Y, X), "cellprob_logit": (N, 1, Y, X)}``
        (e.g. ``dynacell.cellpose_dense.CellposeDense``). Put in eval mode with
        ``requires_grad=False`` here; never the evaluation segmenter.
    slab_halfwidth : int
        Half-width of the max projection around a 3-D crop's focus plane.
    crop_size : int or None
        Center-crop side of the projected image, ``None`` for the whole image.
    """

    def __init__(
        self,
        segmenter: nn.Module,
        slab_halfwidth: int = 1,
        crop_size: int | None = 256,
    ) -> None:
        super().__init__()
        if slab_halfwidth < 0:
            raise ValueError(f"slab_halfwidth must be >= 0, got {slab_halfwidth}")
        if crop_size is not None and crop_size < 1:
            raise ValueError(f"crop_size must be >= 1 or None, got {crop_size}")
        segmenter.eval()
        segmenter.requires_grad_(False)
        # Bypass nn.Module.__setattr__ so the segmenter is not a registered submodule.
        self.__dict__["segmenter"] = segmenter
        self.slab_halfwidth = slab_halfwidth
        self.crop_size = crop_size

    def project(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """Reduce ``pred`` and ``target`` to the 2-D images the segmenter scores.

        Parameters
        ----------
        pred, target : Tensor
            Shape ``(B, 1, Z, Y, X)``.

        Returns
        -------
        tuple of (Tensor, Tensor)
            Projected ``pred`` and ``target``, each ``(B, 1, Y', X')``.
        """
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must share a shape; got {tuple(pred.shape)}, {tuple(target.shape)}")
        if pred.ndim != 5 or pred.shape[1] != 1:
            raise ValueError(f"expected (B, 1, Z, Y, X), got {tuple(pred.shape)}")
        planes = nucleus_area_planes(target[:, 0].detach())
        pred = _slab_mip(pred[:, 0], planes, self.slab_halfwidth).unsqueeze(1)
        target = _slab_mip(target[:, 0], planes, self.slab_halfwidth).unsqueeze(1)
        return _center_crop(pred, self.crop_size), _center_crop(target, self.crop_size)

    def _segment(self, img: Tensor) -> dict[str, Tensor]:
        seg = self.segmenter
        if next(seg.parameters()).device != img.device:
            seg.to(img.device)
        with torch.autocast(device_type=img.device.type, enabled=False):
            return seg(img.to(torch.promote_types(img.dtype, torch.float32)))

    def forward(
        self, pred: Tensor, target: Tensor, return_components: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Compute the consistency loss.

        Parameters
        ----------
        pred : Tensor
            Prediction, ``(B, 1, Z, Y, X)``.
        target : Tensor
            Ground truth, same shape.
        return_components : bool
            Also return ``{"segcons", "segcons_flow", "segcons_prob"}``.

        Returns
        -------
        Tensor or tuple of (Tensor, dict of str to Tensor)
            Scalar float32 loss, optionally with its flow and probability terms.
        """
        pred2d, target2d = self.project(pred, target.detach())
        with torch.no_grad():
            ref = self._segment(target2d)
        out = self._segment(pred2d)
        flow_sq = ((out["flow"] - ref["flow"]) ** 2).mean(1)
        kl = _binary_kl(out["cellprob_logit"], ref["cellprob_logit"])[:, 0]
        flow_term, prob_term = flow_sq.mean(), kl.mean()
        loss = flow_term + prob_term
        if return_components:
            return loss, {"segcons": loss, "segcons_flow": flow_term, "segcons_prob": prob_term}
        return loss
