"""Self-calibrating soft-Dice auxiliary loss for virtual staining.

The prediction is soft-thresholded at a level derived per patch from the target
and the foreground mask it is scored against, so the loss needs no knowledge of
the normalization applied upstream (z-score, median/IQR, MinMax to [-1, 1]).

For every (sample, channel) patch:

- ``tau = quantile(target, 1 - mean(fg_mask))``: the level at which the target's
  own foreground fraction equals the mask's.
- ``s = c * (median(target | fg) - median(target | bg))``: the knee width
  relative to the patch's own foreground/background contrast.
- ``p = sigmoid((pred - tau) / s)``: bounded, with non-zero gradient everywhere.
- ``dice = 1 - 2 * sum(p * r) / (sum(p**2) + sum(r**2) + eps)``, where the
  reference ``r`` is the binary mask (``label="mask"``) or the target through
  the same sigmoid, ``sigmoid((target - tau) / s)`` (``label="target"``).

``tau``, ``s`` and ``r`` are computed from the target without gradient.

With ``label="mask"`` the loss is not zero at ``pred == target`` wherever the
mask is not a threshold of the target (0.07-0.25 on the iPSC baselines'
validation patches, 2026-09-24), so it also pulls the prediction toward the
mask. ``label="target"`` keeps the mask's role in setting ``tau``, ``s`` and
patch validity but is minimised exactly at ``pred == target``: it isolates a
segmentation-shaped penalty from that pull.
"""

import torch
from torch import Tensor, nn

__all__ = ["SegAuxDice"]


def _quantile_sorted(sorted_rows: Tensor, q: Tensor) -> Tensor:
    """Linearly interpolated per-row quantile of pre-sorted rows.

    Matches ``torch.quantile(..., interpolation="linear")`` but takes one
    quantile level per row and has no input-size limit, which
    ``torch.quantile`` imposes (2**24 elements) and large 3D patches exceed.

    Parameters
    ----------
    sorted_rows : Tensor
        Rows sorted ascending along the last dim, shape ``(R, N)``.
    q : Tensor
        Quantile level per row in ``[0, 1]``, shape ``(R,)``.

    Returns
    -------
    Tensor
        Quantile per row, shape ``(R,)``.
    """
    n = sorted_rows.shape[-1]
    pos = q.clamp(0.0, 1.0) * (n - 1)
    lo = pos.floor().long()
    hi = (lo + 1).clamp(max=n - 1)
    frac = pos - lo.to(pos.dtype)
    v_lo = sorted_rows.gather(-1, lo.unsqueeze(-1)).squeeze(-1)
    v_hi = sorted_rows.gather(-1, hi.unsqueeze(-1)).squeeze(-1)
    return v_lo + frac * (v_hi - v_lo)


def _masked_median(values: Tensor, keep: Tensor) -> Tensor:
    """Linearly interpolated per-row median over the entries where ``keep`` is set.

    Parameters
    ----------
    values : Tensor
        Values, shape ``(R, N)``.
    keep : Tensor
        Boolean selection, shape ``(R, N)``.

    Returns
    -------
    Tensor
        Median per row, shape ``(R,)``; meaningless for rows with no kept entry.
    """
    k = keep.sum(-1)
    # Dropped entries sort to the end, so the first k sorted values are the kept ones.
    sorted_rows = torch.where(keep, values, torch.full_like(values, float("inf"))).sort(dim=-1).values
    pos = 0.5 * (k - 1).clamp(min=0).to(values.dtype)
    lo = pos.floor().long()
    hi = torch.minimum(lo + 1, (k - 1).clamp(min=0))
    frac = pos - lo.to(pos.dtype)
    v_lo = sorted_rows.gather(-1, lo.unsqueeze(-1)).squeeze(-1)
    v_hi = sorted_rows.gather(-1, hi.unsqueeze(-1)).squeeze(-1)
    return v_lo + frac * (v_hi - v_lo)


class SegAuxDice(nn.Module):
    """Squared-denominator soft Dice on a self-calibrated sigmoid of the prediction.

    Patches whose mask is empty or full, or whose target is not brighter inside
    the mask than outside it (no contrast to set a knee width from), carry no
    segmentation signal and are excluded from the mean.

    The knee width is set from the foreground/background contrast rather than
    the patch IQR: on thin-structure patches (membrane) the IQR is ~0.65x the
    contrast, which left 11-25% of foreground voxels with a vanishing sigmoid
    gradient, against 0.5-7.5% with the contrast knee (measured on the iPSC
    baselines' validation patches, 2026-09-24). If every patch is excluded the loss is a zero
    that stays connected to ``pred``'s graph.

    Computed in float32 regardless of autocast.

    Parameters
    ----------
    c : float
        Knee width as a fraction of the patch's foreground/background contrast.
    eps : float
        Dice denominator stabilizer, and the floor on the knee width.
    label : {"mask", "target"}
        Dice reference: the binary foreground mask, or the target soft-thresholded
        like the prediction (self-consistent; zero loss at ``pred == target``).
    """

    def __init__(self, c: float = 0.1, eps: float = 1e-6, label: str = "mask") -> None:
        super().__init__()
        if c <= 0:
            raise ValueError(f"c must be > 0, got {c}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if label not in ("mask", "target"):
            raise ValueError(f"label must be 'mask' or 'target', got {label!r}")
        self.c = c
        self.eps = eps
        self.label = label

    def per_channel(self, pred: Tensor, target: Tensor, fg_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the Dice loss per (sample, channel) and which entries are valid.

        Parameters
        ----------
        pred : Tensor
            Prediction, shape ``(B, C, *spatial)``.
        target : Tensor
            Ground truth, same shape.
        fg_mask : Tensor
            Foreground mask, same shape; binarized at 0.5.

        Returns
        -------
        tuple of (Tensor, Tensor)
            Dice loss ``(B, C)`` in float32 (entries for invalid patches are
            finite but meaningless) and the boolean validity mask ``(B, C)``.
        """
        if pred.shape != target.shape or fg_mask.shape != target.shape:
            raise ValueError(
                f"pred, target and fg_mask must share a shape; got "
                f"{tuple(pred.shape)}, {tuple(target.shape)}, {tuple(fg_mask.shape)}"
            )
        b, c = target.shape[:2]
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred_f = pred.float().reshape(b * c, -1)
            mask = (fg_mask.reshape(b * c, -1) > 0.5).float()
            n = mask.shape[-1]
            fg_count = mask.sum(-1)
            with torch.no_grad():
                sorted_t = target.detach().float().reshape(b * c, -1).sort(dim=-1).values
                q = 1.0 - fg_count / n
                tau = _quantile_sorted(sorted_t, q)
                flat_t = target.detach().float().reshape(b * c, -1)
                fg = mask > 0.5
                contrast = _masked_median(flat_t, fg) - _masked_median(flat_t, ~fg)
                valid = (fg_count > 0) & (fg_count < n) & (contrast > 0)
                # An empty or full mask makes one median inf; substitute a finite
                # width so invalid entries stay finite and the mask-multiply in
                # forward() can zero them (inf * 0 would be NaN).
                s = torch.where(valid, self.c * contrast, torch.ones_like(contrast)).clamp(min=self.eps)
                if self.label == "target":
                    ref = torch.sigmoid((flat_t - tau.unsqueeze(-1)) / s.unsqueeze(-1))
                else:
                    ref = mask
            p = torch.sigmoid((pred_f - tau.unsqueeze(-1)) / s.unsqueeze(-1))
            inter = (p * ref).sum(-1)
            denom = (p * p).sum(-1) + (ref * ref).sum(-1) + self.eps
            dice = 1.0 - 2.0 * inter / denom
        return dice.reshape(b, c), valid.reshape(b, c)

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        fg_mask: Tensor,
        return_components: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Compute the mean Dice loss over valid (sample, channel) patches.

        Parameters
        ----------
        pred : Tensor
            Prediction, shape ``(B, C, *spatial)``.
        target : Tensor
            Ground truth, same shape.
        fg_mask : Tensor
            Foreground mask, same shape; binarized at 0.5.
        return_components : bool
            When ``True``, also return ``{"dice": loss, "n_valid": count}``.

        Returns
        -------
        Tensor or tuple of (Tensor, dict of str to Tensor)
            Scalar float32 loss, optionally with its components.
        """
        dice, valid = self.per_channel(pred, target, fg_mask)
        n_valid = valid.sum()
        # Invalid entries are finite, so multiplying by the mask zeroes them while
        # keeping the graph: with no valid patch this is a connected 0 / 1, and no
        # host sync is needed to branch on n_valid.
        loss = (dice * valid).sum() / n_valid.clamp(min=1)
        if return_components:
            return loss, {"dice": loss, "n_valid": n_valid.float()}
        return loss
