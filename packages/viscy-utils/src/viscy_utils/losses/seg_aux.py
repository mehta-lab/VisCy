"""Self-calibrating soft-Dice auxiliary loss for virtual staining.

The prediction is soft-thresholded at a level derived per patch from the target
and the foreground mask it is scored against, so the loss needs no knowledge of
the normalization applied upstream (z-score, median/IQR, MinMax to [-1, 1]).

For every (sample, channel) patch:

- ``tau = quantile(target, 1 - mean(fg_mask))``: the level at which the target's
  own foreground fraction equals the mask's.
- ``s = c * IQR(target)``: the knee width relative to the patch's dynamic range.
- ``p = sigmoid((pred - tau) / s)``: bounded, with non-zero gradient everywhere.
- ``dice = 1 - 2 * sum(p * m) / (sum(p**2) + sum(m**2) + eps)``.

``tau`` and ``s`` are computed from the target without gradient.
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


class SegAuxDice(nn.Module):
    """Squared-denominator soft Dice on a self-calibrated sigmoid of the prediction.

    Patches whose mask is empty or full, or whose target has zero IQR (no
    dynamic range to set a knee width from), carry no segmentation signal and
    are excluded from the mean. If every patch is excluded the loss is a zero
    that stays connected to ``pred``'s graph.

    Computed in float32 regardless of autocast.

    Parameters
    ----------
    c : float
        Knee width as a fraction of the target patch IQR.
    eps : float
        Dice denominator stabilizer, and the floor on the knee width.
    """

    def __init__(self, c: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        if c <= 0:
            raise ValueError(f"c must be > 0, got {c}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.c = c
        self.eps = eps

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
                iqr = _quantile_sorted(sorted_t, torch.full_like(q, 0.75)) - _quantile_sorted(
                    sorted_t, torch.full_like(q, 0.25)
                )
                valid = (fg_count > 0) & (fg_count < n) & (iqr > 0)
                s = (self.c * iqr).clamp(min=self.eps)
            p = torch.sigmoid((pred_f - tau.unsqueeze(-1)) / s.unsqueeze(-1))
            inter = (p * mask).sum(-1)
            denom = (p * p).sum(-1) + fg_count + self.eps
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
