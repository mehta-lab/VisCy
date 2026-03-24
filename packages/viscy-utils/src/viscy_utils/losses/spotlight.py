"""Spotlight foreground-aware loss for virtual staining.

Implements the Spotlight training strategy from Kalinin et al. 2025,
which combines masked MSE (restricting supervision to foreground voxels)
with Dice loss on soft-thresholded predictions for shape preservation.

Reference
---------
Kalinin, A.A. et al. Foreground-aware Virtual Staining for Accurate
3D Cell Morphological Profiling. ICML GenBio Workshop, 2025.
arXiv:2507.05383
"""

import torch
from torch import Tensor, nn

__all__ = ["SpotlightLoss"]


def _tunable_sigmoid(x: Tensor, k: float) -> Tensor:
    """Apply normalized tunable sigmoid (Emery 2022).

    Maps real-valued input through a soft threshold centered at 0.
    Raw output is in [-1, 1]; clamped to [0, 1] for use in Dice loss.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    k : float
        Sharpness parameter in ``(-1, 0)``. More negative = sharper.

    Returns
    -------
    Tensor
        Soft-thresholded output clamped to [0, 1].
    """
    raw = (x - k * x) / (k - 2 * k * x.abs() + 1)
    # Clamp to [0, 1] for Dice stability — raw maps to [-1, 1] on [-1, 1]
    # inputs and can exceed that range outside. The paper does not specify
    # clamping; this prevents negative values from destabilizing the Dice
    # denominator (sum of soft predictions must be non-negative).
    return raw.clamp(0, 1)


def _otsu_threshold(x: Tensor, n_bins: int = 256) -> Tensor:
    """Compute Otsu threshold for a single flat tensor.

    Parameters
    ----------
    x : Tensor
        1-D tensor of values.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    Tensor
        Scalar threshold value.
    """
    lo, hi = x.min(), x.max()
    if lo == hi:
        return lo
    hist = torch.histc(x, bins=n_bins, min=lo.item(), max=hi.item())
    bin_edges = torch.linspace(lo.item(), hi.item(), n_bins + 1, device=x.device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    cum_sum = hist.cumsum(0)
    cum_mean = (hist * bin_centers).cumsum(0)
    global_mean = cum_mean[-1]

    # Inter-class variance for each possible threshold
    w0 = cum_sum
    w1 = total - cum_sum
    mu0_minus_mu = cum_mean * total - global_mean * cum_sum
    inter_class_var = mu0_minus_mu**2 / (w0 * w1 + 1e-10)

    best_idx = inter_class_var.argmax()
    return bin_centers[best_idx]


def _otsu_threshold_batch(target: Tensor, n_bins: int = 256) -> Tensor:
    """Compute per-(sample, channel) Otsu threshold for a batch.

    Parameters
    ----------
    target : Tensor
        Batch tensor of shape ``(B, C, ...)``.
    n_bins : int
        Number of histogram bins per element.

    Returns
    -------
    Tensor
        Thresholds of shape ``(B, C, 1, ...)`` for broadcasting against
        spatial dimensions of ``target``.
    """
    B, C = target.shape[:2]
    spatial_ndim = target.ndim - 2
    # torch.histc has no dim parameter — must loop over (B, C) elements
    flat = target.reshape(B * C, -1)
    thresholds = torch.stack([_otsu_threshold(sample, n_bins) for sample in flat.unbind(0)])
    return thresholds.reshape(B, C, *([1] * spatial_ndim))


class SpotlightLoss(nn.Module):
    """Foreground-aware loss for virtual staining.

    Combines masked MSE (supervision restricted to foreground) with
    Dice loss on soft-thresholded predictions for shape preservation.

    When ``fg_threshold`` is ``None`` (default), the foreground mask is
    estimated per-sample using Otsu thresholding at runtime. When set to
    a float (e.g., ``0.0``), a fixed threshold is used instead — this is
    correct when the target is normalized with the Otsu threshold as
    subtrahend (paper's approach: ``NormalizeSampled(subtrahend="otsu_threshold")``),
    which centers the FG/BG boundary at exactly 0.

    Parameters
    ----------
    lambda_mse : float
        Weight for the masked MSE term. Dice weight is ``1 - lambda_mse``.
    sigmoid_k : float
        Sharpness of the normalized tunable sigmoid. Must be in ``(-1, 0)``;
        more negative = sharper threshold. Paper default: -0.95.
    eps : float
        Epsilon for Dice denominator stability.
    fg_threshold : float or None
        Fixed foreground threshold. When ``None``, Otsu thresholding is
        computed per-sample at runtime (backward compatible). When a float
        (e.g., ``0.0``), used directly as the FG/BG boundary.
    """

    def __init__(
        self,
        lambda_mse: float = 0.5,
        sigmoid_k: float = -0.95,
        eps: float = 1e-6,
        fg_threshold: float | None = None,
    ) -> None:
        super().__init__()
        if not -1 < sigmoid_k < 0:
            raise ValueError(f"sigmoid_k must be in (-1, 0), got {sigmoid_k}")
        if not 0 < lambda_mse < 1:
            raise ValueError(f"lambda_mse must be in (0, 1), got {lambda_mse}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.lambda_mse = lambda_mse
        self.sigmoid_k = sigmoid_k
        self.eps = eps
        self.fg_threshold = fg_threshold

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, pred: Tensor, target: Tensor, fg_mask: Tensor | None = None) -> Tensor:
        """Compute the Spotlight loss.

        Loss is computed per (batch, channel) pair and averaged. Channels
        with foreground mask data use masked MSE + Dice; channels without
        (all-zero mask) fall back to regular MSE and are excluded from Dice.

        Parameters
        ----------
        pred : Tensor
            Predicted tensor of shape ``(B, C, Z, Y, X)`` or ``(B, C, Y, X)``.
        target : Tensor
            Ground truth tensor of the same shape.
        fg_mask : Tensor or None
            Precomputed binary foreground mask of the same shape as target.
            When provided, used directly instead of computing from target.
            When None (default), mask is computed at runtime using
            ``fg_threshold`` or Otsu thresholding.

        Returns
        -------
        Tensor
            Scalar loss value.
        """
        # Foreground mask: precomputed > fixed threshold > per-(sample, channel) Otsu
        if fg_mask is not None:
            mask = fg_mask.float()
        elif self.fg_threshold is not None:
            mask = (target >= self.fg_threshold).float()
        else:
            threshold = _otsu_threshold_batch(target)
            mask = (target >= threshold).float()

        spatial_dims = tuple(range(2, pred.ndim))

        # Per-(B, C) foreground counts
        fg_per_ch = mask.sum(dim=spatial_dims)  # (B, C)
        has_fg = fg_per_ch > 0  # (B, C)

        # Masked MSE per (B, C) — channels without mask fall back to regular MSE
        sq_err = (pred - target) ** 2
        masked_sum = (sq_err * mask).sum(dim=spatial_dims)  # (B, C)
        regular_mse = sq_err.mean(dim=spatial_dims)  # (B, C)
        channel_mse = torch.where(has_fg, masked_sum / (fg_per_ch + self.eps), regular_mse)
        masked_mse = channel_mse.mean()

        # Dice per (B, C) — only channels with masks contribute
        soft_pred = _tunable_sigmoid(pred, self.sigmoid_k)
        intersection = (soft_pred * mask).sum(dim=spatial_dims)  # (B, C)
        soft_sum = soft_pred.sum(dim=spatial_dims)  # (B, C)
        channel_dice = 1 - (2 * intersection) / (soft_sum + fg_per_ch + self.eps)
        n_masked = has_fg.sum()
        if n_masked > 0:
            dice = (channel_dice * has_fg.float()).sum() / n_masked
        else:
            dice = pred.new_tensor(0.0)

        return self.lambda_mse * masked_mse + (1 - self.lambda_mse) * dice
