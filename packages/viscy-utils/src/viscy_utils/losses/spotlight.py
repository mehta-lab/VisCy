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
    """Compute per-sample Otsu threshold for a batch.

    Parameters
    ----------
    target : Tensor
        Batch tensor of shape ``(B, ...)``.
    n_bins : int
        Number of histogram bins per sample.

    Returns
    -------
    Tensor
        Thresholds of shape ``(B, 1, ...)`` for broadcasting against ``target``.
    """
    # torch.histc has no dim parameter — must loop over batch
    thresholds = torch.stack([_otsu_threshold(sample.flatten(), n_bins) for sample in target.unbind(0)])
    return thresholds.view(-1, *([1] * (target.ndim - 1)))


class SpotlightLoss(nn.Module):
    """Foreground-aware loss for virtual staining.

    Combines masked MSE (supervision restricted to foreground) with
    Dice loss on soft-thresholded predictions for shape preservation.

    The foreground mask is estimated from the target using Otsu
    thresholding. The prediction is soft-thresholded with a normalized
    tunable sigmoid before computing Dice loss.

    After z-score normalization (mean ~0), Otsu separates the bimodal
    FG/BG distribution around 0, which is correct for the tunable
    sigmoid operating around that point.

    Parameters
    ----------
    lambda_mse : float
        Weight for the masked MSE term. Dice weight is ``1 - lambda_mse``.
    sigmoid_k : float
        Sharpness of the normalized tunable sigmoid. Must be in ``(-1, 0)``;
        more negative = sharper threshold. Paper default: -0.95.
    eps : float
        Epsilon for Dice denominator stability.
    n_bins : int
        Number of histogram bins for Otsu thresholding.
    min_foreground_fraction : float
        Minimum fraction of foreground voxels per sample. Samples below
        this threshold contribute zero loss (effectively skipped).
        Paper uses 0.001 (0.1%). Default 0.0 disables filtering.
    """

    def __init__(
        self,
        lambda_mse: float = 0.5,
        sigmoid_k: float = -0.95,
        eps: float = 1e-6,
        n_bins: int = 256,
        min_foreground_fraction: float = 0.0,
    ) -> None:
        super().__init__()
        if not -1 < sigmoid_k < 0:
            raise ValueError(f"sigmoid_k must be in (-1, 0), got {sigmoid_k}")
        if not 0 < lambda_mse < 1:
            raise ValueError(f"lambda_mse must be in (0, 1), got {lambda_mse}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        if not 0 <= min_foreground_fraction < 1:
            raise ValueError(f"min_foreground_fraction must be in [0, 1), got {min_foreground_fraction}")
        self.lambda_mse = lambda_mse
        self.sigmoid_k = sigmoid_k
        self.eps = eps
        self.n_bins = n_bins
        self.min_foreground_fraction = min_foreground_fraction

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the Spotlight loss.

        Parameters
        ----------
        pred : Tensor
            Predicted tensor of shape ``(B, C, Z, Y, X)`` or ``(B, C, Y, X)``.
        target : Tensor
            Ground truth tensor of the same shape.

        Returns
        -------
        Tensor
            Scalar loss value.
        """
        # Foreground mask from Otsu thresholding on target
        threshold = _otsu_threshold_batch(target, self.n_bins)
        mask = (target >= threshold).float()

        # Per-sample foreground filtering (paper: ≥0.1% FG voxels)
        if self.min_foreground_fraction > 0:
            B = target.shape[0]
            fg_fractions = mask.view(B, -1).mean(dim=1)
            keep = fg_fractions >= self.min_foreground_fraction
            if not keep.any():
                return (pred * 0).sum()
            # Zero out below-threshold samples in both mask and soft predictions
            sample_weight = keep.float().view(B, *([1] * (pred.ndim - 1)))
            mask = mask * sample_weight

        # Masked MSE: only foreground voxels contribute
        sq_err = (pred - target) ** 2
        fg_count = mask.sum()
        if fg_count > 0:
            masked_mse = (sq_err * mask).sum() / fg_count
        else:
            masked_mse = sq_err.mean()

        # Dice loss on soft-thresholded prediction vs binary mask
        soft_pred = _tunable_sigmoid(pred, self.sigmoid_k)
        if self.min_foreground_fraction > 0:
            # Exclude filtered samples from Dice denominator
            soft_pred = soft_pred * sample_weight
        intersection = (soft_pred * mask).sum()
        dice = 1 - (2 * intersection) / (soft_pred.sum() + mask.sum() + self.eps)

        return self.lambda_mse * masked_mse + (1 - self.lambda_mse) * dice
