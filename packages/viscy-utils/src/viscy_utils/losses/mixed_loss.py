"""Mixed reconstruction loss module.

Provides a configurable combination of L1, L2, and MS-DSSIM losses
for image reconstruction tasks, adapted from Zhao et al.
"""

import torch
import torch.nn.functional as F
from torch import nn

from viscy_utils.evaluation.metrics import ms_ssim_25d


class MixedLoss(nn.Module):
    """Mixed reconstruction loss.

    Adapted from Zhao et al, https://arxiv.org/pdf/1511.08861.pdf
    Reduces to simple distances if only one weight is non-zero.

    Parameters
    ----------
    l1_alpha : float, optional
        L1 loss weight, by default 0.5.
    l2_alpha : float, optional
        L2 loss weight, by default 0.0.
    ms_dssim_alpha : float, optional
        MS-DSSIM weight, by default 0.5.
    """

    def __init__(
        self,
        l1_alpha: float = 0.5,
        l2_alpha: float = 0.0,
        ms_dssim_alpha: float = 0.5,
    ):
        super().__init__()
        if not any([l1_alpha, l2_alpha, ms_dssim_alpha]):
            raise ValueError("Loss term weights cannot be all zero!")
        self.l1_alpha = l1_alpha
        self.l2_alpha = l2_alpha
        self.ms_dssim_alpha = ms_dssim_alpha

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, preds, target):
        """Compute the mixed reconstruction loss.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted tensor of shape (B, C, D, H, W).
        target : torch.Tensor
            Target tensor of the same shape as preds.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        loss = 0
        if self.l1_alpha:
            # the gaussian in the reference is not used
            # because the SSIM here uses a uniform window
            loss += F.l1_loss(preds, target) * self.l1_alpha
        if self.l2_alpha:
            loss += F.mse_loss(preds, target) * self.l2_alpha
        if self.ms_dssim_alpha:
            ms_ssim = ms_ssim_25d(preds, target, clamp=True)
            # the 1/2 factor in the original DSSIM is not used
            # since the MS-SSIM here is stabilized with ReLU
            loss += (1 - ms_ssim) * self.ms_dssim_alpha
        return loss
