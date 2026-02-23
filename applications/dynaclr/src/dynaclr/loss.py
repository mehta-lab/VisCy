"""NT-Xent loss with hard-negative concentration (HCL).

Provides NTXentHCL, a drop-in replacement for pytorch_metric_learning's NTXentLoss
that up-weights hard negatives via a beta parameter.
"""

from __future__ import annotations

import torch
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.utils import common_functions as c_f
from torch import Tensor


class NTXentHCL(NTXentLoss):
    """NT-Xent loss with hard-negative concentration.

    When beta=0.0, produces identical results to standard NTXentLoss.
    When beta>0, up-weights hard negatives (high cosine similarity)
    in the denominator, focusing learning on difficult examples.

    The HCL reweighting multiplies each negative pair's contribution
    in the denominator by exp(beta * sim(i, k)), concentrating gradient
    signal on negatives that are close to the anchor in embedding space.

    Parameters
    ----------
    temperature : float
        Temperature scaling for cosine similarities. Default: 0.07.
    beta : float
        Hard-negative concentration strength. 0.0 = standard NT-Xent.
        Higher values concentrate more on hard negatives. Default: 0.5.
    """

    def __init__(self, temperature: float = 0.07, beta: float = 0.5, **kwargs):
        super().__init__(temperature=temperature, **kwargs)
        self.beta = beta
        self.add_to_recordable_attributes(list_of_names=["beta"], is_stat=False)

    def _compute_loss(
        self,
        pos_pairs: Tensor,
        neg_pairs: Tensor,
        indices_tuple: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> dict:
        """Compute NTXent loss with optional hard-negative concentration.

        When beta=0.0, this delegates to the parent NTXentLoss._compute_loss
        for exact numerical equivalence. When beta>0, it applies HCL
        reweighting to the negative pairs in the log-softmax denominator.
        """
        if self.beta == 0.0:
            return super()._compute_loss(pos_pairs, neg_pairs, indices_tuple)

        a1, p, a2, _ = indices_tuple

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype

            # If dealing with actual distances, use negative distances
            if not self.distance.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs

            pos_pairs_scaled = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs_scaled = neg_pairs / self.temperature

            # Build per-anchor negative mask: n_per_p[i, j] = 1 if neg j
            # belongs to anchor i
            n_per_p = c_f.to_dtype(
                a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype
            )

            # HCL reweighting: multiply each negative by exp(beta * sim)
            # neg_pairs are raw similarities (before /temperature)
            # We use them directly for the reweighting factor
            hcl_weights = torch.exp(self.beta * neg_pairs) * n_per_p

            # Normalize weights per anchor so they sum to the count of
            # negatives for that anchor (preserves loss magnitude)
            neg_counts = n_per_p.sum(dim=1, keepdim=True)
            weight_sums = hcl_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            hcl_weights = hcl_weights * neg_counts / weight_sums

            # Apply temperature scaling and masks
            neg_pairs_masked = neg_pairs_scaled * n_per_p
            neg_pairs_masked[n_per_p == 0] = c_f.neg_inf(dtype)

            # Numerical stability: subtract max
            max_val = torch.max(
                pos_pairs_scaled,
                torch.max(neg_pairs_masked, dim=1, keepdim=True)[0],
            ).detach()

            numerator = torch.exp(pos_pairs_scaled - max_val).squeeze(1)
            # Apply HCL weights to the exponentiated negatives
            weighted_neg = hcl_weights * torch.exp(neg_pairs_masked - max_val)
            denominator = torch.sum(weighted_neg, dim=1) + numerator

            log_exp = torch.log(
                (numerator / denominator) + c_f.small_val(dtype)
            )
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()
