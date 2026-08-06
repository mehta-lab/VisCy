"""NT-Xent losses with optional temperature scheduling.

Provides:
- ``NTXentLoss``: re-exported with a ``step()`` method for temperature scheduling.
- ``NTXentHCL``: adds hard-negative concentration on top.
- ``TemporalStraighteningLoss``: curvature loss on consecutive track embeddings.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss as _NTXentLossBase
from pytorch_metric_learning.utils import common_functions as c_f
from torch import Tensor, nn

from viscy_models.schedule import cosine_anneal


class NTXentLoss(_NTXentLossBase):
    """NT-Xent loss with optional temperature schedule.

    Drop-in replacement for ``pytorch_metric_learning.losses.NTXentLoss``
    that adds a :meth:`step` method for cosine temperature annealing.

    Call :meth:`step` at the start of each epoch to apply the schedule.

    Parameters
    ----------
    temperature : float
        Temperature scaling for cosine similarities. Default: 0.07.
    temperature_schedule : {"cosine", "constant"}
        Schedule for temperature. ``"cosine"`` anneals from
        ``temperature_start`` down to ``temperature`` over
        ``temperature_warmup_epochs``. Default: ``"constant"``.
    temperature_start : float
        Initial temperature when using ``"cosine"`` schedule. Default: 0.1.
    temperature_warmup_epochs : int
        Epochs over which to anneal temperature. Default: 50.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        temperature_schedule: Literal["cosine", "constant"] = "constant",
        temperature_start: float = 0.1,
        temperature_warmup_epochs: int = 50,
        **kwargs,
    ):
        super().__init__(temperature=temperature, **kwargs)
        self.temperature_schedule = temperature_schedule
        self.temperature_start = temperature_start
        self.temperature_end = temperature
        self.temperature_warmup_epochs = temperature_warmup_epochs

    def step(self, epoch: int) -> None:
        """Update temperature for the given epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch (0-indexed).
        """
        if self.temperature_schedule == "cosine":
            self.temperature = cosine_anneal(
                self.temperature_start,
                self.temperature_end,
                epoch,
                self.temperature_warmup_epochs,
            )


class NTXentHCL(NTXentLoss):
    """NT-Xent loss with hard-negative concentration and optional temperature schedule.

    When beta=0.0, produces identical results to standard NTXentLoss.
    When beta>0, up-weights hard negatives (high cosine similarity)
    in the denominator, focusing learning on difficult examples.

    The HCL reweighting multiplies each negative pair's contribution
    in the denominator by exp(beta * sim(i, k)), concentrating gradient
    signal on negatives that are close to the anchor in embedding space.

    Call :meth:`step` at the start of each epoch to apply the temperature schedule.

    Parameters
    ----------
    temperature : float
        Temperature scaling for cosine similarities. Default: 0.07.
    beta : float
        Hard-negative concentration strength. 0.0 = standard NT-Xent.
        Higher values concentrate more on hard negatives. Default: 0.5.
    temperature_schedule : {"cosine", "constant"}
        Inherited from :class:`NTXentLoss`. Default: ``"constant"``.
    temperature_start : float
        Inherited from :class:`NTXentLoss`. Default: 0.1.
    temperature_warmup_epochs : int
        Inherited from :class:`NTXentLoss`. Default: 50.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        beta: float = 0.5,
        temperature_schedule: Literal["cosine", "constant"] = "constant",
        temperature_start: float = 0.1,
        temperature_warmup_epochs: int = 50,
        **kwargs,
    ):
        super().__init__(
            temperature=temperature,
            temperature_schedule=temperature_schedule,
            temperature_start=temperature_start,
            temperature_warmup_epochs=temperature_warmup_epochs,
            **kwargs,
        )
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
            n_per_p = c_f.to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)

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

            log_exp = torch.log((numerator / denominator) + c_f.small_val(dtype))
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()


class TemporalStraighteningLoss(nn.Module):
    """Curvature loss straightening latent trajectories (Wang et al. 2026).

    Penalizes turning between consecutive latent velocity vectors along a track,
    ``L_curv = mean_t (1 - cos(v_t, v_{t+1}))`` with ``v_t = z_{t+1} - z_t``.
    Minimizing it makes successive velocities parallel, so the trajectory
    approaches a straight line and Euclidean distance tracks progression.

    Acts directly on the encoder embedding ``z`` (no projection head): cosine is
    scale-invariant, so no normalization is applied inside the loss. Composes
    additively with any anti-collapse objective (e.g. NT-Xent).

    Parameters
    ----------
    eps : float
        Numerical floor for the cosine similarity denominator. Default: 1e-8.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def step(self, epoch: int) -> None:
        """No-op; the loss weight schedule lives on the LightningModule."""

    def forward(self, z_seq: Tensor, valid: Tensor | None = None) -> Tensor:
        """Compute mean curvature over a batch of consecutive-frame sequences.

        Parameters
        ----------
        z_seq : Tensor
            Embeddings of shape ``(B, K, D)`` with ``K >= 3`` consecutive frames
            of the same track in time order.
        valid : Tensor or None
            Boolean mask of shape ``(B,)`` selecting samples with a real
            (non-placeholder) sequence. Invalid samples are dropped. When every
            sample is invalid, a graph-connected zero (``(z_seq * 0).sum()``) is
            returned so DDP all-reduce stays balanced. Default: None (all used).

        Returns
        -------
        Tensor
            Scalar curvature loss in ``[0, 2]``.
        """
        if z_seq.size(1) < 3:
            raise ValueError(f"z_seq must have K >= 3 frames, got K={z_seq.size(1)}")
        v = z_seq[:, 1:, :] - z_seq[:, :-1, :]  # (B, K-1, D) velocities
        v1, v2 = v[:, :-1, :], v[:, 1:, :]  # (B, K-2, D) consecutive pairs
        cos = F.cosine_similarity(v1, v2, dim=-1, eps=self.eps)  # (B, K-2)
        per_sample = (1.0 - cos).mean(dim=1)  # (B,)
        if valid is not None:
            per_sample = per_sample[valid]
            if per_sample.numel() == 0:
                # Graph-connected zero: keeps every DDP rank producing a
                # gradient-tracked scalar so all-reduce stays balanced.
                return (z_seq * 0.0).sum()
        return per_sample.mean()
