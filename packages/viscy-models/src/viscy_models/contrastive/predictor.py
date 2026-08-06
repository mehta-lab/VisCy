"""Latent predictor for temporal-dynamics learning (Wang et al. 2026 ``f_theta``).

A single MLP shared across all cells that maps a latent state to its next state,
``z_hat_{t+1} = predictor(z_t)``. Trained with a stop-gradient target
(``L_pred = ||predictor(z_t) - sg(z_{t+1})||^2``). One shared predictor working
across many cells is what rewards a *common* transition structure; the
stop-gradient on the target prevents the trivial collapse where the encoder maps
all frames to a constant.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

__all__ = ["Predictor"]


class Predictor(nn.Module):
    """Shared latent next-state predictor: an MLP ``dim -> hidden -> dim``.

    Parameters
    ----------
    dim : int
        Latent dimension (matches the encoder embedding, e.g. 768). Input and
        output dimension.
    hidden_dim : int or None
        Hidden layer width. Defaults to ``dim`` when None.
    """

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Predict the next latent state.

        Parameters
        ----------
        z : Tensor
            Current latent states of shape ``(N, dim)``.

        Returns
        -------
        Tensor
            Predicted next latent states of shape ``(N, dim)``.
        """
        return self.net(z)
