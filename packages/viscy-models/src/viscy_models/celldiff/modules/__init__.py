"""CellDiff shared modules for model building blocks."""

import math

import torch
import torch.nn as nn

__all__ = ["TimestepEmbedder"]


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into vector representations.

    Uses sinusoidal frequency embeddings followed by an MLP.

    Parameters
    ----------
    hidden_size : int
        Output embedding dimension.
    frequency_embedding_size : int
        Dimension of intermediate sinusoidal embeddings.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Parameters
        ----------
        t : torch.Tensor
            Scalar timesteps of shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Embeddings of shape ``(B, hidden_size)``.
        """
        args = t[:, None].float() * self.freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        t_emb = self.mlp(t_freq)
        return t_emb
