"""Causal temporal models for fate anticipation.

A small dilated 1D-conv stack with left-only padding (so the prediction at timestep ``t``
depends only on frames ``<= t``) followed by a binary head. Scaled-down analog of the TCN
in Soelistyo et al. 2022.
"""

from __future__ import annotations

import torch
from torch import nn


class CausalTCN(nn.Module):
    """Causal dilated 1D-conv stack with a binary fate head.

    Parameters
    ----------
    in_dim : int, optional
        Input feature dimension per timestep (e.g. 32 for PCA, 768 for the raw embedding).
    hidden : int, optional
        Number of channels in each convolution layer.
    dilations : tuple of int, optional
        Dilation factor per layer; the receptive field grows with the largest dilation.
    out_dim : int, optional
        Number of target logits. ``1`` preserves the historical binary-head
        interface; values greater than one provide independent output heads on
        the same temporal trunk.
    dropout : float, optional
        Dropout after each causal convolution. Default: 0.3.
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        out_dim: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if out_dim < 1:
            raise ValueError("out_dim must be at least 1")
        self.convs = nn.ModuleList()
        self.pads: list[int] = []
        c_in = in_dim
        for d in dilations:
            self.pads.append((3 - 1) * d)  # kernel_size=3, left-pad only -> causal
            self.convs.append(nn.Conv1d(c_in, hidden, kernel_size=3, dilation=d))
            c_in = hidden
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the fate logit at the current (last) timestep.

        Parameters
        ----------
        x : torch.Tensor
            Input windows, shape ``(B, T, C)``.

        Returns
        -------
        torch.Tensor
            Fate logit per window, shape ``(B,)`` for ``out_dim=1`` or
            ``(B, out_dim)`` for a multi-output head.
        """
        h = x.transpose(1, 2)  # (B, C, T)
        for conv, pad in zip(self.convs, self.pads):
            h = self.drop(self.act(conv(nn.functional.pad(h, (pad, 0)))))
        return self.head(h[:, :, -1]).squeeze(-1)


class CausalTransformer(nn.Module):
    """Small causal Transformer with learned relative-lag embeddings.

    The input contains only observations through the prediction time. A strict
    upper-triangular attention mask additionally guarantees that the hidden
    state at position ``i`` cannot depend on positions after ``i``. Learned
    positions represent fixed lags within the causal window; RoPE is not used.

    The default configuration has approximately the same parameter count as a
    five-frame ``CausalTCN(in_dim=18, hidden=64, dilations=(1, 2))``.
    """

    def __init__(
        self,
        in_dim: int = 18,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 48,
        dropout: float = 0.3,
        out_dim: int = 1,
        max_window: int = 5,
    ) -> None:
        super().__init__()
        if out_dim < 1:
            raise ValueError("out_dim must be at least 1")
        if max_window < 1:
            raise ValueError("max_window must be at least 1")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")

        self.max_window = max_window
        self.input_projection = nn.Linear(in_dim, d_model)
        self.position = nn.Parameter(torch.empty(1, max_window, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)
        causal_mask = torch.triu(
            torch.ones(max_window, max_window, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("_causal_mask", causal_mask, persistent=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return one strictly causal hidden state per input position."""
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, time, features)")
        time = x.shape[1]
        if time > self.max_window:
            raise ValueError(f"Input window {time} exceeds max_window={self.max_window}")
        h = self.input_projection(x) + self.position[:, self.max_window - time :]
        h = self.encoder(h, mask=self._causal_mask[:time, :time])
        return self.final_norm(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict current-time logits from the final causal token."""
        h = self.encode(x)
        return self.head(h[:, -1]).squeeze(-1)
