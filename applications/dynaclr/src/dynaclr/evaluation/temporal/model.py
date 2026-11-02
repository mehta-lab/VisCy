"""Causal temporal models for fate anticipation.

A small dilated 1D-conv stack with left-only padding (so the prediction at timestep ``t``
depends only on frames ``<= t``) followed by a binary head. Scaled-down analog of the TCN
in Soelistyo et al. 2022.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding used by the ``vs_cell_cycle`` temporal head.

    Positions default to the integer slot index. Subclasses may supply
    arbitrary real-valued positions (for example elapsed hours) through
    :meth:`from_positions`; the rotation angle depends only on pairwise position
    differences, so any global offset cancels inside the attention product.
    """

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        inverse_frequency = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def from_positions(
        self,
        positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cosine and sine tables for explicit positions.

        Parameters
        ----------
        positions : torch.Tensor
            Real-valued positions, shape ``(T,)`` or ``(B, T)``.
        dtype : torch.dtype
            Output dtype matching the query/key tensors.

        Returns
        -------
        tuple of torch.Tensor
            Cosine and sine tables broadcastable against ``(B, H, T, head_dim)``:
            shape ``(T, head_dim)`` for 1D positions, ``(B, 1, T, head_dim)``
            for batched positions.
        """
        frequencies = positions.to(torch.float32).unsqueeze(-1) * self.inverse_frequency.to(positions.device)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        if angles.ndim == 3:
            angles = angles.unsqueeze(1)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    def forward(
        self,
        sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
        return self.from_positions(positions, dtype)


class ContinuousRotaryEmbedding(RotaryEmbedding):
    """Rotary embedding over real-valued time in hours.

    Inverse frequencies are log-spaced so that the rotation periods span
    ``[min_period_hours, max_period_hours]``. Because angles depend only on
    pairwise differences, the embedding is invariant to a global time shift,
    and a 10-minute and a 2-hour frame interval produce different attention
    patterns without any change to the model.

    Parameters
    ----------
    head_dim : int
        Per-head feature dimension; must be even.
    min_period_hours : float, optional
        Shortest rotation period; should be near the finest frame interval.
    max_period_hours : float, optional
        Longest rotation period; should cover the longest track duration.
    """

    def __init__(
        self,
        head_dim: int,
        min_period_hours: float = 1.0 / 6.0,
        max_period_hours: float = 96.0,
    ) -> None:
        super().__init__(head_dim)
        if min_period_hours <= 0 or max_period_hours <= min_period_hours:
            raise ValueError("Require 0 < min_period_hours < max_period_hours")
        periods = torch.logspace(
            math.log10(min_period_hours),
            math.log10(max_period_hours),
            head_dim // 2,
            dtype=torch.float32,
        )
        self.inverse_frequency = 2.0 * math.pi / periods


class RoPESelfAttention(nn.Module):
    """Bidirectional self-attention with rotary query/key positions.

    Parameters
    ----------
    d_model : int
        Model width.
    n_heads : int
        Number of attention heads.
    dropout : float
        Attention dropout probability.
    rope : nn.Module, optional
        Rotary embedding module. Defaults to slot-index :class:`RotaryEmbedding`;
        pass :class:`ContinuousRotaryEmbedding` to rotate by elapsed time.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        rope: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.rope = rope if rope is not None else RotaryEmbedding(self.head_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend over the sequence with rotary positions.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens, shape ``(B, T, d_model)``.
        positions : torch.Tensor, optional
            Real-valued positions per token, shape ``(B, T)``. Defaults to the
            slot index.
        key_padding_mask : torch.Tensor, optional
            Boolean ``(B, T)`` mask where ``True`` marks keys that may be
            attended to. Hidden or padded keys are ``False``.

        Returns
        -------
        torch.Tensor
            Contextualized tokens, shape ``(B, T, d_model)``.
        """
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        if positions is None:
            cosine, sine = self.rope(length, x.device, query.dtype)
        else:
            cosine, sine = self.rope.from_positions(positions, query.dtype)
        query = query * cosine + _rotate_half(query) * sine
        key = key * cosine + _rotate_half(key) * sine
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.projection(output.transpose(1, 2).reshape(batch, length, width))


class RoPETransformerLayer(nn.Module):
    """Pre-normalized RoPE attention and feed-forward residual block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        feedforward_multiplier: int,
        dropout: float,
        rope: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = RoPESelfAttention(d_model, n_heads, dropout, rope=rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * feedforward_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * feedforward_multiplier, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), positions=positions, key_padding_mask=key_padding_mask))
        return x + self.dropout(self.feedforward(self.norm2(x)))


class BidirectionalRoPETransformer(nn.Module):
    """Dihan/``vs_cell_cycle``-style temporal RoPE classifier.

    The architecture matches the frozen FUCCI head: a per-frame projection,
    four bidirectional RoPE Transformer blocks by default, and a per-token
    LayerNorm/linear head. ``readout_index`` selects the scored token: ``-1``
    supports history windows ending at prediction time, while the comparison
    uses the center token of a past/current/future window for offline inference.
    """

    def __init__(
        self,
        in_dim: int = 18,
        d_model: int = 384,
        n_heads: int = 8,
        num_layers: int = 4,
        feedforward_multiplier: int = 4,
        dropout: float = 0.1,
        out_dim: int = 1,
        readout_index: int = -1,
    ) -> None:
        super().__init__()
        if out_dim < 1:
            raise ValueError("out_dim must be at least 1")
        self.readout_index = readout_index
        self.input_projection = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList(
            [
                RoPETransformerLayer(
                    d_model,
                    n_heads,
                    feedforward_multiplier,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return one bidirectionally contextualized token per input frame."""
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, time, features)")
        hidden = self.input_projection(x)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.final_norm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict logits from the configured contextualized token."""
        hidden = self.encode(x)
        return self.head(hidden[:, self.readout_index]).squeeze(-1)
