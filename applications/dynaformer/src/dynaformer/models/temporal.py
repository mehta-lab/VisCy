"""Bidirectional transformer with rotary position embedding over the time axis.

The temporal module is the only part of the network that sees more than one frame.
It consumes one token per frame and returns one token per frame, so the spatial
encoder and the image decoder stay purely per-frame.

Rotary position embedding (RoPE) rather than a learned positional parameter is a
deliberate choice: there is no length-dependent weight, so the same module trains
on short clips and runs a whole track in a single forward pass.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "RoPEEncoderLayer",
    "RoPESelfAttention",
    "RoPETransformerEncoder",
    "RotaryEmbedding",
    "apply_rope",
]


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the two halves of the last dimension into each other."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embedding to queries or keys.

    Parameters
    ----------
    x : Tensor
        ``(B, heads, T, head_dim)``.
    cos : Tensor
        ``(T, head_dim)`` cosine table, broadcast over batch and heads.
    sin : Tensor
        ``(T, head_dim)`` sine table, broadcast over batch and heads.

    Returns
    -------
    Tensor
        Rotated tensor of the same shape as ``x``.
    """
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Build ``(cos, sin)`` tables for the current sequence length.

    Parameters
    ----------
    head_dim : int
        Attention head dimension. Must be even.
    base : float
        Geometric base of the inverse frequencies, by default 10000.
    """

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Return cosine and sine tables of shape ``(seq_len, head_dim)``."""
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inv_freq.to(device))
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)


class RoPESelfAttention(nn.Module):
    """Multi-head self-attention with rotary position embedding, bidirectional.

    Parameters
    ----------
    d_model : int
        Token width. Must be divisible by ``n_heads``.
    n_heads : int
        Number of attention heads.
    dropout : float
        Attention dropout probability, by default 0.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        """Attend over the sequence axis of a ``(B, T, d_model)`` tensor."""
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        cos, sin = self.rope(t, x.device, query.dtype)
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)
        # No causal mask: every frame may attend to past and future frames.
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(attended.transpose(1, 2).reshape(b, t, d))


class RoPEEncoderLayer(nn.Module):
    """Pre-norm encoder layer with RoPE self-attention and a GELU feed-forward block.

    Parameters
    ----------
    d_model : int
        Token width.
    n_heads : int
        Number of attention heads.
    ff_mult : int
        Feed-forward expansion ratio, by default 4.
    dropout : float
        Dropout probability applied to both residual branches, by default 0.1.
    """

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Run one pre-norm attention and feed-forward residual block."""
        x = x + self.drop(self.attn(self.norm1(x)))
        return x + self.drop(self.feed_forward(self.norm2(x)))


class RoPETransformerEncoder(nn.Module):
    """Stack of :class:`RoPEEncoderLayer` blocks over the temporal axis.

    Accepts any sequence length, so a model trained on short clips can run a whole
    track in one forward pass.

    Parameters
    ----------
    d_model : int
        Token width, matching the frame encoder's token dimension.
    n_heads : int
        Number of attention heads, by default 8.
    n_layers : int
        Number of encoder layers, by default 4.
    ff_mult : int
        Feed-forward expansion ratio, by default 4.
    dropout : float
        Dropout probability, by default 0.1.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([RoPEEncoderLayer(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Mix a ``(B, T, d_model)`` sequence of frame tokens along ``T``."""
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
