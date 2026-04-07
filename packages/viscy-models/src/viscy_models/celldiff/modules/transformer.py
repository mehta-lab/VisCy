"""Transformer blocks and output layers for CellDiff models."""

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation: ``x * (1 + scale) + shift``."""
    return x * (1 + scale) + shift


class TransformerBlock(nn.Module):
    """Transformer block with optional adaLN-Zero timestep conditioning.

    When ``time_embed_dim`` is None, uses standard pre-LayerNorm attention
    and feed-forward. When provided, uses adaptive layer norm conditioning
    from timestep embeddings (adaLN-Zero).

    Parameters
    ----------
    hidden_size : int
        Hidden dimension.
    num_heads : int
        Number of attention heads.
    dim_head : int
        Dimension per attention head.
    dropout : float
        Dropout rate for attention.
    final_dropout : float
        Dropout rate for the feed-forward output.
    norm_eps : float
        Layer norm epsilon.
    attention_bias : bool
        Whether attention QKV projections have bias.
    attention_out_bias : bool
        Whether attention output projection has bias.
    activation_fn : str
        Feed-forward activation function name (passed to diffusers FeedForward).
    ff_inner_dim : int | None
        Feed-forward inner dimension override.
    ff_bias : bool
        Whether feed-forward layers have bias.
    time_embed_dim : int | None
        Timestep embedding dimension for adaLN-Zero conditioning.
        Pass None for unconditional (e.g. deterministic virtual staining).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dim_head: int,
        dropout: float = 0.0,
        final_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        attention_bias: bool = False,
        attention_out_bias: bool = True,
        activation_fn: str = "geglu",
        ff_inner_dim: int | None = None,
        ff_bias: bool = True,
        time_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)

        self.attn = Attention(
            query_dim=hidden_size,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            upcast_attention=False,
            out_bias=attention_out_bias,
        )

        self.ff = FeedForward(
            hidden_size,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        if time_embed_dim is not None:
            self.adaLN = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, 6 * hidden_size, bias=True),
            )
            nn.init.zeros_(self.adaLN[1].weight)
            nn.init.zeros_(self.adaLN[1].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Token sequence of shape ``(B, L, C)``.
        time_embeds : torch.Tensor | None
            Timestep embeddings of shape ``(B, time_embed_dim)``.
            Pass None for unconditional forward pass.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, L, C)``.
        """
        B, L, C = hidden_states.shape
        if C != self.hidden_size:
            raise ValueError(f"hidden_states last dim {C} != hidden_size {self.hidden_size}")

        if time_embeds is not None:
            # adaLN-Zero conditioned path
            shift_msa, scale_msa, gate_msa, shift_ff, scale_ff, gate_ff = self.adaLN(time_embeds).chunk(6, dim=-1)

            normed = modulate(
                self.norm1(hidden_states),
                shift_msa.unsqueeze(1),
                scale_msa.unsqueeze(1),
            )
            attn_out = self.attn(normed)
            hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_out

            normed = modulate(
                self.norm2(hidden_states),
                shift_ff.unsqueeze(1),
                scale_ff.unsqueeze(1),
            )
            ff_out = self.ff(normed)
            hidden_states = hidden_states + gate_ff.unsqueeze(1) * ff_out
        else:
            # Simple pre-LN path (no time conditioning)
            hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
            hidden_states = hidden_states + self.ff(self.norm2(hidden_states))

        return hidden_states


def unpatchify(
    x: torch.Tensor,
    out_channels: int,
    latent_grid_size: list[int],
    patch_size: int,
) -> torch.Tensor:
    """Reconstruct a 3D volume from patch tokens.

    Parameters
    ----------
    x : torch.Tensor
        Patch tokens of shape ``(N, T, patch_size**3 * out_channels)``.
    out_channels : int
        Number of output channels.
    latent_grid_size : list[int]
        Latent grid dimensions ``[D, H, W]`` such that ``D*H*W == T``.
    patch_size : int
        Cubic patch side length.

    Returns
    -------
    torch.Tensor
        Reconstructed volume of shape ``(N, out_channels, D*p, H*p, W*p)``.
    """
    c = out_channels
    p = patch_size
    d, h, w = latent_grid_size
    if d * h * w != x.shape[1]:
        raise ValueError(f"Expected {d * h * w} tokens (grid {d}x{h}x{w}), got {x.shape[1]}")

    x = x.reshape(x.shape[0], d, h, w, p, p, p, c)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    imgs = x.reshape(x.shape[0], c, d * p, h * p, w * p)
    return imgs


class FinalLayer(nn.Module):
    """Final projection layer with optional adaLN-Zero conditioning.

    Parameters
    ----------
    hidden_size : int
        Transformer hidden dimension.
    patch_size : int
        Cubic patch size (output is ``patch_size**3 * out_channels`` per token).
    out_channels : int
        Number of output channels after unpatchifying.
    time_embed_dim : int | None
        If provided, adds adaLN-Zero conditioning from time embeddings.
        Pass None for unconditional (e.g. end-to-end virtual staining).
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        time_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True)
        if time_embed_dim is not None:
            self.adaLN = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, 2 * hidden_size, bias=True),
            )
            nn.init.zeros_(self.adaLN[1].weight)
            nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Token sequence of shape ``(B, L, hidden_size)``.
        c : torch.Tensor | None
            Conditioning embeddings of shape ``(B, time_embed_dim)``.

        Returns
        -------
        torch.Tensor
            Projected tokens of shape ``(B, L, patch_size**3 * out_channels)``.
        """
        if c is not None:
            shift, scale = self.adaLN(c).chunk(2, dim=1)
            x = modulate(
                self.norm_final(x),
                shift.unsqueeze(1),
                scale.unsqueeze(1),
            )
        else:
            x = self.norm_final(x)
        x = self.linear(x)
        return x
