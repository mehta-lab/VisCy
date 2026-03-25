import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward
from transformers.activations import ACT2FN

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class BertPredictionHeadTransform(nn.Module):
    def __init__(
            self, 
            hidden_size, 
            hidden_act: str = 'gelu', 
            norm_eps: float = 1e-5,
        ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = ACT2FN[hidden_act]
        self.LayerNorm = nn.LayerNorm(hidden_size, norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MLMHead(nn.Module):
    def __init__(
            self,
            hidden_size, 
            vocab_size, 
            hidden_act: str = 'gelu', 
            norm_eps: float = 1e-5, 
            weight=None
        ):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size, hidden_act, norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        dim_head,
        dropout: float = 0.0,
        final_dropout: float = 0.0,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_bias: bool = False,
        attention_out_bias: bool = True,
        activation_fn: str = "geglu",
        ff_inner_dim=None,
        ff_bias: bool = True,
        time_embed_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.time_embed_dim = time_embed_dim

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

    def forward(self, hidden_states, time_embeds=None):
        """
        hidden_states: (B, L, C)
        time_embeds:   (B, time_dim) or None for unconditional forward pass
        """
        B, L, C = hidden_states.shape
        if C != self.hidden_size:
            raise ValueError(f"hidden_states last dim {C} != hidden_size {self.hidden_size}")

        if time_embeds is not None:
            # adaLN-Zero conditioned path
            shift_msa, scale_msa, gate_msa, shift_ff, scale_ff, gate_ff = self.adaLN(time_embeds).chunk(6, dim=-1)

            normed = modulate(self.norm1(hidden_states), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
            attn_out = self.attn(normed)
            hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_out

            normed = modulate(self.norm2(hidden_states), shift_ff.unsqueeze(1), scale_ff.unsqueeze(1))
            ff_out = self.ff(normed)
            hidden_states = hidden_states + gate_ff.unsqueeze(1) * ff_out
        else:
            # Simple pre-LN path (no time conditioning)
            hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
            hidden_states = hidden_states + self.ff(self.norm2(hidden_states))

        return hidden_states


def unpatchify(x, out_channels, latent_grid_size, patch_size):
    """
    x: (N, T, patch_size**3 * C)
    imgs: (N, C, D, H, W)
    """
    c = out_channels
    p = patch_size
    d, h, w = latent_grid_size
    assert d * h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], d, h, w, p, p, p, c))
    x = torch.einsum('ndhwkpqc->ncdkhpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, d * p, h * p, w * p))
    return imgs


class FinalLayer(nn.Module):
    """
    The final layer of DiT.

    Parameters
    ----------
    hidden_size : int
        Transformer hidden dimension.
    patch_size : int
        Cubic patch size (output is patch_size**3 * out_channels per token).
    out_channels : int
        Number of output channels after unpatchifying.
    time_embed_dim : int | None
        If provided, adds adaLN-Zero conditioning from time embeddings.
        Pass None for unconditional (e.g. end-to-end virtual staining).
    """

    def __init__(self, hidden_size, patch_size, out_channels, time_embed_dim: int | None = None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True)
        if time_embed_dim is not None:
            self.adaLN = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)
            )
            nn.init.zeros_(self.adaLN[1].weight)
            nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, c=None):
        if c is not None:
            shift, scale = self.adaLN(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift.unsqueeze(1).expand(-1, x.shape[1], -1), scale.unsqueeze(1).expand(-1, x.shape[1], -1))
        else:
            x = self.norm_final(x)
        x = self.linear(x)
        return x