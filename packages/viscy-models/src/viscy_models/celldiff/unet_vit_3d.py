"""End-to-end 3D U-Net with Vision Transformer bottleneck for virtual staining.

Deterministic (no diffusion) model mapping label-free phase contrast to
fluorescence virtual staining.  Architecture: CNN encoder with skip connections,
transformer bottleneck, CNN decoder with skip connections.
"""

import torch
import torch.nn as nn

from viscy_models.celldiff.modules.patch_embed_3d import PatchEmbed3D
from viscy_models.celldiff.modules.positional_embedding import get_3d_sincos_pos_embed
from viscy_models.celldiff.modules.simple_diffusion import ResnetBlock
from viscy_models.celldiff.modules.transformer import (
    FinalLayer,
    TransformerBlock,
    unpatchify,
)


class UNetViT3D(nn.Module):
    """3D U-Net with Vision Transformer bottleneck for end-to-end virtual staining.

    Takes a 3D label-free input (e.g. phase contrast) and directly predicts
    fluorescence virtual staining.  No diffusion / timestep conditioning.

    Spatial downsampling uses stride ``(1, 2, 2)`` so the Z dimension is
    preserved through the encoder/decoder (``downsamples_z = False``).

    Parameters
    ----------
    input_spatial_size : list[int]
        Expected input spatial size ``[D, H, W]``.  Used for positional
        embedding computation and forward-pass assertions.
    in_channels : int
        Number of input channels (label-free modality).
    out_channels : int
        Number of output channels (fluorescence targets).
    dims : list[int]
        Channel widths at each encoder level, length ``L``.
        Must have ``len(dims) == len(num_res_block) + 1`` so the last entry
        is the bottleneck channel count.
    num_res_block : list[int]
        Number of residual blocks at each encoder/decoder level, length ``L-1``.
    hidden_size : int
        Transformer hidden dimension.
    num_heads : int
        Number of self-attention heads.
    dim_head : int
        Dimension per attention head.
    dropout : float
        Attention dropout rate.
    final_dropout : float
        Feed-forward output dropout rate.
    num_hidden_layers : int
        Number of transformer blocks in the bottleneck.
    patch_size : int
        Cubic patch size for the 3D patch embedding.
    """

    def __init__(
        self,
        input_spatial_size: list[int] = [8, 512, 512],
        in_channels: int = 1,
        out_channels: int = 1,
        dims: list[int] = [32, 64, 128],
        num_res_block: list[int] = [2, 2],
        hidden_size: int = 512,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        final_dropout: float = 0.0,
        num_hidden_layers: int = 2,
        patch_size: int = 4,
    ) -> None:
        super().__init__()

        assert len(dims) == len(num_res_block) + 1, (
            f"len(dims)={len(dims)} must equal len(num_res_block)+1={len(num_res_block) + 1}"
        )

        self.input_spatial_size = input_spatial_size
        self.num_res_block = num_res_block
        self._dims = dims
        self._patch_size = patch_size

        # ── Input projection ────────────────────────────────────────────────
        self.inconv = nn.Conv3d(in_channels, dims[0], kernel_size=3, stride=1, padding=1)

        # ── Encoder (downsampling) ───────────────────────────────────────────
        downs: dict[str, nn.Module] = {}
        for i_level in range(len(num_res_block)):
            for i_block in range(num_res_block[i_level]):
                downs[f"{i_level}{i_block}"] = ResnetBlock(dims[i_level], dims[i_level])
            # Halve H and W only; preserve depth.
            downs[f"down{i_level}"] = nn.Conv3d(
                dims[i_level],
                dims[i_level + 1],
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            )
        self.downs = nn.ModuleDict(downs)

        # ── Transformer bottleneck ──────────────────────────────────────────
        self.img_embedding = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=dims[-1],
            embed_dim=hidden_size,
            bias=True,
        )

        # Fixed 3-D sinusoidal positional embedding (non-learnable).
        n_down = len(num_res_block)
        latent_size = input_spatial_size[:1] + [s // (2**n_down) for s in input_spatial_size[1:]]

        # Validate that every latent dimension is exactly divisible by patch_size.
        # Integer division silently truncates remainders, which causes the decoder
        # to crash on torch.cat with mismatched skip-connection shapes.
        dim_names = ["D", "H", "W"]
        for dim_val, name, orig in zip(latent_size, dim_names, input_spatial_size):
            if dim_val % patch_size != 0:
                raise ValueError(
                    f"Latent {name} dimension {dim_val} (from input {name}={orig}) "
                    f"is not divisible by patch_size={patch_size}. "
                    f"Each spatial dimension after {n_down} encoder downsamples "
                    f"(stride (1,2,2)) must be divisible by patch_size."
                )
            if dim_val == 0:
                raise ValueError(
                    f"Latent {name} dimension is 0 (from input {name}={orig}). "
                    f"input_spatial_size is too small for {n_down} downsamples."
                )

        self.latent_grid_size = [s // patch_size for s in latent_size]

        img_pos_embed = (
            torch.from_numpy(get_3d_sincos_pos_embed(hidden_size, self.latent_grid_size)).float().unsqueeze(0)
        )
        self.img_pos_embed = nn.Parameter(img_pos_embed, requires_grad=False)

        self.mids = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    final_dropout=final_dropout,
                    # time_embed_dim=None → no adaLN conditioning
                )
                for _ in range(num_hidden_layers)
            ]
        )

        self.img_proj_out = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=dims[-1],
            # time_embed_dim=None → no adaLN conditioning
        )

        # ── Decoder (upsampling) ────────────────────────────────────────────
        ups: dict[str, nn.Module] = {}
        for i_level in reversed(range(len(num_res_block))):
            ups[f"up{i_level}"] = nn.ConvTranspose3d(
                dims[i_level + 1],
                dims[i_level],
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1),
            )
            for i_block in range(num_res_block[i_level]):
                # Skip connection doubles the channel count before this block.
                ups[f"{i_level}{i_block}"] = ResnetBlock(dims[i_level] * 2, dims[i_level])
        self.ups = nn.ModuleDict(ups)

        # ── Output projection ───────────────────────────────────────────────
        self.outconv = nn.Conv3d(dims[0], out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def num_blocks(self) -> int:
        """Number of spatial downsampling stages (each halves H and W by 2)."""
        return len(self.num_res_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run end-to-end virtual staining prediction.

        Parameters
        ----------
        x : torch.Tensor
            Input volume of shape ``(B, in_channels, D, H, W)``.
            Spatial dimensions must match ``input_spatial_size``.

        Returns
        -------
        torch.Tensor
            Predicted staining of shape ``(B, out_channels, D, H, W)``.
        """
        if x.shape[2:] != torch.Size(self.input_spatial_size):
            raise ValueError(f"x spatial size {list(x.shape[2:])} does not match expected {self.input_spatial_size}")

        x = self.inconv(x)

        # ── Encode ──────────────────────────────────────────────────────────
        skips: list[torch.Tensor] = []
        for i_level in range(len(self.num_res_block)):
            for i_block in range(self.num_res_block[i_level]):
                x = self.downs[f"{i_level}{i_block}"](x)
                skips.append(x)
            x = self.downs[f"down{i_level}"](x)

        # ── Transformer bottleneck ──────────────────────────────────────────
        x_embeds = self.img_embedding(x) + self.img_pos_embed
        for block in self.mids:
            x_embeds = block(x_embeds)
        x = self.img_proj_out(x_embeds)
        x = unpatchify(x, self._dims[-1], self.latent_grid_size, self._patch_size)

        # ── Decode ──────────────────────────────────────────────────────────
        for i_level in reversed(range(len(self.num_res_block))):
            x = self.ups[f"up{i_level}"](x)
            for i_block in range(self.num_res_block[i_level]):
                x = torch.cat((x, skips.pop()), dim=1)
                x = self.ups[f"{i_level}{i_block}"](x)

        return self.outconv(x)
