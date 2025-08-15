from typing import Tuple

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


class Autoencoder3DKL(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_down_blocks: int = 2,
        num_up_blocks: int = 2,
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            num_down_blocks=num_down_blocks,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            num_up_blocks=num_up_blocks,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, 1) if use_post_quant_conv else None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:

        enc = self.encoder(x)
        if self.quant_conv is not None:
            enc = self.quant_conv(enc)

        return enc

    def encode(self, x: torch.Tensor):
        h = self._encode(x)
        mean, logvar = torch.chunk(h, 2, dim=1)

        return mean, logvar

    def _decode(self, z: torch.Tensor):

        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)

        return dec

    def decode(self, z: torch.FloatTensor):        
        decoded = self._decode(z)

        return decoded

    def forward(self, x):
        # placeholder forward
        return x