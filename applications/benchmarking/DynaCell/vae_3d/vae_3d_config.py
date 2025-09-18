# -*- coding: utf-8 -*-
from dataclasses import dataclass
from transformers import PretrainedConfig
from dataclasses import field

@dataclass
class VAE3DConfig(PretrainedConfig):
    model_type: str = 'vae'

    # Model parameters
    in_channels: int = 1
    out_channels: int = 1
    num_down_blocks: int = 5
    latent_channels: int = 2
    vae_block_out_channels: list = field(default_factory=lambda: [32, 64, 128, 256, 256])
    loadcheck_path: str = ""