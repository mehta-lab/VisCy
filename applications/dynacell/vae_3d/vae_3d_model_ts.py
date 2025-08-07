import os
import torch
import torch.nn as nn
from .modules.autoencoders_ts import Autoencoder3DKL
from .vae_3d_config import VAE3DConfig


class VAE3DModel(nn.Module):
    def __init__(self, config: VAE3DConfig):
        super().__init__()
        self.config = config

        self.num_down_blocks = config.num_down_blocks
        self.num_up_blocks = self.num_down_blocks

        # Initialize Autoencoder3DKL
        self.vae = Autoencoder3DKL(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            num_down_blocks=self.num_down_blocks,
            num_up_blocks=self.num_up_blocks,
            block_out_channels=config.vae_block_out_channels,
            latent_channels=config.latent_channels, 
        )

        self.load_pretrained_weights(checkpoint_path=config.loadcheck_path)

    def load_pretrained_weights(self, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.
        """
                
        if os.path.splitext(checkpoint_path)[1] == '.safetensors':
            from safetensors.torch import load_file
            checkpoints_state = load_file(checkpoint_path)
        else:
            checkpoints_state = torch.load(checkpoint_path, map_location="cpu")

        if "model" in checkpoints_state:
            checkpoints_state = checkpoints_state["model"]
        elif "module" in checkpoints_state:
            checkpoints_state = checkpoints_state["module"]

        IncompatibleKeys = self.load_state_dict(checkpoints_state, strict=True)
        IncompatibleKeys = IncompatibleKeys._asdict()

        missing_keys = []
        for keys in IncompatibleKeys["missing_keys"]:
            if keys.find("dummy") == -1:
                missing_keys.append(keys)

        unexpected_keys = []
        for keys in IncompatibleKeys["unexpected_keys"]:
            if keys.find("dummy") == -1:
                unexpected_keys.append(keys)

        if len(missing_keys) > 0:
            print(
                "Missing keys in {}: {}".format(
                    checkpoint_path,
                    missing_keys,
                )
            )

        if len(unexpected_keys) > 0:
            print(
                "Unexpected keys {}: {}".format(
                    checkpoint_path,
                    unexpected_keys,
                )
            )

    def encode(self, x):
        """Encodes input into latent space."""
        return self.vae.encode(x)

    def decode(self, latents):
        """Decodes latent space into reconstructed input."""
        return self.vae.decode(latents)

    def forward(self, x):
        # placeholder forward
        return x

    def reconstruct(self, x):
        mean, logvar = self.encode(x)
        latents = mean + torch.exp(0.5 * logvar) * torch.randn_like(logvar)  # Reparameterization trick
        recon_x = self.decode(latents)

        return recon_x
