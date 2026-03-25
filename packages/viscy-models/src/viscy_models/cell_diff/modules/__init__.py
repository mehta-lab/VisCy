import torch
import torch.nn as nn
import math

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class CondConvNet(nn.Module):
    def __init__(self, in_channels, cond_out_channels):
        """
        Parameters:
        - in_channels (int): Number of channels per input image.
        - cond_out_channels (list of int): Output channels for each conv layer.
        - num_cell_images (int): Number of comma-separated cell images in config.cell_image.
        """
        super(CondConvNet, self).__init__()
        layers = []

        # First convolution layer
        layers.append(
            nn.Conv3d(
                in_channels, 
                cond_out_channels[0],
                kernel_size=2,
                stride=2,
                padding=0
            )
        )

        # Intermediate layers
        for i in range(len(cond_out_channels) - 1):
            layers.append(nn.SiLU())
            layers.append(
                nn.Conv3d(
                    cond_out_channels[i],
                    cond_out_channels[i + 1],
                    kernel_size=2,
                    stride=2,
                    padding=0
                )
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
