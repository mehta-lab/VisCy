import torch
from torch import nn as nn

class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans, embed_dim, bias=True):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        B, C, D, H, W = x.shape
        x = x.view(B, C, -1)
        x = x.transpose(1, 2)

        return x
    