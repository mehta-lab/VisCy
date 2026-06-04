"""3D patch embedding for vision transformers."""

import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """Convert a 3D volume into a sequence of patch embeddings.

    Uses a single strided 3D convolution to project non-overlapping
    cubic patches into an embedding space.

    Parameters
    ----------
    patch_size : int
        Cubic patch side length.
    in_chans : int
        Number of input channels.
    embed_dim : int
        Embedding dimension per patch.
    bias : bool
        Whether to include bias in the projection convolution.
    """

    def __init__(self, patch_size: int, in_chans: int, embed_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input volume into patch token sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Patch embeddings of shape ``(B, num_patches, embed_dim)``.
        """
        x = self.proj(x)
        B, C, D, H, W = x.shape
        x = x.view(B, C, -1)
        x = x.transpose(1, 2)
        return x
