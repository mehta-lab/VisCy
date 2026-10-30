"""3D patch embedding for vision transformers."""

from collections.abc import Sequence

import torch
import torch.nn as nn

__all__ = ["PatchEmbed3D", "normalize_patch_size"]


def normalize_patch_size(patch_size: int | Sequence[int]) -> tuple[int, int, int]:
    """Expand a patch size to a per-axis ``(D, H, W)`` triple.

    An ``int`` means a cubic patch, matching the original CELL-Diff recipe. A
    3-sequence gives per-axis control, which is what lets the same network run
    Z-preserving at ``D=1`` — a cubic ``patch_size=4`` would need a Conv3d with
    a size-4 kernel over a size-1 axis.

    Parameters
    ----------
    patch_size : int | Sequence[int]
        Cubic side length, or ``(D, H, W)`` patch extents.

    Returns
    -------
    tuple[int, int, int]
        Per-axis patch extents.
    """
    dims = (patch_size,) * 3 if isinstance(patch_size, int) else tuple(int(p) for p in patch_size)
    if len(dims) != 3:
        raise ValueError(f"patch_size must be an int or a length-3 sequence, got {patch_size!r}")
    if any(p < 1 for p in dims):
        raise ValueError(f"patch_size entries must be >= 1, got {patch_size!r}")
    return dims


class PatchEmbed3D(nn.Module):
    """Convert a 3D volume into a sequence of patch embeddings.

    Uses a single strided 3D convolution to project non-overlapping
    patches into an embedding space.

    Parameters
    ----------
    patch_size : int | Sequence[int]
        Cubic patch side length, or per-axis ``(D, H, W)`` extents.
    in_chans : int
        Number of input channels.
    embed_dim : int
        Embedding dimension per patch.
    bias : bool
        Whether to include bias in the projection convolution.
    """

    def __init__(self, patch_size: int | Sequence[int], in_chans: int, embed_dim: int, bias: bool = True) -> None:
        super().__init__()
        patch = normalize_patch_size(patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch, stride=patch, bias=bias)

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
