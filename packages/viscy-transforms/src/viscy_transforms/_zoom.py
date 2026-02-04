"""Batch-aware zoom transforms.

This module provides GPU-efficient batched zoom (resize/rescale) transforms
using PyTorch's native interpolation functions.
"""

from typing import Sequence

import torch
from monai.transforms import MapTransform, Transform
from torch import Tensor
from typing_extensions import Literal

__all__ = ["BatchedZoom", "BatchedZoomd"]


class BatchedZoom(Transform):
    """Zoom (resize) a batched tensor by a scale factor.

    Uses ``torch.nn.functional.interpolate`` for GPU-efficient resizing
    of batched 3D data. Supports various interpolation modes.

    Parameters
    ----------
    scale_factor : float | tuple[float, float, float]
        Multiplier for spatial size. If float, same factor is used for
        all dimensions. If tuple, specifies (depth, height, width) factors.
    mode : str
        Interpolation algorithm. Options:
        - "nearest": Nearest neighbor interpolation
        - "nearest-exact": Exact nearest neighbor
        - "linear": Linear interpolation (1D)
        - "bilinear": Bilinear interpolation (2D)
        - "bicubic": Bicubic interpolation (2D)
        - "trilinear": Trilinear interpolation (3D)
        - "area": Area interpolation
    align_corners : bool | None
        If True, aligns corners of input and output tensors.
        Only applicable for linear, bilinear, bicubic, trilinear modes.
        Default: None.
    recompute_scale_factor : bool | None
        If True, recomputes scale_factor for use in interpolation.
        Default: None.
    antialias : bool
        If True, applies anti-aliasing when downsampling.
        Only effective for bilinear and bicubic modes. Default: False.

    Returns
    -------
    Tensor
        Resized tensor with scaled spatial dimensions.

    Examples
    --------
    >>> zoom = BatchedZoom(scale_factor=0.5, mode="trilinear")
    >>> x = torch.randn(2, 1, 32, 64, 64)  # [B, C, D, H, W]
    >>> y = zoom(x)
    >>> y.shape
    torch.Size([2, 1, 16, 32, 32])
    """

    def __init__(
        self,
        scale_factor: float | tuple[float, float, float],
        mode: Literal[
            "nearest",
            "nearest-exact",
            "linear",
            "bilinear",
            "bicubic",
            "trilinear",
            "area",
        ],
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None,
        antialias: bool = False,
    ) -> None:
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def __call__(self, sample: Tensor) -> Tensor:
        """Zoom the input tensor.

        Parameters
        ----------
        sample : Tensor
            Input tensor with shape (B, C, D, H, W).

        Returns
        -------
        Tensor
            Resized tensor with scaled spatial dimensions.
        """
        return torch.nn.functional.interpolate(
            sample,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.antialias,
        )


class BatchedZoomd(MapTransform):
    """Dictionary wrapper for BatchedZoom transform.

    Applies zoom (resize) to specified keys in a data dictionary.

    Parameters
    ----------
    keys : Sequence[str]
        Keys of the data dictionary to apply zoom to.
    scale_factor : float | tuple[float, float, float]
        Multiplier for spatial size. If float, same factor is used for
        all dimensions. If tuple, specifies (depth, height, width) factors.
    mode : str
        Interpolation algorithm. See :class:`BatchedZoom` for options.
    align_corners : bool | None
        If True, aligns corners of input and output tensors. Default: None.
    recompute_scale_factor : bool | None
        If True, recomputes scale_factor for interpolation. Default: None.
    antialias : bool
        If True, applies anti-aliasing when downsampling. Default: False.

    Returns
    -------
    dict[str, Tensor]
        Dictionary with zoomed tensors for specified keys.

    See Also
    --------
    BatchedZoom : Underlying zoom transform.

    Examples
    --------
    >>> zoom = BatchedZoomd(keys=["image"], scale_factor=2.0, mode="trilinear")
    >>> sample = {"image": torch.randn(2, 1, 16, 32, 32)}
    >>> output = zoom(sample)
    >>> output["image"].shape
    torch.Size([2, 1, 32, 64, 64])
    """

    def __init__(
        self,
        keys: Sequence[str],
        scale_factor: float | tuple[float, float, float],
        mode: Literal[
            "nearest",
            "nearest-exact",
            "linear",
            "bilinear",
            "bicubic",
            "trilinear",
            "area",
        ],
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None,
        antialias: bool = False,
    ) -> None:
        super().__init__(keys)
        self.transform = BatchedZoom(
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )

    def __call__(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply zoom to the specified keys.

        Parameters
        ----------
        data : dict[str, Tensor]
            Dictionary containing tensors with shape (B, C, D, H, W).

        Returns
        -------
        dict[str, Tensor]
            Dictionary with zoomed tensors for specified keys.
        """
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])
        return d
