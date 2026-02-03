"""Percentile-based intensity scaling transforms.

This module provides GPU-optimized transforms for scaling image intensity
based on percentile ranges, designed for batched microscopy data.
"""

from warnings import warn

import numpy as np
import torch
from monai.transforms import MapTransform, ScaleIntensityRangePercentiles
from numpy.typing import DTypeLike
from torch import Tensor
from typing_extensions import Iterable

__all__ = [
    "BatchedScaleIntensityRangePercentiles",
    "BatchedScaleIntensityRangePercentilesd",
]


class BatchedScaleIntensityRangePercentiles(ScaleIntensityRangePercentiles):
    """Scale batched tensor intensity based on percentile range.

    GPU-optimized version of MONAI's ScaleIntensityRangePercentiles that
    operates on batched data. Computes percentiles across spatial dimensions
    for each sample in the batch independently.

    Parameters
    ----------
    lower : float
        Lower percentile for input range (0-100).
    upper : float
        Upper percentile for input range (0-100).
    b_min : float | None
        Minimum value of output range. None to skip scaling.
    b_max : float | None
        Maximum value of output range. None to skip scaling.
    clip : bool
        Whether to clip output values to [b_min, b_max]. Default: False.
    relative : bool
        Whether to compute relative percentile range. Default: False.
    channel_wise : bool
        Whether to compute percentiles per channel. Default: False.
    dtype : DTypeLike
        Output data type. Default: np.float32.

    Returns
    -------
    Tensor
        Scaled tensor with normalized intensity values.

    See Also
    --------
    monai.transforms.ScaleIntensityRangePercentiles : Parent MONAI transform.
    BatchedScaleIntensityRangePercentilesd : Dictionary wrapper.
    """

    def _normalize(self, img: Tensor) -> Tensor:
        q_low = self.lower / 100.0
        q_high = self.upper / 100.0
        batch_size, *_ = img.shape
        # TODO: address pytorch#64947 to improve performance
        a_min, a_max = torch.quantile(
            img.view(batch_size, -1),
            torch.tensor([q_low, q_high], dtype=img.dtype, device=img.device),
            dim=1,
        ).reshape(2, batch_size, 1, 1, 1, 1)
        b_min = self.b_min
        b_max = self.b_max

        if self.relative:
            if (self.b_min is None) or (self.b_max is None):
                raise ValueError("If it is relative, b_min and b_max should not be None.")
            b_min = ((self.b_max - self.b_min) * (q_low)) + self.b_min
            b_max = ((self.b_max - self.b_min) * (q_high)) + self.b_min

        if (a_min == a_max).any():
            warn("Divide by zero (a_min == a_max)")
            if b_min is None:
                return img - a_min
            return img - a_min + b_min

        img = (img - a_min) / (a_max - a_min)
        if (b_min is not None) and (b_max is not None):
            img = img * (b_max - b_min) + b_min
        if self.clip:
            img = img.clip(b_min, b_max)

        return img

    def __call__(self, img: Tensor) -> Tensor:
        """Scale the input tensor based on percentile range.

        Parameters
        ----------
        img : Tensor
            Input tensor with shape (B, C, D, H, W).

        Returns
        -------
        Tensor
            Scaled tensor with normalized intensity values.
        """
        if self.channel_wise:
            channels = [self._normalize(img[:, c : c + 1]) for c in range(img.shape[1])]
            return torch.cat(channels, dim=1)
        else:
            return self._normalize(img=img)


class BatchedScaleIntensityRangePercentilesd(MapTransform):
    """Dictionary wrapper for batched percentile intensity scaling.

    Applies BatchedScaleIntensityRangePercentiles to specified keys in
    a data dictionary.

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys of the data dictionary to scale.
    lower : float
        Lower percentile for input range (0-100).
    upper : float
        Upper percentile for input range (0-100).
    b_min : float | None
        Minimum value of output range.
    b_max : float | None
        Maximum value of output range.
    clip : bool
        Whether to clip output values. Default: False.
    relative : bool
        Whether to compute relative percentile range. Default: False.
    channel_wise : bool
        Whether to compute percentiles per channel. Default: False.
    dtype : DTypeLike
        Output data type. Default: np.float32.
    allow_missing_keys : bool
        Whether to allow missing keys. Default: False.

    Returns
    -------
    dict[str, Tensor]
        Dictionary with scaled tensors for specified keys.

    See Also
    --------
    BatchedScaleIntensityRangePercentiles : Underlying transform.
    """

    def __init__(
        self,
        keys: str | Iterable[str],
        lower: float,
        upper: float,
        b_min: float | None,
        b_max: float | None,
        clip: bool = False,
        relative: bool = False,
        channel_wise: bool = False,
        dtype: DTypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = BatchedScaleIntensityRangePercentiles(
            lower, upper, b_min, b_max, clip, relative, channel_wise, dtype
        )

    def __call__(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Scale intensity of specified keys.

        Parameters
        ----------
        data : dict[str, Tensor]
            Dictionary containing tensors with shape (B, C, D, H, W).

        Returns
        -------
        dict[str, Tensor]
            Dictionary with scaled tensors for specified keys.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d
