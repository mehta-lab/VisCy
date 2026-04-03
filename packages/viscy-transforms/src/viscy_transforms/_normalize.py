"""Normalization transforms for microscopy data.

This module provides transforms for normalizing image data using
precomputed statistics from FOV or dataset-level computations.
"""

from monai.transforms import MapTransform
from torch import Tensor
from typing_extensions import Iterable, Literal

from viscy_transforms._typing import Sample

__all__ = ["NormalizeSampled", "MinMaxSampled"]

_DATA_RANGE_KEYS: dict[str, tuple[str, str]] = {
    "min_max": ("min", "max"),
    "p1_p99": ("p1", "p99"),
    "p5_p95": ("p5", "p95"),
}


def _match_image(tensor: Tensor, target: Tensor) -> Tensor:
    """Reshape a scalar or ``(B,)`` stat tensor to broadcast against an image."""
    return tensor.reshape(tensor.shape + (1,) * (target.ndim - tensor.ndim)).to(device=target.device)


class NormalizeSampled(MapTransform):
    is_spatial = False

    """Normalize using precomputed statistics stored in ``sample["norm_meta"]``.

    Expects ``norm_meta`` to have structure::

        {channel_label: {level: {stat_name: Tensor, ...}, ...}, ...}

    For ``timepoint_statistics``, the dataset must pre-resolve the correct
    timepoint so that the level value is ``{stat_name: Tensor}`` directly
    (not nested by timepoint index).

    Stats tensors may be scalar ``()`` or batched ``(B,)``.
    ``_match_image`` reshapes them to broadcast against
    ``(B, 1, Z, Y, X)`` image tensors.

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys to normalize.
    level : {'fov_statistics', 'dataset_statistics', 'timepoint_statistics'}
        Level of normalization.
    subtrahend : str, optional
        Subtrahend for normalization, defaults to "mean".
    divisor : str, optional
        Divisor for normalization, defaults to "std".
    remove_meta : bool, optional
        Whether to remove metadata after normalization, defaults to False.
    """

    def __init__(
        self,
        keys: str | Iterable[str],
        level: Literal["fov_statistics", "dataset_statistics", "timepoint_statistics"],
        subtrahend="mean",
        divisor="std",
        remove_meta: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.level = level
        self.remove_meta = remove_meta

    def __call__(self, sample: Sample) -> Sample:
        for key in self.keys:
            level_meta = sample["norm_meta"][key][self.level]
            subtrahend_val = _match_image(level_meta[self.subtrahend], sample[key])
            divisor_val = _match_image(level_meta[self.divisor], sample[key]) + 1e-8
            sample[key] = (sample[key] - subtrahend_val) / divisor_val
        if self.remove_meta:
            sample.pop("norm_meta")
        return sample


class MinMaxSampled(MapTransform):
    is_spatial = False

    """Normalize to [-1, 1] by clipping then rescaling with precomputed range statistics.

    Applies::

        x_clipped = clamp(x, low, high)
        x_norm = 2 * (x_clipped - low) / (high - low) - 1

    where ``low`` and ``high`` are determined by ``data_range``.

    Expects ``norm_meta`` to have structure::

        {channel_label: {level: {stat_name: Tensor, ...}, ...}, ...}

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys to normalize.
    level : {'fov_statistics', 'dataset_statistics', 'timepoint_statistics'}
        Level of normalization.
    data_range : {'min_max', 'p1_p99', 'p5_p95'}
        Statistic pair to use as ``[low, high]``. Defaults to ``'p1_p99'``.
    remove_meta : bool
        Whether to remove metadata after normalization, defaults to False.
    """

    def __init__(
        self,
        keys: str | Iterable[str],
        level: Literal["fov_statistics", "dataset_statistics", "timepoint_statistics"],
        data_range: Literal["min_max", "p1_p99", "p5_p95"] = "p1_p99",
        remove_meta: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.level = level
        if data_range not in _DATA_RANGE_KEYS:
            raise ValueError(f"Invalid data_range: {data_range}")
        self._low_key, self._high_key = _DATA_RANGE_KEYS[data_range]
        self.remove_meta = remove_meta

    def __call__(self, sample: Sample) -> Sample:
        for key in self.keys:
            level_meta = sample["norm_meta"][key][self.level]
            min_val = _match_image(level_meta[self._low_key], sample[key])
            max_val = _match_image(level_meta[self._high_key], sample[key])
            sample[key] = sample[key].clamp(min_val, max_val)
            sample[key] = 2.0 * (sample[key] - min_val) / (max_val - min_val + 1e-8) - 1.0
        if self.remove_meta:
            sample.pop("norm_meta")
        return sample
