"""Normalization transforms for microscopy data.

This module provides transforms for normalizing image data using
precomputed statistics from FOV or dataset-level computations.
"""

from monai.transforms import MapTransform
from torch import Tensor
from typing_extensions import Iterable, Literal

from viscy_transforms._typing import Sample

__all__ = ["NormalizeSampled"]


class NormalizeSampled(MapTransform):
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

    @staticmethod
    def _match_image(tensor: Tensor, target: Tensor) -> Tensor:
        return tensor.reshape(tensor.shape + (1,) * (target.ndim - tensor.ndim)).to(device=target.device)

    def __call__(self, sample: Sample) -> Sample:
        for key in self.keys:
            level_meta = sample["norm_meta"][key][self.level]
            subtrahend_val = level_meta[self.subtrahend]
            subtrahend_val = self._match_image(subtrahend_val, sample[key])
            divisor_val = level_meta[self.divisor] + 1e-8
            divisor_val = self._match_image(divisor_val, sample[key])
            sample[key] = (sample[key] - subtrahend_val) / divisor_val
        if self.remove_meta:
            sample.pop("norm_meta")
        return sample
