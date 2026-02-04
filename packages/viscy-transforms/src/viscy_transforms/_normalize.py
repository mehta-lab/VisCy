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
    """
    Normalize the sample.

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys to normalize.
    level : {'fov_statistics', 'dataset_statistics'}
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
        level: Literal["fov_statistics", "dataset_statistics"],
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
        """Normalize the sample using precomputed statistics.

        Parameters
        ----------
        sample : Sample
            Dictionary containing tensors and norm_meta with statistics.

        Returns
        -------
        Sample
            Dictionary with normalized tensors for specified keys.
        """
        for key in self.keys:
            level_meta = sample["norm_meta"][key][self.level]
            subtrahend_val = level_meta[self.subtrahend]
            subtrahend_val = self._match_image(subtrahend_val, sample[key])
            divisor_val = level_meta[self.divisor] + 1e-8  # avoid div by zero
            divisor_val = self._match_image(divisor_val, sample[key])
            sample[key] = (sample[key] - subtrahend_val) / divisor_val
        if self.remove_meta:
            sample.pop("norm_meta")
        return sample

    def _normalize():
        NotImplementedError("_normalization() not implemented")
