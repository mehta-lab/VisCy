"""Intensity inversion transforms for microscopy data.

This module provides transforms for randomly inverting image intensity
values, useful for augmentation in microscopy imaging.
"""

import torch
from monai.transforms import MapTransform, RandomizableTransform
from typing_extensions import Iterable

from viscy_transforms._typing import Sample

__all__ = ["BatchedRandInvertIntensityd", "RandInvertIntensityd"]


class BatchedRandInvertIntensityd(MapTransform, RandomizableTransform):
    """Randomly invert intensity per sample in a batch.

    For each sample in the batch, independently decides whether to
    negate the tensor values. Uses MONAI-style ``randomize()`` to
    generate per-sample decisions, matching the pattern in
    :class:`~viscy_transforms.BatchedRandAdjustContrastd`.

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys of the data dictionary to potentially invert.
    prob : float
        Probability of applying inversion per sample. Default: 0.1.
    allow_missing_keys : bool
        Whether to allow missing keys. Default: False.
    """

    is_spatial = False

    def __init__(
        self,
        keys: str | Iterable[str],
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys=allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

    def randomize(self, data: torch.Tensor) -> None:
        """Generate per-sample inversion decisions.

        Parameters
        ----------
        data : Tensor
            Reference tensor to determine batch size and device.
        """
        batch_size = data.shape[0]
        self._do_transform = torch.rand(batch_size, device=data.device) < self.prob

    def __call__(self, sample: Sample) -> Sample:
        """Randomly invert intensities with per-sample randomization.

        Parameters
        ----------
        sample : Sample
            Dictionary with batched tensors ``(B, C, Z, Y, X)``.

        Returns
        -------
        Sample
            Dictionary with potentially inverted tensors.
        """
        first_key = next(iter(self.key_iterator(sample)))
        self.randomize(sample[first_key])
        sign = torch.where(self._do_transform, -1.0, 1.0).to(
            dtype=sample[first_key].dtype, device=sample[first_key].device
        )
        sign = sign.view(-1, *([1] * (sample[first_key].ndim - 1)))
        for key in self.key_iterator(sample):
            sample[key] = sample[key] * sign
        return sample


class RandInvertIntensityd(MapTransform, RandomizableTransform):
    """Randomly invert the intensity of the image.

    Multiplies intensity values by -1 to invert the image contrast.
    Useful for augmentation in microscopy where structures may appear
    as bright or dark depending on imaging modality.

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys of the data dictionary to potentially invert.
    prob : float
        Probability of applying inversion. Default: 0.1.
    allow_missing_keys : bool
        Whether to allow missing keys in the data dictionary. Default: False.

    Returns
    -------
    Sample
        Dictionary with potentially inverted tensors for specified keys.
    """

    def __init__(
        self,
        keys: str | Iterable[str],
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys=allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

    def __call__(self, sample: Sample) -> Sample:
        """Randomly invert the sample intensities.

        Parameters
        ----------
        sample : Sample
            Dictionary containing tensors.

        Returns
        -------
        Sample
            Dictionary with potentially inverted tensors.
        """
        self.randomize(None)
        if not self._do_transform:
            return sample
        for key in self.keys:
            if key in sample:
                sample[key] = -sample[key]
        return sample
