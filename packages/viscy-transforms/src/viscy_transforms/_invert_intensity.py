"""Intensity inversion transforms for microscopy data.

This module provides transforms for randomly inverting image intensity
values, useful for augmentation in microscopy imaging.
"""

from monai.transforms import MapTransform, RandomizableTransform
from typing_extensions import Iterable

from viscy_transforms._typing import Sample

__all__ = ["RandInvertIntensityd"]


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
