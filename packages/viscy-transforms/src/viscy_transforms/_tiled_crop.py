"""Tiled cropping transforms for microscopy data.

This module provides transforms for generating multiple tiled crops
from images, useful for deterministic validation cropping.
"""

import numpy as np
from monai.transforms import MapTransform, MultiSampleTrait
from torch import Tensor
from typing_extensions import Iterable

from viscy_transforms._typing import Sample

__all__ = ["TiledSpatialCropSamplesd"]


class TiledSpatialCropSamplesd(MapTransform, MultiSampleTrait):
    """Crop multiple tiled ROIs from an image.

    Generates multiple non-overlapping crops arranged in a grid pattern.
    Used for deterministic cropping in validation to ensure reproducible
    evaluation across the full field of view.

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys of the data dictionary to crop.
    roi_size : tuple[int, int, int]
        Size of each crop region as (D, H, W).
    num_samples : int
        Number of crops to generate. Must not exceed the maximum number
        of non-overlapping crops that fit in the image.

    Returns
    -------
    list[Sample]
        List of num_samples dictionaries, each containing cropped regions.

    Raises
    ------
    ValueError
        If num_samples exceeds the number of possible non-overlapping crops.
    """

    def __init__(
        self,
        keys: str | Iterable[str],
        roi_size: tuple[int, int, int],
        num_samples: int,
    ) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.roi_size = roi_size
        self.num_samples = num_samples

    def _check_num_samples(self, spatial_size: np.ndarray, offset: int) -> np.ndarray:
        max_grid_shape = spatial_size // self.roi_size
        max_num_samples = max_grid_shape.prod()
        if offset >= max_num_samples:
            raise ValueError(f"Number of samples {self.num_samples} should be smaller than {max_num_samples}.")
        grid_idx = np.asarray(np.unravel_index(offset, max_grid_shape))
        return grid_idx * self.roi_size

    def _crop(self, img: Tensor, offset: int) -> Tensor:
        spatial_size = np.array(img.shape[-3:])
        crop_start = self._check_num_samples(spatial_size, offset)
        crop_end = crop_start + np.array(self.roi_size)
        return img[
            ...,
            crop_start[0] : crop_end[0],
            crop_start[1] : crop_end[1],
            crop_start[2] : crop_end[2],
        ]

    def __call__(self, sample: Sample) -> list[Sample]:
        """Generate tiled crops from the sample.

        Parameters
        ----------
        sample : Sample
            Dictionary containing tensors with shape (C, D, H, W).

        Returns
        -------
        list[Sample]
            List of num_samples dictionaries with cropped regions.
        """
        results = []
        for i in range(self.num_samples):
            result = {}
            for key in self.keys:
                result[key] = self._crop(sample[key], i)
            if "norm_meta" in sample:
                result["norm_meta"] = sample["norm_meta"]
            results.append(result)
        return results
