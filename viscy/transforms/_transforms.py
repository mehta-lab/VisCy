import numpy as np
import torch
from monai.transforms import (
    MapTransform,
    MultiSampleTrait,
    RandomizableTransform,
    Transform,
)
from torch import Tensor
from typing_extensions import Iterable, Literal

from viscy.data.typing import ChannelMap, Sample


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

    # TODO: need to implement the case where the preprocessing already exists
    def __call__(self, sample: Sample) -> Sample:
        for key in self.keys:
            level_meta = sample["norm_meta"][key][self.level]
            subtrahend_val = level_meta[self.subtrahend]
            divisor_val = level_meta[self.divisor] + 1e-8  # avoid div by zero
            sample[key] = (sample[key] - subtrahend_val) / divisor_val
        if self.remove_meta:
            sample.pop("norm_meta")
        return sample

    def _normalize():
        NotImplementedError("_normalization() not implemented")


class RandInvertIntensityd(MapTransform, RandomizableTransform):
    """
    Randomly invert the intensity of the image.
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
        self.randomize(None)
        if not self._do_transform:
            return sample
        for key in self.keys:
            if key in sample:
                sample[key] = -sample[key]
        return sample


class TiledSpatialCropSamplesd(MapTransform, MultiSampleTrait):
    """
    Crop multiple tiled ROIs from an image.
    Used for deterministic cropping in validation.
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
            raise ValueError(
                f"Number of samples {self.num_samples} should be "
                f"smaller than {max_num_samples}."
            )
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

    def __call__(self, sample: Sample) -> Sample:
        results = []
        for i in range(self.num_samples):
            result = {}
            for key in self.keys:
                result[key] = self._crop(sample[key], i)
            if "norm_meta" in sample:
                result["norm_meta"] = sample["norm_meta"]
            results.append(result)
        return results


class StackChannelsd(MapTransform):
    """Stack source and target channels."""

    def __init__(self, channel_map: ChannelMap) -> None:
        channel_names = []
        for channels in channel_map.values():
            channel_names.extend(channels)
        super().__init__(channel_names, allow_missing_keys=False)
        self.channel_map = channel_map

    def __call__(self, sample: Sample) -> Sample:
        results = {}
        for key, channels in self.channel_map.items():
            results[key] = torch.cat([sample[ch] for ch in channels], dim=0)
        return results


class BatchedZoom(Transform):
    "Batched zoom transform using ``torch.nn.functional.interpolate``."

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
        return torch.nn.functional.interpolate(
            sample,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.antialias,
        )
