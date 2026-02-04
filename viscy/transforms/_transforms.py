from warnings import warn

import numpy as np
import torch
from kornia.augmentation import RandomAffine3D
from monai.transforms import (
    MapTransform,
    MultiSampleTrait,
    RandomizableTransform,
    ScaleIntensityRangePercentiles,
)
from numpy.typing import DTypeLike
from torch import Tensor
from typing_extensions import Iterable, Literal, Sequence

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

    @staticmethod
    def _match_image(tensor: Tensor, target: Tensor) -> Tensor:
        return tensor.reshape(tensor.shape + (1,) * (target.ndim - tensor.ndim)).to(
            device=target.device
        )

    # TODO: need to implement the case where the preprocessing already exists
    def __call__(self, sample: Sample) -> Sample:
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


class BatchedScaleIntensityRangePercentiles(ScaleIntensityRangePercentiles):
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
                raise ValueError(
                    "If it is relative, b_min and b_max should not be None."
                )
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
        if self.channel_wise:
            channels = [self._normalize(img[:, c : c + 1]) for c in range(img.shape[1])]
            return torch.cat(channels, dim=1)
        else:
            return self._normalize(img=img)


class BatchedScaleIntensityRangePercentilesd(MapTransform):
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
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d


class BatchedRandAffined(MapTransform):
    def __init__(
        self,
        keys: str | Iterable[str],
        prob: float = 0.1,
        rotate_range: Sequence[tuple[float, float] | float] | float | None = None,
        shear_range: Sequence[tuple[float, float] | float] | float | None = None,
        translate_range: Sequence[tuple[float, float] | float] | float | None = None,
        scale_range: Sequence[tuple[float, float] | float] | float | None = None,
        mode: str = "bilinear",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        rotate_range = self._radians_to_degrees(
            self._maybe_invert_sequence(rotate_range)
        )
        if rotate_range is None:
            rotate_range = (0.0, 0.0, 0.0)
        shear_range = self._radians_to_degrees(self._maybe_invert_sequence(shear_range))
        translate_range = self._maybe_invert_sequence(translate_range)
        scale_range = self._maybe_invert_sequence(scale_range)
        self.random_affine = RandomAffine3D(
            degrees=rotate_range,
            translate=translate_range,
            scale=scale_range,
            shears=shear_range,
            resample=mode,
            p=prob,
        )
        # disable unnecessary transfer to CPU
        self.random_affine.disable_features = True

    @staticmethod
    def _maybe_invert_sequence(
        value: Sequence[tuple[float, float] | float] | float | None,
    ) -> Sequence[tuple[float, float] | float] | float | None:
        """Translate MONAI's ZYX order to Kornia's XYZ order."""
        if isinstance(value, Sequence):
            return tuple(reversed(value))
        return value

    @staticmethod
    def _radians_to_degrees(
        rotate_range: Sequence[tuple[float, float] | float] | float | None,
    ) -> Sequence[tuple[float, float] | float] | float | None:
        if rotate_range is None:
            return None
        return torch.from_numpy(np.rad2deg(rotate_range))

    @torch.no_grad()
    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        d = dict(sample)
        for key in self.key_iterator(d):
            data = d[key]
            try:
                d[key] = self.random_affine(data)
            except RuntimeError:
                # retry
                d[key] = self.random_affine(data)
            assert d[key].device == data.device
        return d
