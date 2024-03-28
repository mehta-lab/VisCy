"""Redefine transforms from MONAI for jsonargparse."""

from typing import Sequence, Union

from monai.transforms import (
    MapTransform,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandomizableTransform,
    RandScaleIntensityd,
    RandWeightedCropd,
)
from monai.transforms.transform import Randomizable
from numpy.random.mtrand import RandomState as RandomState
from typing_extensions import Iterable, Literal

from viscy.data.typing import Sample


class RandWeightedCropd(RandWeightedCropd):
    def __init__(
        self,
        keys: Union[Sequence[str], str],
        w_key: str,
        spatial_size: Sequence[int],
        num_samples: int = 1,
        **kwargs,
    ):
        super().__init__(
            keys=keys,
            w_key=w_key,
            spatial_size=spatial_size,
            num_samples=num_samples,
            **kwargs,
        )


class RandAffined(RandAffined):
    def __init__(
        self,
        keys: Union[Sequence[str], str],
        prob: float,
        rotate_range: Union[Sequence[float], float],
        shear_range: Union[Sequence[float], float],
        scale_range: Union[Sequence[float], float],
        **kwargs,
    ):
        super().__init__(
            keys=keys,
            prob=prob,
            rotate_range=rotate_range,
            shear_range=shear_range,
            scale_range=scale_range,
            **kwargs,
        )


class RandAdjustContrastd(RandAdjustContrastd):
    def __init__(
        self,
        keys: Union[Sequence[str], str],
        prob: float,
        gamma: Union[Sequence[float], float],
        **kwargs,
    ):
        super().__init__(
            keys=keys,
            prob=prob,
            gamma=gamma,
            **kwargs,
        )


class RandScaleIntensityd(RandScaleIntensityd):
    def __init__(
        self,
        keys: Union[Sequence[str], str],
        factors: Union[Sequence[float], float],
        prob: float,
        **kwargs,
    ):
        super().__init__(
            keys=keys,
            factors=factors,
            prob=prob,
            **kwargs,
        )


class RandGaussianNoised(RandGaussianNoised):
    def __init__(
        self,
        keys: Union[Sequence[str], str],
        prob: float,
        mean: Union[Sequence[float], float],
        std: Union[Sequence[float], float],
        **kwargs,
    ):
        super().__init__(
            keys=keys,
            prob=prob,
            mean=mean,
            std=std,
            **kwargs,
        )


class RandGaussianSmoothd(RandGaussianSmoothd):
    def __init__(
        self,
        keys: Union[Sequence[str], str],
        prob: float,
        sigma_x: Union[Sequence[float], float],
        sigma_y: Union[Sequence[float], float],
        sigma_z: Union[Sequence[float], float],
        **kwargs,
    ):
        super().__init__(
            keys=keys,
            prob=prob,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_z=sigma_z,
            **kwargs,
        )


class NormalizeSampled(MapTransform):
    """
    Normalize the sample
    :param Union[str, Iterable[str]] keys: keys to normalize
    :param str fov: fov path with respect to Plate
    :param str subtrahend: subtrahend for normalization, defaults to "mean"
    :param str divisor: divisor for normalization, defaults to "std"
    """

    def __init__(
        self,
        keys: Union[str, Iterable[str]],
        level: Literal["fov_statistics", "dataset_statistics"],
        subtrahend="mean",
        divisor="std",
    ) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.level = level

    # TODO: need to implement the case where the preprocessing already exists
    def __call__(self, sample: Sample) -> Sample:
        for key in self.keys:
            level_meta = sample["norm_meta"][key][self.level]
            subtrahend_val = level_meta[self.subtrahend]
            divisor_val = level_meta[self.divisor] + 1e-8  # avoid div by zero
            sample[key] = (sample[key] - subtrahend_val) / divisor_val
        return sample

    def _normalize():
        NotImplementedError("_normalization() not implemented")


class RandInvertIntensityd(MapTransform, RandomizableTransform):
    """
    Randomly invert the intensity of the image.
    """

    def __init__(
        self,
        keys: Union[str, Iterable[str]],
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys=allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

    def __call__(self, sample: Sample) -> Sample:
        self.randomize(None)
        for key in self.keys:
            if key in sample:
                sample[key] = -sample[key]
        return sample

    def set_random_state(
        self, seed: int | None = None, state: RandomState | None = None
    ) -> Randomizable:
        super().set_random_state(seed, state)
        return self
