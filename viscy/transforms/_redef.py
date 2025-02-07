"""Redefine transforms from MONAI for jsonargparse."""

from typing import Sequence

from monai.transforms import (
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandWeightedCropd,
    ScaleIntensityRangePercentilesd,
)
from numpy.typing import DTypeLike


class RandWeightedCropd(RandWeightedCropd):
    def __init__(
        self,
        keys: Sequence[str] | str,
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
        keys: Sequence[str] | str,
        prob: float,
        rotate_range: Sequence[float] | float,
        shear_range: Sequence[float] | float,
        scale_range: Sequence[float] | float,
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
        keys: Sequence[str] | str,
        prob: float,
        gamma: tuple[float, float] | float,
        **kwargs,
    ):
        super().__init__(keys=keys, prob=prob, gamma=gamma, **kwargs)


class RandScaleIntensityd(RandScaleIntensityd):
    def __init__(
        self,
        keys: Sequence[str] | str,
        factors: tuple[float, float] | float,
        prob: float,
        **kwargs,
    ):
        super().__init__(keys=keys, factors=factors, prob=prob, **kwargs)


class RandGaussianNoised(RandGaussianNoised):
    def __init__(
        self,
        keys: Sequence[str] | str,
        prob: float,
        mean: float,
        std: float,
        **kwargs,
    ):
        super().__init__(keys=keys, prob=prob, mean=mean, std=std, **kwargs)


class RandGaussianSmoothd(RandGaussianSmoothd):
    def __init__(
        self,
        keys: Sequence[str] | str,
        prob: float,
        sigma_x: tuple[float, float] | float,
        sigma_y: tuple[float, float] | float,
        sigma_z: tuple[float, float] | float,
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


class ScaleIntensityRangePercentilesd(ScaleIntensityRangePercentilesd):
    def __init__(
        self,
        keys: Sequence[str] | str,
        lower: float,
        upper: float,
        b_min: float | None,
        b_max: float | None,
        clip: bool = False,
        relative: bool = False,
        channel_wise: bool = False,
        dtype: DTypeLike | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(
            keys=keys,
            lower=lower,
            upper=upper,
            b_min=b_min,
            b_max=b_max,
            clip=clip,
            relative=relative,
            channel_wise=channel_wise,
            dtype=dtype,
            allow_missing_keys=allow_missing_keys,
        )
