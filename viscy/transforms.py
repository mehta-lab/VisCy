"""Redefine transforms from MONAI for jsonargparse."""

from typing import Sequence, Union

from monai.transforms import (
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandWeightedCropd,
)


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
