"""Redefine transforms from MONAI for jsonargparse."""

from collections.abc import Sequence
from typing import Any

from monai.transforms import (
    CenterSpatialCropd,
    Decollated,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandSpatialCropd,
    RandWeightedCropd,
    ScaleIntensityRangePercentilesd,
    ToDeviced,
)
from numpy.typing import DTypeLike


class Decollated(Decollated):
    """Decollate data wrapper for jsonargparse compatibility.

    See parent class documentation for details.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        detach: bool = True,
        pad_batch: bool = True,
        fill_value: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            keys=keys,
            detach=detach,
            pad_batch=pad_batch,
            fill_value=fill_value,
            **kwargs,
        )


class ToDeviced(ToDeviced):
    """Transfer data to device wrapper for jsonargparse compatibility.

    See parent class documentation for details.
    """

    def __init__(self, keys: Sequence[str] | str, **kwargs: Any) -> None:
        super().__init__(keys=keys, **kwargs)


class RandWeightedCropd(RandWeightedCropd):
    """Random weighted crop wrapper for jsonargparse compatibility.

    See parent class documentation for details.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        w_key: str,
        spatial_size: Sequence[int],
        num_samples: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            keys=keys,
            w_key=w_key,
            spatial_size=spatial_size,
            num_samples=num_samples,
            **kwargs,
        )


class RandAffined(RandAffined):
    """Random affine transform wrapper for jsonargparse compatibility.

    See parent class documentation for details.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        prob: float,
        rotate_range: Sequence[float | Sequence[float]] | float,
        shear_range: Sequence[float | Sequence[float]] | float,
        scale_range: Sequence[float | Sequence[float]] | float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            keys=keys,
            prob=prob,
            rotate_range=rotate_range,
            shear_range=shear_range,
            scale_range=scale_range,
            **kwargs,
        )


class RandAdjustContrastd(RandAdjustContrastd):
    """Random contrast adjustment wrapper for jsonargparse compatibility.

    See parent class documentation for details.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        prob: float,
        gamma: tuple[float, float] | float,
        **kwargs: Any,
    ) -> None:
        super().__init__(keys=keys, prob=prob, gamma=gamma, **kwargs)


class RandScaleIntensityd(RandScaleIntensityd):
    """Random intensity scaling wrapper for jsonargparse compatibility.

    See parent class documentation for details.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        factors: tuple[float, float] | float,
        prob: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(keys=keys, factors=factors, prob=prob, **kwargs)


class RandGaussianNoised(RandGaussianNoised):
    """Random Gaussian noise wrapper for jsonargparse compatibility.

    See parent class documentation for details.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        prob: float,
        mean: float,
        std: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(keys=keys, prob=prob, mean=mean, std=std, **kwargs)


class RandGaussianSmoothd(RandGaussianSmoothd):
    """Random Gaussian smoothing wrapper for jsonargparse compatibility.

    See parent class documentation for details.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        prob: float,
        sigma_x: tuple[float, float] | float,
        sigma_y: tuple[float, float] | float,
        sigma_z: tuple[float, float] | float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            keys=keys,
            prob=prob,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_z=sigma_z,
            **kwargs,
        )


class ScaleIntensityRangePercentilesd(ScaleIntensityRangePercentilesd):
    """Scale intensity by percentile range wrapper for jsonargparse compatibility.

    See parent class documentation for details.
    """

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
    ) -> None:
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


class RandSpatialCropd(RandSpatialCropd):
    """Random spatial crop wrapper for jsonargparse compatibility.

    See parent class documentation for details.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        roi_size: Sequence[int] | int,
        random_center: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            keys=keys,
            roi_size=roi_size,
            random_center=random_center,
            **kwargs,
        )


class CenterSpatialCropd(CenterSpatialCropd):
    """Center spatial crop wrapper for jsonargparse compatibility.

    See parent class documentation for details.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        roi_size: Sequence[int] | int,
        **kwargs: Any,
    ) -> None:
        super().__init__(keys=keys, roi_size=roi_size, **kwargs)


class RandFlipd(RandFlipd):
    """Random flip wrapper for jsonargparse compatibility.

    See parent class documentation for details.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        prob: float,
        spatial_axis: Sequence[int] | int,
        **kwargs: Any,
    ) -> None:
        super().__init__(keys=keys, prob=prob, spatial_axis=spatial_axis, **kwargs)
