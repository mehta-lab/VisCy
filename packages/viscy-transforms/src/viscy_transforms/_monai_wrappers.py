"""MONAI transform wrappers with explicit signatures for jsonargparse.

MONAI transforms use **kwargs in constructors, which prevents jsonargparse
from introspecting parameters for config-driven pipelines (e.g., LightningCLI).
These wrappers re-declare constructors with explicit type hints.

All transforms in this module are dictionary-based ("d" suffix) and operate
on specified keys within a data dictionary.
"""

from typing import Sequence

from monai.transforms import (
    CenterSpatialCropd as _CenterSpatialCropd,
)
from monai.transforms import (
    Decollated as _Decollated,
)
from monai.transforms import (
    NormalizeIntensityd as _NormalizeIntensityd,
)
from monai.transforms import (
    RandAdjustContrastd as _RandAdjustContrastd,
)
from monai.transforms import (
    RandAffined as _RandAffined,
)
from monai.transforms import (
    RandFlipd as _RandFlipd,
)
from monai.transforms import (
    RandGaussianNoised as _RandGaussianNoised,
)
from monai.transforms import (
    RandGaussianSmoothd as _RandGaussianSmoothd,
)
from monai.transforms import (
    RandScaleIntensityd as _RandScaleIntensityd,
)
from monai.transforms import (
    RandSpatialCropd as _RandSpatialCropd,
)
from monai.transforms import (
    RandWeightedCropd as _RandWeightedCropd,
)
from monai.transforms import (
    ScaleIntensityRangePercentilesd as _ScaleIntensityRangePercentilesd,
)
from monai.transforms import (
    ToDeviced as _ToDeviced,
)
from numpy.typing import DTypeLike

__all__ = [
    "Decollated",
    "ToDeviced",
    "RandWeightedCropd",
    "RandAffined",
    "RandAdjustContrastd",
    "RandScaleIntensityd",
    "RandGaussianNoised",
    "RandGaussianSmoothd",
    "ScaleIntensityRangePercentilesd",
    "RandSpatialCropd",
    "CenterSpatialCropd",
    "RandFlipd",
    "NormalizeIntensityd",
]


class Decollated(_Decollated):
    """Decollate batch data back into a list of samples.

    Wrapper around MONAI's Decollated transform with explicit constructor
    signature for jsonargparse compatibility.

    Parameters
    ----------
    keys : Sequence[str] | str
        Keys of the data dictionary to decollate.
    detach : bool
        Whether to detach tensors from the computation graph. Default: True.
    pad_batch : bool
        Whether to pad smaller tensors to match the batch size. Default: True.
    fill_value : float | None
        Value used for padding when pad_batch is True. Default: None.
    **kwargs
        Additional arguments passed to the parent class.

    See Also
    --------
    monai.transforms.Decollated : Parent MONAI transform.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        detach: bool = True,
        pad_batch: bool = True,
        fill_value: float | None = None,
        **kwargs,
    ):
        super().__init__(
            keys=keys,
            detach=detach,
            pad_batch=pad_batch,
            fill_value=fill_value,
            **kwargs,
        )


class ToDeviced(_ToDeviced):
    """Move data to a specified device.

    Wrapper around MONAI's ToDeviced transform with explicit constructor
    signature for jsonargparse compatibility.

    Parameters
    ----------
    keys : Sequence[str] | str
        Keys of the data dictionary to move to the device.
    **kwargs
        Additional arguments passed to the parent class, including:
        - device : torch.device or str
            Target device (e.g., "cuda:0", "cpu").
        - non_blocking : bool
            Whether to use non-blocking transfer.

    See Also
    --------
    monai.transforms.ToDeviced : Parent MONAI transform.
    """

    def __init__(self, keys: Sequence[str] | str, **kwargs):
        super().__init__(keys=keys, **kwargs)


class RandWeightedCropd(_RandWeightedCropd):
    """Randomly crop regions weighted by a spatial importance map.

    Wrapper around MONAI's RandWeightedCropd transform with explicit
    constructor signature for jsonargparse compatibility. Crops are sampled
    with probability proportional to the weight map values.

    Parameters
    ----------
    keys : Sequence[str] | str
        Keys of the data dictionary to apply the crop to.
    w_key : str
        Key of the weight map in the data dictionary.
    spatial_size : Sequence[int]
        Size of the crop region as (D, H, W) for 3D data.
    num_samples : int
        Number of crop samples to generate per input. Default: 1.
    **kwargs
        Additional arguments passed to the parent class.

    See Also
    --------
    monai.transforms.RandWeightedCropd : Parent MONAI transform.
    """

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


class RandAffined(_RandAffined):
    """Apply random affine transformations to data.

    Wrapper around MONAI's RandAffined transform with explicit constructor
    signature for jsonargparse compatibility. Applies rotation, shearing,
    and scaling transformations.

    Parameters
    ----------
    keys : Sequence[str] | str
        Keys of the data dictionary to transform.
    prob : float
        Probability of applying the transform (0.0 to 1.0).
    rotate_range : Sequence[float | Sequence[float]] | float
        Range of rotation angles in radians for each axis.
        Can be a single value or (min, max) tuple per axis.
    shear_range : Sequence[float | Sequence[float]] | float
        Range of shear factors for each axis.
        Can be a single value or (min, max) tuple per axis.
    scale_range : Sequence[float | Sequence[float]] | float
        Range of scale factors for each axis.
        Can be a single value or (min, max) tuple per axis.
    **kwargs
        Additional arguments passed to the parent class, including:
        - mode : str
            Interpolation mode (e.g., "bilinear", "nearest").
        - padding_mode : str
            Padding mode for out-of-bounds values.

    See Also
    --------
    monai.transforms.RandAffined : Parent MONAI transform.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        prob: float,
        rotate_range: Sequence[float | Sequence[float]] | float,
        shear_range: Sequence[float | Sequence[float]] | float,
        scale_range: Sequence[float | Sequence[float]] | float,
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


class RandAdjustContrastd(_RandAdjustContrastd):
    """Randomly adjust image contrast using gamma correction.

    Wrapper around MONAI's RandAdjustContrastd transform with explicit
    constructor signature for jsonargparse compatibility. Applies the
    transformation: output = input^gamma.

    Parameters
    ----------
    keys : Sequence[str] | str
        Keys of the data dictionary to transform.
    prob : float
        Probability of applying the transform (0.0 to 1.0).
    gamma : tuple[float, float] | float
        Gamma value range. If tuple, samples uniformly between min and max.
        Values < 1 increase contrast, values > 1 decrease contrast.
    **kwargs
        Additional arguments passed to the parent class.

    See Also
    --------
    monai.transforms.RandAdjustContrastd : Parent MONAI transform.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        prob: float,
        gamma: tuple[float, float] | float,
        **kwargs,
    ):
        super().__init__(keys=keys, prob=prob, gamma=gamma, **kwargs)


class RandScaleIntensityd(_RandScaleIntensityd):
    """Randomly scale image intensity by a multiplicative factor.

    Wrapper around MONAI's RandScaleIntensityd transform with explicit
    constructor signature for jsonargparse compatibility. Multiplies
    intensity values by a randomly sampled factor.

    Parameters
    ----------
    keys : Sequence[str] | str
        Keys of the data dictionary to transform.
    factors : tuple[float, float] | float
        Scale factor range. If tuple, samples uniformly between min and max.
        Factor of 1.0 means no change, < 1 darkens, > 1 brightens.
    prob : float
        Probability of applying the transform (0.0 to 1.0).
    **kwargs
        Additional arguments passed to the parent class.

    See Also
    --------
    monai.transforms.RandScaleIntensityd : Parent MONAI transform.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        factors: tuple[float, float] | float,
        prob: float,
        **kwargs,
    ):
        super().__init__(keys=keys, factors=factors, prob=prob, **kwargs)


class RandGaussianNoised(_RandGaussianNoised):
    """Randomly add Gaussian noise to image data.

    Wrapper around MONAI's RandGaussianNoised transform with explicit
    constructor signature for jsonargparse compatibility. Adds noise
    sampled from a Gaussian distribution.

    Parameters
    ----------
    keys : Sequence[str] | str
        Keys of the data dictionary to transform.
    prob : float
        Probability of applying the transform (0.0 to 1.0).
    mean : float
        Mean of the Gaussian noise distribution.
    std : float
        Standard deviation of the Gaussian noise distribution.
    **kwargs
        Additional arguments passed to the parent class.

    See Also
    --------
    monai.transforms.RandGaussianNoised : Parent MONAI transform.
    BatchedRandGaussianNoised : Batch-optimized version for GPU efficiency.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        prob: float,
        mean: float,
        std: float,
        **kwargs,
    ):
        super().__init__(keys=keys, prob=prob, mean=mean, std=std, **kwargs)


class RandGaussianSmoothd(_RandGaussianSmoothd):
    """Randomly apply Gaussian smoothing (blur) to image data.

    Wrapper around MONAI's RandGaussianSmoothd transform with explicit
    constructor signature for jsonargparse compatibility. Applies Gaussian
    blur with independently sampled sigma values per axis.

    Parameters
    ----------
    keys : Sequence[str] | str
        Keys of the data dictionary to transform.
    prob : float
        Probability of applying the transform (0.0 to 1.0).
    sigma_x : tuple[float, float] | float
        Standard deviation range for x-axis blur.
        If tuple, samples uniformly between min and max.
    sigma_y : tuple[float, float] | float
        Standard deviation range for y-axis blur.
        If tuple, samples uniformly between min and max.
    sigma_z : tuple[float, float] | float
        Standard deviation range for z-axis blur.
        If tuple, samples uniformly between min and max.
    **kwargs
        Additional arguments passed to the parent class.

    See Also
    --------
    monai.transforms.RandGaussianSmoothd : Parent MONAI transform.
    BatchedRandGaussianSmoothd : Batch-optimized version for GPU efficiency.
    """

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


class ScaleIntensityRangePercentilesd(_ScaleIntensityRangePercentilesd):
    """Scale intensity values based on percentile range.

    Wrapper around MONAI's ScaleIntensityRangePercentilesd transform with
    explicit constructor signature for jsonargparse compatibility. Maps
    intensity values from a percentile-defined range to a target range.

    Parameters
    ----------
    keys : Sequence[str] | str
        Keys of the data dictionary to transform.
    lower : float
        Lower percentile for input range (0-100).
    upper : float
        Upper percentile for input range (0-100).
    b_min : float | None
        Minimum value of output range. None to skip scaling.
    b_max : float | None
        Maximum value of output range. None to skip scaling.
    clip : bool
        Whether to clip output values to [b_min, b_max]. Default: False.
    relative : bool
        Whether to compute relative percentile range. Default: False.
    channel_wise : bool
        Whether to compute percentiles per channel. Default: False.
    dtype : DTypeLike | None
        Output data type. Default: None (preserve input dtype).
    allow_missing_keys : bool
        Whether to allow missing keys in data dictionary. Default: False.

    See Also
    --------
    monai.transforms.ScaleIntensityRangePercentilesd : Parent MONAI transform.
    BatchedScaleIntensityRangePercentilesd : Batch-optimized version.
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


class RandSpatialCropd(_RandSpatialCropd):
    """Randomly crop a region of specified size from the input.

    Wrapper around MONAI's RandSpatialCropd transform with explicit
    constructor signature for jsonargparse compatibility.

    Parameters
    ----------
    keys : Sequence[str] | str
        Keys of the data dictionary to crop.
    roi_size : Sequence[int] | int
        Size of the crop region. If int, applies same size to all dimensions.
    random_center : bool
        Whether to randomly select the crop center. If False, crops from
        the center of the input. Default: True.
    **kwargs
        Additional arguments passed to the parent class.

    See Also
    --------
    monai.transforms.RandSpatialCropd : Parent MONAI transform.
    CenterSpatialCropd : Deterministic center cropping.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        roi_size: Sequence[int] | int,
        random_center: bool = True,
        **kwargs,
    ):
        super().__init__(
            keys=keys,
            roi_size=roi_size,
            random_center=random_center,
            **kwargs,
        )


class CenterSpatialCropd(_CenterSpatialCropd):
    """Crop a region of specified size from the center of the input.

    Wrapper around MONAI's CenterSpatialCropd transform with explicit
    constructor signature for jsonargparse compatibility.

    Parameters
    ----------
    keys : Sequence[str] | str
        Keys of the data dictionary to crop.
    roi_size : Sequence[int] | int
        Size of the crop region. If int, applies same size to all dimensions.
    **kwargs
        Additional arguments passed to the parent class.

    See Also
    --------
    monai.transforms.CenterSpatialCropd : Parent MONAI transform.
    RandSpatialCropd : Random position cropping.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        roi_size: Sequence[int] | int,
        **kwargs,
    ):
        super().__init__(keys=keys, roi_size=roi_size, **kwargs)


class RandFlipd(_RandFlipd):
    """Randomly flip the input along specified spatial axes.

    Wrapper around MONAI's RandFlipd transform with explicit constructor
    signature for jsonargparse compatibility.

    Parameters
    ----------
    keys : Sequence[str] | str
        Keys of the data dictionary to flip.
    prob : float
        Probability of applying the flip (0.0 to 1.0).
    spatial_axis : Sequence[int] | int
        Spatial axis or axes along which to flip. For 3D data:
        0 = depth (Z), 1 = height (Y), 2 = width (X).
    **kwargs
        Additional arguments passed to the parent class.

    See Also
    --------
    monai.transforms.RandFlipd : Parent MONAI transform.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        prob: float,
        spatial_axis: Sequence[int] | int,
        **kwargs,
    ):
        super().__init__(keys=keys, prob=prob, spatial_axis=spatial_axis, **kwargs)


class NormalizeIntensityd(_NormalizeIntensityd):
    """Normalize intensity values using mean and standard deviation.

    Wrapper around MONAI's NormalizeIntensityd transform with explicit
    constructor signature for jsonargparse compatibility. Computes
    z-score normalization: (x - mean) / std.

    Parameters
    ----------
    keys : Sequence[str] | str
        Keys of the data dictionary to normalize.
    **kwargs
        Additional arguments passed to the parent class, including:
        - subtrahend : float | None
            Value to subtract (overrides computed mean).
        - divisor : float | None
            Value to divide by (overrides computed std).
        - nonzero : bool
            Whether to compute statistics only on non-zero values.
        - channel_wise : bool
            Whether to normalize each channel independently.

    See Also
    --------
    monai.transforms.NormalizeIntensityd : Parent MONAI transform.
    NormalizeSampled : Custom normalization using precomputed statistics.
    """

    def __init__(self, keys: Sequence[str] | str, **kwargs):
        super().__init__(keys=keys, **kwargs)
