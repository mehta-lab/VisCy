"""VisCy Transforms - Image transforms for virtual staining microscopy.

This package provides PyTorch-based image transforms for preprocessing
microscopy data in virtual staining workflows. Transforms follow the
MONAI Dictionary transform pattern for use with DataLoader pipelines.

Public API:
    All transforms are exported at the package level.
    Example: `from viscy_transforms import NormalizeSampled`

Version:
    Use `importlib.metadata.version('viscy-transforms')` to get version.
"""

from importlib.metadata import version

from viscy_transforms._adjust_contrast import (
    BatchedRandAdjustContrast,
    BatchedRandAdjustContrastd,
)
from viscy_transforms._affine import BatchedRandAffined
from viscy_transforms._crop import (
    BatchedCenterSpatialCrop,
    BatchedCenterSpatialCropd,
    BatchedRandSpatialCrop,
    BatchedRandSpatialCropd,
)
from viscy_transforms._decollate import Decollate
from viscy_transforms._elastic import BatchedRand3DElasticd
from viscy_transforms._flip import BatchedRandFlip, BatchedRandFlipd
from viscy_transforms._gaussian_smooth import (
    BatchedRandGaussianSmooth,
    BatchedRandGaussianSmoothd,
)
from viscy_transforms._histogram_shift import BatchedRandHistogramShiftd
from viscy_transforms._invert_intensity import RandInvertIntensityd
from viscy_transforms._monai_wrappers import (
    CenterSpatialCropd,
    Decollated,
    NormalizeIntensityd,
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
from viscy_transforms._noise import (
    BatchedRandGaussianNoise,
    BatchedRandGaussianNoised,
    RandGaussianNoiseTensor,
    RandGaussianNoiseTensord,
)
from viscy_transforms._normalize import NormalizeSampled
from viscy_transforms._percentile_scale import (
    BatchedScaleIntensityRangePercentiles,
    BatchedScaleIntensityRangePercentilesd,
)
from viscy_transforms._pixel_shuffle import BatchedRandLocalPixelShufflingd
from viscy_transforms._scale_intensity import (
    BatchedRandScaleIntensity,
    BatchedRandScaleIntensityd,
)
from viscy_transforms._sharpen import BatchedRandSharpend
from viscy_transforms._stack_channels import StackChannelsd
from viscy_transforms._tiled_crop import TiledSpatialCropSamplesd
from viscy_transforms._zoom import BatchedZoom, BatchedZoomd
from viscy_transforms._zstack_shift import BatchedRandZStackShiftd

__version__ = version("viscy-transforms")

__all__ = [
    "BatchedCenterSpatialCrop",
    "BatchedCenterSpatialCropd",
    "BatchedRandAdjustContrast",
    "BatchedRandAdjustContrastd",
    "BatchedRandAffined",
    "BatchedRand3DElasticd",
    "BatchedRandFlip",
    "BatchedRandFlipd",
    "BatchedRandGaussianSmooth",
    "BatchedRandGaussianSmoothd",
    "BatchedRandGaussianNoise",
    "BatchedRandGaussianNoised",
    "BatchedRandHistogramShiftd",
    "BatchedRandLocalPixelShufflingd",
    "BatchedRandScaleIntensity",
    "BatchedRandScaleIntensityd",
    "BatchedRandSharpend",
    "BatchedRandSpatialCrop",
    "BatchedRandSpatialCropd",
    "BatchedRandZStackShiftd",
    "BatchedScaleIntensityRangePercentiles",
    "BatchedScaleIntensityRangePercentilesd",
    "BatchedZoom",
    "BatchedZoomd",
    "CenterSpatialCropd",
    "Decollate",
    "Decollated",
    "NormalizeSampled",
    "NormalizeIntensityd",
    "RandAdjustContrastd",
    "RandAffined",
    "RandFlipd",
    "RandGaussianNoised",
    "RandGaussianNoiseTensor",
    "RandGaussianNoiseTensord",
    "RandGaussianSmoothd",
    "RandInvertIntensityd",
    "RandScaleIntensityd",
    "RandSpatialCropd",
    "RandWeightedCropd",
    "ScaleIntensityRangePercentilesd",
    "StackChannelsd",
    "TiledSpatialCropSamplesd",
    "ToDeviced",
]
