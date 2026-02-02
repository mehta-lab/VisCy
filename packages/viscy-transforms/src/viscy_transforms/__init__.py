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
from viscy_transforms._crop import (
    BatchedCenterSpatialCrop,
    BatchedCenterSpatialCropd,
    BatchedRandSpatialCrop,
    BatchedRandSpatialCropd,
)
from viscy_transforms._decollate import Decollate
from viscy_transforms._flip import BatchedRandFlip, BatchedRandFlipd
from viscy_transforms._gaussian_smooth import (
    BatchedRandGaussianSmooth,
    BatchedRandGaussianSmoothd,
)
from viscy_transforms._noise import (
    BatchedRandGaussianNoise,
    BatchedRandGaussianNoised,
    RandGaussianNoiseTensor,
    RandGaussianNoiseTensord,
)
from viscy_transforms._redef import (
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
from viscy_transforms._scale_intensity import (
    BatchedRandScaleIntensity,
    BatchedRandScaleIntensityd,
)
from viscy_transforms._transforms import (
    BatchedRandAffined,
    BatchedScaleIntensityRangePercentiles,
    BatchedScaleIntensityRangePercentilesd,
    NormalizeSampled,
    RandInvertIntensityd,
    StackChannelsd,
    TiledSpatialCropSamplesd,
)
from viscy_transforms._zoom import BatchedZoom, BatchedZoomd
from viscy_transforms.batched_rand_3d_elasticd import BatchedRand3DElasticd
from viscy_transforms.batched_rand_histogram_shiftd import BatchedRandHistogramShiftd
from viscy_transforms.batched_rand_local_pixel_shufflingd import (
    BatchedRandLocalPixelShufflingd,
)
from viscy_transforms.batched_rand_sharpend import BatchedRandSharpend
from viscy_transforms.batched_rand_zstack_shiftd import BatchedRandZStackShiftd

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
