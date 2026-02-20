"""VisCy Data - Data loading and Lightning DataModules for AI x Imaging tasks.

This package provides PyTorch Lightning DataModules and Datasets for loading
and preprocessing microscopy data in virtual staining workflows.

Public API:
    All DataModules, Datasets, and type definitions are exported at the package level.
    Example: ``from viscy_data import HCSDataModule, Sample, NormMeta``

Optional Extras:
    Some modules require optional dependencies:
    - ``pip install 'viscy-data[triplet]'`` for TripletDataModule (tensorstore, pandas)
    - ``pip install 'viscy-data[livecell]'`` for LiveCellDataModule (pycocotools, tifffile, torchvision)
    - ``pip install 'viscy-data[mmap]'`` for MmappedDataModule (tensordict)
    - ``pip install 'viscy-data[all]'`` for all optional dependencies

Version:
    Use ``importlib.metadata.version('viscy-data')`` to get version.
"""

# Type definitions (from _typing.py)
# Utility modules (from _select.py, distributed.py)
from viscy_data._select import SelectWell
from viscy_data._typing import (
    ULTRACK_INDEX_COLUMNS,
    AnnotationColumns,
    ChannelMap,
    ChannelNormStats,
    DictTransform,
    HCSStackIndex,
    LevelNormStats,
    NormMeta,
    OneOrSeq,
    Sample,
    SegmentationSample,
    TrackingIndex,
    TripletSample,
)

# Cell classification (from cell_classification.py -- requires pandas at runtime)
from viscy_data.cell_classification import (
    ClassificationDataModule,
    ClassificationDataset,
)

# Cell division triplet (from cell_division_triplet.py)
from viscy_data.cell_division_triplet import (
    CellDivisionTripletDataModule,
    CellDivisionTripletDataset,
)

# Combined/Concat DataModules (from combined.py)
from viscy_data.combined import (
    BatchedConcatDataModule,
    BatchedConcatDataset,
    CachedConcatDataModule,
    CombinedDataModule,
    CombineMode,
    ConcatDataModule,
)

# CTMC v1 (from ctmc_v1.py)
from viscy_data.ctmc_v1 import CTMCv1DataModule
from viscy_data.distributed import ShardedDistributedSampler

# GPU augmentation DataModules (from gpu_aug.py)
from viscy_data.gpu_aug import (
    CachedOmeZarrDataModule,
    CachedOmeZarrDataset,
    GPUTransformDataModule,
)

# Core DataModules (from hcs.py)
from viscy_data.hcs import HCSDataModule, MaskTestDataset, SlidingWindowDataset

# LiveCell benchmark (from livecell.py -- requires [livecell] extra at runtime)
from viscy_data.livecell import LiveCellDataModule, LiveCellDataset, LiveCellTestDataset

# Memory-mapped cache (from mmap_cache.py -- requires [mmap] extra at runtime)
from viscy_data.mmap_cache import MmappedDataModule, MmappedDataset

# Segmentation (from segmentation.py)
from viscy_data.segmentation import SegmentationDataModule, SegmentationDataset

# Triplet learning (from triplet.py -- requires [triplet] extra at runtime)
from viscy_data.triplet import TripletDataModule, TripletDataset

__all__ = [
    # Types
    "AnnotationColumns",
    "ChannelMap",
    "ChannelNormStats",
    "DictTransform",
    "HCSStackIndex",
    "ULTRACK_INDEX_COLUMNS",
    "LevelNormStats",
    "NormMeta",
    "OneOrSeq",
    "Sample",
    "SegmentationSample",
    "TrackingIndex",
    "TripletSample",
    # Utilities
    "SelectWell",
    "ShardedDistributedSampler",
    # Core
    "HCSDataModule",
    "MaskTestDataset",
    "SlidingWindowDataset",
    # GPU augmentation
    "CachedOmeZarrDataModule",
    "CachedOmeZarrDataset",
    "GPUTransformDataModule",
    # Triplet
    "TripletDataModule",
    "TripletDataset",
    # Cell classification
    "ClassificationDataModule",
    "ClassificationDataset",
    # Cell division
    "CellDivisionTripletDataModule",
    "CellDivisionTripletDataset",
    # Memory-mapped cache
    "MmappedDataModule",
    "MmappedDataset",
    # LiveCell
    "LiveCellDataModule",
    "LiveCellDataset",
    "LiveCellTestDataset",
    # CTMC
    "CTMCv1DataModule",
    # Segmentation
    "SegmentationDataModule",
    "SegmentationDataset",
    # Combined
    "BatchedConcatDataModule",
    "BatchedConcatDataset",
    "CachedConcatDataModule",
    "CombinedDataModule",
    "CombineMode",
    "ConcatDataModule",
]
