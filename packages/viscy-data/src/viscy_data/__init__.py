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

import importlib
from typing import Any

# Lightweight, always-needed types and utilities -- keep eager
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

# Mapping of lazily-loaded names to their submodules.
# Submodules are only imported on first access.
_LAZY_IMPORTS: dict[str, str] = {
    # Cell classification
    "ClassificationDataModule": "viscy_data.cell_classification",
    "ClassificationDataset": "viscy_data.cell_classification",
    # Cell division triplet
    "CellDivisionTripletDataModule": "viscy_data.cell_division_triplet",
    "CellDivisionTripletDataset": "viscy_data.cell_division_triplet",
    # Combined/Concat
    "BatchedConcatDataModule": "viscy_data.combined",
    "BatchedConcatDataset": "viscy_data.combined",
    "CachedConcatDataModule": "viscy_data.combined",
    "CombinedDataModule": "viscy_data.combined",
    "CombineMode": "viscy_data.combined",
    "ConcatDataModule": "viscy_data.combined",
    # CTMC v1
    "CTMCv1DataModule": "viscy_data.ctmc_v1",
    # Distributed
    "ShardedDistributedSampler": "viscy_data.distributed",
    # GPU augmentation
    "CachedOmeZarrDataModule": "viscy_data.gpu_aug",
    "CachedOmeZarrDataset": "viscy_data.gpu_aug",
    "GPUTransformDataModule": "viscy_data.gpu_aug",
    # Core HCS
    "HCSDataModule": "viscy_data.hcs",
    "MaskTestDataset": "viscy_data.hcs",
    "SlidingWindowDataset": "viscy_data.hcs",
    # LiveCell
    "LiveCellDataModule": "viscy_data.livecell",
    "LiveCellDataset": "viscy_data.livecell",
    "LiveCellTestDataset": "viscy_data.livecell",
    # Memory-mapped cache
    "MmappedDataModule": "viscy_data.mmap_cache",
    "MmappedDataset": "viscy_data.mmap_cache",
    # Segmentation
    "SegmentationDataModule": "viscy_data.segmentation",
    "SegmentationDataset": "viscy_data.segmentation",
    # Triplet
    "TripletDataModule": "viscy_data.triplet",
    "TripletDataset": "viscy_data.triplet",
}

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


def __getattr__(name: str) -> Any:
    """Lazily import a public name from its submodule on first access."""
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List public API names and standard module attributes."""
    return list(__all__) + [k for k in globals() if k.startswith("__")]
