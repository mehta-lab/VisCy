"""VisCy Data - Data loading and Lightning DataModules for virtual staining microscopy.

This package provides PyTorch Lightning DataModules and Datasets for loading
and preprocessing microscopy data in virtual staining workflows.

Public API:
    Type definitions are exported at the package level.
    Example: ``from viscy_data import Sample, NormMeta``

Version:
    Use ``importlib.metadata.version('viscy-data')`` to get version.
"""

from viscy_data._typing import (
    INDEX_COLUMNS,
    LABEL_CELL_CYCLE_STATE,
    LABEL_CELL_DIVISION_STATE,
    LABEL_CELL_REMODELING_STATE,
    LABEL_INFECTION_STATE,
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

__all__ = [
    "AnnotationColumns",
    "ChannelMap",
    "ChannelNormStats",
    "DictTransform",
    "HCSStackIndex",
    "INDEX_COLUMNS",
    "LABEL_CELL_CYCLE_STATE",
    "LABEL_CELL_DIVISION_STATE",
    "LABEL_CELL_REMODELING_STATE",
    "LABEL_INFECTION_STATE",
    "LevelNormStats",
    "NormMeta",
    "OneOrSeq",
    "Sample",
    "SegmentationSample",
    "TrackingIndex",
    "TripletSample",
]
