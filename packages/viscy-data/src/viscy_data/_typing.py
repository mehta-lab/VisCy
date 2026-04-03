"""Type definitions for viscy-data.

Copied verbatim from ``viscy/data/typing.py`` with the following additions:
- ``ULTRACK_INDEX_COLUMNS`` extracted from ``viscy/data/triplet.py``
- ``__all__`` for explicit public API
- Updated ``typing_extensions.NotRequired`` to ``typing.NotRequired`` (Python >=3.11)
"""

from typing import (
    Callable,
    Literal,
    NamedTuple,
    NotRequired,
    Sequence,
    TypedDict,
    TypeVar,
)

from torch import ShortTensor, Tensor

__all__ = [
    "AnnotationColumns",
    "CELL_INDEX_BIOLOGY_COLUMNS",
    "CELL_INDEX_CORE_COLUMNS",
    "CELL_INDEX_GROUPING_COLUMNS",
    "CELL_INDEX_IMAGING_COLUMNS",
    "CELL_INDEX_NORMALIZATION_COLUMNS",
    "CELL_INDEX_OPS_COLUMNS",
    "CELL_INDEX_TIMELAPSE_COLUMNS",
    "CellIndex",
    "ChannelMap",
    "ChannelNormStats",
    "DictTransform",
    "HCSStackIndex",
    "ULTRACK_INDEX_COLUMNS",
    "LABEL_CELL_CYCLE_STATE",
    "LABEL_CELL_DIVISION_STATE",
    "LABEL_CELL_REMODELING_STATE",
    "LABEL_INFECTION_STATE",
    "LevelNormStats",
    "NormMeta",
    "OneOrSeq",
    "Sample",
    "SampleMeta",
    "SegmentationSample",
    "TripletSample",
]

DictTransform = Callable[[dict[str, Tensor | dict]], dict[str, Tensor]]


T = TypeVar("T")
OneOrSeq = T | Sequence[T]


class LevelNormStats(TypedDict):
    """Per-level normalization statistics."""

    mean: Tensor
    std: Tensor
    median: Tensor
    iqr: Tensor


class ChannelNormStats(TypedDict, total=False):
    """Per-channel normalization statistics."""

    dataset_statistics: LevelNormStats
    fov_statistics: LevelNormStats
    timepoint_statistics: dict[str, LevelNormStats]


NormMeta = dict[str, ChannelNormStats]


class HCSStackIndex(NamedTuple):
    """HCS stack index."""

    # name of the image array, e.g. "A/1/0/0"
    image: str
    time: int
    z: int


class Sample(TypedDict, total=False):
    """Image sample type for mini-batches.

    All fields are optional.
    """

    index: HCSStackIndex
    # Image data
    source: OneOrSeq[Tensor]
    target: OneOrSeq[Tensor]
    weight: OneOrSeq[Tensor]
    # Instance segmentation masks
    labels: OneOrSeq[Tensor]
    # None: not available
    norm_meta: NormMeta | None


class SegmentationSample(TypedDict):
    """Segmentation sample type for mini-batches."""

    pred: ShortTensor
    target: ShortTensor
    position_idx: OneOrSeq[int]
    time_idx: OneOrSeq[int]


class ChannelMap(TypedDict):
    """Source channel names."""

    source: OneOrSeq[str]
    target: NotRequired[OneOrSeq[str]]


class CellIndex(TypedDict, total=False):
    """Ultrack tracking index carried in predict-mode batches.

    All fields optional — presence depends on the source CSV columns.
    (fov_name, track_id, t) together uniquely identify a cell observation
    and are the join key back to valid_anchors.
    """

    fov_name: OneOrSeq[str]
    track_id: OneOrSeq[int]
    t: OneOrSeq[int]
    id: OneOrSeq[int]
    parent_track_id: OneOrSeq[int]
    parent_id: OneOrSeq[int]
    z: OneOrSeq[float]
    y: OneOrSeq[float]
    x: OneOrSeq[float]


class SampleMeta(TypedDict, total=False):
    """Biological metadata carried in train-mode batches for sampler debugging.

    Joinable against valid_anchors on (global_track_id, t).

    Core fields are defined here. Domain-specific fields should be added by
    subclassing ``SampleMeta`` (e.g. ``OpsSampleMeta``). The ``labels`` field
    is an open-ended dict of integer labels that auxiliary heads can consume
    via ``batch_key`` without requiring a subclass.
    """

    experiment: OneOrSeq[str]
    perturbation: OneOrSeq[str]
    microscope: OneOrSeq[str]
    fov_name: OneOrSeq[str]
    global_track_id: OneOrSeq[str]
    t: OneOrSeq[int]
    hours_post_perturbation: OneOrSeq[float]
    lineage_id: OneOrSeq[int]
    labels: dict[str, int]


class TripletSample(TypedDict):
    """Triplet sample type for mini-batches."""

    anchor: Tensor
    positive: NotRequired[Tensor]
    negative: NotRequired[Tensor]
    index: NotRequired[list[CellIndex]]
    anchor_meta: NotRequired[list[SampleMeta]]
    positive_meta: NotRequired[list[SampleMeta]]


# NOTE: these are the only columns that are allowed for the annotation dataframe.
AnnotationColumns = Literal[
    "infection_state",
    "cell_division_state",
    "cell_remodeling_state",
    "cell_cycle_state",
]


# NOTE: The following labels are not mutable.
# They are used to map the labels to the integer values.
LABEL_INFECTION_STATE = {"uninfected": 0, "infected": 1, "unknown": -1}

LABEL_CELL_DIVISION_STATE = {
    "interphase": 0,
    "mitosis": 1,
    "unknown": -1,
}

LABEL_CELL_CYCLE_STATE = {
    "G1": 0,
    "S": 1,
    "G2": 2,
    "prophase": 3,
    "metaphase": 4,
    "anaphase": 5,
    "telophase": 6,
    "unknown": -1,
}

LABEL_CELL_REMODELING_STATE = {
    "no_remodel": 0,
    "remodeling": 1,
    "unknown": -1,
}

CELL_INDEX_CORE_COLUMNS = [
    "cell_id",
    "experiment",
    "store_path",
    "tracks_path",
    "fov",
    "well",
    "y",
    "x",
    "z",
]

CELL_INDEX_GROUPING_COLUMNS = ["perturbation", "channel_name", "microscope"]

CELL_INDEX_BIOLOGY_COLUMNS = ["marker", "organelle"]

CELL_INDEX_TIMELAPSE_COLUMNS = [
    "t",
    "track_id",
    "global_track_id",
    "lineage_id",
    "parent_track_id",
    "hours_post_perturbation",
    "interval_minutes",
]

CELL_INDEX_OPS_COLUMNS = ["gene_name", "reporter", "sgRNA"]

CELL_INDEX_IMAGING_COLUMNS = [
    "pixel_size_xy_um",
    "pixel_size_z_um",
    "T_shape",
    "C_shape",
    "Z_shape",
    "Y_shape",
    "X_shape",
    "z_focus_mean",
]

CELL_INDEX_NORMALIZATION_COLUMNS = [
    "norm_mean",
    "norm_std",
    "norm_median",
    "norm_iqr",
    "norm_max",
    "norm_min",
]

# Extracted from viscy/data/triplet.py for shared access
ULTRACK_INDEX_COLUMNS = [
    "fov_name",
    "track_id",
    "t",
    "id",
    "parent_track_id",
    "parent_id",
    "z",
    "y",
    "x",
]
