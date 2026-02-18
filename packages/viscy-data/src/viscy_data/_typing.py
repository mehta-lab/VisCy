"""Type definitions for viscy-data.

Copied verbatim from ``viscy/data/typing.py`` with the following additions:
- ``ULTRACK_INDEX_COLUMNS`` extracted from ``viscy/data/triplet.py``
- ``__all__`` for explicit public API
- Updated ``typing_extensions.NotRequired`` to ``typing.NotRequired`` (Python >=3.11)
"""

from typing import Callable, Literal, NamedTuple, NotRequired, Sequence, TypedDict, TypeVar

from torch import ShortTensor, Tensor

__all__ = [
    "AnnotationColumns",
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
    "SegmentationSample",
    "TrackingIndex",
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


class ChannelNormStats(TypedDict):
    """Per-channel normalization statistics."""

    dataset_statistics: LevelNormStats
    fov_statistics: LevelNormStats


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


class TrackingIndex(TypedDict):
    """Tracking index extracted from ultrack result.

    Potentially collated by the dataloader.
    """

    fov_name: OneOrSeq[str]
    id: OneOrSeq[int]


class TripletSample(TypedDict):
    """Triplet sample type for mini-batches."""

    anchor: Tensor
    positive: NotRequired[Tensor]
    negative: NotRequired[Tensor]
    index: NotRequired[TrackingIndex]


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
