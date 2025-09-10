"""Type definitions for VisCy data modules and structures."""

from collections.abc import Callable, Sequence
from typing import NamedTuple, TypedDict, TypeVar

from torch import ShortTensor, Tensor

# TODO: use typing.NotRequired when upgrading to Python 3.11
from typing_extensions import NotRequired

DictTransform = Callable[[dict[str, Tensor | dict]], dict[str, Tensor]]


T = TypeVar("T")
OneOrSeq = T | Sequence[T]


class LevelNormStats(TypedDict):
    """Statistics for normalization at a specific level (dataset or FOV)."""

    mean: Tensor
    std: Tensor
    median: Tensor
    iqr: Tensor


class ChannelNormStats(TypedDict):
    """Normalization statistics for a channel at different levels."""

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
    """
    Image sample type for mini-batches.

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
    """
    Tracking index extracted from ultrack result.

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
