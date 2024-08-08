from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Sequence, TypedDict, TypeVar

if TYPE_CHECKING:
    from torch import Tensor

T = TypeVar("T")
OneOrSeq = T | Sequence[T]


class LevelNormStats(TypedDict):
    mean: Tensor
    std: Tensor
    median: Tensor
    iqr: Tensor


class ChannelNormStats(TypedDict):
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
    norm_meta: NormMeta


class _ChannelMap(TypedDict):
    """Source channel names."""

    source: OneOrSeq[str]


class ChannelMap(_ChannelMap, total=False):
    """Source and target channel names."""

    # TODO: use typing.NotRequired when upgrading to Python 3.11
    target: OneOrSeq[str]
