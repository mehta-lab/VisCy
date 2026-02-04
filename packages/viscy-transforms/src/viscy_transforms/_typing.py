"""Type definitions for viscy-transforms.

This module provides type definitions extracted from viscy.data.typing
for use in the standalone viscy-transforms package. These types define
the data structures used by transform classes.

Original source: https://github.com/mehta-lab/VisCy/blob/main/viscy/data/typing.py
"""

from typing import NamedTuple, Sequence, TypedDict, TypeVar

from torch import Tensor
from typing_extensions import NotRequired

__all__ = [
    "OneOrSeq",
    "HCSStackIndex",
    "LevelNormStats",
    "ChannelNormStats",
    "NormMeta",
    "Sample",
    "ChannelMap",
]

# Generic type for single value or sequence
T = TypeVar("T")
OneOrSeq = T | Sequence[T]


class LevelNormStats(TypedDict):
    """Normalization statistics at a single level (FOV or dataset)."""

    mean: Tensor
    std: Tensor
    median: Tensor
    iqr: Tensor


class ChannelNormStats(TypedDict):
    """Normalization statistics for a channel at both levels."""

    dataset_statistics: LevelNormStats
    fov_statistics: LevelNormStats


# Mapping from channel name to its normalization statistics
NormMeta = dict[str, ChannelNormStats]


class HCSStackIndex(NamedTuple):
    """HCS stack index identifying a specific image location."""

    # name of the image array, e.g. "A/1/0/0"
    image: str
    time: int
    z: int


class Sample(TypedDict, total=False):
    """Image sample type for mini-batches.

    All fields are optional (total=False).
    Used as input/output type for transform classes.
    """

    index: HCSStackIndex
    # Image data
    source: OneOrSeq[Tensor]
    target: OneOrSeq[Tensor]
    weight: OneOrSeq[Tensor]
    # Instance segmentation masks
    labels: OneOrSeq[Tensor]
    # Normalization metadata (None if not available)
    norm_meta: NormMeta | None


class ChannelMap(TypedDict):
    """Mapping of source and target channel names.

    Used by StackChannelsd to select which channels to stack.
    """

    source: OneOrSeq[str]
    target: NotRequired[OneOrSeq[str]]
