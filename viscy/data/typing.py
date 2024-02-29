from typing import Sequence, TypedDict, Union

from torch import Tensor


class Sample(TypedDict, total=False):
    """Image sample type for mini-batches."""

    # all optional
    index: tuple[str, int, int]
    source: Union[Tensor, Sequence[Tensor]]
    target: Union[Tensor, Sequence[Tensor]]
    labels: Union[Tensor, Sequence[Tensor]]
    norm_meta: dict[str, dict]


class ChannelMap(TypedDict, total=False):
    """Source and target channel names."""

    source: Union[str, Sequence[str]]
    # optional
    target: Union[str, Sequence[str]]
