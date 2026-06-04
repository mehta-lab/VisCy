"""Channel stacking transforms for microscopy data.

This module provides transforms for combining multiple single-channel
tensors into multi-channel tensors based on channel mapping configurations.
"""

import torch
from monai.transforms import MapTransform

from viscy_transforms._typing import ChannelMap, Sample

__all__ = ["BatchedStackChannelsd", "StackChannelsd"]


class StackChannelsd(MapTransform):
    is_spatial = False

    """Stack source and target channels from multiple keys.

    Combines multiple single-channel tensors into multi-channel tensors
    based on a channel mapping configuration. Concatenates along ``dim=0``
    for per-sample ``(C, Z, Y, X)`` tensors.

    Parameters
    ----------
    channel_map : ChannelMap
        Dictionary mapping output keys to lists of input channel keys.
        Example: {"source": ["phase", "bf"], "target": ["nuclei", "membrane"]}

    Examples
    --------
    >>> stack = StackChannelsd({"source": ["ch1", "ch2"], "target": ["ch3"]})
    >>> sample = {"ch1": tensor1, "ch2": tensor2, "ch3": tensor3}
    >>> result = stack(sample)
    >>> result["source"].shape  # Combined ch1 and ch2
    """

    _cat_dim = 0

    def __init__(self, channel_map: ChannelMap) -> None:
        channel_names = []
        for channels in channel_map.values():
            channel_names.extend(channels)
        super().__init__(channel_names, allow_missing_keys=False)
        self.channel_map = channel_map

    def __call__(self, sample: Sample) -> Sample:
        """Stack channels according to the channel map.

        Parameters
        ----------
        sample : Sample
            Dictionary containing single-channel tensors.

        Returns
        -------
        Sample
            Dictionary with stacked multi-channel tensors.
        """
        results = {}
        for key, channels in self.channel_map.items():
            results[key] = torch.cat([sample[ch] for ch in channels], dim=self._cat_dim)
        return results


class BatchedStackChannelsd(StackChannelsd):
    """Stack source and target channels from batched tensors.

    Like :class:`StackChannelsd` but for batched ``(B, C, Z, Y, X)``
    tensors where concatenation is along ``dim=1`` (channel) instead
    of ``dim=0``.

    Parameters
    ----------
    channel_map : ChannelMap
        Dictionary mapping output keys to lists of input channel keys.
        Example: ``{"source": ["phase"], "target": ["nuclei", "membrane"]}``
    """

    is_spatial = False

    _cat_dim = 1
