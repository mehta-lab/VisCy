"""Batched histogram shift transform.

This module provides a GPU-efficient batched histogram shift transform
for simulating intensity variations in microscopy images.
"""

import torch
from monai.transforms import MapTransform, RandomizableTransform
from torch import Tensor
from typing_extensions import Iterable

__all__ = ["BatchedRandHistogramShiftd"]


class BatchedRandHistogramShiftd(MapTransform, RandomizableTransform):
    """Randomly shift intensity histogram by adding a constant offset.

    Simulates global intensity variations that can occur in microscopy
    due to illumination changes, photobleaching, or detector drift.
    Each sample in the batch receives an independently sampled shift.

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys of the data dictionary to apply the shift to.
    shift_range : tuple[float, float]
        Range for the intensity shift value. Samples uniformly between
        min and max values. Default: (-0.1, 0.1).
    prob : float
        Probability of applying the transform. Default: 0.1.
    allow_missing_keys : bool
        Whether to allow missing keys in the data dictionary. Default: False.

    Returns
    -------
    dict[str, Tensor]
        Dictionary with intensity-shifted tensors for specified keys.

    Examples
    --------
    >>> shift = BatchedRandHistogramShiftd(
    ...     keys=["image"],
    ...     shift_range=(-0.2, 0.2),
    ...     prob=0.5,
    ... )
    >>> sample = {"image": torch.randn(2, 1, 32, 64, 64)}
    >>> output = shift(sample)
    """

    def __init__(
        self,
        keys: str | Iterable[str],
        shift_range: tuple[float, float] = (-0.1, 0.1),
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.shift_range = shift_range

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply histogram shift to the sample.

        Parameters
        ----------
        sample : dict[str, Tensor]
            Dictionary containing tensors with shape (B, C, D, H, W).

        Returns
        -------
        dict[str, Tensor]
            Dictionary with intensity-shifted tensors for specified keys.
        """
        self.randomize(None)
        d = dict(sample)

        for key in self.key_iterator(d):
            data = d[key]
            if self.R.rand() < self.prob:
                batch_size = data.shape[0]

                # Generate random shifts for the batch
                shifts = torch.empty(batch_size, device=data.device, dtype=data.dtype)
                shift_min, shift_max = self.shift_range
                shifts.uniform_(shift_min, shift_max)

                # Apply shifts to batch
                shifts = shifts.view(batch_size, 1, 1, 1, 1)
                d[key] = data + shifts

        return d
