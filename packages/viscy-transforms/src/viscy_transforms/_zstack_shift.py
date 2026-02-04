"""Batched Z-stack shift transform.

This module provides a GPU-efficient batched Z-axis shift transform
for 3D microscopy data, simulating focal plane drift.
"""

import torch
from monai.transforms import MapTransform, RandomizableTransform
from torch import Tensor
from typing_extensions import Iterable

__all__ = ["BatchedRandZStackShiftd"]


class BatchedRandZStackShiftd(MapTransform, RandomizableTransform):
    """Randomly shift Z-stack slices along the depth axis.

    Simulates focal plane drift or sample movement that can occur during
    3D microscopy acquisition. Each sample receives an independent random
    shift in the Z (depth) dimension.

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys of the data dictionary to apply the shift to.
    max_shift : int
        Maximum number of slices to shift in either direction.
        The actual shift is sampled uniformly from [-max_shift, max_shift].
        Default: 3.
    prob : float
        Probability of applying the transform. Default: 0.1.
    mode : str
        Padding mode for empty slices after shifting. Currently only
        "constant" is implemented. Default: "constant".
    cval : float
        Constant value used for padding when mode="constant". Default: 0.0.
    allow_missing_keys : bool
        Whether to allow missing keys in the data dictionary. Default: False.

    Returns
    -------
    dict[str, Tensor]
        Dictionary with Z-shifted tensors for specified keys.

    Examples
    --------
    >>> zshift = BatchedRandZStackShiftd(
    ...     keys=["image"],
    ...     max_shift=5,
    ...     prob=0.5,
    ... )
    >>> sample = {"image": torch.randn(2, 1, 32, 64, 64)}
    >>> output = zshift(sample)
    """

    def __init__(
        self,
        keys: str | Iterable[str],
        max_shift: int = 3,
        prob: float = 0.1,
        mode: str = "constant",
        cval: float = 0.0,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.max_shift = max_shift
        self.mode = mode
        self.cval = cval

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply Z-stack shift to the sample.

        Parameters
        ----------
        sample : dict[str, Tensor]
            Dictionary containing tensors with shape (B, C, D, H, W).

        Returns
        -------
        dict[str, Tensor]
            Dictionary with Z-shifted tensors for specified keys.
        """
        self.randomize(None)
        d = dict(sample)

        for key in self.key_iterator(d):
            data = d[key]
            if self.R.rand() < self.prob:
                batch_size, channels, depth, height, width = data.shape

                # Generate random shifts for the batch
                shifts = torch.randint(
                    -self.max_shift,
                    self.max_shift + 1,
                    (batch_size,),
                    device=data.device,
                )

                # Process samples with shifts
                result = data.clone()
                for b in range(batch_size):
                    shift = shifts[b].item()
                    if shift != 0:
                        if shift > 0:
                            # Shift down, pad at top
                            result[b, :, :shift] = self.cval
                            result[b, :, shift:] = data[b, :, :-shift]
                        else:
                            # Shift up, pad at bottom
                            shift = -shift
                            result[b, :, :-shift] = data[b, :, shift:]
                            result[b, :, -shift:] = self.cval

                d[key] = result

        return d
