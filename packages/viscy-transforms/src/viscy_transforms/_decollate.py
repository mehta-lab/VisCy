"""Decollate batch transform.

This module provides a simple transform to split batched tensors back into
individual samples, useful for post-processing pipeline steps.
"""

from monai.data import decollate_batch
from monai.transforms import Transform
from torch import Tensor

__all__ = ["Decollate"]


class Decollate(Transform):
    """Decollate a batched tensor into a list of individual samples.

    This transform splits a batched tensor along the batch dimension,
    returning a list of individual sample tensors. Useful for applying
    per-sample post-processing operations.

    Parameters
    ----------
    detach : bool
        Whether to detach tensors from the computation graph before
        decollating. Default: True.
    pad_batch : bool
        Whether to pad smaller tensors to match the batch size when
        samples have different sizes. Default: True.
    fill_value : float | None
        Value used for padding when pad_batch is True and samples have
        different sizes. Default: None (use zeros).

    Returns
    -------
    list[Tensor]
        List of individual sample tensors, one per batch element.

    Examples
    --------
    >>> decollate = Decollate(detach=True)
    >>> batch = torch.randn(4, 1, 32, 64, 64)  # [B, C, D, H, W]
    >>> samples = decollate(batch)
    >>> len(samples)
    4
    >>> samples[0].shape
    torch.Size([1, 32, 64, 64])

    See Also
    --------
    monai.data.decollate_batch : Underlying MONAI function.
    Decollated : Dictionary-based version for keyed data.
    """

    def __init__(self, detach: bool = True, pad_batch: bool = True, fill_value=None) -> None:
        super().__init__()
        self.detach = detach
        self.pad_batch = pad_batch
        self.fill_value = fill_value

    def __call__(self, data: Tensor) -> list[Tensor]:
        """Decollate a batched tensor into individual samples.

        Parameters
        ----------
        data : Tensor
            Batched input tensor with shape (B, ...) where B is batch size.

        Returns
        -------
        list[Tensor]
            List of B tensors, each with shape (...).
        """
        return decollate_batch(
            batch=data,
            detach=self.detach,
            pad=self.pad_batch,
            fill_value=self.fill_value,
        )
