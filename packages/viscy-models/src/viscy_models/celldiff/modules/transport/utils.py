"""Utility functions for flow-matching transport."""

import torch
from torch import Tensor


def expand_t_like_x(t: Tensor, x: Tensor) -> Tensor:
    """Reshape time vector to be broadcastable with data tensor.

    Parameters
    ----------
    t : Tensor
        Time vector of shape ``(B,)``.
    x : Tensor
        Data tensor of shape ``(B, ...)``.

    Returns
    -------
    Tensor
        Reshaped time of shape ``(B, 1, 1, ...)``.
    """
    dims = [1] * (len(x.size()) - 1)
    return t.view(t.size(0), *dims)


def mean_flat(x: Tensor) -> Tensor:
    """Take the mean over all non-batch dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(B, ...)``.

    Returns
    -------
    Tensor
        Mean over dims 1..N, shape ``(B,)``.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))
