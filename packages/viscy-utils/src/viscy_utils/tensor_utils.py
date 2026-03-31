"""Tensor conversion utilities for mixed-precision safety."""

import numpy as np
import torch
from torch import Tensor


def to_numpy(t: Tensor) -> np.ndarray:
    """Convert a tensor to a NumPy array, handling mixed-precision dtypes.

    Floating-point tensors are cast to float32 because NumPy does not
    support bfloat16 or other AMP-produced dtypes. Integer and boolean
    tensors preserve their dtype.

    Parameters
    ----------
    t : Tensor
        Input tensor (any device, any dtype).

    Returns
    -------
    np.ndarray
        NumPy array on CPU.
    """
    t = t.detach()
    if t.is_floating_point():
        t = t.to(device="cpu", dtype=torch.float32)
    else:
        t = t.cpu()
    return t.contiguous().numpy()
