"""Tensor conversion utilities for mixed-precision safety."""

import numpy as np
import torch
from torch import Tensor


def to_numpy(t: Tensor) -> np.ndarray:
    """Convert a tensor to a NumPy array, handling mixed-precision dtypes.

    Bfloat16 tensors (from AMP/autocast) are cast to float32 because
    NumPy does not support bfloat16. All other dtypes (float16, float32,
    float64, integers, booleans) are preserved.

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
    if t.dtype == torch.bfloat16:
        t = t.to(device="cpu", dtype=torch.float32)
    else:
        t = t.cpu()
    return t.contiguous().numpy()
