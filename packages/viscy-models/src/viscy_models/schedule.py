"""Parameter scheduling utilities for loss terms and head weights."""

from __future__ import annotations

import math


def cosine_anneal(start: float, end: float, epoch: int, warmup_epochs: int) -> float:
    """Cosine anneal from ``start`` to ``end`` over ``warmup_epochs``.

    After ``warmup_epochs``, returns ``end`` unchanged.

    Parameters
    ----------
    start : float
        Value at epoch 0.
    end : float
        Value at and beyond ``warmup_epochs``.
    epoch : int
        Current epoch (0-indexed).
    warmup_epochs : int
        Number of epochs over which to anneal.

    Returns
    -------
    float
        Annealed value.
    """
    if epoch >= warmup_epochs:
        return end
    progress = epoch / warmup_epochs
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))
