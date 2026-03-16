"""Affine transforms using Kornia for GPU-optimized operations.

This module provides batched affine transformations using Kornia's
RandomAffine3D for efficient GPU execution on microscopy data.
"""

import numpy as np
import torch
from kornia.augmentation import RandomAffine3D
from monai.transforms import MapTransform
from torch import Tensor
from typing_extensions import Iterable, Sequence

__all__ = ["BatchedRandAffined"]


class BatchedRandAffined(MapTransform):
    """Randomly apply 3D affine transformations using Kornia.

    GPU-optimized affine transform using Kornia's RandomAffine3D for
    batched data. Supports rotation, shearing, translation, and scaling
    with efficient GPU execution.

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys of the data dictionary to transform.
    prob : float
        Probability of applying the transform. Default: 0.1.
    rotate_range : Sequence[tuple[float, float] | float] | float | None
        Rotation angle range in radians per axis in (Z, Y, X) order.
        Reversed to Kornia's (X, Y, Z) order and converted to degrees. Default: None.
    shear_range : Sequence[tuple[float, float] | float] | float | None
        Shear angle range in radians per facet in (szy, szx, syz, syx, sxz, sxy) order.
        Reversed to Kornia's (sxy, sxz, syx, syz, szx, szy) order and converted to degrees.
        Also accepts a scalar or 2-tuple to apply uniformly to all 6 facets. Default: None.
    translate_range : Sequence[tuple[float, float] | float] | float | None
        Translation range as a fraction of image size per axis in (Z, Y, X) order.
        Reversed to Kornia's (X, Y, Z) order. Default: None.
    scale_range : Sequence[tuple[float, float] | float] | float | None
        Scale factor range per axis in (Z, Y, X) order.
        Reversed to Kornia's (X, Y, Z) order. Default: None.
    mode : str
        Interpolation mode. Default: "bilinear".
    allow_missing_keys : bool
        Whether to allow missing keys. Default: False.

    Returns
    -------
    dict[str, Tensor]
        Dictionary with transformed tensors for specified keys.

    Notes
    -----
    Parameter ordering follows MONAI convention (Z, Y, X) but is internally
    converted to Kornia's convention (X, Y, Z).

    See Also
    --------
    kornia.augmentation.RandomAffine3D : Underlying Kornia transform.
    """

    def __init__(
        self,
        keys: str | Iterable[str],
        prob: float = 0.1,
        rotate_range: Sequence[tuple[float, float] | float] | float | None = None,
        shear_range: Sequence[tuple[float, float] | float] | float | None = None,
        translate_range: Sequence[tuple[float, float] | float] | float | None = None,
        scale_range: Sequence[tuple[float, float] | float] | float | None = None,
        mode: str = "bilinear",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        rotate_range = self._radians_to_degrees(self._maybe_invert_sequence(rotate_range))
        if rotate_range is None:
            rotate_range = (0.0, 0.0, 0.0)
        shear_range = self._radians_to_degrees(self._maybe_invert_sequence(shear_range))
        translate_range = self._maybe_invert_sequence(translate_range)
        scale_range = self._maybe_invert_sequence(scale_range)
        self.random_affine = RandomAffine3D(
            degrees=rotate_range,
            translate=translate_range,
            scale=scale_range,
            shears=shear_range,
            resample=mode,
            p=prob,
        )
        # disable unnecessary transfer to CPU
        self.random_affine.disable_features = True

    @staticmethod
    def _maybe_invert_sequence(
        value: Sequence[tuple[float, float] | float] | float | None,
    ) -> Sequence[tuple[float, float] | float] | float | None:
        """Translate MONAI's ZYX order to Kornia's XYZ order."""
        if isinstance(value, Sequence):
            return tuple(reversed(value))
        return value

    @staticmethod
    def _radians_to_degrees(
        rotate_range: Sequence[tuple[float, float] | float] | float | None,
    ) -> tuple[tuple[float, float], ...] | None:
        if rotate_range is None:
            return None
        result = []
        for v in rotate_range:
            if isinstance(v, (tuple, list)):
                result.append((float(np.rad2deg(v[0])), float(np.rad2deg(v[1]))))
            else:
                deg = float(np.rad2deg(v))
                result.append((-deg, deg))
        return tuple(result)

    @torch.no_grad()
    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply random affine transformation to specified keys.

        Parameters
        ----------
        sample : dict[str, Tensor]
            Dictionary containing tensors with shape (B, C, D, H, W).

        Returns
        -------
        dict[str, Tensor]
            Dictionary with transformed tensors for specified keys.
        """
        d = dict(sample)
        for key in self.key_iterator(d):
            data = d[key]
            try:
                d[key] = self.random_affine(data)
            except RuntimeError:
                # retry
                d[key] = self.random_affine(data)
            assert d[key].device == data.device
        return d
