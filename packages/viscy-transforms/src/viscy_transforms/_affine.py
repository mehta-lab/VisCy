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
        Rotation angle range in radians for each axis (Z, Y, X order).
        Converted to degrees for Kornia (X, Y, Z order). Default: None.
    shear_range : Sequence[tuple[float, float] | float] | float | None
        Shear factor range for each axis (Z, Y, X order).
        Converted to degrees for Kornia. Default: None.
    translate_range : Sequence[tuple[float, float] | float] | float | None
        Translation range for each axis (Z, Y, X order).
        Converted to XYZ order for Kornia. Default: None.
    scale_range : Sequence[tuple[float, float] | float] | float | None
        Scale factor range for each axis (Z, Y, X order).
        Converted to XYZ order for Kornia. Default: None.
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
    ) -> Sequence[tuple[float, float] | float] | float | None:
        if rotate_range is None:
            return None
        return torch.from_numpy(np.rad2deg(rotate_range))

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
