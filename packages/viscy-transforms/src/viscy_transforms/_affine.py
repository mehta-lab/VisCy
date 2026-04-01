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

    Random parameters are generated once per ``__call__`` and reused
    across all keys so that paired inputs (e.g. source/target) receive
    identical spatial transforms.

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
    scale_range : tuple[float, float] | Sequence[tuple[float, float]] | None
        Scale factor range (absolute, not offset from 1.0). Two forms:

        * ``(min, max)`` — same range sampled independently per axis.
        * ``[(z_min, z_max), (y_min, y_max), (x_min, x_max)]`` —
          per-axis ranges in ZYX order (converted internally to XYZ).

        Default: None (no scaling).
    isotropic_scale : bool
        When True and ``scale_range`` is provided, a single random scale
        factor is drawn and applied identically to all three axes.
        Default: False.
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
    Rotation, shear, and per-axis scale parameter ordering follows MONAI
    convention (Z, Y, X) but is internally converted to Kornia's convention
    (X, Y, Z).  A flat ``scale_range=(min, max)`` is **not** axis-inverted.

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
        scale_range: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
        isotropic_scale: bool = False,
        mode: str = "bilinear",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        rotate_range = self._radians_to_degrees(self._invert_per_axis(rotate_range))
        if rotate_range is None:
            rotate_range = (0.0, 0.0, 0.0)
        shear_range = self._radians_to_degrees(self._invert_per_axis(shear_range))
        translate_range = self._invert_per_axis(translate_range)
        scale_range, per_axis = self._parse_scale_range(scale_range)
        if isotropic_scale and per_axis:
            raise ValueError(
                "isotropic_scale=True cannot be combined with per-axis scale_range. "
                "Use a flat (min, max) range instead."
            )
        self._isotropic_scale = isotropic_scale and scale_range is not None
        self.random_affine = RandomAffine3D(
            degrees=rotate_range,
            translate=translate_range,
            scale=scale_range,
            shears=shear_range,
            resample=mode,
            p=prob,
        )

    @staticmethod
    def _invert_per_axis(
        value: Sequence[tuple[float, float] | float] | float | None,
    ) -> Sequence[tuple[float, float] | float] | float | None:
        """Translate MONAI's ZYX per-axis order to Kornia's XYZ order."""
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

    @staticmethod
    def _parse_scale_range(
        scale_range: tuple[float, float] | Sequence[tuple[float, float]] | None,
    ) -> tuple[tuple[tuple[float, float], ...] | tuple[float, float] | None, bool]:
        """Parse scale_range into Kornia format.

        Returns
        -------
        kornia_scale
            Value passed to ``RandomAffine3D(scale=...)``.
        per_axis
            True if per-axis ZYX ranges were provided.
        """
        if scale_range is None:
            return None, False
        # Per-axis: list of 3 (min, max) pairs in ZYX order → reverse to XYZ.
        if len(scale_range) == 3 and isinstance(scale_range[0], (list, tuple)):
            z, y, x = scale_range
            return (tuple(x), tuple(y), tuple(z)), True
        # Flat (min, max) — Kornia samples independently per axis from this range.
        return tuple(scale_range), False

    @staticmethod
    def _make_scale_isotropic(params: dict[str, Tensor]) -> dict[str, Tensor]:
        """Replace per-axis scale with a single isotropic factor per sample."""
        scale = params["scale"]
        iso = scale[:, 0:1].expand_as(scale).contiguous()
        params["scale"] = iso
        return params

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
        # Find the first present key; return unchanged if none match.
        first_key = self.first_key(d)
        if first_key not in d:
            return d
        # Generate random parameters once from the first key's shape.
        params = self.random_affine.forward_parameters(d[first_key].shape)
        if self._isotropic_scale:
            params = self._make_scale_isotropic(params)
        # Apply with the same parameters to every key.
        for key in self.key_iterator(d):
            d[key] = self.random_affine(d[key], params=params)
        return d
