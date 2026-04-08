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
    shear_range : Sequence[float] | Sequence[tuple[float, float]] | tuple[float, float] | None
        Shear range in degrees. Three forms:

        * ``(min, max)`` — same range for all 6 shear facets.
        * ``[s_zy, s_zx, s_yz]`` — 3 values matching MONAI's upper-triangle
          convention. Maps to Kornia facets ``szy``, ``szx``, ``syz``.
        * 6 ``(min, max)`` pairs for ``(sxy, sxz, syx, syz, szx, szy)``.

        Default: None (no shearing).
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
    scale_z_shear : bool
        When True and ``shear_range`` uses the 3-value shorthand, Z-related
        shear facets are scaled by ``z_depth / yx_size`` at call time so
        that the pixel displacement in Z is proportional to the Z depth
        rather than the much larger YX extent. Default: True.
        Set to False for unscaled (raw) shear values.
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

    is_spatial = True

    def __init__(
        self,
        keys: str | Iterable[str],
        prob: float = 0.1,
        rotate_range: Sequence[tuple[float, float] | float] | float | None = None,
        shear_range: Sequence[tuple[float, float] | float] | float | None = None,
        translate_range: Sequence[tuple[float, float] | float] | float | None = None,
        scale_range: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
        isotropic_scale: bool = False,
        scale_z_shear: bool = True,
        mode: str = "bilinear",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        rotate_range = self._radians_to_degrees(self._invert_per_axis(rotate_range))
        if rotate_range is None:
            rotate_range = (0.0, 0.0, 0.0)
        self._scale_z_shear = scale_z_shear
        shear_range = self._parse_shear_range(shear_range)
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
    def _parse_shear_range(
        shear_range: Sequence | tuple | None,
    ) -> tuple | None:
        """Parse shear_range into Kornia format.

        The 3-value shorthand ``[z, y, x]`` (degrees) matches MONAI's
        convention: 3 values fill the upper-triangle of the shear matrix
        in ZYX order as ``(s_zy, s_zx, s_yz)``, which maps to Kornia
        facets ``(szy=±z, szx=±y, syz=±x)``.
        """
        if shear_range is None:
            return None
        # (min, max) isotropic — pass through.
        if len(shear_range) == 2 and not isinstance(shear_range[0], (list, tuple)):
            return tuple(shear_range)
        # 6 (min, max) pairs — pass through.
        if len(shear_range) == 6:
            return tuple(tuple(p) if isinstance(p, (list, tuple)) else (-p, p) for p in shear_range)
        # 3-value MONAI shorthand: [s_zy, s_zx, s_yz] in ZYX order.
        # MONAI matrix:  [[1, v0, v1, 0], [v2, 1, 0, 0], [0, 0, 1, 0]]
        # Maps to Kornia: (sxy=0, sxz=0, syx=0, syz=±v2, szx=±v1, szy=±v0)
        if len(shear_range) == 3 and not isinstance(shear_range[0], (list, tuple)):
            s_zy, s_zx, s_yz = shear_range
            return ((0, 0), (0, 0), (0, 0), (-s_yz, s_yz), (-s_zx, s_zx), (-s_zy, s_zy))
        raise ValueError(
            f"shear_range must be (min, max), [s_zy, s_zx, s_yz] (3-value), or 6 (min, max) pairs. Got {shear_range!r}."
        )

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
        if len(scale_range) != 2:
            raise ValueError(
                f"scale_range must be (min, max) or [(z_min, z_max), (y_min, y_max), "
                f"(x_min, x_max)]. Got {scale_range!r} with length {len(scale_range)}."
            )
        # Flat (min, max) — Kornia samples independently per axis from this range.
        return tuple(scale_range), False

    @staticmethod
    def _scale_z_shear_facets(params: dict[str, Tensor], shape: torch.Size) -> dict[str, Tensor]:
        """Scale Z-related shear facets by z_depth / yx_size.

        Shear magnitude is proportional to the axis extent it operates on.
        Without scaling, a shear applied across the 512-pixel YX extent
        produces ~64x more displacement than the same angle across Z=8.
        This method reduces Z-related facets so the pixel displacement
        in Z is proportional to the Z depth.
        """
        z_depth = shape[2]
        yx_size = max(shape[3], shape[4])
        if yx_size <= 1 or z_depth >= yx_size:
            return params
        ratio = z_depth / yx_size
        for key in ("sxz", "syz", "szx", "szy"):
            params[key].mul_(ratio)
        return params

    @staticmethod
    def _make_scale_isotropic(params: dict[str, Tensor]) -> dict[str, Tensor]:
        """Replace per-axis scale with a single isotropic factor per sample."""
        scale = params["scale"]
        iso = scale[:, 0:1].expand_as(scale)
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
        ref = d[first_key]
        params = self.random_affine.forward_parameters(ref.shape)
        if self._isotropic_scale:
            params = self._make_scale_isotropic(params)
        if self._scale_z_shear:
            params = self._scale_z_shear_facets(params, ref.shape)
        # Apply with the same parameters to every key.
        for key in self.key_iterator(d):
            d[key] = self.random_affine(d[key], params=params)
        return d
