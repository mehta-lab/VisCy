"""Affine transforms using Kornia for GPU-optimized operations.

This module provides batched affine transformations using Kornia's
RandomAffine3D for efficient GPU execution on microscopy data.
"""

import logging

import numpy as np
import torch
from kornia.augmentation import RandomAffine3D
from kornia.geometry.transform import warp_affine3d
from monai.transforms import MapTransform
from torch import Tensor
from typing_extensions import Iterable, Sequence

__all__ = ["BatchedRandAffined"]

_logger = logging.getLogger(__name__)


class _PaddedRandomAffine3D(RandomAffine3D):
    """RandomAffine3D with configurable padding_mode.

    Kornia 0.8.x hard-codes ``padding_mode='zeros'`` in apply_transform.
    This subclass overrides that call to forward the user-specified mode.
    """

    def __init__(self, *args: object, padding_mode: str = "zeros", **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._padding_mode = padding_mode

    def apply_transform(
        self,
        input: Tensor,
        params: dict,
        flags: dict,
        transform: Tensor | None = None,
    ) -> Tensor:
        return warp_affine3d(
            input,
            transform[:, :3, :],
            (input.shape[-3], input.shape[-2], input.shape[-1]),
            flags["resample"].name.lower(),
            padding_mode=self._padding_mode,
            align_corners=flags["align_corners"],
        )


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
    padding_mode : str
        Padding mode for areas outside the rotated image boundary.
        ``"zeros"`` fills with 0, ``"border"`` replicates edge pixels,
        ``"reflection"`` mirrors the image. Default: ``"zeros"``.

        Use ``"border"`` when the oversized crop border is insufficient
        to absorb large rotation angles (i.e. crop/output ratio < √2).
    safe_crop_size : Sequence[int] | None
        ZYX size of the downstream center crop. When set, the sampled
        scale is clamped so that the rotated source covers this crop
        region, reducing zero-corner artifacts.

        The per-sample lower bound on Kornia scale is:

        ``s_min_i = coverage * (sum_j |R_ij| * d_j) / h_i``

        where ``d = safe_crop_size / 2``, ``h = input_size / 2``,
        ``R`` is the rotation matrix, and ``coverage`` is
        ``safe_crop_coverage``. Default: None (no clamping).
    safe_crop_coverage : float
        Fraction of the ``safe_crop_size`` that must be covered by
        the source after the affine transform. ``1.0`` eliminates all
        zero-corner artifacts; lower values (e.g. ``0.85``) allow
        small corners to remain as extra augmentation while still
        preventing the worst cases. Default: 1.0.
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
        padding_mode: str = "zeros",
        safe_crop_size: Sequence[int] | None = None,
        safe_crop_coverage: float = 1.0,
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
        self._safe_crop_size = tuple(safe_crop_size) if safe_crop_size is not None else None
        self._safe_crop_coverage = safe_crop_coverage
        self.random_affine = _PaddedRandomAffine3D(
            degrees=rotate_range,
            translate=translate_range,
            scale=scale_range,
            shears=shear_range,
            resample=mode,
            p=prob,
            padding_mode=padding_mode,
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

    @staticmethod
    def _compute_scale_floor(
        angles: Tensor,
        input_shape: torch.Size,
        safe_crop_size: tuple[int, ...],
    ) -> Tensor:
        """Per-axis minimum Kornia scale for full source coverage.

        For Z-only rotation by θ in the YX plane, the backward-warp
        footprint along each axis is ``D_i * k(θ) / s_i`` where
        ``k = |cos θ| + |sin θ|``. Requiring this ≤ ``S_i`` gives
        ``s_i ≥ k(θ) * D_i / S_i``.

        Parameters
        ----------
        angles : Tensor
            Sampled rotation angles in degrees, shape ``(B, 3)``,
            Kornia ``(X, Y, Z)`` order. Matches the ``"angles"`` key
            from ``RandomAffine3D.forward_parameters()``.
        input_shape : torch.Size
            Input tensor shape ``(B, C, D, H, W)``.
        safe_crop_size : tuple[int, ...]
            Downstream crop size in ``(Z, Y, X)`` order.

        Returns
        -------
        Tensor
            Minimum scale per axis, shape ``(B, 3)``, Kornia
            ``(X, Y, Z)`` order.
        """
        theta_z = torch.deg2rad(angles[:, 2])
        cos_z = theta_z.cos().abs()
        sin_z = theta_z.sin().abs()

        dz = safe_crop_size[0] / 2.0
        dy = safe_crop_size[1] / 2.0
        dx = safe_crop_size[2] / 2.0
        hz = input_shape[2] / 2.0
        hy = input_shape[3] / 2.0
        hx = input_shape[4] / 2.0

        # Z rotation mixes X and Y in the backward warp.
        s_min_x = (cos_z * dx + sin_z * dy) / hx
        s_min_y = (sin_z * dx + cos_z * dy) / hy
        s_min_z = torch.full_like(s_min_x, dz / hz)

        return torch.stack([s_min_x, s_min_y, s_min_z], dim=-1)

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
        if self._safe_crop_size is not None:
            xy_angles = params["angles"][:, :2]
            if (xy_angles.abs() > 1e-3).any():
                _logger.warning(
                    "safe_crop_size only accounts for Z-axis rotation; "
                    "X/Y rotations (%.1f, %.1f deg) may cause zero-corner artifacts.",
                    xy_angles[:, 0].abs().max().item(),
                    xy_angles[:, 1].abs().max().item(),
                )
            s_floor = self._compute_scale_floor(params["angles"], ref.shape, self._safe_crop_size)
            s_floor *= self._safe_crop_coverage
            if self._isotropic_scale:
                s_floor = s_floor.max(dim=-1, keepdim=True).values.expand_as(s_floor)
            params["scale"] = torch.max(params["scale"], s_floor)
        if self._scale_z_shear:
            params = self._scale_z_shear_facets(params, ref.shape)
        # Apply with the same parameters to every key.
        for key in self.key_iterator(d):
            d[key] = self.random_affine(d[key], params=params)
        return d
