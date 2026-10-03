"""Background low-pass of a training target, foreground kept at full resolution.

``x' = w * x + (1 - w) * B`` with ``m = dilate(fg_mask, r)``,
``w = clamp(G_f(m), 0, 1)`` and the background estimate
``B = G_lp(x * (1 - m)) / G_lp(1 - m)``: inside the dilated foreground the
target is unchanged, in the background it is a Gaussian low-pass of the
background pixels alone, and a Gaussian feather of the dilated mask blends the
two without a seam.

``B`` is a normalized convolution: a plain ``G_lp(x)`` would bleed bright
foreground into the background as a halo, while ``B`` averages background
pixels only, so a foreground blob never reaches it. With ``sigma_lp=None`` (its
``sigma_lp -> inf`` limit) ``B`` is flat: the mean of each plane's background
pixels, which also removes autofluorescence and uneven illumination.

It changes the target, never the loss: the background stays supervised toward a
smooth field. In flow matching, masking the background out of the loss would
leave the initial Gaussian noise there unremoved (the Spotlight v1 collapse).

The Gaussians are normalized, edge-replicated (``scipy.ndimage`` ``mode="nearest"``,
``truncate=4.0``) and separable, applied as shifted multiply-adds in float32, so
no convolution backend (cuDNN TF32, autocast) can lower their precision. ``B`` is a
weighted mean of background pixels, so it keeps the local background level and
the op needs no change to the target's normalization.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["BackgroundLowPass"]

# At or below this fraction of the low-pass kernel's mass on background pixels the
# normalized convolution has no background in reach; B falls back to the plane's flat mean.
_MIN_BACKGROUND_WEIGHT = 1e-6


def _gaussian_kernel(sigma: float, device: torch.device) -> Tensor:
    """Return a normalized float32 1D Gaussian of radius ``int(4 * sigma + 0.5)``, as scipy does."""
    radius = int(4.0 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _gaussian_blur(x: Tensor, sigmas: tuple[float, float, float]) -> Tensor:
    """Separable Gaussian blur of ``(N, D, H, W)`` with edge-replicated borders.

    Parameters
    ----------
    x : Tensor
        Float32, shape ``(N, D, H, W)``.
    sigmas : tuple of float
        Standard deviation in voxels along ``(D, H, W)``; 0 leaves that axis alone.

    Returns
    -------
    Tensor
        Blurred tensor, shape of ``x``.
    """
    for dim, sigma in zip((1, 2, 3), sigmas):
        length = x.shape[dim]
        # A size-1 axis replicates its one value, so the blur along it is the identity.
        if sigma <= 0 or length == 1:
            continue
        k = _gaussian_kernel(sigma, x.device)
        r = k.numel() // 2
        # F.pad lists the last dim first; replicate supports any pad width.
        pad = [0] * 6
        pad[2 * (3 - dim)] = pad[2 * (3 - dim) + 1] = r
        padded = F.pad(x.unsqueeze(1), pad, mode="replicate").squeeze(1)
        out = k[0] * padded.narrow(dim, 0, length)
        for i in range(1, k.numel()):
            out = out + k[i] * padded.narrow(dim, i, length)
        x = out
    return x


class BackgroundLowPass(nn.Module):
    """Replace a target's background by a low-pass estimate from background pixels.

    ``x' = w * x + (1 - w) * B``, ``m = dilate(fg_mask > 0.5, r)``,
    ``w = clamp(G_f(m), 0, 1)``, ``B = G_lp(x * (1 - m)) / G_lp(1 - m)``.
    ``w`` is exactly 1 where the feather kernel lies entirely inside ``m`` (the
    dilated mask shrunk by its radius, ``int(4 * sigma_feather + 0.5)``), so
    ``x' = x`` exactly there, and exactly 0 where the kernel misses ``m``, so
    ``x' = B`` exactly there.

    ``sigma_lp=None`` makes ``B`` flat: per sample, channel and Z plane, the mean
    of ``x`` over the plane's background pixels, the ``sigma_lp -> inf`` limit of
    the normalized convolution. Where the normalized convolution has at most
    ``1e-6`` of its kernel mass on background (a thin background ring narrower
    than its support, say), ``B`` falls back to that flat mean. A plane with no
    background pixel at all has no estimate: ``B = x`` there, and with the
    per-plane feather ``w == 1``, so the plane is returned unchanged.

    By default every operation acts in XY, per Z plane: Z is anisotropic, and
    2D models have ``D = 1``. The ``*_z`` parameters extend each to Z.

    Computed in float32 regardless of autocast; returned in the target's dtype.

    Parameters
    ----------
    sigma_lp : float or None
        Gaussian sigma in XY, in voxels, of the background estimate (> 0);
        ``None`` (or ``inf``) for a flat background.
    sigma_feather : float
        Gaussian sigma in XY, in voxels, that feathers the dilated mask into
        ``w`` (>= 0; 0 keeps the hard dilated mask).
    dilate_radius : int
        Dilation radius in XY, in voxels (>= 0): a ``(2r + 1)``-wide square.
    sigma_lp_z : float
        Background-estimate sigma along Z, in voxels (>= 0; 0 = per plane).
        Must be 0 with a flat background, which is per plane.
    sigma_feather_z : float
        Feather sigma along Z, in voxels (>= 0; 0 = per plane).
    dilate_radius_z : int
        Dilation radius along Z, in voxels (>= 0; 0 = per plane).
    """

    def __init__(
        self,
        sigma_lp: float | None,
        sigma_feather: float,
        dilate_radius: int,
        sigma_lp_z: float = 0.0,
        sigma_feather_z: float = 0.0,
        dilate_radius_z: int = 0,
    ) -> None:
        super().__init__()
        if sigma_lp is not None and math.isinf(sigma_lp):
            sigma_lp = None
        if sigma_lp is None:
            if sigma_lp_z != 0:
                raise ValueError(f"sigma_lp_z={sigma_lp_z} has no effect with a flat background (sigma_lp=None)")
        elif sigma_lp <= 0:
            raise ValueError(f"sigma_lp must be > 0 or None, got {sigma_lp}")
        for name, value in (
            ("sigma_feather", sigma_feather),
            ("sigma_lp_z", sigma_lp_z),
            ("sigma_feather_z", sigma_feather_z),
            ("dilate_radius", dilate_radius),
            ("dilate_radius_z", dilate_radius_z),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        self.sigma_lp = None if sigma_lp is None else float(sigma_lp)
        self.sigma_feather = float(sigma_feather)
        self.dilate_radius = int(dilate_radius)
        self.sigma_lp_z = float(sigma_lp_z)
        self.sigma_feather_z = float(sigma_feather_z)
        self.dilate_radius_z = int(dilate_radius_z)

    def dilated_mask(self, fg_mask: Tensor) -> Tensor:
        """Return ``m = dilate(fg_mask > 0.5, r)`` as float32 in {0, 1}.

        Parameters
        ----------
        fg_mask : Tensor
            Foreground mask, shape ``(B, C, D, H, W)``; binarized at 0.5.

        Returns
        -------
        Tensor
            Float32, shape of ``fg_mask``.
        """
        b, c = fg_mask.shape[:2]
        rz, r = self.dilate_radius_z, self.dilate_radius
        mask = (fg_mask > 0.5).float().reshape(b * c, 1, *fg_mask.shape[2:])
        if rz or r:
            # max_pool pads with -inf, so the border never dilates the mask.
            mask = F.max_pool3d(mask, (2 * rz + 1, 2 * r + 1, 2 * r + 1), stride=1, padding=(rz, r, r))
        return mask.reshape(fg_mask.shape)

    def blend_weight(self, dilated: Tensor) -> Tensor:
        """Return ``w = clamp(G_f(m), 0, 1)`` in float32, exactly 1 under an all-``m`` kernel.

        Parameters
        ----------
        dilated : Tensor
            Dilated mask ``m`` from :meth:`dilated_mask`, shape ``(B, C, D, H, W)``.

        Returns
        -------
        Tensor
            Float32 weights in [0, 1], shape of ``dilated``.
        """
        b, c = dilated.shape[:2]
        spatial = dilated.shape[2:]
        with torch.autocast(device_type=dilated.device.type, enabled=False):
            sigmas = (self.sigma_feather_z, self.sigma_feather, self.sigma_feather)
            blurred = _gaussian_blur(dilated.float().reshape(b * c, *spatial), sigmas)
            # The normalized kernel sums to 1 only up to rounding (0.99999982 at sigma 0.5).
            # Edge replication makes the blur of ones the same value at every voxel, computed
            # by the same operations as any all-ones window of m: dividing by it gives exactly
            # 1 there. Size-2 axes reproduce it (size 1 is skipped, like in the full tensor).
            ones = torch.ones((1, *(min(n, 2) for n in spatial)), device=dilated.device)
            unit = _gaussian_blur(ones, sigmas)[0, 0, 0, 0]
            return (blurred / unit).clamp(0.0, 1.0).reshape(dilated.shape)

    def background(self, target: Tensor, dilated: Tensor) -> Tensor:
        """Return the background estimate ``B`` in float32; ``target`` on planes without background.

        Parameters
        ----------
        target : Tensor
            Target, shape ``(B, C, D, H, W)``.
        dilated : Tensor
            Dilated mask ``m`` from :meth:`dilated_mask`, same shape.

        Returns
        -------
        Tensor
            Float32, shape of ``target``.
        """
        b, c = target.shape[:2]
        spatial = target.shape[2:]
        with torch.autocast(device_type=target.device.type, enabled=False):
            x = target.float()
            bg = 1.0 - dilated.float()
            count = bg.sum(dim=(-2, -1), keepdim=True)
            plane_mean = (x * bg).sum(dim=(-2, -1), keepdim=True) / count.clamp(min=1.0)
            flat = torch.where(count > 0, plane_mean, x)
            if self.sigma_lp is None:
                return flat
            sigmas = (self.sigma_lp_z, self.sigma_lp, self.sigma_lp)
            num = _gaussian_blur((x * bg).reshape(b * c, *spatial), sigmas).reshape(x.shape)
            den = _gaussian_blur(bg.reshape(b * c, *spatial), sigmas).reshape(x.shape)
            defined = den > _MIN_BACKGROUND_WEIGHT
            return torch.where(defined, num / den.clamp(min=_MIN_BACKGROUND_WEIGHT), flat)

    def forward(self, target: Tensor, fg_mask: Tensor) -> Tensor:
        """Return the target with its background replaced by ``B``.

        Parameters
        ----------
        target : Tensor
            Target, shape ``(B, C, D, H, W)``.
        fg_mask : Tensor
            Foreground mask, same shape; binarized at 0.5.

        Returns
        -------
        Tensor
            ``w * target + (1 - w) * B``, shape and dtype of ``target``.
        """
        if target.ndim != 5:
            raise ValueError(f"target must be (B, C, D, H, W), got shape {tuple(target.shape)}")
        if fg_mask.shape != target.shape:
            raise ValueError(f"fg_mask {tuple(fg_mask.shape)} must match target {tuple(target.shape)}")
        dilated = self.dilated_mask(fg_mask)
        w = self.blend_weight(dilated)
        x = target.float()
        return (w * x + (1.0 - w) * self.background(target, dilated)).to(target.dtype)
