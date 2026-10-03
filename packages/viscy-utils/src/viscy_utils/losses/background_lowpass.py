"""Background low-pass of a training target, foreground kept at full resolution.

``x' = w * x + (1 - w) * G_lp(x)`` with ``w = clamp(G_f(dilate(fg_mask, r)), 0, 1)``:
inside the dilated foreground the target is unchanged, in the background it is a
Gaussian low-pass of itself, and a Gaussian feather of the dilated mask blends
the two without a seam.

It changes the target, never the loss: the background stays supervised toward a
smooth field. In flow matching, masking the background out of the loss would
leave the initial Gaussian noise there unremoved (the Spotlight v1 collapse).

The Gaussians are normalized, edge-replicated (``scipy.ndimage`` ``mode="nearest"``,
``truncate=4.0``) and separable, applied as shifted multiply-adds in float32, so
no convolution backend (cuDNN TF32, autocast) can lower their precision. A
constant field passes through ``G_lp`` unchanged, so the low-pass keeps the local
mean and the op needs no change to the target's normalization.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["BackgroundLowPass"]


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
    """Low-pass a target outside its (dilated, feathered) foreground mask.

    ``x' = w * x + (1 - w) * G_lp(x)``, ``w = clamp(G_f(dilate(fg_mask, r)), 0, 1)``.
    Where ``w == 1`` (the dilated mask shrunk by the feather's radius,
    ``int(4 * sigma_feather + 0.5)``) ``x'`` equals ``x`` up to float32 rounding
    of the kernel sum; where ``w == 0`` it is ``G_lp(x)``.

    By default every operation acts in XY, per Z plane: Z is anisotropic, and
    2D models have ``D = 1``. The ``*_z`` parameters extend each to Z.

    Computed in float32 regardless of autocast; returned in the target's dtype.

    Parameters
    ----------
    sigma_lp : float
        Background low-pass Gaussian sigma in XY, in voxels (> 0).
    sigma_feather : float
        Gaussian sigma in XY, in voxels, that feathers the dilated mask into
        ``w`` (>= 0; 0 keeps the hard dilated mask).
    dilate_radius : int
        Dilation radius in XY, in voxels (>= 0): a ``(2r + 1)``-wide square.
    sigma_lp_z : float
        Low-pass sigma along Z, in voxels (>= 0; 0 = per plane).
    sigma_feather_z : float
        Feather sigma along Z, in voxels (>= 0; 0 = per plane).
    dilate_radius_z : int
        Dilation radius along Z, in voxels (>= 0; 0 = per plane).
    """

    def __init__(
        self,
        sigma_lp: float,
        sigma_feather: float,
        dilate_radius: int,
        sigma_lp_z: float = 0.0,
        sigma_feather_z: float = 0.0,
        dilate_radius_z: int = 0,
    ) -> None:
        super().__init__()
        if sigma_lp <= 0:
            raise ValueError(f"sigma_lp must be > 0, got {sigma_lp}")
        for name, value in (
            ("sigma_feather", sigma_feather),
            ("sigma_lp_z", sigma_lp_z),
            ("sigma_feather_z", sigma_feather_z),
            ("dilate_radius", dilate_radius),
            ("dilate_radius_z", dilate_radius_z),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        self.sigma_lp = float(sigma_lp)
        self.sigma_feather = float(sigma_feather)
        self.dilate_radius = int(dilate_radius)
        self.sigma_lp_z = float(sigma_lp_z)
        self.sigma_feather_z = float(sigma_feather_z)
        self.dilate_radius_z = int(dilate_radius_z)

    def lowpass(self, target: Tensor) -> Tensor:
        """Return ``G_lp(target)`` in float32.

        Parameters
        ----------
        target : Tensor
            Shape ``(B, C, D, H, W)``.

        Returns
        -------
        Tensor
            Float32, shape of ``target``.
        """
        b, c = target.shape[:2]
        with torch.autocast(device_type=target.device.type, enabled=False):
            flat = target.float().reshape(b * c, *target.shape[2:])
            sigmas = (self.sigma_lp_z, self.sigma_lp, self.sigma_lp)
            return _gaussian_blur(flat, sigmas).reshape(target.shape)

    def blend_weight(self, fg_mask: Tensor) -> Tensor:
        """Return ``w = clamp(G_f(dilate(fg_mask > 0.5, r)), 0, 1)`` in float32.

        Parameters
        ----------
        fg_mask : Tensor
            Foreground mask, shape ``(B, C, D, H, W)``; binarized at 0.5.

        Returns
        -------
        Tensor
            Float32 weights in [0, 1], shape of ``fg_mask``.
        """
        b, c = fg_mask.shape[:2]
        rz, r = self.dilate_radius_z, self.dilate_radius
        with torch.autocast(device_type=fg_mask.device.type, enabled=False):
            mask = (fg_mask > 0.5).float().reshape(b * c, 1, *fg_mask.shape[2:])
            if rz or r:
                # max_pool pads with -inf, so the border never dilates the mask.
                mask = F.max_pool3d(mask, (2 * rz + 1, 2 * r + 1, 2 * r + 1), stride=1, padding=(rz, r, r))
            sigmas = (self.sigma_feather_z, self.sigma_feather, self.sigma_feather)
            w = _gaussian_blur(mask.squeeze(1), sigmas).clamp(0.0, 1.0)
            return w.reshape(fg_mask.shape)

    def forward(self, target: Tensor, fg_mask: Tensor) -> Tensor:
        """Return the target with its background low-passed.

        Parameters
        ----------
        target : Tensor
            Target, shape ``(B, C, D, H, W)``.
        fg_mask : Tensor
            Foreground mask, same shape; binarized at 0.5.

        Returns
        -------
        Tensor
            ``w * target + (1 - w) * G_lp(target)``, shape and dtype of ``target``.
        """
        if target.ndim != 5:
            raise ValueError(f"target must be (B, C, D, H, W), got shape {tuple(target.shape)}")
        if fg_mask.shape != target.shape:
            raise ValueError(f"fg_mask {tuple(fg_mask.shape)} must match target {tuple(target.shape)}")
        w = self.blend_weight(fg_mask)
        x = target.float()
        return (w * x + (1.0 - w) * self.lowpass(target)).to(target.dtype)
