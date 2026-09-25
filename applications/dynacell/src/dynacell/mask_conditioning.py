"""Foreground-mask helpers for CellDiff's mask-aware variants (Spotlight v2 Stage 1b).

- :func:`encode_mask` maps a binary mask to CellDiff's target range, so a mask
  channel lives in the same ``[-1, 1]`` units as the ``MinMaxSampled`` target.
- :class:`MaskCorruption` degrades a ground-truth mask on the GPU during
  C-cond training, so the model learns to tolerate the imperfect masks it is
  conditioned on at test time.
- :class:`CondMaskSource` supplies that test-time mask: it reads a thresholded
  channel of another store (e.g. an FNet ``_segaux`` prediction) aligned with
  each predict window.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from iohub.ngff import open_ome_zarr
from torch import Tensor

__all__ = ["CondMaskSource", "MaskCorruption", "binarize_mask", "encode_mask"]


def binarize_mask(fg_mask: Tensor) -> Tensor:
    """Binarize a (possibly interpolated) foreground mask at 0.5, keeping its dtype."""
    return (fg_mask > 0.5).to(fg_mask.dtype)


def encode_mask(mask: Tensor) -> Tensor:
    """Map a binary ``{0, 1}`` mask to ``{-1, 1}``, CellDiff's target range."""
    return 2.0 * mask - 1.0


class MaskCorruption:
    """Random GPU corruption of a binary foreground mask, applied per sample.

    In order: a YX erosion or dilation (equal odds) by a radius drawn from
    ``[min_radius, max_radius]`` px; dropping of random blobs; and, with
    probability ``blank_prob``, replacing the whole mask by background.

    Blob dropping approximates "drop a fraction of the instances" without
    connected-component labelling, which has no cheap GPU implementation in
    torch: a coarse random field with ``drop_cell_size`` px cells (random
    grid offset, spanning all of Z) removes each cell with probability
    ``drop_fraction``. At the default 64 px a cell is about one iPSC nucleus,
    so it mostly removes whole objects, but it can also cut one at a cell
    edge, a partial miss that a thresholded prediction also produces.

    Morphology is YX-only (``(1, k, k)`` kernels), so it means the same for
    the 2D (``Z=1``) and 3D models and does not erode anisotropic Z.

    Parameters
    ----------
    min_radius : int
        Smallest erode/dilate radius in px.
    max_radius : int
        Largest erode/dilate radius in px.
    drop_fraction : float
        Probability that each coarse cell is dropped.
    drop_cell_size : int
        YX side of a coarse cell in px.
    blank_prob : float
        Probability that a sample's mask is replaced by all-background.
    """

    def __init__(
        self,
        min_radius: int = 1,
        max_radius: int = 3,
        drop_fraction: float = 0.3,
        drop_cell_size: int = 64,
        blank_prob: float = 0.1,
    ) -> None:
        if not 0 <= min_radius <= max_radius:
            raise ValueError(f"need 0 <= min_radius <= max_radius, got {min_radius}, {max_radius}")
        if not 0.0 <= drop_fraction <= 1.0:
            raise ValueError(f"drop_fraction must be in [0, 1], got {drop_fraction}")
        if drop_cell_size < 1:
            raise ValueError(f"drop_cell_size must be >= 1, got {drop_cell_size}")
        if not 0.0 <= blank_prob <= 1.0:
            raise ValueError(f"blank_prob must be in [0, 1], got {blank_prob}")
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.drop_fraction = drop_fraction
        self.drop_cell_size = drop_cell_size
        self.blank_prob = blank_prob

    def _morph(self, mask: Tensor) -> Tensor:
        r = int(torch.randint(self.min_radius, self.max_radius + 1, ()))
        if r == 0:
            return mask
        kernel, pad = (1, 2 * r + 1, 2 * r + 1), (0, r, r)
        if torch.rand(()) < 0.5:
            return F.max_pool3d(mask, kernel, stride=1, padding=pad)
        # Erosion; max_pool pads with -inf, so borders are not eroded.
        return -F.max_pool3d(-mask, kernel, stride=1, padding=pad)

    def _drop(self, mask: Tensor) -> Tensor:
        h, w = mask.shape[-2:]
        cell = self.drop_cell_size
        gh, gw = h // cell + 2, w // cell + 2
        keep = (torch.rand(gh, gw, device=mask.device) >= self.drop_fraction).to(mask.dtype)
        keep = keep.repeat_interleave(cell, 0).repeat_interleave(cell, 1)
        oy, ox = (int(o) for o in torch.randint(0, cell, (2,)))
        return mask * keep[oy : oy + h, ox : ox + w]

    def __call__(self, mask: Tensor) -> Tensor:
        """Corrupt each sample of a binary mask independently.

        Parameters
        ----------
        mask : Tensor
            Binary mask ``(B, C, D, H, W)`` with values in ``{0, 1}``.

        Returns
        -------
        Tensor
            Corrupted binary mask, same shape and dtype.
        """
        if mask.dim() != 5:
            raise ValueError(f"mask must be (B, C, D, H, W), got shape {tuple(mask.shape)}")
        out = []
        for sample in mask.split(1):
            sample = self._morph(sample)
            if self.drop_fraction > 0:
                sample = self._drop(sample)
            if torch.rand(()) < self.blank_prob:
                sample = torch.zeros_like(sample)
            out.append(sample)
        return torch.cat(out)


class CondMaskSource:
    """Predict-time conditioning mask read from another OME-Zarr HCS store.

    The store must share the predict store's position layout and image shape
    (true of a prediction store written by ``HCSPredictionWriter`` from the
    same predict set). Each predict window ``(position, t, z)`` is read at the
    same place, so the mask stays pixel-aligned with the phase input: predict
    applies no spatial transform.

    Parameters
    ----------
    data_path : str
        Path to the HCS store holding the mask source, e.g. an FNet
        ``_segaux`` ``prediction.zarr``.
    channel : str
        Channel to threshold, e.g. ``"Nuclei_prediction"``.
    threshold : float
        Voxels ``>= threshold`` are foreground, in the channel's own units.
    """

    def __init__(self, data_path: str, channel: str, threshold: float) -> None:
        self.data_path = Path(data_path)
        self.channel = channel
        self.threshold = threshold

    def read(self, index: tuple, spatial: tuple[int, int, int]) -> Tensor:
        """Read the binary mask windows of a predict batch.

        Parameters
        ----------
        index : tuple
            The batch's ``index``: image paths (``"/row/col/pos/array"``),
            time indices and Z start indices, one per sample.
        spatial : tuple of int
            Window ``(Z, Y, X)`` of the batch's source.

        Returns
        -------
        Tensor
            Float32 mask ``(B, 1, Z, Y, X)`` with values in ``{0, 1}``.

        Raises
        ------
        ValueError
            If the source image's YX shape differs from the window or the
            window holds NaN (an unwritten prediction).
        """
        img_names, t_indices, z_indices = index
        depth = spatial[0]
        windows = []
        for img_name, t, z in zip(img_names, t_indices, z_indices):
            _, row, col, pos, arr = img_name.split("/")
            with open_ome_zarr(self.data_path / row / col / pos, mode="r") as position:
                ch = position.get_channel_index(self.channel)
                image = position[arr]
                window = np.asarray(image[int(t), ch, int(z) : int(z) + depth])
            if window.shape != tuple(spatial):
                raise ValueError(
                    f"{self.data_path}/{row}/{col}/{pos} window at t={int(t)}, z={int(z)} has shape "
                    f"{window.shape}, expected {tuple(spatial)}: the mask store must match the predict store."
                )
            if np.isnan(window).any():
                raise ValueError(
                    f"{self.data_path}/{row}/{col}/{pos} channel {self.channel!r} holds NaN at t={int(t)}, "
                    f"z={int(z)}; the mask source is incomplete."
                )
            windows.append(torch.from_numpy(window >= self.threshold).float())
        return torch.stack(windows).unsqueeze(1)
