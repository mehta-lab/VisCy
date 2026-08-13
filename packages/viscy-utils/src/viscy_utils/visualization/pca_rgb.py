"""DINO-style PCA-RGB visualization of dense patch features.

Pure tensor math — no model imports.  Given dense patch features
``(B, N, D)`` from any vision backbone (ViT patch tokens, CNN feature map
flattened over space), fits PCA, takes the top three components, and
paints each spatial location with the corresponding RGB triple.  Optional
foreground masking via the first principal component reproduces the
original DINO paper's "object-centric" overlay.

The technique works on any backbone that produces dense features —
ViT, ConvNeXt, MAE encoders — and is not tied to attention.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

PCAMode = Literal["per_image", "batch", "reference"]


def fit_pca_components(
    patches: Tensor,
    n_components: int = 3,
) -> tuple[Tensor, Tensor]:
    """Fit a PCA basis on a flat ``(M, D)`` patch matrix.

    Parameters
    ----------
    patches : Tensor
        ``(M, D)`` patch features.  Caller is responsible for pooling
        across whichever images / batches should share the basis.
    n_components : int
        Number of principal components to retain, by default ``3``.

    Returns
    -------
    mean : Tensor
        ``(D,)`` per-feature mean used to center new data.
    components : Tensor
        ``(D, n_components)`` orthonormal basis ordered by descending
        explained variance.
    """
    if patches.ndim != 2:
        raise ValueError(f"expected (M, D) patches, got shape {tuple(patches.shape)}")
    mean = patches.mean(dim=0)
    centered = patches - mean
    q = max(n_components + 4, n_components * 2)
    q = min(q, min(centered.shape))
    _, _, v = torch.pca_lowrank(centered, q=q, center=False)
    return mean, v[:, :n_components].contiguous()


def _project(patches: Tensor, mean: Tensor, components: Tensor) -> Tensor:
    """Project ``(B, N, D)`` patches to ``(B, N, K)`` PC scores."""
    return (patches - mean) @ components


def _minmax_normalize(
    scores: Tensor,
    dim: tuple[int, ...],
    clip_percentile: tuple[float, float] | None = None,
) -> Tensor:
    """Min-max scale ``scores`` to ``[0, 1]`` over the given dims.

    Parameters
    ----------
    scores : Tensor
    dim : tuple of int
        Dims to reduce when computing the per-channel min/max.
    clip_percentile : (low, high) in [0, 100] or None
        If given, use the ``low``/``high`` percentile across ``dim``
        instead of strict min/max, then clip the output to ``[0, 1]``.
        Robust to outlier patches.  ``None`` (default) uses strict
        ``amin``/``amax``.
    """
    if clip_percentile is None:
        lo = scores.amin(dim=dim, keepdim=True)
        hi = scores.amax(dim=dim, keepdim=True)
        return (scores - lo) / (hi - lo).clamp(min=1e-8)

    low_q, high_q = clip_percentile
    if not (0.0 <= low_q < high_q <= 100.0):
        raise ValueError(f"clip_percentile must be (low, high) with 0 <= low < high <= 100, got {clip_percentile}")
    # torch.quantile only collapses one dim at a time; iterate.
    lo = scores
    hi = scores
    for d in sorted(dim, reverse=True):
        lo = lo.quantile(low_q / 100.0, dim=d, keepdim=True)
        hi = hi.quantile(high_q / 100.0, dim=d, keepdim=True)
    return ((scores - lo) / (hi - lo).clamp(min=1e-8)).clamp(0.0, 1.0)


def _infer_grid_hw(n_patches: int) -> tuple[int, int]:
    """Infer ``(H, W)`` from a square patch count."""
    side = int(round(math.sqrt(n_patches)))
    if side * side != n_patches:
        raise ValueError(f"cannot infer square grid from {n_patches} patches; pass grid_hw explicitly")
    return side, side


def pca_rgb_from_patches(
    patches: Tensor,
    *,
    mode: PCAMode = "batch",
    mask_fg: bool = True,
    fg_threshold: float = 0.5,
    components: Tensor | None = None,
    mean: Tensor | None = None,
    grid_hw: tuple[int, int] | None = None,
    upsample_to: tuple[int, int] | None = None,
    bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    clip_percentile: tuple[float, float] | None = None,
    interp_mode: Literal["nearest", "bilinear"] = "nearest",
) -> Tensor:
    """Render DINO-style PCA-RGB overlays from dense patch features.

    Parameters
    ----------
    patches : Tensor
        ``(B, N, D)`` patch features.  ``N`` is the number of spatial
        locations (e.g. 196 for a ViT-L/16 at 224 px), ``D`` is the
        feature dimension.
    mode : {'per_image', 'batch', 'reference'}
        How the PCA basis is fit.

        - ``'per_image'`` — one PCA per image.  Colors are not comparable
          across images.
        - ``'batch'`` (default) — fit one PCA on patches pooled across
          the batch.  Colors are consistent within the batch.
        - ``'reference'`` — use the basis passed via ``components`` /
          ``mean``.  Colors are consistent across any data projected
          with the same basis.

    mask_fg : bool
        If ``True`` (default), patches whose first principal component
        is below ``fg_threshold`` (after min-max normalization to
        ``[0,1]``) are painted with ``bg_color`` instead of their RGB
        triple.  This reproduces the original DINO paper's foreground
        masking.
    fg_threshold : float
        Threshold on the normalized PC1 score, by default ``0.5``
        (the median).
    components, mean : Tensor or None
        ``(D, 3)`` basis and ``(D,)`` mean for ``mode='reference'``.
        Ignored otherwise.  Use :func:`fit_pca_components` to compute
        them on a held-out set.
    grid_hw : tuple[int, int] or None
        Spatial layout of the patches.  Inferred as ``(sqrt(N), sqrt(N))``
        when ``None``.
    upsample_to : tuple[int, int] or None
        If given, bilinearly upsample the RGB grid to ``(H, W)`` before
        returning.
    bg_color : tuple[float, float, float]
        RGB triple in ``[0, 1]`` used for masked-out background patches,
        by default white.
    clip_percentile : (low, high) in [0, 100] or None
        If given, clip PC scores to the ``low``/``high`` percentile per
        channel before scaling to ``[0, 1]``.  Robust to outlier patches
        (debris, FOV-edge artifacts) that would otherwise peg the RGB
        range.  ``None`` (default) uses strict min/max — fine for clean
        data, but a few outliers will wash everything else out.
    interp_mode : {'nearest', 'bilinear'}
        Upsampling kernel used when ``upsample_to`` is set.  Defaults to
        ``'nearest'`` so the output honestly reflects the model's patch
        grid (e.g. a 14×14 ViT-L/16 grid renders as crisp 16×16-pixel
        squares at 224 px output).  ``'bilinear'`` smooths the grid into
        a continuous heatmap, which is prettier but lies about spatial
        resolution.

    Returns
    -------
    Tensor
        ``(B, 3, H, W)`` RGB tensor in ``[0, 1]``.  ``H, W`` is
        ``upsample_to`` if given, otherwise ``grid_hw``.
    """
    if patches.ndim != 3:
        raise ValueError(f"expected (B, N, D) patches, got shape {tuple(patches.shape)}")
    b, n, d = patches.shape
    grid_h, grid_w = grid_hw if grid_hw is not None else _infer_grid_hw(n)
    if grid_h * grid_w != n:
        raise ValueError(f"grid_hw {grid_h}x{grid_w} = {grid_h * grid_w} != N={n}")

    if mode == "reference":
        if components is None or mean is None:
            raise ValueError("mode='reference' requires components and mean")
        if components.shape != (d, 3):
            raise ValueError(f"components must be (D=3, 3), got {tuple(components.shape)}")
        scores = _project(patches, mean.to(patches), components.to(patches))
    elif mode == "batch":
        flat = patches.reshape(b * n, d)
        m, comps = fit_pca_components(flat, n_components=3)
        scores = _project(patches, m, comps)
    elif mode == "per_image":
        scores_list = []
        for i in range(b):
            m, comps = fit_pca_components(patches[i], n_components=3)
            scores_list.append(_project(patches[i : i + 1], m, comps))
        scores = torch.cat(scores_list, dim=0)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    pc1 = _minmax_normalize(scores[..., 0:1], dim=(1,), clip_percentile=clip_percentile)
    rgb_norm_dim = (1,) if mode == "per_image" else (0, 1)
    rgb = _minmax_normalize(scores, dim=rgb_norm_dim, clip_percentile=clip_percentile)

    if mask_fg:
        keep = (pc1 >= fg_threshold).to(rgb)
        bg = torch.tensor(bg_color, device=rgb.device, dtype=rgb.dtype).view(1, 1, 3)
        rgb = keep * rgb + (1 - keep) * bg

    rgb_grid = rgb.transpose(1, 2).reshape(b, 3, grid_h, grid_w)
    if upsample_to is not None:
        if interp_mode == "nearest":
            rgb_grid = F.interpolate(rgb_grid, size=upsample_to, mode="nearest")
        elif interp_mode == "bilinear":
            rgb_grid = F.interpolate(rgb_grid, size=upsample_to, mode="bilinear", align_corners=False)
        else:
            raise ValueError(f"unknown interp_mode {interp_mode!r}; expected 'nearest' or 'bilinear'")
    return rgb_grid.clamp(0.0, 1.0)
