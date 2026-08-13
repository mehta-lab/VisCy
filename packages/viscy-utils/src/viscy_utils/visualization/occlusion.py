"""Occlusion-based saliency for any frozen embedding model.

Pure tensor math — no model imports.  Given a callable that maps
``(B, C, H, W)`` images to ``(B, D)`` embeddings, slides a fill patch
across each image and measures how the embedding changes.  Regions
where occlusion shifts the embedding the most are the regions the
model "relies on" for that cell's representation.

Unlike PCA-RGB, the spatial resolution of the output is set by the
occlusion *stride* — not the model's patch grid.  This makes occlusion
the right tool for backbones like ConvNeXt where the native feature
map is coarse (e.g. 5×5).

Cost: one forward pass per occlusion position.  At ``stride=8`` on a
160×160 image that's 400 forwards per image — batchable, and cheap
relative to a full evaluation run.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


def _grid_positions(image_size: int, occ_size: int, stride: int) -> list[int]:
    """Return top-left occlusion positions covering ``image_size``."""
    if occ_size <= 0 or stride <= 0:
        raise ValueError(f"occ_size and stride must be positive, got {occ_size}, {stride}")
    if occ_size > image_size:
        raise ValueError(f"occ_size {occ_size} > image_size {image_size}")
    last = max(image_size - occ_size, 0)
    positions = list(range(0, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    return positions


def _gaussian_blur(images: Tensor, ksize: int, sigma: float) -> Tensor:
    """Depthwise separable Gaussian blur of ``(B, C, H, W)`` images (reflect pad)."""
    device, dtype = images.device, images.dtype
    coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    k1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    k1d = k1d / k1d.sum()
    c = images.shape[1]
    pad = ksize // 2
    kx = k1d.view(1, 1, 1, ksize).expand(c, 1, 1, ksize)
    ky = k1d.view(1, 1, ksize, 1).expand(c, 1, ksize, 1)
    x = F.pad(images, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=c)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, ky, groups=c)
    return x


@torch.no_grad()
def occlusion_saliency(
    images: Tensor,
    embed_fn: Callable[[Tensor], Tensor],
    *,
    occ_size: int = 16,
    stride: int = 8,
    fill_value: float | str = 0.0,
    batch_size: int = 32,
    distance: str = "l2",
) -> Tensor:
    """Compute occlusion-based saliency per image.

    Parameters
    ----------
    images : Tensor
        ``(B, C, H, W)`` input.  All occlusions for a single image
        share its baseline embedding.
    embed_fn : callable
        Function that maps ``(N, C, H, W)`` to ``(N, D)`` global
        embeddings.  Must work for arbitrary batch sizes.
    occ_size : int
        Side length of the square occlusion window in input pixels.
    stride : int
        Step between occlusion positions.  Smaller stride = denser
        saliency map but more forward passes.
    fill_value : float or {'mean', 'zero', 'blur'}
        How to fill the occluded square.  ``'mean'`` uses the per-image
        mean (best for z-scored input where ``0`` is already the FOV
        mean).  ``'zero'`` is a constant 0.0.  A float is used verbatim.
        ``'blur'`` replaces the square with a Gaussian-blurred copy of
        the *same region* (a spatially varying fill), which preserves
        gross shape / signed density while destroying fine texture —
        the faithful occluder for texture/density models (e.g. phase),
        with no hard-edged out-of-distribution flat square.
    batch_size : int
        How many occluded copies to forward through ``embed_fn`` at
        once.  Lower if you OOM.
    distance : {'l2', 'cosine'}
        How to measure the embedding shift.  ``'l2'`` (default) returns
        Euclidean distance; ``'cosine'`` returns ``1 - cos_sim``.

    Returns
    -------
    Tensor
        ``(B, H_out, W_out)`` saliency map at the occlusion-stride
        resolution.  Values are non-negative; higher = larger embedding
        shift = more important region.
    """
    if images.ndim != 4:
        raise ValueError(f"expected (B, C, H, W) images, got shape {tuple(images.shape)}")

    device = images.device
    b, c, h, w = images.shape
    ys = _grid_positions(h, occ_size, stride)
    xs = _grid_positions(w, occ_size, stride)
    n_pos = len(ys) * len(xs)

    baseline = embed_fn(images)
    if baseline.shape[0] != b:
        raise RuntimeError(f"embed_fn returned batch {baseline.shape[0]} for input batch {b}")

    saliency = torch.zeros(b, len(ys), len(xs), device=device, dtype=baseline.dtype)

    # ``blurred`` is a full-image Gaussian-blurred copy used for the spatially
    # varying 'blur' fill; ``fill`` is the broadcast scalar used otherwise.
    blurred: Tensor | None = None
    if isinstance(fill_value, str):
        if fill_value == "mean":
            fill = images.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        elif fill_value == "zero":
            fill = torch.zeros(b, c, 1, 1, device=device, dtype=images.dtype)
        elif fill_value == "blur":
            # kernel/sigma scale with the occluder so the patch is fully smoothed.
            ksize = max(3, (occ_size // 2) * 2 + 1)
            sigma = occ_size / 3.0
            blurred = _gaussian_blur(images, ksize, sigma)
            fill = torch.zeros(b, c, 1, 1, device=device, dtype=images.dtype)  # unused
        else:
            raise ValueError(f"fill_value string must be 'mean', 'zero', or 'blur', got {fill_value!r}")
    else:
        fill = torch.full((b, c, 1, 1), float(fill_value), device=device, dtype=images.dtype)

    # Iterate one image at a time so we can batch occlusion positions
    # without materializing (B * n_pos, C, H, W).
    for i in range(b):
        img = images[i : i + 1]
        base = baseline[i : i + 1]
        fill_i = fill[i : i + 1]
        blurred_i = blurred[i : i + 1] if blurred is not None else None

        # Build all occluded copies for this image: (n_pos, C, H, W).
        flat_positions = [(yy, xx) for yy in ys for xx in xs]
        for start in range(0, n_pos, batch_size):
            chunk = flat_positions[start : start + batch_size]
            occluded = img.expand(len(chunk), -1, -1, -1).clone()
            for k, (yy, xx) in enumerate(chunk):
                if blurred_i is not None:
                    occluded[k, :, yy : yy + occ_size, xx : xx + occ_size] = blurred_i[
                        :, :, yy : yy + occ_size, xx : xx + occ_size
                    ]
                else:
                    occluded[k, :, yy : yy + occ_size, xx : xx + occ_size] = fill_i
            occ_emb = embed_fn(occluded)

            if distance == "l2":
                d = (occ_emb - base).norm(dim=-1)
            elif distance == "cosine":
                d = 1.0 - F.cosine_similarity(occ_emb, base.expand_as(occ_emb), dim=-1)
            elif distance == "signed_delta":
                # Signed difference for scalar-output embed_fns (typically
                # a single-class probability). Positive = occlusion *raises*
                # the score, negative = occlusion *drops* it. Pair with a
                # diverging colormap centred at 0 (e.g. icefire).
                if occ_emb.ndim != 2 or occ_emb.shape[-1] != 1:
                    raise ValueError(
                        f"distance='signed_delta' requires (B, 1) output from embed_fn, got {tuple(occ_emb.shape)}"
                    )
                d = (occ_emb - base).squeeze(-1)
            else:
                raise ValueError(f"unknown distance {distance!r}")

            for k, (yy, xx) in enumerate(chunk):
                yi = yy // stride
                xi = xx // stride
                saliency[i, yi, xi] = d[k]

    return saliency


def saliency_to_rgb(
    saliency: Tensor,
    *,
    cmap: str = "magma",
    clip_percentile: tuple[float, float] | None = (2.0, 98.0),
    clip_value: float | None = None,
    upsample_to: tuple[int, int] | None = None,
    interp_mode: str = "nearest",
) -> Tensor:
    """Render a ``(B, H, W)`` saliency map as ``(B, 3, H_out, W_out)`` RGB.

    Parameters
    ----------
    saliency : Tensor
        ``(B, H, W)`` saliency.  May be signed (e.g. signed_delta
        attribution) — pair with a diverging cmap and ``clip_value``.
    cmap : str
        Any matplotlib or seaborn colormap name.  ``'icefire'`` is
        recommended for signed attribution; ``'magma'`` for unsigned
        magnitude.
    clip_percentile : (low, high) or None
        Per-image percentile clipping → ``[0, 1]`` rescale.  Used when
        ``clip_value`` is None.  Default ``(2, 98)``.
    clip_value : float or None
        If given, *fixed* symmetric clipping to ``[-clip_value,
        clip_value]`` mapped to ``[0, 1]``.  Use this for signed
        attribution so zero stays at the colormap centre and the same
        scale applies across frames — colors are then comparable
        across batch / time.  Overrides ``clip_percentile``.
    upsample_to : tuple[int, int] or None
        If given, resize the colored map to ``(H, W)`` via the chosen
        ``interp_mode``.
    interp_mode : {'nearest', 'bilinear'}
        Upsampling kernel.  Default ``'nearest'`` keeps occlusion-grid
        cells crisp.

    Returns
    -------
    Tensor
        ``(B, 3, H_out, W_out)`` float tensor in ``[0, 1]``.
    """
    import matplotlib

    if saliency.ndim != 3:
        raise ValueError(f"expected (B, H, W), got {tuple(saliency.shape)}")

    sal = saliency.detach().cpu().float()
    b = sal.shape[0]
    out = []
    try:
        cm = matplotlib.colormaps.get_cmap(cmap)
    except (ValueError, KeyError):
        # Fall back to seaborn for names matplotlib doesn't ship (e.g. icefire).
        import seaborn as sns

        cm = sns.color_palette(cmap, as_cmap=True)
    for i in range(b):
        s = sal[i]
        if clip_value is not None:
            cv = float(clip_value)
            s = ((s.clamp(-cv, cv) + cv) / (2 * cv)).clamp(0.0, 1.0)
        elif clip_percentile is not None:
            lo = s.quantile(clip_percentile[0] / 100.0)
            hi = s.quantile(clip_percentile[1] / 100.0)
            s = ((s - lo) / (hi - lo).clamp(min=1e-8)).clamp(0.0, 1.0)
        else:
            lo, hi = s.min(), s.max()
            s = ((s - lo) / (hi - lo).clamp(min=1e-8)).clamp(0.0, 1.0)
        rgba = cm(s.numpy())  # (H, W, 4)
        out.append(torch.from_numpy(rgba[..., :3]).permute(2, 0, 1))
    rgb = torch.stack(out, dim=0).to(saliency.device)

    if upsample_to is not None:
        if interp_mode == "nearest":
            rgb = F.interpolate(rgb, size=upsample_to, mode="nearest")
        elif interp_mode == "bilinear":
            rgb = F.interpolate(rgb, size=upsample_to, mode="bilinear", align_corners=False)
        else:
            raise ValueError(f"unknown interp_mode {interp_mode!r}")
    return rgb.clamp(0.0, 1.0)
