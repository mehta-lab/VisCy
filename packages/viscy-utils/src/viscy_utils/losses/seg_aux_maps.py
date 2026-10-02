"""Mask-derived maps for :class:`~viscy_utils.losses.SegAuxDice`.

- :func:`squared_edt`: exact, separable, anisotropic squared Euclidean distance
  transform in torch (no cupy/cucim needed), on CPU or GPU.
- :func:`sauna_weight_map`: the SAUNA combined boundary/thickness uncertainty
  map ("h" variant) of a binary mask, as a per-voxel Dice weight.
- :func:`soft_skeleton`: the clDice soft skeleton.

SAUNA: Oulu-IMEDS/SAUNA, ``mlpipeline/utils/generate_uncertainty_masks.py``
(``extract_boundary_uncertainty_map``, ``extract_thickness_uncertainty_map``
with ``target_c_label="h"``, ``extract_combined_uncertainty_map``).
clDice: jocpae/clDice, ``soft_skeleton.py``.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

__all__ = ["box_max", "sauna_weight_map", "soft_skeleton", "squared_edt"]

# Elements in one (lines, L, L) block of the brute-force EDT pass: 2**26 float32 = 256 MiB.
_EDT_BLOCK_ELEMENTS = 2**26


def _nearest_seed_1d_squared(seeds: Tensor, spacing: float, dim: int) -> Tensor:
    """Squared distance to the nearest seed along ``dim``, by two scans.

    Parameters
    ----------
    seeds : Tensor
        Boolean, any shape.
    spacing : float
        Physical size of one step along ``dim``.
    dim : int
        Axis to scan.

    Returns
    -------
    Tensor
        Float32 squared distance, shape of ``seeds``; ``inf`` on lines with no seed.
    """
    shape = [1] * seeds.ndim
    shape[dim] = seeds.shape[dim]
    idx = torch.arange(seeds.shape[dim], device=seeds.device, dtype=torch.float32).reshape(shape)
    left = torch.where(seeds, idx, float("-inf")).cummax(dim=dim).values
    right = torch.where(seeds, idx, float("inf")).flip(dim).cummin(dim=dim).values.flip(dim)
    dist = torch.minimum(idx - left, right - idx) * spacing
    return dist * dist


def _lower_envelope_1d(f: Tensor, spacing: float, band: int | None) -> Tensor:
    """``out[x] = min_y f[y] + (spacing * (x - y))**2`` along the last dim, brute force.

    Parameters
    ----------
    f : Tensor
        Float32, shape ``(M, L)``; may hold ``inf``.
    spacing : float
        Physical size of one step along the last dim.
    band : int or None
        Only consider ``|x - y| <= band``; ``None`` searches the whole line.

    Returns
    -------
    Tensor
        Shape ``(M, L)``.
    """
    m, length = f.shape
    out = torch.empty_like(f)
    if band is None or 2 * band + 1 >= length:
        pos = torch.arange(length, device=f.device, dtype=torch.float32) * spacing
        offsets = (pos[:, None] - pos[None, :]) ** 2  # (x, y)
        rows = max(1, _EDT_BLOCK_ELEMENTS // (length * length))
        for start in range(0, m, rows):
            out[start : start + rows] = (f[start : start + rows, None, :] + offsets).amin(dim=-1)
        return out
    steps = torch.arange(-band, band + 1, device=f.device, dtype=torch.float32) * spacing
    offsets = steps * steps  # (2 * band + 1,)
    rows = max(1, _EDT_BLOCK_ELEMENTS // (length * (2 * band + 1)))
    for start in range(0, m, rows):
        padded = F.pad(f[start : start + rows], (band, band), value=float("inf"))
        out[start : start + rows] = (padded.unfold(-1, 2 * band + 1, 1) + offsets).amin(dim=-1)
    return out


def _along_axis(x: Tensor, axis: int, fn) -> Tensor:
    """Apply a ``(M, L) -> (M, L)`` function along ``axis`` of ``x``."""
    moved = x.movedim(axis, -1)
    shape = moved.shape
    return fn(moved.reshape(-1, shape[-1])).reshape(shape).movedim(-1, axis)


def squared_edt(seeds: Tensor, spacing: Sequence[float], max_distance: float | None = None) -> Tensor:
    """Exact squared Euclidean distance from every voxel to the nearest seed.

    Separable (Saito-Toriwaki): an exact two-scan 1D pass along the largest
    axis, then a brute-force lower-envelope pass along each other axis, blocked
    to bound memory. Size-1 axes are skipped. Matches
    ``scipy.ndimage.distance_transform_edt(~seeds, sampling=spacing) ** 2``.

    With ``max_distance`` the result is ``min(d**2, max_distance**2)``, still
    exact: every value is capped at ``max_distance**2`` before each pass, so no
    minimizer that matters lies further than ``max_distance`` along any line,
    and each pass only searches that band.

    Parameters
    ----------
    seeds : Tensor
        Boolean, shape ``(N, *spatial)``; each of the ``N`` rows is independent.
    spacing : Sequence[float]
        Physical voxel size per spatial dim.
    max_distance : float or None
        Cap on the returned distance; ``None`` for none.

    Returns
    -------
    Tensor
        Float32 squared distances, shape of ``seeds``; 0 on seeds and, without
        a cap, ``inf`` everywhere in a row without any seed.
    """
    spatial = seeds.shape[1:]
    if len(spacing) != len(spatial):
        raise ValueError(f"spacing has {len(spacing)} entries for {len(spatial)} spatial dims")
    cap2 = None if max_distance is None else float(max_distance) ** 2
    axes = [a for a in range(len(spatial)) if spatial[a] > 1]
    first = max(axes, key=lambda a: spatial[a]) if axes else None
    if first is None:
        d2 = torch.where(seeds, 0.0, float("inf")).float()
    else:
        d2 = _nearest_seed_1d_squared(seeds, float(spacing[first]), first + 1)
    if cap2 is not None:
        d2 = d2.clamp(max=cap2)
    for a in axes:
        if a != first:
            band = None if max_distance is None else math.ceil(max_distance / spacing[a])
            d2 = _along_axis(d2, a + 1, lambda f, a=a, band=band: _lower_envelope_1d(f, float(spacing[a]), band))
    # Passes keep values exact where the true distance is within the cap and >= the cap elsewhere.
    return d2 if cap2 is None else d2.clamp(max=cap2)


def _axis_distance_bound(seeds: Tensor, spacing: Sequence[float]) -> Tensor:
    """Per-voxel upper bound on the distance to the nearest seed.

    The smallest distance to a seed along any single axis: 1D scans only, cheap.
    """
    bound2 = torch.full(seeds.shape, float("inf"), device=seeds.device)
    for a in range(seeds.ndim - 1):
        if seeds.shape[a + 1] > 1:
            d2 = _nearest_seed_1d_squared(seeds, float(spacing[a]), a + 1)
            bound2 = torch.minimum(bound2, d2)
    return bound2.sqrt()


def _window_max(x: Tensor, k: int, dim: int) -> Tensor:
    """Max over a centred odd window ``k`` along ``dim``, ``-inf`` beyond the edges.

    Doubling (sparse-table) maxima: ``log2(k)`` elementwise passes, no pooling
    indices. Equal to ``max_pool1d(k, stride=1, padding=k // 2)`` along ``dim``.
    """
    length = x.shape[dim]
    pad = [0, 0] * (x.ndim - 1 - dim) + [k // 2, k // 2]
    m = F.pad(x, pad, value=float("-inf"))  # length + k - 1 along dim
    span = 1
    while 2 * span <= k:
        n = m.shape[dim] - span
        m = torch.maximum(m.narrow(dim, 0, n), m.narrow(dim, span, n))
        span *= 2
    # m[j] = max over [j, j + span); combine two overlapping spans to cover k.
    return torch.maximum(m.narrow(dim, 0, length), m.narrow(dim, k - span, length))


def box_max(x: Tensor, windows: Sequence[int]) -> Tensor:
    """Box max filter, stride 1, same size, as sequential 1D window maxima.

    Parameters
    ----------
    x : Tensor
        Float, shape ``(N, *spatial)``.
    windows : Sequence[int]
        Odd window length per spatial dim; 1 leaves that dim alone.

    Returns
    -------
    Tensor
        Filtered tensor, shape of ``x``.
    """
    for a, k in enumerate(windows):
        length = x.shape[a + 1]
        # A window wider than 2L - 1 already covers the whole line from every voxel.
        k = min(k, 2 * length - 1)
        if k <= 1:
            continue
        x = _window_max(x, k, a + 1)
    return x


def _odd_windows(radius: float, spacing: Sequence[float]) -> list[int]:
    """SAUNA's ``do_max_pooling`` kernel per axis: ``ceil(radius)`` voxels, made odd."""
    windows = []
    for s in spacing:
        k = math.ceil(radius / s)
        windows.append(k + 1 if k % 2 == 0 else k)
    return windows


def sauna_weight_map(mask: Tensor, spacing: Sequence[float]) -> Tensor:
    """Per-voxel Dice weight ``|y~|`` from SAUNA's combined "h" uncertainty map.

    With ``fg_dist`` the distance of each voxel to the nearest background voxel
    and ``bg_dist`` to the nearest foreground voxel (physical units),
    ``fg_max = max(fg_dist)``:

    - boundary map ``gt_b = (fg_dist - min(bg_dist, fg_max)) / fg_max``;
    - thickness map ``gt_t``: box max of ``fg_dist`` over a window of
      ``fg_max`` per axis, over ``fg_max`` on foreground; box max of
      ``min(bg_dist, fg_max)`` (window from that map's own maximum, as SAUNA
      does) over ``fg_max``, clipped to [0, 1], on background;
    - ``y~ = clip(gt_b + (1 - gt_t), -1, 1)`` on foreground and
      ``clip(gt_b - (1 - gt_t), -1, 1)`` on background.

    Rows with no foreground or no background get weight 1.

    Parameters
    ----------
    mask : Tensor
        Boolean foreground, shape ``(N, *spatial)``.
    spacing : Sequence[float]
        Physical voxel size per spatial dim.

    Returns
    -------
    Tensor
        Float32 weights in [0, 1], shape of ``mask``.
    """
    y = sauna_signed_map(mask, spacing)
    ok = _has_fg_and_bg(mask)
    return torch.where(_expand_rows(ok, mask), y.abs(), torch.ones_like(y))


def _has_fg_and_bg(mask: Tensor) -> Tensor:
    flat = mask.reshape(mask.shape[0], -1)
    return flat.any(-1) & (~flat).any(-1)


def _expand_rows(row_values: Tensor, like: Tensor) -> Tensor:
    return row_values.reshape(-1, *([1] * (like.ndim - 1)))


def sauna_signed_map(mask: Tensor, spacing: Sequence[float]) -> Tensor:
    """SAUNA's signed combined map ``y~`` in [-1, 1] (see :func:`sauna_weight_map`).

    Parameters
    ----------
    mask : Tensor
        Boolean foreground, shape ``(N, *spatial)``.
    spacing : Sequence[float]
        Physical voxel size per spatial dim.

    Returns
    -------
    Tensor
        Float32, shape of ``mask``; 0 in rows with no foreground or no background.
    """
    n = mask.shape[0]
    ok_rows = _has_fg_and_bg(mask)
    ok = _expand_rows(ok_rows, mask)
    if not ok_rows.any():
        return torch.zeros(mask.shape, device=mask.device)
    # Both EDTs only need to be exact up to the largest foreground distance:
    # fg_dist never exceeds it and bg_dist is clipped to it. The fg cap is a
    # cheap upper bound on fg_max, taken over rows that have background.
    fg_bound = _axis_distance_bound(~mask, spacing).reshape(n, -1).amax(-1)[ok_rows].amax().item()
    fg_cap = fg_bound if math.isfinite(fg_bound) else None
    # Rows lacking fg or bg have inf / capped distances; zero them so everything stays finite.
    fg_dist = torch.where(ok, squared_edt(~mask, spacing, max_distance=fg_cap).sqrt(), 0.0)
    fg_max = fg_dist.reshape(n, -1).amax(-1)
    fg_max = torch.where(fg_max > 0, fg_max, torch.ones_like(fg_max))
    bg_dist = torch.where(ok, squared_edt(mask, spacing, max_distance=fg_max.max().item()).sqrt(), 0.0)
    bg_clip = torch.minimum(bg_dist, _expand_rows(fg_max, mask))
    bg_max = bg_clip.reshape(n, -1).amax(-1)
    # Windows differ per row (each patch has its own fg_max): filter row by row.
    fg_pool = torch.empty_like(fg_dist)
    bg_pool = torch.empty_like(bg_clip)
    for i, (fg_r, bg_r) in enumerate(zip(fg_max.tolist(), bg_max.tolist())):
        fg_pool[i : i + 1] = box_max(fg_dist[i : i + 1], _odd_windows(fg_r, spacing))
        bg_pool[i : i + 1] = box_max(bg_clip[i : i + 1], _odd_windows(bg_r, spacing))
    fg_max_b = _expand_rows(fg_max, mask)
    gt_b = (fg_dist - bg_clip) / fg_max_b
    gt_t = torch.where(mask, fg_pool / fg_max_b, (bg_pool / fg_max_b).clamp(0.0, 1.0))
    y = torch.where(mask, gt_b + (1.0 - gt_t), gt_b - (1.0 - gt_t)).clamp(-1.0, 1.0)
    return torch.where(ok, y, 0.0)


def _max_pool(x: Tensor, kernel: tuple[int, ...]) -> Tensor:
    """Stride-1, same-size max-pool of ``(N, 1, *spatial)`` over 1-3 spatial dims."""
    pool = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}[len(kernel)]
    return pool(x, kernel, stride=1, padding=tuple(k // 2 for k in kernel))


def _soft_erode(x: Tensor) -> Tensor:
    """ClDice ``soft_erode``: min of the 1D 3-voxel min-pools along each axis (the cross)."""
    nd = x.ndim - 2
    out = None
    for a in range(nd):
        kernel = tuple(3 if i == a else 1 for i in range(nd))
        eroded = -_max_pool(-x, kernel)
        out = eroded if out is None else torch.minimum(out, eroded)
    return out


def _soft_dilate(x: Tensor) -> Tensor:
    """ClDice ``soft_dilate``: 3^n box max-pool."""
    return _max_pool(x, (3,) * (x.ndim - 2))


def _soft_open(x: Tensor) -> Tensor:
    return _soft_dilate(_soft_erode(x))


def _skeleton_step(img: Tensor, skel: Tensor) -> tuple[Tensor, Tensor]:
    img = _soft_erode(img)
    delta = F.relu(img - _soft_open(img))
    return img, skel + F.relu(delta - skel * delta)


def soft_skeleton(img: Tensor, iters: int) -> Tensor:
    """ClDice soft skeleton (jocpae/clDice ``soft_skel``), in 1-3 spatial dims.

    When a gradient is needed every iteration is activation-checkpointed: each
    one holds ~a dozen full-size intermediates (pool values and int64 indices),
    which for a
    ``(8, 1, 32, 384, 384)`` batch and 10 iterations exceeded a 48 GB GPU.
    Checkpointing keeps two tensors per iteration and recomputes the rest in
    backward; values and gradients are unchanged by it.

    Parameters
    ----------
    img : Tensor
        Soft mask in [0, 1], shape ``(N, 1, *spatial)``.
    iters : int
        Erosion iterations; should exceed the largest foreground radius in voxels.

    Returns
    -------
    Tensor
        Soft skeleton, shape of ``img``.
    """
    checkpointed = img.requires_grad and torch.is_grad_enabled()
    skel = F.relu(img - _soft_open(img))
    for _ in range(iters):
        if checkpointed:
            img, skel = checkpoint(_skeleton_step, img, skel, use_reentrant=False)
        else:
            img, skel = _skeleton_step(img, skel)
    return skel
