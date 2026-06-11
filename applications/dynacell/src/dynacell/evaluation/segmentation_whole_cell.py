"""GPU whole-cell instance segmentation: nuclei seeds + membrane EDT watershed.

Productionizes the ``.tmp/wholecell_*_cellgrid.py`` prototypes. Given a membrane
fluorescence image, the matching nucleus fluorescence, and pre-computed nucleus
instance labels (the watershed seeds), it segments whole cells and returns
cytoplasm-only instance labels:

1. robust-clip + CLAHE the membrane and nucleus channels (GPU, via ``cubic``);
2. (optional) downscale both to an isotropic ``cell_voxel_um`` working grid and
   NN-resize the seeds to match — the morphological size parameters are physical
   (um / um^2 / um^3) and converted to working-grid pixels so they are
   grid-independent;
3. build a solid cell mask: grayscale-close ``clip(membrane + nucleus)`` per XY
   plane, threshold at the lower multi-Otsu boundary and fill holes (tissue);
   subtract membrane "walls" (upper multi-Otsu boundary of a blurred membrane);
   union the seed footprint so every nucleus is interior;
4. marker-controlled EDT watershed (``cubic.segment_watershed``) seeded by the
   nucleus labels — IDs are preserved;
5. drop sub-``min_cell`` cells, sequential-relabel, NN-upscale to native;
6. carve the nucleus footprint out (``cells[seed > 0] = 0``) so metrics score the
   cytoplasmic shell only.

Supports a single 3-D ``(Z, Y, X)`` volume and a single 2-D ``(Y, X)`` slice; the
branch is selected by ``memb_img.ndim``. The 3-D path runs the recipe on the full
volume (the per-XY-plane closing loops over Z) — it never segments per-slice and
stitches.
"""

import numpy as np
from cubic.cuda import ascupy, asnumpy, get_array_module
from cubic.image_utils import rescale_isotropic, rescale_xy
from cubic.segmentation.segment_utils import (
    _binary_fill_holes,
    _remove_small_holes,
    _remove_small_objects,
    segment_watershed,
)
from cubic.skimage import filters as _filters
from cubic.skimage import morphology as _morphology
from cubic.skimage import transform as _transform

from dynacell.evaluation.segmentation_cellpose import _robust_clahe

CELL_VOXEL_UM = 0.3
"""Default isotropic working-grid voxel size (um) for the cell stage. ``None``
runs the recipe at native resolution (using the lateral voxel size for the
physical-to-pixel conversions; assumes near-isotropic for volume floors)."""

CLOSE_UM = 2.5
"""Grayscale-closing disk radius (um) that bridges dim cytoplasm in the sum."""

WALL_SIGMA_UM = 0.35
"""Gaussian sigma (um) applied to the membrane before wall thresholding."""

WALL_MIN_UM = 1.0
"""Drop membrane-wall specks below this physical size (um^ndim)."""

HOLE_UM = 3.0
"""Fill cell-mask holes below this physical size (um^ndim)."""

MIN_CELL_UM = 15.0
"""Drop whole cells below this physical size. Default is the 2-D area floor
(um^2); set to a um^3 volume floor (e.g. 50) for ``dimension=3d`` runs."""


def slice_index(memb_vol: np.ndarray, *, selection: str = "frac", fraction: float = 0.30) -> int:
    """Pick a representative z-plane index from a membrane volume.

    Parameters
    ----------
    memb_vol : numpy.ndarray
        3-D ``(Z, Y, X)`` membrane fluorescence volume.
    selection : str
        ``"frac"`` returns ``round(fraction * (Z - 1))``; ``"sharpest"`` returns
        the plane with the highest intensity variance.
    fraction : float
        Fractional depth used when ``selection="frac"``. Defaults to 0.30.

    Returns
    -------
    int
        The chosen z index.
    """
    z = memb_vol.shape[0]
    if selection == "frac":
        return int(round(fraction * (z - 1)))
    if selection == "sharpest":
        return int(np.argmax(memb_vol.reshape(z, -1).var(axis=1)))
    raise ValueError(f"Unknown slice_selection: {selection!r} (expected 'frac' or 'sharpest').")


def focus_slab(memb_vol: np.ndarray, *, halfwidth: int, selection: str = "frac", fraction: float = 0.30) -> slice:
    """Return an in-focus z-slab centered on :func:`slice_index`.

    Builds a ``slice`` of width ``2*halfwidth + 1`` planes centered on the plane
    :func:`slice_index` would pick, clipped to ``[0, Z)``. Reused to restrict a
    max-Z projection (deep-feature crops, per-cell SSIM) to the in-focus band so
    the projection is not dominated by out-of-focus caps — and so those tracks
    share the plane the 2-D instance segmentation already selects.

    Parameters
    ----------
    memb_vol : numpy.ndarray
        3-D ``(Z, Y, X)`` reference volume (GT, so GT and prediction share the
        slab). ``selection``/``fraction`` are forwarded to :func:`slice_index`.
    halfwidth : int
        Planes on each side of the center; ``0`` reproduces a single-slice pick.

    Returns
    -------
    slice
        ``slice(z_start, z_end)`` over the Z axis.
    """
    z0 = slice_index(memb_vol, selection=selection, fraction=fraction)
    z = memb_vol.shape[0]
    return slice(max(0, z0 - halfwidth), min(z, z0 + halfwidth + 1))


def _relabel_sequential_device(labels):
    """Relabel a (possibly device) integer label image to a dense ``1..K`` uint16."""
    xp = get_array_module(labels)
    ids = np.unique(labels)
    ids = ids[ids > 0]
    out = xp.zeros(labels.shape, dtype=xp.uint16)
    if int(ids.size) == 0:
        return out
    remap = xp.zeros(int(ids.max()) + 1, dtype=xp.uint16)
    remap[ids] = xp.arange(1, int(ids.size) + 1, dtype=xp.uint16)
    return remap[labels]


def _cells_on_grid(memb_g, nuc_g, seed_g, *, close_px, wall_sigma_px, wall_min_px, hole_px, min_cell_px):
    """Run the channel-sum + wall-subtraction + watershed recipe on a working grid.

    All inputs are GPU arrays at the same (working) resolution: ``memb_g`` /
    ``nuc_g`` are robust-clipped + CLAHE'd float membrane / nucleus images,
    ``seed_g`` the uint16 nucleus labels. Returns sequentially relabeled uint16
    whole-cell instances on the same device.
    """
    xp = get_array_module(memb_g)
    combined = np.clip(memb_g + nuc_g, 0.0, 1.0)
    disk_fp = ascupy(_morphology.disk(close_px))
    # grayscale closing per XY plane (a single plane in 2D, looped over Z in 3D)
    if combined.ndim == 3:
        closed = xp.zeros_like(combined)
        for z in range(combined.shape[0]):
            closed[z] = _morphology.closing(combined[z], disk_fp)
    else:
        closed = _morphology.closing(combined, disk_fp)
    # lower multi-Otsu boundary (inclusive); Otsu fallback for near-binary images
    try:
        thr0 = _filters.threshold_multiotsu(closed, classes=3, nbins=128)[0]
    except ValueError:
        thr0 = _filters.threshold_otsu(closed)
    tissue = _binary_fill_holes(closed > thr0)
    memb_s = _filters.gaussian(memb_g, sigma=wall_sigma_px, preserve_range=True)
    walls = memb_s > _filters.threshold_multiotsu(memb_s, classes=3, nbins=128)[1]
    walls = _remove_small_objects(walls, min_size=wall_min_px)
    # the seed OR guarantees every nucleus voxel is interior to the mask, so the
    # marker-controlled watershed needs no orphan-seed restoration loop
    cell_mask = (tissue & ~walls) | (seed_g > 0)
    cell_mask = _remove_small_holes(cell_mask, area_threshold=hole_px)
    cells = segment_watershed(cell_mask, markers=seed_g, mask=cell_mask)
    cells = _remove_small_objects(cells, min_size=min_cell_px)
    return _relabel_sequential_device(cells)


def segment_whole_cell(
    memb_img: np.ndarray,
    nuc_img: np.ndarray,
    seed_labels: np.ndarray,
    spacing: tuple[float, ...],
    *,
    cell_voxel_um: float | None = CELL_VOXEL_UM,
    close_um: float = CLOSE_UM,
    wall_sigma_um: float = WALL_SIGMA_UM,
    wall_min_um: float = WALL_MIN_UM,
    hole_um: float = HOLE_UM,
    min_cell_um: float = MIN_CELL_UM,
    memb_clahe: bool = True,
    subtract_nuclei: bool = True,
) -> np.ndarray:
    """Segment whole cells from membrane + nucleus + seed labels.

    Parameters
    ----------
    memb_img : numpy.ndarray
        Membrane fluorescence: 3-D ``(Z, Y, X)`` volume or 2-D ``(Y, X)`` slice.
    nuc_img : numpy.ndarray
        Nucleus fluorescence, same shape as *memb_img*. Summed with the membrane
        to fill the dark nuclear core of the membrane channel.
    seed_labels : numpy.ndarray
        Native-resolution uint16 nucleus instance labels (watershed markers),
        same shape as *memb_img*.
    spacing : tuple of float
        Physical voxel size in micrometers: ``(z, y, x)`` for 3-D, ``(y, x)`` for
        2-D.
    cell_voxel_um : float or None
        Isotropic working-grid voxel size (um). ``None`` runs at native
        resolution. The morphological size parameters below are physical and
        converted to working-grid pixels.
    close_um, wall_sigma_um, wall_min_um, hole_um, min_cell_um : float
        Physical morphological parameters (see module constants). ``wall_min_um``,
        ``hole_um`` and ``min_cell_um`` are areas (2-D) or volumes (3-D).
    memb_clahe : bool
        Apply CLAHE to the membrane channel (the nucleus channel is always
        CLAHE'd).
    subtract_nuclei : bool
        Carve the nucleus footprint out of each whole-cell label so the result is
        cytoplasm-only.

    Returns
    -------
    numpy.ndarray
        uint16 whole-cell (or cytoplasm-only) instance labels at native
        resolution, same spatial shape as *memb_img*.
    """
    ndim = memb_img.ndim
    native_shape = memb_img.shape
    memb_c = _robust_clahe(memb_img, use_clahe=memb_clahe)
    nuc_c = _robust_clahe(nuc_img, use_clahe=True)

    if cell_voxel_um is not None:
        v = cell_voxel_um
        if ndim == 3:
            memb_g = rescale_isotropic(memb_c, spacing, downscale_xy=True, order=3, target_z_voxel_size=v)
            nuc_g = rescale_isotropic(nuc_c, spacing, downscale_xy=True, order=3, target_z_voxel_size=v)
        else:
            scale = spacing[-1] / v
            memb_g = rescale_xy(memb_c, scale=scale, order=3, anti_aliasing=True, preserve_range=True)
            nuc_g = rescale_xy(nuc_c, scale=scale, order=3, anti_aliasing=True, preserve_range=True)
        seed_g = _transform.resize(
            ascupy(seed_labels), memb_g.shape, order=0, anti_aliasing=False, preserve_range=True
        ).astype(np.uint16)
    else:
        v = spacing[-1]
        memb_g, nuc_g = memb_c, nuc_c
        seed_g = ascupy(seed_labels.astype(np.uint16, copy=False))

    close_px = max(3, round(close_um / v))
    wall_sigma_px = max(0.5, wall_sigma_um / v)
    wall_min_px = max(1, round(wall_min_um / v**ndim))
    hole_px = max(1, round(hole_um / v**ndim))
    min_cell_px = max(1, round(min_cell_um / v**ndim))

    cells_g = _cells_on_grid(
        memb_g,
        nuc_g,
        seed_g,
        close_px=close_px,
        wall_sigma_px=wall_sigma_px,
        wall_min_px=wall_min_px,
        hole_px=hole_px,
        min_cell_px=min_cell_px,
    )

    if cell_voxel_um is not None:
        cells = asnumpy(
            _transform.resize(cells_g, native_shape, order=0, anti_aliasing=False, preserve_range=True)
        ).astype(np.uint16)
    else:
        cells = asnumpy(cells_g).astype(np.uint16)

    if subtract_nuclei:
        cells[np.asarray(seed_labels) > 0] = 0
    return cells
