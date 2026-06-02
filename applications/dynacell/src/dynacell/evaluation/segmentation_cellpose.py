"""GPU-resident Cellpose-SAM nucleus segmentation with fnet_astr_vpa-style pre/post.

All preprocessing (robust percentile clip, CLAHE, isotropic downscale) and
postprocessing (size filter, sequential relabel, upsample to native) run on GPU
through ``cubic``. The Cellpose-SAM inference itself runs through
``cubic.segmentation.segment_cpsam``, which uploads the downscaled volume to the
device exactly once and returns only the integer mask to the host — keeping the
flow field and intermediate buffers GPU-resident (vs ``CellposeModel.eval``,
which re-uploads tiles host-side).

Recipe adapted from ``fnet_astr_vpa`` (Allen iPSC nucleus segmentation):
robust(0.1) -> CLAHE(kernel=shape//(2,3,5) in 3D, shape//(3,5) in 2D, clip=0.015)
-> isotropic downscale, then Cellpose-SAM (``do_3D`` for volumes, 2D for single
slices), then an on-device size filter + sequential relabel. The isotropic
downscale makes the volume isotropic at ``target_voxel_um`` so we pass
``anisotropy=1.0`` (3D) and Cellpose performs no internal resize
(``image_scaling=1`` via ``diameter=None``).

Both a single 3-D ``(Z, Y, X)`` volume (``do_3d=True``) and a single 2-D
``(Y, X)`` slice (``do_3d=False``) are supported; ``return_labels`` selects
between an integer instance-label image (uint16) and the binary footprint.
"""

import numpy as np
from cellpose import models
from cubic.cuda import ascupy, asnumpy, get_array_module
from cubic.image_utils import rescale_isotropic, rescale_xy
from cubic.segmentation import segment_cpsam
from cubic.skimage import exposure as _cubic_exposure
from cubic.skimage import transform as _cubic_transform

ROBUST_PERCENTILE = 0.1
"""Lower percentile for robust min/max rescale; upper is ``100 - this``.
Clips bright outliers before CLAHE so the [0, 1] stretch tracks signal, not
shot-noise tips (fnet ``preprocessing.yaml``)."""

CLAHE_KERNEL_DIVISOR = (2, 3, 5)
"""Per-axis (Z, Y, X) divisor for the 3D CLAHE contextual kernel:
``kernel = image.shape // divisor`` (fnet ``clahe``)."""

CLAHE_KERNEL_DIVISOR_2D = (3, 5)
"""Per-axis (Y, X) divisor for the CLAHE contextual kernel on a single 2D slice."""

CLAHE_CLIP_LIMIT = 0.015
CLAHE_NBINS = 256

TARGET_VOXEL_UM = 0.58
"""Isotropic voxel size (um) the volume is downscaled to before Cellpose.
Matches the grid fnet's cellpose ran on (Allen iPSC ~0.58 um isotropic). At
512x512 FOVs this yields lateral <= 256 (iPSC ~99, A549 ~128) so Cellpose runs
single-tile; larger FOVs tile on host automatically."""

CELLPROB_THRESHOLD = 0.0
FLOW_THRESHOLD = 0.4

MIN_OBJECT_SIZE = 500
"""Drop Cellpose instances smaller than this many voxels (at the downscaled
isotropic grid) as spurious — the one cleanup step we keep from fnet's
``postprocessing.yaml``.

We deliberately do **not** call ``cubic.cleanup_segmentation``: its
``clear_xy_borders`` uses ``np.in1d`` (removed in numpy>=2.2) and its
``remove_large_objects`` asserts a non-constant label image mid-pipeline, which
crashes on empty/near-empty FOVs. fnet's ``max_obj_size=7500`` /
``max_hole_size=50`` filters are also dropped for now: those thresholds were
tuned for fnet's legacy ``nuclei`` model on its grid and would wrongly remove
real whole nuclei at Cellpose-SAM's ``TARGET_VOXEL_UM`` resolution — re-tune in
Phase 0 before reinstating. Border-touching nuclei are kept (consistent on GT
and prediction sides for binary Dice)."""


def load_cellpose_model(use_gpu: bool = True) -> "models.CellposeModel":
    """Load the Cellpose-SAM model.

    Parameters
    ----------
    use_gpu : bool
        Place the network on GPU. Defaults to True.

    Returns
    -------
    cellpose.models.CellposeModel
        The Cellpose-SAM model (``cpsam``).
    """
    model = models.CellposeModel(gpu=use_gpu)
    # cellpose 4.1.x's CellposeModel no longer exposes a ``backbone`` attribute,
    # which ``cubic.segmentation.segment_cpsam`` requires (it reads it only to
    # pick the tile size: "sam_vitl" -> 256). Cellpose-SAM is the ViT-L backbone,
    # so set it explicitly to keep the GPU-resident path's precondition satisfied.
    if not hasattr(model, "backbone"):
        model.backbone = "sam_vitl"
    return model


def _robust_clahe(img: np.ndarray, use_clahe: bool = True):
    """Robust percentile clip to ~[0, 1] then (optional) CLAHE, on GPU.

    Shared preprocessing front-end for the nucleus (this module) and whole-cell
    (:mod:`dynacell.evaluation.segmentation_whole_cell`) pipelines. The CLAHE
    contextual kernel divisor is selected by ``img.ndim`` (3-D vs 2-D). Returns a
    CuPy array with the same shape as *img*, float32 in ~[0, 1].
    """
    img_dev = ascupy(img.astype(np.float32, copy=False))
    lo, hi = np.percentile(img_dev, (ROBUST_PERCENTILE, 100.0 - ROBUST_PERCENTILE))
    img_dev = np.clip((img_dev - lo) / (hi - lo), 0.0, 1.0)
    if use_clahe:
        divisor = CLAHE_KERNEL_DIVISOR if img_dev.ndim == 3 else CLAHE_KERNEL_DIVISOR_2D
        kernel = tuple(int(s // d) for s, d in zip(img_dev.shape, divisor))
        img_dev = _cubic_exposure.equalize_adapthist(
            img_dev, kernel_size=kernel, clip_limit=CLAHE_CLIP_LIMIT, nbins=CLAHE_NBINS
        )
    return img_dev


def _preprocess_gpu(
    img: np.ndarray,
    spacing: tuple[float, ...],
    target_voxel_um: float,
    use_clahe: bool = True,
):
    """Robust clip -> (optional) CLAHE -> isotropic downscale, all on GPU.

    Supports a 3-D ``(Z, Y, X)`` volume (3-tuple *spacing*, downscaled to an
    isotropic ``target_voxel_um`` grid) and a 2-D ``(Y, X)`` slice (2-tuple
    *spacing*, XY downscaled to ``target_voxel_um``). The branch is selected by
    ``img.ndim``.

    Parameters
    ----------
    img : numpy.ndarray
        3-D ``(Z, Y, X)`` image or 2-D ``(Y, X)`` slice.
    spacing : tuple of float
        Physical voxel size in micrometers: ``(z, y, x)`` for 3-D, ``(y, x)``
        for 2-D.
    target_voxel_um : float
        Isotropic voxel size (um) to downscale to.
    use_clahe : bool
        Apply CLAHE after the robust clip. Defaults to True.

    Returns
    -------
    cupy.ndarray
        Preprocessed downscaled image on GPU, float32 in ~[0, 1].
    """
    img_dev = _robust_clahe(img, use_clahe=use_clahe)
    if img_dev.ndim == 3:
        return rescale_isotropic(
            img_dev,
            spacing,
            downscale_xy=True,
            order=3,
            target_z_voxel_size=target_voxel_um,
        )
    return rescale_xy(
        img_dev,
        scale=spacing[-1] / target_voxel_um,
        order=3,
        anti_aliasing=True,
        preserve_range=True,
    )


def segment_nucleus(
    img: np.ndarray,
    spacing: tuple[float, ...],
    model: "models.CellposeModel",
    *,
    target_voxel_um: float = TARGET_VOXEL_UM,
    cellprob_threshold: float = CELLPROB_THRESHOLD,
    flow_threshold: float = FLOW_THRESHOLD,
    min_obj_size: int = MIN_OBJECT_SIZE,
    use_clahe: bool = True,
    do_3d: bool = True,
    return_labels: bool = False,
) -> np.ndarray:
    """Segment nuclei in a single z-stack or slice with GPU Cellpose-SAM.

    Parameters
    ----------
    img : numpy.ndarray
        3-D ``(Z, Y, X)`` volume (``do_3d=True``) or 2-D ``(Y, X)`` slice
        (``do_3d=False``) fluorescence image.
    spacing : tuple of float
        Physical voxel size in micrometers: ``(z, y, x)`` for 3-D, ``(y, x)``
        for 2-D (from ``config.pixel_metrics.spacing``).
    model : cellpose.models.CellposeModel
        Pre-loaded Cellpose-SAM model (on CUDA — ``segment_cpsam`` is GPU-only).
    target_voxel_um : float
        Isotropic voxel size (um) to downscale to before inference.
    cellprob_threshold, flow_threshold : float
        Cellpose mask thresholds (``flow_threshold`` is ignored by Cellpose in 3D).
    min_obj_size : int
        Remove instances smaller than this many voxels/pixels at the downscaled
        grid. The default is tuned for the 3-D volume path; lower it for 2-D
        slices (areas are far smaller than volumes at the same voxel size).
    use_clahe : bool
        Apply CLAHE during preprocessing. Defaults to True.
    do_3d : bool
        Run volumetric Cellpose-SAM (``do_3D``) on a ``(Z, Y, X)`` volume. When
        False, run 2-D Cellpose-SAM on a ``(Y, X)`` slice.
    return_labels : bool
        When True, return the uint16 instance-label image; otherwise return the
        binary footprint (``labels > 0``).

    Returns
    -------
    numpy.ndarray
        Instance labels (uint16) or boolean footprint with the same spatial
        shape as *img* (native resolution).
    """
    native_shape = img.shape
    down = _preprocess_gpu(img, spacing, target_voxel_um, use_clahe=use_clahe)
    # channel-first 3 identical channels: (3, Zd, Yd, Xd) for 3D, (3, Yd, Xd) for 2D
    down_3ch = np.repeat(asnumpy(down)[np.newaxis], 3, axis=0)

    masks, _, _ = segment_cpsam(
        model,
        down_3ch,
        channel_axis=0,
        z_axis=1 if do_3d else None,
        do_3D=do_3d,
        anisotropy=1.0 if do_3d else None,
        diameter=None,
        normalize=False,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
    )

    # GPU size-filter + sequential relabel: keep instances >= min_obj_size, remap
    # surviving ids to 1..K via an on-device gather LUT (np.unique dispatches to
    # CuPy on the device array, so this stays GPU-resident).
    labels_dev = ascupy(masks)
    xp = get_array_module(labels_dev)
    ids, counts = np.unique(labels_dev, return_counts=True)
    keep = ids[(ids != 0) & (counts >= min_obj_size)]
    if int(keep.size) == 0:
        dtype = np.uint16 if return_labels else bool
        return np.zeros(native_shape, dtype=dtype)
    remap = xp.zeros(int(ids.max()) + 1, dtype=xp.uint16)
    remap[keep] = xp.arange(1, int(keep.size) + 1, dtype=xp.uint16)
    relabeled = remap[labels_dev]
    native = _cubic_transform.resize(relabeled, native_shape, order=0, anti_aliasing=False, preserve_range=True)
    if return_labels:
        return asnumpy(native).astype(np.uint16)
    return asnumpy(native) > 0


def segment_nucleus_instances(
    img: np.ndarray,
    spacing: tuple[float, ...],
    model: "models.CellposeModel",
    **kwargs,
) -> np.ndarray:
    """Segment nuclei and return uint16 instance labels (native resolution).

    Thin alias for :func:`segment_nucleus` with ``return_labels=True``. Used both
    as the per-slice/volume label generator for the nucleus instance-AP path and
    as the watershed seed generator for the whole-cell path.
    """
    return segment_nucleus(img, spacing, model, return_labels=True, **kwargs)
