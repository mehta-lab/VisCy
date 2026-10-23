"""GPU-resident Cellpose-DINO (cpdino) instance segmentation, faithful to the cpv4 recipe.

Unlike :mod:`dynacell.evaluation.segmentation_cellpose` (which robust-clips + CLAHEs +
isotropically downscales the input and runs Cellpose-SAM with ``normalize=False``), the
cpdino path replicates exactly how the Cellpose-DINO ViT-L model was validated in the
``.tmp/cpv4_*`` benchmark: the **raw** fluorescence image is handed to Cellpose with
``normalize=True`` (cellpose's own 1st/99th-percentile contrast stretch) and **no** CLAHE
or downscale. Preserving that preprocessing is what makes the AP numbers here match the
benchmarked cpdino result.

Inference runs through :func:`cubic.segmentation.segment_cellpose` (cubic 0.8.0a2, PR#51),
which uploads the input to the device exactly once and returns only the integer mask to
the host — handing it a device array makes that upload a no-op. cubic auto-selects the
DINO tile size (384) from ``model.backbone`` (``dino_vitl``). ``segment_cellpose`` is
AP>=0.95-parity with stock ``CellposeModel.eval`` for these models (verified in PR#51).

Both a single 3-D ``(Z, Y, X)`` volume (``do_3d=True`` orthogonal-flow, or
``stitch_threshold>0`` 2D-per-plane + IoU stitch) and a single 2-D ``(Y, X)`` slice are
supported; the eval pipeline runs the 2-D in-focus slice.
"""

import numpy as np
from cubic.cuda import ascupy, asnumpy
from cubic.segmentation import segment_cellpose

NORMALIZE = True
"""Use cellpose's built-in 1st/99th-percentile normalization (the cpv4 recipe). The raw
image is passed through unmodified; no robust-clip/CLAHE front-end."""

FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0

MIN_SIZE = 15
"""cellpose's default spurious-instance filter (pixels/voxels). cpv4 ran with the default
for both the nucleus and membrane channels, so cpdino uses one value for both targets."""

STITCH_THRESHOLD = 0.0
"""IoU stitch threshold for the 2D-per-plane 3D path. 0.0 = single-slice 2D (the eval
pipeline's in-focus path) or true ``do_3D``; set to 0.4 to reproduce the cpv4 stitch sweep."""


def segment_cpdino_instances(
    img: np.ndarray,
    spacing: tuple[float, ...],
    model,
    *,
    do_3d: bool = False,
    normalize: bool = NORMALIZE,
    flow_threshold: float = FLOW_THRESHOLD,
    cellprob_threshold: float = CELLPROB_THRESHOLD,
    min_size: int = MIN_SIZE,
    stitch_threshold: float = STITCH_THRESHOLD,
    return_labels: bool = True,
) -> np.ndarray:
    """Segment instances in a single slice/volume with GPU-resident Cellpose-DINO.

    Parameters
    ----------
    img : numpy.ndarray
        Raw fluorescence: 2-D ``(Y, X)`` slice (``do_3d=False`` and
        ``stitch_threshold=0``) or 3-D ``(Z, Y, X)`` volume. No pre-normalization is
        applied — cellpose normalizes internally when ``normalize=True``.
    spacing : tuple of float
        Physical voxel size in micrometers: ``(z, y, x)`` for 3-D, ``(y, x)`` for 2-D.
        Only used to derive ``anisotropy = z / x`` for the ``do_3d`` path.
    model : cellpose.models.CellposeModel
        Pre-loaded Cellpose-DINO model (``pretrained_model="cpdino"``, backbone
        ``dino_vitl``), on CUDA — ``segment_cellpose`` is GPU-only.
    do_3d : bool
        Run cellpose's orthogonal-flow 3-D path on a ``(Z, Y, X)`` volume.
    normalize : bool
        Apply cellpose's percentile normalization inside inference (the cpv4 recipe).
    flow_threshold, cellprob_threshold : float
        Cellpose mask thresholds (``flow_threshold`` is ignored by cellpose in 3-D).
    min_size : int
        Drop instances smaller than this many pixels/voxels (cellpose default 15).
    stitch_threshold : float
        IoU stitch threshold for the 2D-per-plane 3-D path (0 = single 2-D slice).
    return_labels : bool
        Return the uint16 instance-label image; otherwise the boolean footprint.

    Returns
    -------
    numpy.ndarray
        Instance labels (uint16) or boolean footprint at native resolution, same spatial
        shape as *img*.
    """
    native_shape = img.shape
    img_dev = ascupy(img.astype(np.float32, copy=False))
    z_axis = 0 if (do_3d or stitch_threshold > 0.0) else None
    anisotropy = (spacing[0] / spacing[-1]) if do_3d else None

    masks, _, _ = segment_cellpose(
        model,
        img_dev,
        normalize=normalize,
        channel_axis=None,
        z_axis=z_axis,
        do_3D=do_3d,
        anisotropy=anisotropy,
        stitch_threshold=stitch_threshold,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        diameter=None,
    )

    labels = asnumpy(masks).astype(np.uint16)
    if labels.shape != tuple(native_shape):
        raise ValueError(
            f"cpdino returned shape {labels.shape} != input {tuple(native_shape)}; "
            "the faithful path does not downscale, so shapes must match"
        )
    return labels if return_labels else labels > 0


def segment_whole_cell_cpdino(
    memb_img: np.ndarray,
    seed_labels: np.ndarray | None,
    spacing: tuple[float, ...],
    model,
    *,
    subtract_nuclei: bool = True,
    **infer_kwargs,
) -> np.ndarray:
    """Segment whole cells directly from the membrane channel, then carve the nucleus.

    Replaces the nuclei-seed + EDT-watershed pipeline
    (:func:`dynacell.evaluation.segmentation_whole_cell.segment_whole_cell`): cpdino
    segments the whole cell from the membrane image in one pass, and the nucleus
    footprint (*seed_labels*, cpdino instances of the GT nucleus channel) is carved out so
    metrics score the cytoplasmic shell — same ``subtract_nuclei`` semantics as watershed.

    Parameters
    ----------
    memb_img : numpy.ndarray
        Raw membrane fluorescence, 2-D ``(Y, X)`` or 3-D ``(Z, Y, X)``.
    seed_labels : numpy.ndarray or None
        Native-resolution uint16 nucleus instance labels to carve out. ``None`` or
        all-zero leaves the whole-cell labels intact (a no-op carve).
    spacing : tuple of float
        Physical voxel size (see :func:`segment_cpdino_instances`).
    model : cellpose.models.CellposeModel
        Pre-loaded Cellpose-DINO model.
    subtract_nuclei : bool
        Carve the nucleus footprint out of each whole-cell label.
    **infer_kwargs
        Forwarded to :func:`segment_cpdino_instances` (``do_3d``, ``normalize``,
        ``flow_threshold``, ``cellprob_threshold``, ``min_size``, ``stitch_threshold``).

    Returns
    -------
    numpy.ndarray
        uint16 whole-cell (or cytoplasm-only) instance labels at native resolution.
    """
    cells = segment_cpdino_instances(memb_img, spacing, model, return_labels=True, **infer_kwargs)
    if subtract_nuclei and seed_labels is not None:
        cells = cells.copy()
        cells[np.asarray(seed_labels) > 0] = 0
    return cells
