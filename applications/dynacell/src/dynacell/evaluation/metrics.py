"""Metric computation for evaluation: pixel metrics, mask metrics, MicroMS3IM."""

from typing import Any

import numpy as np
import torch

try:
    from cubic.cuda import ascupy, asnumpy
    from cubic.feature import glcm_features
    from cubic.feature.voxel import regionprops_table
    from cubic.metrics import fsc_resolution, nrmse, pcc, psnr
    from cubic.metrics import ssim as cubic_ssim  # aliased — dynacell keeps a local ssim() wrapper
    from cubic.metrics.bandlimited import spectral_pcc
    from cubic.scipy import ndimage as _cubic_ndimage
    from cubic.skimage import filters as _cubic_filters
except ImportError:
    ascupy = None  # type: ignore[assignment]
    asnumpy = None  # type: ignore[assignment]
    cubic_ssim = None  # type: ignore[assignment]
    fsc_resolution = None  # type: ignore[assignment]
    glcm_features = None  # type: ignore[assignment]
    nrmse = None  # type: ignore[assignment]
    pcc = None  # type: ignore[assignment]
    psnr = None  # type: ignore[assignment]
    regionprops_table = None  # type: ignore[assignment]
    spectral_pcc = None  # type: ignore[assignment]
    _cubic_filters = None  # type: ignore[assignment]
    _cubic_ndimage = None  # type: ignore[assignment]

from dynacell.evaluation.utils import _minmax_norm


def _require_cubic():
    # Only cubic itself is required: the metric helpers below gate the GPU
    # upload on ``torch.cuda.is_available()`` and otherwise run on numpy, where
    # cubic dispatches to its CPU (numpy / scikit-image) path. cucim / cupy are
    # needed only when a GPU is actually present and used — and ``ascupy`` raises
    # a clear "GPU requested but not available" there if they are missing — so
    # this must NOT hard-require the eval_gpu stack (it would block the CPU path).
    if ascupy is None:
        raise ImportError(
            "cubic is required for resolution and feature metrics. "
            "Install via the `eval` extra: `uv sync --extra eval`."
        )


@torch.inference_mode()
def _min_max_normalize(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Min-max normalize a tensor to [0, 1] range."""

    x = x.float()
    x = (x - x.min()) / torch.clamp(x.max() - x.min(), min=eps)

    return x


@torch.inference_mode()
def ssim(img1: torch.Tensor, img2: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute mean structural similarity index (SSIM) for 3D volumetric inputs.

    Parameters
    ----------
    img1, img2 : torch.Tensor
        3-D tensors of shape ``(D, H, W)``.
    eps : float
        Small constant for min-max normalization stability.
    """
    if cubic_ssim is None:
        raise ImportError("cubic is required for SSIM. Install via the `eval` extra: `uv sync --extra eval`.")
    if img1.ndim != 3:
        raise ValueError(f"ssim expects 3-D (D, H, W) input, got {img1.ndim}-D tensor of shape {tuple(img1.shape)}")
    img1 = _min_max_normalize(img1, eps=eps)
    img2 = _min_max_normalize(img2, eps=eps)

    img1 = img1.unsqueeze(0).unsqueeze(0)  # (D,H,W) → (1,1,D,H,W) — cubic's 5D contract
    img2 = img2.unsqueeze(0).unsqueeze(0)

    return cubic_ssim(img1, img2, spatial_dims=3, data_range=1.0, gaussian_weights=True)


def evaluate_segmentations(segmented_pred, segmented_gt) -> dict[str, float]:
    """Evaluate binary segmentation against ground truth.

    Returns
    -------
    dict[str, float]
        A dict with dice, iou, precision, recall, accuracy, tp, fp, fn, tn.

    Notes
    -----
    Non-zero values are treated as foreground.
    Inputs must have the same shape.
    """
    pred = np.asarray(segmented_pred)
    gt = np.asarray(segmented_gt)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: predicted shape {pred.shape} != ground truth shape {gt.shape}")

    # Treat any non-zero value as foreground
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum(dtype=np.int64)
    fp = np.logical_and(pred, ~gt).sum(dtype=np.int64)
    fn = np.logical_and(~pred, gt).sum(dtype=np.int64)
    tn = np.logical_and(~pred, ~gt).sum(dtype=np.int64)

    # Safe division helper
    def _safe_div(num: float, den: float) -> float:
        return float(num / den) if den != 0 else 0.0

    dice = _safe_div(2 * tp, 2 * tp + fp + fn)
    iou = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "TP": float(tp),
        "FP": float(fp),
        "FN": float(fn),
        "TN": float(tn),
    }


def compute_pixel_metrics(prediction, target, spacing, fsc_kwargs=None, spectral_pcc_kwargs=None, use_gpu=True):
    """Compute pixel-level image quality metrics between prediction and target.

    Notes
    -----
    Inputs (numpy, torch CPU/CUDA, or cupy) are coerced to a single
    ``xp`` array module via ``cubic.cuda.ascupy``/``asnumpy`` — cupy
    when ``use_gpu=True`` and CUDA is available, numpy otherwise. The
    converters no-op when the input is already on the target module,
    so a caller that pre-uploaded the full FOV via ``ascupy(predict)``
    once pays zero per-call upload tax here. cubic metrics consume
    ``xp`` directly; the SSIM wrapper consumes a torch view built
    zero-copy from ``xp`` via ``torch.as_tensor`` (CUDA Array Interface
    for cupy, ``from_numpy`` for numpy).
    """
    if pcc is None:
        raise ImportError("cubic is required for pixel metrics. Install via the `eval` extra: `uv sync --extra eval`.")
    _require_cubic()
    use_cuda = bool(use_gpu and torch.cuda.is_available())
    to_xp = ascupy if use_cuda else asnumpy
    pred_xp, target_xp = to_xp(prediction), to_xp(target)

    # ``.contiguous()`` recovers the contiguity guarantee the previous
    # ``.to(device)`` step provided: when ``target_xp`` is a non-contiguous
    # cupy view (e.g. a strided zarr slice), cubic_ssim → MONAI → conv3d's
    # CUDA backend can fail or silently re-materialize on recent torch.
    metrics = {
        "PCC": pcc(target_xp, pred_xp),
        "SSIM": ssim(torch.as_tensor(target_xp).contiguous(), torch.as_tensor(pred_xp).contiguous()),
        "NRMSE": nrmse(target_xp, pred_xp, normalize="min_max"),
        "PSNR": psnr(target_xp, pred_xp, normalize="min_max"),
    }

    if spectral_pcc_kwargs is None and fsc_kwargs is None:
        return metrics

    if spectral_pcc_kwargs is not None:
        metrics["Spectral_PCC"] = spectral_pcc(pred_xp, target_xp, spacing=spacing, **spectral_pcc_kwargs)
    if fsc_kwargs is not None:
        # cubic.fsc_resolution mean-centers internally before every FFT,
        # so we pass the raw arrays.
        resolutions = fsc_resolution(target_xp, pred_xp, spacing=spacing, **fsc_kwargs)
        metrics.update({f"{k.upper()}_FSC_Resolution": float(v) for k, v in resolutions.items()})

    return metrics


def _require_microms3im():
    """Import MicroMS3IM, raising the same install hint both fit/score share."""
    try:
        from cubic.metrics import MicroMS3IM
    except ImportError as e:
        raise ImportError(
            "cubic>=0.7.0a4 is required for MicroMS3IM. Install via the `eval` extra: `uv sync --extra eval`."
        ) from e
    return MicroMS3IM


def fit_microssim(targets: np.ndarray, predictions: np.ndarray, use_gpu: bool = True):
    """Fit a single MicroMS3IM instance on a batch of (target, prediction) pairs.

    Per the microSSIM paper (Ashesh & Jug, 2024, sec. 3.3):

        "we learn a single scalar for the entire dataset. Had we optimized
        for every (x, y) pair, we would get a higher measure value on
        average, but this does not align well with the motivation for
        this measure, which is to estimate an optimal linear
        transformation between the space of predictions to their
        corresponding high-SNR micrographs."

    Callers therefore fit ONCE over all (gt, pred) slices in a leaf and
    reuse the fitted ``sim`` for scoring every FOV/timepoint pair, instead
    of refitting per FOV (which inflates scores and breaks cross-FOV
    comparability).

    Parameters
    ----------
    targets, predictions : np.ndarray
        Arrays of shape ``(N, H, W)`` aligned along the leading axis — the
        full pool of 2D slices used for fitting the relative-intensity
        factor α.
    use_gpu : bool
        When ``True`` and cupy/cucim are available, dispatches to cubic's
        GPU path via ``cubic.cuda.ascupy``.

    Returns
    -------
    MicroMS3IM
        Fitted instance — ``sim.score(target_slice, pred_slice)`` may
        then be called without further fitting.
    """
    MicroMS3IM = _require_microms3im()
    # Convert to cupy when GPU is requested — cubic.skimage dispatches to
    # cucim (GPU Gaussian filters) when inputs carry a .device attribute.
    to_xp = ascupy if (use_gpu and ascupy is not None and torch.cuda.is_available()) else asnumpy
    targets = to_xp(targets)
    predictions = to_xp(predictions)
    sim = MicroMS3IM()
    sim.fit(targets, predictions)
    return sim


def score_microssim(microssim_data, sim, use_gpu: bool = True):
    """Score MicroMS3IM per FOV-T using a pre-fitted ``sim`` (no refit).

    Each entry of ``microssim_data`` contributes one row to the returned
    list, averaging ``sim.score(target_slice, pred_slice)`` over that
    entry's z-slices. ``sim`` must have been fitted via :func:`fit_microssim`
    on the leaf-level pool of pairs — refitting inside this function
    would re-introduce the per-call α drift the leaf-level calibration
    pass is here to prevent.
    """
    targets = np.concatenate([img["target"] for img in microssim_data], axis=0)
    predictions = np.concatenate([img["predict"] for img in microssim_data], axis=0)
    to_xp = ascupy if (use_gpu and ascupy is not None and torch.cuda.is_available()) else asnumpy
    targets = to_xp(targets)
    predictions = to_xp(predictions)

    scores: list[dict[str, float]] = []
    slice_idx = 0
    for img in microssim_data:
        num_slices = len(img["target"])
        if num_slices == 0:
            raise ValueError(
                "score_microssim received a microssim_data entry with zero z-slices; "
                "this signals a stacking bug or an empty FOV upstream."
            )
        img_targets = targets[slice_idx : slice_idx + num_slices]
        img_predictions = predictions[slice_idx : slice_idx + num_slices]
        slice_scores: list[float] = []
        for i in range(num_slices):
            try:
                slice_scores.append(sim.score(img_targets[i], img_predictions[i]))
            except ValueError as exc:
                # cubic's ms_ssim raises ``ValueError("data_range must be finite
                # and positive; got <x>")`` when target or prediction collapses
                # to a constant slice (data_range = max - min = 0) or when a NaN
                # α from the fitted path propagates into pred_norm (data_range =
                # NaN - NaN = NaN). All other ValueErrors (un-fitted sim, shape
                # mismatch, ndim != 2, kernel/spatial-min violations) are real
                # bugs and must propagate. A degenerate slice is scored as 0
                # rather than NaN so that the FOV-T mean is dragged toward the
                # floor — a model that collapses on a subset of slices/FOVs
                # deserves a penalty in leaf-level rankings, not silent removal
                # from the average (a ``nanmean``-style aggregation would let
                # collapsed predictions vanish, leaving a partially-collapsing
                # model indistinguishable from one that scores well everywhere).
                if "data_range" not in str(exc):
                    raise
                slice_scores.append(0.0)
        slice_idx += num_slices
        scores.append({"MicroMS3IM": float(np.asarray(slice_scores, dtype=float).mean())})
    return scores


def _robust_norm(x, p_lo: float = 1.0, p_hi: float = 99.0, eps: float = 1e-8):
    """Percentile-clip ``x`` to ``[p_lo, p_hi]`` then min-max to ``[0, 1]``.

    Replaces the fragile raw min-max (:func:`_minmax_norm`, outlier-dominated)
    for the CP feature track. Device-agnostic — ``np.percentile``/``np.clip``
    dispatch on numpy or cupy. The clipped numerator is bounded by the span, so
    the ``+ eps`` denominator keeps a constant/near-constant image finite
    (output → 0) instead of NaN/inf (mirrors :func:`_minmax_norm`'s eps guard).
    """
    lo, hi = np.percentile(x, (p_lo, p_hi))
    x = np.clip(x, lo, hi)
    return (x - lo) / ((hi - lo) + eps)


# --- CP per-cell distribution-shape extra_properties --------------------------
# skimage/cucim invoke an extra_property as ``func(regionmask, intensity_image)``
# where ``intensity_image`` is the full bounding-box rectangle (background
# included) and ``regionmask`` is the boolean footprint, so each callable must
# reduce over the foreground ``intensity[regionmask]`` only. All ops are ``np.``
# so they run unchanged on numpy (CPU) or cupy (cuCIM GPU); the output column
# name is the function ``__name__``.
def _make_percentile(q: int, name: str):
    """Build a foreground-percentile extra_property named ``name``."""

    def _prop(regionmask, intensity):
        return np.percentile(intensity[regionmask], q)

    _prop.__name__ = name
    return _prop


_p10 = _make_percentile(10, "p10")
_p25 = _make_percentile(25, "p25")
_p50 = _make_percentile(50, "p50")
_p75 = _make_percentile(75, "p75")
_p90 = _make_percentile(90, "p90")


def _make_standardized_moment(order: int, name: str, *, excess: float = 0.0):
    """Build a foreground standardized-moment extra_property named ``name``.

    Reduces over ``intensity[regionmask]``; returns NaN for degenerate regions
    (<2 voxels or zero std). ``excess`` subtracts the Gaussian baseline (3.0 for
    Fisher/excess kurtosis).
    """

    def _prop(regionmask, intensity):
        vals = intensity[regionmask]
        if vals.size < 2:
            return np.nan
        mean = vals.mean()
        std = vals.std()
        if float(std) == 0.0:
            return np.nan
        return ((vals - mean) ** order).mean() / std**order - excess

    _prop.__name__ = name
    return _prop


_skewness = _make_standardized_moment(3, "skewness")
_kurtosis = _make_standardized_moment(4, "kurtosis", excess=3.0)

_DISTRIBUTION_PROPS = (_p10, _p25, _p50, _p75, _p90, _skewness, _kurtosis)

# CP column schema. ``iqr`` is derived from p25/p75 at assembly (no extra
# regionprops pass). Gradient/Laplacian stats come from SEPARATE
# regionprops_table calls whose dict keys (``intensity_mean``/``intensity_std``)
# collide with the base intensity keys, so they are aliased on assembly to
# ``gradient_mean``/``gradient_std``/``laplacian_var``.
_CP_BASE_FEATURE_NAMES: tuple[str, ...] = (
    "intensity_mean",
    "intensity_std",
    "intensity_min",
    "intensity_max",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "iqr",
    "skewness",
    "kurtosis",
    "gradient_mean",
    "gradient_std",
    "laplacian_var",
)

# GLCM Haralick props (opt-in). ``_GLCM_PROP_KEYS`` are the keys returned by
# ``cubic.feature.glcm_features``; the CP columns prefix them with ``glcm_``.
_GLCM_PROP_KEYS: tuple[str, ...] = (
    "contrast",
    "dissimilarity",
    "homogeneity",
    "ASM",
    "energy",
    "correlation",
    "entropy",
)
_CP_GLCM_FEATURE_NAMES: tuple[str, ...] = tuple(f"glcm_{key}" for key in _GLCM_PROP_KEYS)

# Version tag for the CP feature recipe. Recorded in the cache manifest's
# ``cp_features`` identity dict; a bump (or a cp.glcm / cp.norm config change)
# auto-invalidates stale CP caches via
# :func:`pipeline_cache._auto_invalidate_on_artifact_param_mismatch`.
CP_FEATURE_VERSION = "v2_dist_texture"


def active_cp_feature_names(glcm_enabled: bool) -> tuple[str, ...]:
    """Return the ordered CP column names for the active config.

    The schema is GLCM-dependent: the base distribution/texture columns are
    always emitted; the seven ``glcm_*`` columns are appended only when GLCM is
    enabled. Used by both the matrix assembly and the
    ``cp_selected_feature_mask.json`` sidecar so they never drift.
    """
    if glcm_enabled:
        return _CP_BASE_FEATURE_NAMES + _CP_GLCM_FEATURE_NAMES
    return _CP_BASE_FEATURE_NAMES


def drop_paired_nonfinite_rows(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows where either side has any non-finite value.

    Used to sanitize CP regionprops outputs (NaN intensity_std on
    degenerate 1-voxel regions crashes FID covariance via
    ``np.linalg.eigvals``) and to align paired metrics (median cosine
    similarity) to the same row IDs on both sides.
    """
    if pred.shape[0] == 0:
        return pred, target
    valid = np.isfinite(pred).all(axis=1) & np.isfinite(target).all(axis=1)
    if valid.all():
        return pred, target
    return pred[valid], target[valid]


def _region_slices(labels_host: np.ndarray) -> list:
    """Per-label bounding-box slice tuples for a host label array, in one pass.

    ``find_objects(labels)[lab - 1]`` is the slice tuple for label ``lab``
    (``None`` for absent labels) — one O(volume) sweep, versus a per-label
    ``argwhere`` scan of the whole volume.
    """
    return _cubic_ndimage.find_objects(labels_host)


def _per_cell_glcm(img, cell_segmentation, glcm_cfg: dict) -> dict[str, np.ndarray]:
    """Per-cell GLCM Haralick props on the robust-normalized image.

    Each cell's native bbox crop is quantized over the SHARED image-wide range
    ``(0.0, 1.0)`` — ``img`` is already robust-normalized to ``[0, 1]``, so a
    fixed range makes texture comparable across cells and across GT/pred
    (the per-image quantization contract). Returns one ``glcm_<prop>`` array
    per :data:`_CP_GLCM_FEATURE_NAMES`, ordered by ascending label.
    """
    levels = int(glcm_cfg.get("levels", 32))
    distances = tuple(glcm_cfg.get("distances", (1,)))
    # A 2-D eval stores its plane as a singleton-Z volume ``(1, H, W)``. Squeeze
    # that axis so GLCM runs in true 2-D (its correct dimensionality), instead
    # of as a degenerate 3-D volume whose z-direction co-occurrence pairs are
    # empty — empty pairs are tolerated by numpy ``bincount`` but raise on cupy.
    squeeze_z = img.ndim == 3 and img.shape[0] == 1
    labels_host = asnumpy(cell_segmentation)
    objects = _region_slices(labels_host)
    cols: dict[str, list[float]] = {name: [] for name in _CP_GLCM_FEATURE_NAMES}
    for lab in np.unique(labels_host):
        if lab == 0:
            continue
        slices = objects[int(lab) - 1]
        crop = img[slices]
        mask = cell_segmentation[slices] == int(lab)
        if squeeze_z:
            crop = crop[0]
            mask = mask[0]
        props = glcm_features(crop, mask=mask, levels=levels, distances=distances, value_range=(0.0, 1.0))
        for key, name in zip(_GLCM_PROP_KEYS, _CP_GLCM_FEATURE_NAMES):
            cols[name].append(float(props[key]))
    return {name: np.asarray(cols[name], dtype=float) for name in _CP_GLCM_FEATURE_NAMES}


def cp_regionprops(image, cell_segmentation, spacing, *, norm=None, glcm_cfg=None, use_gpu=True):
    """Per-cell conventional ("CP") image features for one image + segmentation.

    Returns ``(n_cells, n_features)`` with columns ordered by
    :func:`active_cp_feature_names`. Same body for GT and prediction. The image
    is robust-normalized per image (percentile-clip + min-max) so intensity and
    texture features stay comparable across the GT/pred intensity-range
    mismatch. Weighted moments are dropped in favor of distribution-shape
    descriptors + gradient/Laplacian texture, plus optional GLCM Haralick.

    Parameters
    ----------
    image, cell_segmentation : np.ndarray
        Single-timepoint intensity volume and its integer label image.
    spacing : list[float]
        Physical voxel spacing forwarded to ``regionprops_table``.
    norm : dict, optional
        ``{"p_lo", "p_hi"}`` percentile-clip bounds; defaults to ``(1, 99)``.
    glcm_cfg : dict, optional
        ``{"enabled", "levels", "distances"}``; GLCM columns are emitted only
        when ``enabled`` is true.
    use_gpu : bool
        Upload inputs via ``ascupy`` (cuCIM dispatch) when CUDA is available.
    """
    _require_cubic()
    norm = dict(norm) if norm is not None else {}
    glcm_cfg = dict(glcm_cfg) if glcm_cfg is not None else {}
    glcm_enabled = bool(glcm_cfg.get("enabled", False))
    img = _robust_norm(image, norm.get("p_lo", 1.0), norm.get("p_hi", 99.0))

    use_cuda = bool(use_gpu and torch.cuda.is_available())
    if use_cuda:
        img = ascupy(img)
        cell_segmentation = ascupy(cell_segmentation)

    base = regionprops_table(
        cell_segmentation,
        img,
        spacing=spacing,
        properties=["intensity_mean", "intensity_std", "intensity_min", "intensity_max"],
        extra_properties=_DISTRIBUTION_PROPS,
    )
    grad = regionprops_table(
        cell_segmentation,
        _cubic_filters.sobel(img),
        spacing=spacing,
        properties=["intensity_mean", "intensity_std"],
    )
    lapt = regionprops_table(
        cell_segmentation,
        _cubic_filters.laplace(img),
        spacing=spacing,
        properties=["intensity_std"],
    )

    # Start from the base regionprops dict — its extra_property columns are
    # already named to match the schema (p10..kurtosis); the extra ``label``
    # column is ignored by ``active_cp_feature_names``. Add the derived/aliased
    # keys: ``iqr`` from p25/p75, and gradient/Laplacian (whose ``intensity_*``
    # keys would collide with the base intensities). Laplacian variance is the
    # built-in ``intensity_std`` squared (var = std**2).
    columns: dict[str, Any] = dict(base)
    columns["iqr"] = base["p75"] - base["p25"]
    columns["gradient_mean"] = grad["intensity_mean"]
    columns["gradient_std"] = grad["intensity_std"]
    columns["laplacian_var"] = lapt["intensity_std"] ** 2
    if use_cuda:
        columns = {name: asnumpy(value) for name, value in columns.items()}
    if glcm_enabled:
        columns.update(_per_cell_glcm(img, cell_segmentation, glcm_cfg))

    names = active_cp_feature_names(glcm_enabled)
    if columns["intensity_mean"].shape[0] == 0:
        return np.empty((0, len(names)), dtype=float)
    return np.stack([np.asarray(columns[name], dtype=float) for name in names], axis=1)


def _cell_ssim(gt_crop, pred_crop, mask, *, min_size: int = 7) -> float:
    """2-D scale-invariant masked SSIM for one cell crop (NaN if too small).

    3-D crops are max-projected to 2-D first; cells smaller than the SSIM
    window in either spatial dim score NaN rather than raising.
    """
    if gt_crop.ndim == 3:
        gt2d = np.max(gt_crop, axis=0)
        pred2d = np.max(pred_crop, axis=0)
        mask2d = np.any(mask, axis=0)
    else:
        gt2d, pred2d, mask2d = gt_crop, pred_crop, mask
    if min(gt2d.shape[-2:]) < min_size:
        return float("nan")
    return float(cubic_ssim(gt2d, pred2d, win_size=min_size, mask=mask2d, scale_invariant=True))


def per_cell_similarity(
    predict_t,
    target_t,
    cell_segmentation_t,
    *,
    metrics: tuple[str, ...] = ("pcc",),
    reduce: tuple[str, ...] = ("mean", "median"),
    use_gpu: bool = True,
) -> dict[str, float]:
    """Per-cell paired GT-vs-pred image similarity, reduced over cells.

    Crops each cell's native bbox from GT and prediction, scores a paired,
    scale-invariant similarity inside the mask (``pcc`` is affine-invariant and
    mask-aware; optional ``ssim`` is 2-D scale-invariant with a size guard), and
    NaN-reduces over cells. Returns ``{f"PerCell_{METRIC}_{reduce}": value}``.
    Unlike the CP feature vector this is a *paired* metric (one scalar per
    cell), aggregated like the pixel metrics — it cannot feed FID/KID.
    """
    _require_cubic()
    use_cuda = bool(use_gpu and torch.cuda.is_available())
    to_xp = ascupy if use_cuda else asnumpy
    pred = to_xp(predict_t)
    tgt = to_xp(target_t)
    lab = to_xp(cell_segmentation_t)
    lab_host = asnumpy(lab)
    objects = _region_slices(lab_host)

    per_metric: dict[str, list[float]] = {m: [] for m in metrics}
    for lab_id in np.unique(lab_host):
        if lab_id == 0:
            continue
        slices = objects[int(lab_id) - 1]
        mask = lab[slices] == int(lab_id)
        gt_crop = tgt[slices]
        pred_crop = pred[slices]
        if "pcc" in metrics:
            per_metric["pcc"].append(float(pcc(gt_crop, pred_crop, mask=mask)))
        if "ssim" in metrics:
            per_metric["ssim"].append(_cell_ssim(gt_crop, pred_crop, mask))

    out: dict[str, float] = {}
    for m in metrics:
        vals = np.asarray(per_metric[m], dtype=float)
        finite = vals[np.isfinite(vals)]
        for r in reduce:
            key = f"PerCell_{m.upper()}_{r}"
            if finite.size == 0:
                out[key] = float("nan")
            elif r == "mean":
                out[key] = float(finite.mean())
            elif r == "median":
                out[key] = float(np.median(finite))
            else:
                raise ValueError(f"unknown reduce {r!r}; use 'mean' or 'median'")
    return out


def _build_per_cell_crops_2d(img_2d, cell_segmentation_3d, patch_size):
    """Build per-cell masked 2-D crops shared across deep-feature extractors.

    Iteration order is ``np.unique(cell_segmentation_3d)`` with the
    background label ``0`` skipped. The result is a list of
    ``(patch_size, patch_size)`` arrays, one per non-background cell.
    """
    crops: list[np.ndarray] = []
    for idx in np.unique(cell_segmentation_3d):
        if idx == 0:
            continue
        cell_mask_2d = np.any(cell_segmentation_3d == idx, axis=0)
        yx_coords = np.argwhere(cell_mask_2d)
        if len(yx_coords) == 0:
            continue
        com_y, com_x = np.mean(yx_coords, axis=0).astype(int)
        half_patch = patch_size // 2
        y_start, y_end = com_y - half_patch, com_y + half_patch
        x_start, x_end = com_x - half_patch, com_x + half_patch
        pad_y_before = max(0, -y_start)
        pad_y_after = max(0, y_end - img_2d.shape[0])
        pad_x_before = max(0, -x_start)
        pad_x_after = max(0, x_end - img_2d.shape[1])
        y_slice = slice(max(0, y_start), min(img_2d.shape[0], y_end))
        x_slice = slice(max(0, x_start), min(img_2d.shape[1], x_end))
        cell_crop = (img_2d * cell_mask_2d)[y_slice, x_slice]
        if pad_y_before or pad_y_after or pad_x_before or pad_x_after:
            pad = ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after))
            cell_crop = np.pad(cell_crop, pad, mode="constant")
        crops.append(cell_crop)
    return crops


def features_from_crops(crops, feature_extractor):
    """Run a deep-feature extractor over a list of masked 2-D crops.

    Uses ``feature_extractor.extract_features_batch(crops)`` when the
    extractor provides it; otherwise falls back to per-cell calls. The
    batch path lets each extractor stack all cells of a (FOV, t) into a
    single forward, amortizing Python overhead and letting cuDNN pick
    wider kernels.

    Extractor contract
    ------------------
    Both code paths require the extractor to return a ``torch.Tensor``
    (``.detach().cpu()`` is called on the result). ``extract_features``
    must return one tensor per crop; ``extract_features_batch`` must
    return a stacked tensor whose leading dim equals ``len(crops)``.
    """
    if not crops:
        return np.empty((0, 0), dtype=np.float32)
    batch_fn = getattr(feature_extractor, "extract_features_batch", None)
    if batch_fn is not None:
        out = batch_fn(crops)
        return np.asarray(out.detach().cpu()).reshape(len(crops), -1).astype(np.float32, copy=False)
    feats = [feature_extractor.extract_features(c).detach().cpu().numpy().reshape(-1) for c in crops]
    return np.stack(feats, axis=0)


def build_crops(image, cell_segmentation, patch_size):
    """Compute the 2-D max-z projection + per-cell crops for one image.

    Shared by every deep-feature extractor in the eval pipeline so the
    max-projection, cell iteration, and crop construction run once per
    (FOV, timepoint) instead of once per backbone.
    """
    if image.shape != cell_segmentation.shape:
        raise ValueError(f"Shape mismatch: image {image.shape} vs cell_segmentation {cell_segmentation.shape}")
    image_2d = _minmax_norm(np.max(image, axis=0))
    return _build_per_cell_crops_2d(image_2d, cell_segmentation, patch_size)


def deep_features(image, cell_segmentation, feature_extractor, patch_size):
    """Per-cell deep embeddings for one image, shape ``(n_cells, feature_dim)``.

    Prefer :func:`build_crops` + :func:`features_from_crops` when the same
    crops feed multiple extractors.
    """
    crops = build_crops(image, cell_segmentation, patch_size)
    return features_from_crops(crops, feature_extractor)
