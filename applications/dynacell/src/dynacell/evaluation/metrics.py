"""Metric computation for evaluation: pixel metrics, mask metrics, MicroMS3IM."""

import numpy as np
import torch

try:
    from cubic.cuda import ascupy, asnumpy
    from cubic.feature.voxel import regionprops_table
    from cubic.metrics import fsc_resolution, nrmse, pcc, psnr
    from cubic.metrics import ssim as cubic_ssim  # aliased — dynacell keeps a local ssim() wrapper
    from cubic.metrics.bandlimited import spectral_pcc
except ImportError:
    ascupy = None  # type: ignore[assignment]
    asnumpy = None  # type: ignore[assignment]
    cubic_ssim = None  # type: ignore[assignment]
    fsc_resolution = None  # type: ignore[assignment]
    nrmse = None  # type: ignore[assignment]
    pcc = None  # type: ignore[assignment]
    psnr = None  # type: ignore[assignment]
    regionprops_table = None  # type: ignore[assignment]
    spectral_pcc = None  # type: ignore[assignment]

from dynacell.evaluation.utils import _minmax_norm


def _require_cubic():
    if ascupy is None:
        raise ImportError(
            "cubic is required for resolution and feature metrics. "
            "Install via the `eval` extra: `uv sync --extra eval`."
        )
    try:
        import cucim  # noqa: F401
        import cupy  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"{e.name} is required for GPU-backed metrics. Install cupy-cuda12x "
            "and cucim-cu12 via the `eval_gpu` extra: `uv sync --extra eval_gpu`."
        ) from e


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
    Tensors are moved to the chosen device (GPU when ``use_gpu=True`` and
    CUDA is available, CPU otherwise) and converted to cupy/numpy via
    ``cubic.cuda.ascupy``/``asnumpy`` before all metric calls. cupy arrays
    pass through cubic's ``@scale_invariant`` unchanged, enabling GPU-backed
    computation (via cucim/cupy) for all metrics when CUDA is available.
    Spectral metrics additionally benefit from zero-copy CUDA Array Interface
    transfer.
    """
    if pcc is None:
        raise ImportError("cubic is required for pixel metrics. Install via the `eval` extra: `uv sync --extra eval`.")
    _require_cubic()
    prediction = torch.as_tensor(prediction)
    target = torch.as_tensor(target)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    prediction = prediction.to(device)
    target = target.to(device)
    to_xp = ascupy if device.type == "cuda" else asnumpy
    pred_xp, target_xp = to_xp(prediction), to_xp(target)

    metrics = {
        "PCC": pcc(target_xp, pred_xp),
        "SSIM": ssim(target, prediction),
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
                # bugs and must propagate. ``np.nanmean`` below drops the NaN
                # entries; the whole FOV-T row is NaN only if every slice trips
                # this guard.
                if "data_range" not in str(exc):
                    raise
                slice_scores.append(float("nan"))
        slice_idx += num_slices
        if np.all(np.isnan(slice_scores)):
            scores.append({"MicroMS3IM": float("nan")})
        else:
            scores.append({"MicroMS3IM": float(np.nanmean(slice_scores))})
    return scores


PROPS_3D = (
    "intensity_max",
    "intensity_mean",
    "intensity_min",
    "intensity_std",
    "moments_weighted",
    "moments_weighted_central",
)


def _cp_raw_regionprops(img, cell_segmentation, spacing):
    """Compute raw per-cell regionprops and return a (n_cells, n_props) matrix.

    No normalization, no column-drop, no z-score — callers are responsible for
    supplying already-normalized ``img`` (via :func:`_minmax_norm`). Columns
    follow the order of :data:`PROPS_3D` as flattened by ``regionprops_table``.
    """
    if torch.cuda.is_available():
        img = ascupy(img)
        cell_segmentation = ascupy(cell_segmentation)
    feats = regionprops_table(cell_segmentation, img, spacing=spacing, properties=list(PROPS_3D))
    feats.pop("label", None)
    if torch.cuda.is_available():
        return np.array([asnumpy(v) for v in feats.values()]).T
    return np.array(list(feats.values())).T


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


def cp_regionprops(image, cell_segmentation, spacing):
    """Raw CP regionprops for one image and its cell segmentation.

    Returns an array of shape ``(n_cells, n_props_raw)``. Same body for GT
    and prediction sides — *image* is min/max normalized before extraction.
    """
    _require_cubic()
    return _cp_raw_regionprops(_minmax_norm(image), cell_segmentation, spacing)


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
