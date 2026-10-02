"""Metric computation for evaluation: pixel metrics, mask metrics, MicroMS3IM."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

try:
    from cubic.cuda import ascupy, asnumpy
    from cubic.feature import glcm_features
    from cubic.feature.voxel import regionprops_table
    from cubic.metrics import frc_resolution, fsc_resolution, nrmse, pcc, psnr
    from cubic.metrics import ssim as cubic_ssim  # aliased — dynacell keeps a local ssim() wrapper
    from cubic.metrics.bandlimited import spectral_pcc
    from cubic.metrics.microssim import compute_norm_parameters, normalize_min_max
    from cubic.scipy import ndimage as _cubic_ndimage
    from cubic.skimage import filters as _cubic_filters
except ImportError:
    ascupy = None  # type: ignore[assignment]
    asnumpy = None  # type: ignore[assignment]
    compute_norm_parameters = None  # type: ignore[assignment]
    cubic_ssim = None  # type: ignore[assignment]
    frc_resolution = None  # type: ignore[assignment]
    fsc_resolution = None  # type: ignore[assignment]
    glcm_features = None  # type: ignore[assignment]
    normalize_min_max = None  # type: ignore[assignment]
    nrmse = None  # type: ignore[assignment]
    pcc = None  # type: ignore[assignment]
    psnr = None  # type: ignore[assignment]
    regionprops_table = None  # type: ignore[assignment]
    spectral_pcc = None  # type: ignore[assignment]
    _cubic_filters = None  # type: ignore[assignment]
    _cubic_ndimage = None  # type: ignore[assignment]


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


def _min_max_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalize a tensor to [0, 1] range; called inside :func:`ssim`'s inference mode."""
    import torch

    x = x.float()
    return (x - x.min()) / torch.clamp(x.max() - x.min(), min=eps)


def ssim(img1: torch.Tensor, img2: torch.Tensor, *, scale_invariant: bool = True, eps: float = 1e-8) -> float:
    """Compute mean structural similarity index (SSIM) for 2D or 3D inputs.

    ``spatial_dims`` is dispatched from the input rank (cubic convention): a 2-D
    ``(H, W)`` input scores an in-plane SSIM, a 3-D ``(D, H, W)`` input scores a
    volumetric SSIM.

    Two scorings, both reported by :func:`compute_pixel_metrics`:

    ``scale_invariant=True`` (default)
        The prediction is fitted to the target by the least-squares affine gain
        that best matches it, and ``data_range`` is derived from the *target*
        alone. Reports agreement with the target rather than the prediction's
        dynamic range, and matches ``per_cell_similarity``, which has always
        scored this way. Reported as ``SI_SSIM``.
    ``scale_invariant=False``
        Each input is min-max rescaled to ``[0, 1]`` by *its own* extremes before
        scoring against ``data_range=1.0``. Two outlier voxels set the two scales
        independently, so a prediction whose dynamic range merely differs from the
        target's is penalized as if it disagreed with it — the metric is partly a
        dynamic-range comparison. Kept because it is the scale-*sensitive*
        convention the published benchmark tables were built on, so both forms can
        be reported side by side. Reported as ``SSIM``.

    Parameters
    ----------
    img1, img2 : torch.Tensor
        2-D ``(H, W)`` or 3-D ``(D, H, W)`` tensors of the same shape. ``img1``
        is the target: it sets the scale-invariant reference.
    scale_invariant : bool
        Select the scoring above.
    eps : float
        Min-max denominator floor; used only when ``scale_invariant=False``.
    """
    # torch is imported lazily: it costs seconds, and cp_reference and the paper
    # scripts import this module only for CP feature names and cache helpers.
    import torch

    with torch.inference_mode():
        if cubic_ssim is None:
            raise ImportError("cubic is required for SSIM. Install via the `eval` extra: `uv sync --extra eval`.")
        if img1.ndim not in (2, 3):
            raise ValueError(
                f"ssim expects 2-D (H, W) or 3-D (D, H, W) input, got {img1.ndim}-D tensor of shape {tuple(img1.shape)}"
            )
        spatial_dims = img1.ndim

        if not scale_invariant:
            img1 = _min_max_normalize(img1, eps=eps)
            img2 = _min_max_normalize(img2, eps=eps)

        # cubic's batched dispatch expects [N, C, (D,) H, W] (ndim = spatial_dims + 2):
        # (H,W) → (1,1,H,W); (D,H,W) → (1,1,D,H,W).
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)

        if scale_invariant:
            # No data_range here: the scale_invariant path derives its own, and cubic
            # raises if both are supplied.
            return cubic_ssim(img1, img2, spatial_dims=spatial_dims, gaussian_weights=True, scale_invariant=True)
        return cubic_ssim(img1, img2, spatial_dims=spatial_dims, data_range=1.0, gaussian_weights=True)


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


def compute_pixel_metrics(
    prediction, target, spacing, fsc_kwargs=None, spectral_pcc_kwargs=None, use_gpu=True, foreground=None
):
    """Compute pixel-level image quality metrics between prediction and target.

    Parameters
    ----------
    foreground : dict or None
        ``{"source", "smooth_sigma_um", "feather_sigma_um"}`` for
        :func:`foreground_weight`. When given, the foreground-limited columns of
        :func:`foreground_pixel_metrics` (``FG_*`` and ``FG_frac``) are added,
        scored against a weight map built from ``target`` alone. ``None``
        (default) adds nothing, so the returned dict is unchanged.

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
    import torch

    use_cuda = bool(use_gpu and torch.cuda.is_available())
    to_xp = ascupy if use_cuda else asnumpy
    pred_xp, target_xp = to_xp(prediction), to_xp(target)

    # ``.contiguous()`` recovers the contiguity guarantee the previous
    # ``.to(device)`` step provided: when ``target_xp`` is a non-contiguous
    # cupy view (e.g. a strided zarr slice), cubic_ssim → MONAI → conv3d's
    # CUDA backend can fail or silently re-materialize on recent torch.
    #
    # SSIM / NRMSE / PSNR are reported in BOTH scalings, from the same arrays in
    # the same pass, so the two columns of a table row can never pair values from
    # different predictions:
    #   - bare ``SSIM``/``NRMSE``/``PSNR`` are scale-SENSITIVE (``normalize="min_max"``:
    #     each array rescaled by its own extremes). This is the convention the
    #     published benchmark tables were built on. Two outlier voxels can set the
    #     denominator: on a hot-pixel probe min-max inflated NRMSE by 7048% against
    #     2224% for the scale-invariant form.
    #   - ``SI_*`` are scale-INVARIANT (least-squares affine fit of the prediction
    #     to the target, ``data_range`` from the target alone), so they measure
    #     agreement with the target rather than the prediction's dynamic range.
    # PCC needs no pair -- it is already affine-invariant, and is the fixed anchor
    # that shows the two scalings differ only in what they were meant to differ in.
    target_pt = torch.as_tensor(target_xp).contiguous()
    pred_pt = torch.as_tensor(pred_xp).contiguous()
    metrics = {
        "PCC": pcc(target_xp, pred_xp),
        "SSIM": ssim(target_pt, pred_pt, scale_invariant=False),
        "NRMSE": nrmse(target_xp, pred_xp, normalize="min_max"),
        "PSNR": psnr(target_xp, pred_xp, normalize="min_max"),
        "SI_SSIM": ssim(target_pt, pred_pt, scale_invariant=True),
        "SI_NRMSE": nrmse(target_xp, pred_xp, scale_invariant=True),
        "SI_PSNR": psnr(target_xp, pred_xp, scale_invariant=True),
    }
    if foreground is not None:
        weight = foreground_weight(target_xp, spacing, **foreground)
        metrics.update(foreground_pixel_metrics(pred_xp, target_xp, weight))

    if spectral_pcc_kwargs is None and fsc_kwargs is None:
        return metrics

    # Match the frequency-domain metrics to the input rank (cubic convention):
    # the in-focus 2D path passes (H, W) arrays → trailing YX spacing + the
    # ring-based FRC; the full-3D path keeps the (Z, Y, X) spacing + shell-based
    # FSC. The 3D path is byte-identical to before (ndim==3 → full spacing, FSC).
    ndim = pred_xp.ndim
    freq_spacing = list(spacing)[-ndim:]

    if spectral_pcc_kwargs is not None:
        metrics["Spectral_PCC"] = spectral_pcc(pred_xp, target_xp, spacing=freq_spacing, **spectral_pcc_kwargs)
    if fsc_kwargs is not None:
        # cubic.{fsc,frc}_resolution mean-center internally before every FFT,
        # so we pass the raw arrays.
        if ndim == 2:
            metrics["FRC_Resolution"] = float(frc_resolution(target_xp, pred_xp, spacing=freq_spacing, **fsc_kwargs))
        else:
            resolutions = fsc_resolution(target_xp, pred_xp, spacing=freq_spacing, **fsc_kwargs)
            metrics.update({f"{k.upper()}_FSC_Resolution": float(v) for k, v in resolutions.items()})

    return metrics


#: Foreground sources :func:`foreground_weight` accepts.
FOREGROUND_SOURCES = ("smooth_otsu", "otsu")
#: Columns :func:`foreground_pixel_metrics` returns, in order.
FOREGROUND_COLUMNS = ("FG_PCC", "FG_SI_SSIM", "FG_SI_NRMSE", "FG_SI_PSNR", "FG_frac")
#: Default ``smooth_otsu`` sigmas (um) per ``target_name``, used when
#: ``pixel_metrics.foreground`` leaves a sigma null. Chosen 2026-10-01 from XY + XZ
#: figures on DynaCell-lite (iPSC, 50 FOVs; A549 mock, 12 FOVs): 0.5 um smoothing
#: gives clean nuclei (raw Otsu is speckled on the noisy iPSC GT; 1 um merges
#: neighbours), membrane walls + the basal sheet, and mitochondria-rich regions; ER
#: takes 1 um so the foreground is the ER-filled cytoplasm, gaps between tubules
#: included, with nuclei excluded. Median ``FG_frac`` (iPSC / A549 mock): nucleus
#: 0.25 / 0.05, membrane 0.32 / 0.21, ER 0.40 / 0.19, mitochondria 0.13 / 0.09.
FOREGROUND_SIGMAS_UM: dict[str, dict[str, float]] = {
    "nucleus": {"smooth_sigma_um": 0.5, "feather_sigma_um": 0.5},
    "membrane": {"smooth_sigma_um": 0.5, "feather_sigma_um": 0.5},
    "er": {"smooth_sigma_um": 1.0, "feather_sigma_um": 0.5},
    "mitochondria": {"smooth_sigma_um": 0.5, "feather_sigma_um": 0.5},
}

# The SSIM window of the whole-image ``SI_SSIM``: skimage's ``gaussian_weights=True``
# (sigma 1.5, truncate 3.5 -> an 11-voxel window, K1/K2 defaults). ``FG_SI_SSIM``
# uses the same window so the two columns differ only in where they look.
_SSIM_SIGMA = 1.5
_SSIM_TRUNCATE = 3.5
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03
#: Weight above which a voxel counts as hard foreground (data range, emptiness).
_HARD_FOREGROUND = 0.5
#: Weighted prediction variance below which the prediction is constant in the foreground.
_CONSTANT_VARIANCE = 1e-12


def _voxel_sigma(sigma_um: float, spacing: Sequence[float]) -> tuple[float, ...]:
    """Convert an isotropic physical sigma (um) to per-axis voxel sigmas."""
    return tuple(float(sigma_um) / float(s) for s in spacing)


def foreground_weight(
    target,
    spacing: Sequence[float],
    *,
    source: str = "smooth_otsu",
    smooth_sigma_um: float = 0.0,
    feather_sigma_um: float = 0.0,
):
    """Soft foreground weight map in ``[0, 1]`` built from the ground truth alone.

    Every prediction scored against one target sees the same weight map, so
    foreground-limited columns compare models on one region.

    ``smooth_otsu``: Gaussian-smooth the target (``smooth_sigma_um``), Otsu-threshold
    the smoothed volume, then feather the binary mask with a Gaussian
    (``feather_sigma_um``). ``otsu``: Otsu on the raw target (Spotlight v1's mask),
    feathered the same way; ``smooth_sigma_um`` is ignored. Sigmas are physical and
    isotropic; each axis gets ``sigma_um / spacing``. A sigma of 0 skips its step,
    so ``feather_sigma_um=0`` returns the hard mask.

    Parameters
    ----------
    target : array
        2-D ``(H, W)`` or 3-D ``(D, H, W)`` NumPy or CuPy array.
    spacing : sequence of float
        Physical voxel spacing (um); its trailing ``target.ndim`` entries are used.
    source : {"smooth_otsu", "otsu"}
        Foreground recipe, see above.
    smooth_sigma_um, feather_sigma_um : float
        Non-negative Gaussian sigmas in um.

    Returns
    -------
    array
        Weight map on ``target``'s device, float32 (float64 for a float64 target).
        Exactly constant z-planes (zero padding) are neither thresholded nor
        foreground. All zeros when the (smoothed) target is constant: there is no
        foreground.

    Raises
    ------
    ValueError
        On an unknown ``source``, a negative sigma, or a target that is not 2-D/3-D.
    """
    _require_cubic()
    if source not in FOREGROUND_SOURCES:
        raise ValueError(f"foreground source must be one of {FOREGROUND_SOURCES}; got {source!r}")
    if smooth_sigma_um < 0 or feather_sigma_um < 0:
        raise ValueError(
            f"foreground sigmas must be >= 0; got smooth={smooth_sigma_um!r}, feather={feather_sigma_um!r}"
        )
    if target.ndim not in (2, 3):
        raise ValueError(f"foreground_weight expects a 2-D or 3-D target; got shape {tuple(target.shape)}")
    spacing = list(spacing)[-target.ndim :]
    image = target.astype(np.result_type(target.dtype, np.float32), copy=False)
    # Exactly constant z-planes carry no signal: some A549 SEC61B/TOMM20 GT FOVs start
    # with all-zero planes, which on their own form one of Otsu's two classes and turn
    # every other voxel into foreground. Like fit_microssim's constant GT slices, they
    # are left out of the threshold and of the foreground.
    planes = image.reshape(image.shape[0] if image.ndim == 3 else 1, -1)
    signal = planes.max(axis=1) > planes.min(axis=1)
    if not bool(signal.any()):
        return np.zeros_like(image)
    if source == "smooth_otsu" and smooth_sigma_um > 0:
        sigma = _voxel_sigma(smooth_sigma_um, spacing)
        if bool(signal.all()):
            image = _cubic_ndimage.gaussian_filter(image, sigma=sigma, mode="reflect")
        else:
            # Average over signal planes only (normalized convolution): blurring the
            # padding into its neighbours would dim them into a third Otsu class.
            support = np.broadcast_to(signal.reshape(-1, 1, 1), image.shape).astype(image.dtype)
            mass = _cubic_ndimage.gaussian_filter(support, sigma=sigma, mode="reflect")
            image = _cubic_ndimage.gaussian_filter(image * support, sigma=sigma, mode="reflect")
            image = image / np.where(mass > 0, mass, 1.0)
    values = image[signal] if image.ndim == 3 else image
    if not float(values.max()) > float(values.min()):
        return np.zeros_like(image)
    binary = image > float(_cubic_filters.threshold_otsu(values))
    if image.ndim == 3:
        binary[~signal] = False
    weight = binary.astype(image.dtype)
    if feather_sigma_um > 0:
        weight = _cubic_ndimage.gaussian_filter(weight, sigma=_voxel_sigma(feather_sigma_um, spacing), mode="reflect")
        weight = np.clip(weight, 0.0, 1.0)
    return weight


def _weighted_mean(values, weight, total: float) -> float:
    """``sum(weight * values) / total``, accumulated in float64."""
    return float((values * weight).sum(dtype=np.float64)) / total


def _weighted_ssim(image_true, image_test, weight, data_range: float) -> float:
    """Normalized-convolution SSIM averaged with ``weight``.

    Local means, variances and covariance come from ``weight``-weighted Gaussian
    windows normalized by the local weight mass (normalized convolution), so
    out-of-foreground voxels inform no local statistic and nothing is zeroed or
    eroded. The per-voxel SSIM is then averaged with ``weight`` over skimage's
    border crop. This follows the masked SSIM of Gao et al. 2022 (dycheck), which
    instead normalizes by the count of mask pixels in the window (partial
    convolution) and averages the map over every pixel, so windows holding no mask
    score 1 there. Window, ``K1``/``K2`` and the sample-covariance factor are
    skimage's, so ``weight == 1`` reproduces
    ``skimage.metrics.structural_similarity(..., gaussian_weights=True)``.
    Returns NaN when the crop holds no weight (all foreground within the border).
    """
    win_size = 2 * int(_SSIM_TRUNCATE * _SSIM_SIGMA + 0.5) + 1
    if min(image_true.shape) < win_size:
        raise ValueError(f"FG_SI_SSIM needs every axis >= {win_size}; got shape {tuple(image_true.shape)}")

    def blur(x):
        return _cubic_ndimage.gaussian_filter(x, sigma=_SSIM_SIGMA, truncate=_SSIM_TRUNCATE, mode="reflect")

    # A voxel with weight > 0 has mass >= (kernel centre) * weight > 0; a voxel whose
    # window holds no weight gets a placeholder mass and is dropped from the mean below.
    mass = blur(weight)
    inv_mass = (mass > 0) / np.where(mass > 0, mass, 1.0)
    del mass
    mu_true = blur(weight * image_true) * inv_mass
    mu_test = blur(weight * image_test) * inv_mass
    cov_norm = win_size**image_true.ndim / (win_size**image_true.ndim - 1.0)
    var_true = cov_norm * (blur(weight * image_true * image_true) * inv_mass - mu_true * mu_true)
    var_test = cov_norm * (blur(weight * image_test * image_test) * inv_mass - mu_test * mu_test)
    covar = cov_norm * (blur(weight * image_true * image_test) * inv_mass - mu_true * mu_test)
    del inv_mass
    c1 = (_SSIM_K1 * data_range) ** 2
    c2 = (_SSIM_K2 * data_range) ** 2
    ssim_map = ((2 * mu_true * mu_test + c1) * (2 * covar + c2)) / (
        (mu_true * mu_true + mu_test * mu_test + c1) * (var_true + var_test + c2)
    )
    pad = (win_size - 1) // 2
    crop = tuple(slice(pad, n - pad) for n in ssim_map.shape)
    ssim_map, weight = ssim_map[crop], weight[crop]
    total = float(weight.sum(dtype=np.float64))
    if total == 0:
        return float("nan")
    weighted = np.where(weight > 0, weight * ssim_map, 0.0)
    return float(weighted.sum(dtype=np.float64)) / total


def foreground_pixel_metrics(prediction, target, weight) -> dict[str, float]:
    """Scale-invariant pixel metrics restricted to a soft foreground weight map.

    The prediction is fitted to the target by a ``weight``-weighted least-squares
    affine fit (the form of cubic's ``scale_invariant`` branch: target standardized
    by its weighted mean and std, centred prediction scaled by the weighted
    least-squares gain), and every column is scored against ``weight``:

    - ``FG_PCC``: weighted Pearson correlation.
    - ``FG_SI_NRMSE``, ``FG_SI_PSNR``: weighted MSE; ``data_range`` is the target's
      range inside the hard foreground (``weight > 0.5``) over its weighted std.
    - ``FG_SI_SSIM``: normalized-convolution SSIM (see :func:`_weighted_ssim`) on the
      same scaled pair and ``data_range``.
    - ``FG_frac``: ``mean(weight)``.

    ``weight == 1`` reproduces ``PCC``, ``SI_NRMSE``, ``SI_PSNR`` and ``SI_SSIM``.
    Nothing is zeroed or eroded, so a thin foreground keeps its score. A feathered
    weight reaches past the GT boundary, so the object edges count too: the columns
    reward placing the objects as well as matching their interior (a hard weight,
    ``feather_sigma_um=0``, scores the interior alone).

    Degenerate inputs return values instead of raising: no hard foreground (or a
    constant target inside it) gives NaN for every ``FG_*`` column; a prediction
    constant inside the foreground gives ``FG_PCC = 0`` and a zero gain (its best
    affine fit is the target's mean), so a collapsed prediction is scored, not
    dropped from a mean. A foreground lying wholly inside SSIM's border crop gives
    NaN for ``FG_SI_SSIM`` alone.

    Parameters
    ----------
    prediction, target : array
        Same-shape 2-D or 3-D NumPy or CuPy arrays.
    weight : array
        Weight map in ``[0, 1]`` of the same shape and device, e.g. from
        :func:`foreground_weight` on ``target``.

    Returns
    -------
    dict[str, float]
        The :data:`FOREGROUND_COLUMNS`.
    """
    if not prediction.shape == target.shape == weight.shape:
        raise ValueError(
            f"Shape mismatch: prediction {tuple(prediction.shape)}, target {tuple(target.shape)}, "
            f"weight {tuple(weight.shape)}"
        )
    frac = float(weight.mean(dtype=np.float64))
    empty = dict.fromkeys(FOREGROUND_COLUMNS[:-1], float("nan")) | {"FG_frac": frac}
    hard = weight > _HARD_FOREGROUND
    if not bool(hard.any()):
        return empty
    dtype = np.result_type(target.dtype, prediction.dtype, np.float32)
    target = target.astype(dtype, copy=False)
    prediction = prediction.astype(dtype, copy=False)
    weight = weight.astype(dtype, copy=False)
    total = float(weight.sum(dtype=np.float64))

    # Python-float scalars keep the arrays in their own precision; reductions run in float64.
    target_zero = target - _weighted_mean(target, weight, total)
    target_std = _weighted_mean(target_zero * target_zero, weight, total) ** 0.5
    if target_std == 0:
        return empty
    target_norm = target_zero / target_std
    del target_zero
    pred_zero = prediction - _weighted_mean(prediction, weight, total)
    pred_var = _weighted_mean(pred_zero * pred_zero, weight, total)
    if pred_var < _CONSTANT_VARIANCE:
        fg_pcc, gain = 0.0, 0.0
    else:
        covar = _weighted_mean(target_norm * pred_zero, weight, total)
        fg_pcc = float(np.clip(covar / pred_var**0.5, -1.0, 1.0))
        gain = covar / pred_var
    pred_scaled = pred_zero * gain
    del pred_zero
    data_range = float(target[hard].max() - target[hard].min()) / target_std
    residual = target_norm - pred_scaled
    mse = _weighted_mean(residual * residual, weight, total)
    del residual
    return {
        "FG_PCC": fg_pcc,
        "FG_SI_SSIM": _weighted_ssim(target_norm, pred_scaled, weight, data_range),
        "FG_SI_NRMSE": mse**0.5 / data_range,
        "FG_SI_PSNR": 10 * float(np.log10(data_range**2 / mse)) if mse > 0 else float("inf"),
        "FG_frac": frac,
    }


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
    MicroMS3IM or None
        Fitted instance — ``sim.score(target_slice, pred_slice)`` may
        then be called without further fitting. Constant target slices
        carry no data range, so they are dropped from the pool before
        fitting (``score_microssim`` still scores them 0.0). ``None`` when
        every slice is constant, or when a remaining normalized target
        slice has a data range that is not finite and positive (a
        non-finite GT pixel, or a degenerate pool normalization), where α
        is undefined.
    """
    MicroMS3IM = _require_microms3im()
    import torch

    # Convert to cupy when GPU is requested — cubic.skimage dispatches to
    # cucim (GPU Gaussian filters) when inputs carry a .device attribute.
    to_xp = ascupy if (use_gpu and ascupy is not None and torch.cuda.is_available()) else asnumpy
    targets = to_xp(targets)
    predictions = to_xp(predictions)
    # A constant GT slice (the all-zero z-slices in A549 TOMM20_mock.zarr) has no
    # data range, so cubic cannot fit α on it. Drop it from the pool and fit on the
    # rest. Only exact constants go: a NaN/inf GT pixel gives a non-finite range,
    # which is kept so the check below still makes the leaf NaN.
    constant = targets.max(axis=(1, 2)) - targets.min(axis=(1, 2)) == 0
    if constant.all():
        print(
            f"[microssim] all {len(constant)} calibration GT slices are constant; MicroMS3IM will be NaN for all FOVs."
        )
        return None
    if constant.any():
        print(f"[microssim] dropped {int(constant.sum())} of {len(constant)} constant calibration GT slices.")
        targets = targets[~constant]
        predictions = predictions[~constant]
    # cubic fits α with each normalized GT slice's own data_range (max - min) and
    # raises unless it is finite and positive: a non-finite GT pixel, or a pool
    # whose background percentile equals its max (max_val == 0). The normalization
    # is monotone, so normalizing each slice's extrema with cubic's own parameters
    # reproduces that range; the same parameters are then handed to the fit.
    offset_gt, offset_pred, max_val = compute_norm_parameters(targets, predictions)
    with np.errstate(divide="ignore", invalid="ignore"):
        gt_range = asnumpy(
            normalize_min_max(targets.max(axis=(1, 2)), offset_gt, max_val)
            - normalize_min_max(targets.min(axis=(1, 2)), offset_gt, max_val)
        )
    degenerate = ~(np.isfinite(gt_range) & (gt_range > 0))
    if degenerate.any():
        print(
            f"[microssim] {int(degenerate.sum())} of {len(gt_range)} calibration GT slices have a "
            "normalized data range that is not finite and positive; MicroMS3IM will be NaN for all FOVs."
        )
        return None
    sim = MicroMS3IM(offset_gt=offset_gt, offset_pred=offset_pred, max_val=max_val)
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
    import torch

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

    Replaces the raw min-max this track used to run on, which a single hot pixel
    could dominate. Device-agnostic — ``np.percentile``/``np.clip`` dispatch on
    numpy or cupy. The clipped numerator is bounded by the span, so the ``+ eps``
    denominator keeps a constant/near-constant image finite (output → 0) instead
    of NaN/inf.
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

#: The CP column order each recipe version wrote, frozen as literals (GLCM on; with
#: GLCM off the ``glcm_*`` columns are absent). CP caches written before the feature
#: names were recorded in the manifest carry only ``cp_feature_version`` +
#: ``cp_glcm_enabled``, and this table reads their exact column order back from
#: those -- no re-cache needed. It is deliberately NOT derived from
#: :func:`active_cp_feature_names`: a reorder of the live tuples without a version
#: bump then disagrees with this table (and fails its test), instead of
#: silently re-labelling old caches (see tests/test_cp_feature_names.py).
CP_FEATURE_NAMES_BY_VERSION: dict[str, tuple[str, ...]] = {
    "v2_dist_texture": (
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
        "glcm_contrast",
        "glcm_dissimilarity",
        "glcm_homogeneity",
        "glcm_ASM",
        "glcm_energy",
        "glcm_correlation",
        "glcm_entropy",
    ),
}


def active_cp_feature_names(glcm_enabled: bool) -> tuple[str, ...]:
    """Return the ordered CP column names for the active config.

    The schema is GLCM-dependent: the base distribution/texture columns are
    always emitted; the seven ``glcm_*`` columns are appended only when GLCM is
    enabled. Used by both the matrix assembly and the CP reference's recipe
    identity (``cp_reference.cp_space``), so a reference built for another
    column set is refused instead of silently misaligned.
    """
    if glcm_enabled:
        return _CP_BASE_FEATURE_NAMES + _CP_GLCM_FEATURE_NAMES
    return _CP_BASE_FEATURE_NAMES


#: CP columns whose value depends on the device that computed them. cuCIM's GPU
#: regionprops reduces the built-in min/max in float32, while skimage on CPU keeps
#: float64 (~1e-8 rel apart; every other column agrees to <= 1e-14). That is enough
#: to move the GT-only variance filter: a cell holding the p99-clipped pixel has
#: intensity_max exactly 1.0 on GPU but ``1 - eps / (hi - lo)`` on CPU, which
#: changes the exact-tie counts and hence the feature mask.
_DEVICE_DEPENDENT_CP_COLUMNS: tuple[str, ...] = ("intensity_min", "intensity_max")


def round_device_dependent_cp_columns(features: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Round the device-dependent CP columns to float32, returning float64.

    The single definition of the CP min/max rounding, applied both by
    :func:`cp_regionprops` and to every CP array read from a cache. It aligns
    CPU to GPU, the device that built every production CP cache; on GPU output
    (already float32-exact) it is a no-op, and on a CPU-built cache it yields
    exactly what the fixed extractor computes, so such a cache needs no
    recompute.

    Parameters
    ----------
    features : np.ndarray
        ``(n_cells, n_features)`` CP matrix. A zero-row array (including the
        ``(0, 0)`` empty-FOV cache sentinel) is returned unchanged.
    feature_names : sequence of str
        Column names of ``features``, in order; the columns are located by name.

    Returns
    -------
    np.ndarray
        A float64 copy with :data:`_DEVICE_DEPENDENT_CP_COLUMNS` rounded.

    Raises
    ------
    ValueError
        If ``features`` is not 2-D or its column count does not match
        ``feature_names``.
    """
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"CP features must be 2-D (n_cells, n_features); got shape {features.shape}")
    if features.shape[0] == 0:
        return features
    names = list(feature_names)
    if features.shape[1] != len(names):
        raise ValueError(f"CP features have {features.shape[1]} columns but {len(names)} names: {names}")
    out = features.astype(np.float64, copy=True)
    for name in _DEVICE_DEPENDENT_CP_COLUMNS:
        j = names.index(name)
        out[:, j] = out[:, j].astype(np.float32).astype(np.float64)
    return out


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

    names = active_cp_feature_names(glcm_enabled)
    # Drop 1-voxel regions on both devices: cuCIM's GPU regionprops_table raises
    # TypeError on one, and GLCM raises ValueError (no voxel pair) on either device.
    # Their CP row was non-finite (zero std -> NaN skewness/kurtosis), and every
    # metric already dropped non-finite rows. The kept labels are renumbered 1..n
    # in order, because a gap in the label sequence makes the GPU extra_properties
    # callbacks hit an illegal memory access.
    labels_host = asnumpy(cell_segmentation)
    counts = np.bincount(labels_host.ravel())
    kept = np.flatnonzero(counts[1:] >= 2) + 1
    # cuCIM's GPU regionprops_dict indexes results[0] on an empty list and raises
    # IndexError when the label image has no regions, so short-circuit a no-cell
    # FOV before any regionprops_table call. Some iPSC nucleus/membrane FOVs have
    # an empty cleaned segmentation; er/mito FOVs always have cells.
    if kept.size == 0:
        return np.empty((0, len(names)), dtype=float)
    # Labels already 1..n with none dropped (the common case): skip the full-volume copy.
    if kept.size < counts.size - 1:
        relabel = np.zeros(counts.size, dtype=labels_host.dtype)
        relabel[kept] = np.arange(1, kept.size + 1)
        cell_segmentation = relabel[labels_host]

    import torch

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

    if columns["intensity_mean"].shape[0] == 0:
        return np.empty((0, len(names)), dtype=float)
    # Round min/max to float32 so CPU output matches GPU (no-op on GPU).
    return round_device_dependent_cp_columns(
        np.stack([np.asarray(columns[name], dtype=float) for name in names], axis=1), names
    )


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
    z_slab: slice | None = None,
) -> dict[str, float]:
    """Per-cell paired GT-vs-pred image similarity, reduced over cells.

    Crops each cell's native bbox from GT and prediction, scores a paired,
    scale-invariant similarity inside the mask (``pcc`` is affine-invariant and
    mask-aware; optional ``ssim`` is 2-D scale-invariant with a size guard), and
    NaN-reduces over cells. Returns ``{f"PerCell_{METRIC}_{reduce}": value}``.
    Unlike the CP feature vector this is a *paired* metric (one scalar per
    cell), aggregated like the pixel metrics — it cannot feed FID/KID.

    Raises ``ValueError`` for an empty or unknown ``metrics``/``reduce`` rather
    than silently emitting all-NaN (or no) ``PerCell_*`` columns — a silent
    miss would hide a config typo and confuse the final-metrics cache gate,
    which keys off the expected ``PerCell_*`` columns.

    Parameters
    ----------
    z_slab : slice or None
        When provided, restrict the volume (and hence each cell's bbox + the
        ``ssim`` max-Z projection) to this in-focus band of planes before
        scoring. Mirrors :func:`build_crops`'s ``z_slab`` so the per-cell
        similarity and the deep-feature embeddings see the same slab. ``None``
        (default) uses the full stack — the legacy behavior.
    """
    _require_cubic()
    if not metrics or set(metrics) - {"pcc", "ssim"}:
        raise ValueError(f"cell_similarity.metrics must be a non-empty subset of {{'pcc', 'ssim'}}; got {metrics!r}")
    if not reduce or set(reduce) - {"mean", "median"}:
        raise ValueError(f"cell_similarity.reduce must be a non-empty subset of {{'mean', 'median'}}; got {reduce!r}")
    if z_slab is not None:
        predict_t = predict_t[z_slab]
        target_t = target_t[z_slab]
        cell_segmentation_t = cell_segmentation_t[z_slab]
    import torch

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
            else:  # "median" — the only remaining option after up-front validation
                out[key] = float(np.median(finite))
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
    # float32 to match the batch path, so both write the same dtype to the cache.
    return np.stack(feats, axis=0).astype(np.float32, copy=False)


def build_crops(image, cell_segmentation, patch_size, *, z_slab: slice | None = None):
    """Compute the 2-D max-z projection + per-cell crops for one image.

    Shared by every deep-feature extractor in the eval pipeline so the
    max-projection, cell iteration, and crop construction run once per
    (FOV, timepoint) instead of once per backbone.

    The projection is robust-normalized per image (percentile-clip
    ``[1, 99]`` then min-max to ``[0, 1]`` via :func:`_robust_norm`) — the
    same recipe :func:`cp_regionprops` uses — so GT and prediction crops
    land on comparable, outlier-robust ranges before the backbones. Raw
    min-max (the previous recipe) let a single hot/saturated pixel anywhere
    in the max-projection set the scale and compress every cell crop toward
    black, and did so asymmetrically for GT vs prediction (real fluorescence
    carries hot pixels/debris that model outputs rarely reproduce) — a
    GT↔pred intensity-range mismatch injected straight into the features.
    The clip is not affine, so it changes even the z-score-based backbones'
    (CELL-DINO, MorphEm) inputs, not just DINOv3's.

    Parameters
    ----------
    z_slab : slice or None
        When provided, restrict both the max-Z projection and the per-cell
        label footprint to this band of planes — an in-focus slab centered on
        the GT phase focus plane (see :func:`focus.build_focus_slabs`).
        Out-of-focus caps (e.g. A549's all-membrane top planes) are excluded so
        the MIP is not dominated by them. ``None`` (default) projects the whole
        stack — the legacy behavior.
    """
    if image.shape != cell_segmentation.shape:
        raise ValueError(f"Shape mismatch: image {image.shape} vs cell_segmentation {cell_segmentation.shape}")
    if z_slab is not None:
        image = image[z_slab]
        cell_segmentation = cell_segmentation[z_slab]
    # ``_robust_norm`` upcasts to float64 (``np.percentile`` returns float64);
    # cast back to float32 so crops match the float32 model weights. DINOv3's
    # HF processor and DynaCLR (no explicit dtype) would otherwise feed float64
    # into float32 conv/linear layers → "Input type (double) and bias type
    # (float) should be the same". CELL-DINO/MorphEm are already dtype-guarded.
    image_2d = _robust_norm(np.max(image, axis=0)).astype(np.float32, copy=False)
    return _build_per_cell_crops_2d(image_2d, cell_segmentation, patch_size)


def deep_features(image, cell_segmentation, feature_extractor, patch_size, *, z_slab: slice | None = None):
    """Per-cell deep embeddings for one image, shape ``(n_cells, feature_dim)``.

    Prefer :func:`build_crops` + :func:`features_from_crops` when the same
    crops feed multiple extractors. ``z_slab`` is forwarded to
    :func:`build_crops` (in-focus slab projection; ``None`` = full stack).
    """
    crops = build_crops(image, cell_segmentation, patch_size, z_slab=z_slab)
    return features_from_crops(crops, feature_extractor)
