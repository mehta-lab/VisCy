"""Metric computation for evaluation: pixel metrics, mask metrics, MicroSSIM."""

import numpy as np
import torch

try:
    from microssim import MicroMS3IM
except ImportError:
    MicroMS3IM = None  # type: ignore[assignment, misc]

try:
    from cubic.cuda import ascupy, asnumpy
    from cubic.feature.voxel import regionprops_table
    from cubic.metrics import fsc_resolution
    from cubic.metrics.bandlimited import spectral_pcc
except ImportError:
    ascupy = None  # type: ignore[assignment]
    asnumpy = None  # type: ignore[assignment]
    fsc_resolution = None  # type: ignore[assignment]
    regionprops_table = None  # type: ignore[assignment]
    spectral_pcc = None  # type: ignore[assignment]

from dynacell.evaluation.torch_ssim import ssim as torch_ssim
from dynacell.evaluation.utils import _minmax_norm, _pairwise_feature_metrics


def _require_microssim():
    if MicroMS3IM is None:
        raise ImportError("microssim is required for MicroMS3IM computation. Install it with: pip install microssim")


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
def _normalize_to_target_scale(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map both tensors onto the target's intensity scale."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    y_true = y_true.float()
    y_pred = y_pred.float()

    target_min = y_true.min()
    target_range = y_true.max() - target_min
    denom = target_range.clamp_min(eps)

    return (y_true - target_min) / denom, (y_pred - target_min) / denom

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
def corr_coef(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Calculate the Pearson correlation coefficient between two PyTorch tensors."""
    if a.shape != b.shape:
        raise ValueError(f"Inputs must be same shape, got {a.shape} and {b.shape}")
    num = (a - a.mean()) * (b - b.mean())
    denom = a.std(correction=0) * b.std(correction=0)
    if denom <= 1e-12:
        return torch.tensor(float("nan"), device=a.device)
    return num.mean() / denom


@torch.inference_mode()
def nrmse(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute normalized root mean squared error (NRMSE) for two PyTorch tensors.

    Both tensors are mapped onto the ground-truth intensity scale before
    computing RMSE, so gain and offset errors remain visible.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth tensor.
    y_pred : torch.Tensor
        Predicted tensor, same shape as y_true.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    torch.Tensor
        A scalar tensor containing the NRMSE.
    """
    y_true_norm = _min_max_normalize(y_true, eps=eps)
    y_pred_norm = _min_max_normalize(y_pred, eps=eps)
    mse = torch.mean((y_true_norm - y_pred_norm) ** 2)
    rmse = torch.sqrt(mse)

    return rmse


@torch.inference_mode()
def psnr(image_true: torch.Tensor, image_test: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute peak signal-to-noise ratio (PSNR) for two PyTorch tensors.

    Both tensors are mapped onto the ground-truth intensity scale before
    computing PSNR, so gain and offset errors remain visible.

    Parameters
    ----------
    image_true : torch.Tensor
        Ground-truth tensor.
    image_test : torch.Tensor
        Predicted / reconstructed tensor, same shape as image_true.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    torch.Tensor
        A scalar tensor containing the PSNR value in dB.
    """
    image_true = _min_max_normalize(image_true, eps=eps)
    image_test = _min_max_normalize(image_test, eps=eps)
    mse = torch.mean((image_true - image_test) ** 2)

    if mse <= eps:
        return torch.tensor(float("inf"), device=image_true.device)

    psnr_val = 20 * torch.log10(torch.tensor(1.0, device=image_true.device)) - 10 * torch.log10(mse)
    return psnr_val


@torch.inference_mode()
def ssim(img1: torch.Tensor, img2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute mean structural similarity index (SSIM)."""
    img1 = _min_max_normalize(img1, eps=eps)
    img2 = _min_max_normalize(img2, eps=eps)

    img1 = img1.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    img2 = img2.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

    return torch_ssim(img1, img2, data_range=1.0)


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
    """Compute pixel-level image quality metrics between prediction and target."""
    _require_cubic()
    prediction = torch.as_tensor(prediction)
    target = torch.as_tensor(target)

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    prediction = prediction.to(device)
    target = target.to(device)

    metrics = {
        "PCC": corr_coef(target, prediction).item(),
        "SSIM": ssim(target, prediction).item(),
        "NRMSE": nrmse(target, prediction).item(),
        "PSNR": psnr(target, prediction).item(),
    }
    target, prediction = target.cpu().numpy(), prediction.cpu().numpy()

    if spectral_pcc_kwargs is not None:
        metrics["Spectral_PCC"] = spectral_pcc(prediction, target, spacing=spacing, **spectral_pcc_kwargs)
    if fsc_kwargs is not None:
        resolutions = fsc_resolution(
            target - target.mean(),
            prediction - prediction.mean(),
            spacing=spacing,
            **fsc_kwargs,
        )
        metrics.update({f"{k.upper()}_FSC_Resolution": float(v) for k, v in resolutions.items()})

    return metrics


def calculate_microssim(microssim_data):
    """Calculate MicroMS3IM scores across a collection of images."""
    _require_microssim()
    _require_cubic()
    targets = np.concatenate([img["target"] for img in microssim_data], axis=0)
    predictions = np.concatenate([img["predict"] for img in microssim_data], axis=0)

    def microssim_with_condition(condition):
        masked_targets = asnumpy(np.where(condition, targets, 0))
        masked_predictions = asnumpy(np.where(condition, predictions, 0))

        sim = MicroMS3IM()
        sim.fit(masked_targets, masked_predictions)

        scores = []
        slice_idx = 0
        for img in microssim_data:
            num_slices = len(img["target"])
            img_masked_targets = masked_targets[slice_idx : slice_idx + num_slices]
            img_masked_predictions = masked_predictions[slice_idx : slice_idx + num_slices]

            slice_scores = []
            for i in range(num_slices):
                slice_scores.append(sim.score(img_masked_targets[i], img_masked_predictions[i]))

            slice_idx += num_slices
            scores.append({"MicroMS3IM": np.mean(np.nan_to_num(slice_scores))})

        return scores

    return microssim_with_condition(np.ones_like(targets, dtype=bool))


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


def cp_target_regionprops(target, cell_segmentation, spacing):
    """GT-side raw CP regionprops, shape ``(n_cells, n_props_raw)``.

    Cacheable per ``(gt_path, cell_segmentation_path, spacing)`` since no
    prediction data is involved.
    """
    _require_cubic()
    return _cp_raw_regionprops(_minmax_norm(target), cell_segmentation, spacing)


def cp_pred_regionprops(prediction, cell_segmentation, spacing):
    """Prediction-side raw CP regionprops, shape ``(n_cells, n_props_raw)``."""
    _require_cubic()
    return _cp_raw_regionprops(_minmax_norm(prediction), cell_segmentation, spacing)


def cp_pairwise(pred_raw, target_raw):
    """Pair raw CP regionprops into CP_FID / CP_KID / CP_Median_Cosine_Similarity.

    Applies the target-side all-zero column drop and per-matrix z-score that
    the original monolithic ``cp_feature_similarity`` applied, then delegates
    to :func:`_pairwise_feature_metrics`. Returns NaN metrics for empty inputs.
    """
    if pred_raw.shape != target_raw.shape:
        raise ValueError(f"Feature shape mismatch: pred {pred_raw.shape} vs target {target_raw.shape}")
    if pred_raw.size == 0:
        return _nan_pairwise("CP")
    non_zero_cols = ~np.all(target_raw == 0, axis=0)
    pred_mat = pred_raw[:, non_zero_cols]
    target_mat = target_raw[:, non_zero_cols]
    pred_mat = (pred_mat - pred_mat.mean(axis=0)) / (pred_mat.std(axis=0) + 1e-8)
    target_mat = (target_mat - target_mat.mean(axis=0)) / (target_mat.std(axis=0) + 1e-8)
    if pred_mat.size == 0:
        return _nan_pairwise("CP")
    return _pairwise_feature_metrics(pred_mat, target_mat, "CP")


def _extract_per_cell_features(img_2d, cell_segmentation_3d, feature_extractor, patch_size):
    """Iterate cells in the shared 3-D segmentation and extract 2-D per-cell features.

    Iteration order is ``np.unique(cell_segmentation_3d)`` with the
    background label ``0`` skipped. Both GT and prediction loops use the
    same segmentation, so their returned arrays align row-by-row.
    """
    feats = []
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
        feat = feature_extractor.extract_features(cell_crop).detach().cpu().numpy().reshape(-1)
        feats.append(feat)
    if not feats:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(feats, axis=0)


def deep_target_features(target, cell_segmentation, feature_extractor, patch_size):
    """GT-side per-cell deep embeddings, shape ``(n_cells, feature_dim)``.

    Cacheable per ``(gt_path, cell_segmentation_path, patch_size, feature_extractor_identity)``.
    """
    if target.shape != cell_segmentation.shape:
        raise ValueError(f"Shape mismatch: target {target.shape} vs cell_segmentation {cell_segmentation.shape}")
    target_2d = _minmax_norm(np.max(target, axis=0))
    return _extract_per_cell_features(target_2d, cell_segmentation, feature_extractor, patch_size)


def deep_pred_features(prediction, cell_segmentation, feature_extractor, patch_size):
    """Prediction-side per-cell deep embeddings, shape ``(n_cells, feature_dim)``."""
    if prediction.shape != cell_segmentation.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} vs cell_segmentation {cell_segmentation.shape}"
        )
    prediction_2d = _minmax_norm(np.max(prediction, axis=0))
    return _extract_per_cell_features(prediction_2d, cell_segmentation, feature_extractor, patch_size)


def deep_pairwise(pred_feats, target_feats, name):
    """Pair per-cell deep features into ``{name}_FID`` / ``_KID`` / ``_Median_Cosine_Similarity``.

    Empty inputs (no cells) produce NaN metrics.
    """
    if name not in ("DINOv3", "DynaCLR"):
        raise ValueError(f"Unsupported feature extractor: {name}")
    if pred_feats.shape != target_feats.shape:
        raise ValueError(f"Feature shape mismatch: pred {pred_feats.shape} vs target {target_feats.shape}")
    if pred_feats.size == 0:
        return _nan_pairwise(name)
    return _pairwise_feature_metrics(pred_feats, target_feats, name)


def _nan_pairwise(name):
    """Return a dict of NaN placeholders matching the pairwise-metrics schema."""
    return {
        f"{name}_Median_Cosine_Similarity": float("nan"),
        f"{name}_FID": float("nan"),
        f"{name}_KID": float("nan"),
    }
