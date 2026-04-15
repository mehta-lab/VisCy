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
        raise ImportError("cubic is required for resolution and feature metrics. Install it with: pip install cubic-s2")


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
    y_true_norm, y_pred_norm = _normalize_to_target_scale(y_true, y_pred, eps=eps)
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
    image_true, image_test = _normalize_to_target_scale(image_true, image_test, eps=eps)
    mse = torch.mean((image_true - image_test) ** 2)

    if mse <= eps:
        return torch.tensor(float("inf"), device=image_true.device)

    psnr_val = 20 * torch.log10(torch.tensor(1.0, device=image_true.device)) - 10 * torch.log10(mse)
    return psnr_val


@torch.inference_mode()
def ssim(img1: torch.Tensor, img2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute mean structural similarity index (SSIM)."""
    img1, img2 = _normalize_to_target_scale(img1, img2, eps=eps)

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


def cp_feature_similarity(prediction, target, cell_segmentation, spacing):
    """Compute CP feature metrics between prediction and target."""
    _require_cubic()
    if prediction.shape != target.shape:
        raise ValueError(f"Input shape mismatch: pred {prediction.shape} vs target {target.shape}")

    prediction = _minmax_norm(prediction)
    target = _minmax_norm(target)

    if torch.cuda.is_available():
        prediction = ascupy(prediction)
        target = ascupy(target)
        cell_segmentation = ascupy(cell_segmentation)

    pred_features = regionprops_table(cell_segmentation, prediction, spacing=spacing, properties=list(PROPS_3D))
    target_features = regionprops_table(cell_segmentation, target, spacing=spacing, properties=list(PROPS_3D))

    pred_features.pop("label", None)
    target_features.pop("label", None)

    if torch.cuda.is_available():
        pred_mat = np.array([asnumpy(v) for v in pred_features.values()]).T
        target_mat = np.array([asnumpy(v) for v in target_features.values()]).T
    else:
        pred_mat = np.array(list(pred_features.values())).T
        target_mat = np.array(list(target_features.values())).T

    # drop columns that are all zero in the target
    non_zero_cols = ~np.all(target_mat == 0, axis=0)
    pred_mat = pred_mat[:, non_zero_cols]
    target_mat = target_mat[:, non_zero_cols]

    if pred_mat.shape != target_mat.shape:
        raise ValueError(f"Feature shape mismatch: pred {pred_mat.shape} vs target {target_mat.shape}")

    # z-score each column
    pred_mat = (pred_mat - pred_mat.mean(axis=0)) / (pred_mat.std(axis=0) + 1e-8)
    target_mat = (target_mat - target_mat.mean(axis=0)) / (target_mat.std(axis=0) + 1e-8)

    if pred_mat.size == 0:
        return {
            "CP_Median_Cosine_Similarity": float("nan"),
            "CP_FID": float("nan"),
            "CP_KID": float("nan"),
        }

    return _pairwise_feature_metrics(pred_mat, target_mat, "CP")


def deep_feature_similarity(
    prediction,
    target,
    feature_extractor,
    cell_segmentation,
    patch_size,
    feature_extractor_name,
):
    """Compute deep learning model feature metrics between prediction and target."""
    if feature_extractor_name not in ("DINOv3", "DynaCLR"):
        raise ValueError(f"Unsupported feature extractor: {feature_extractor_name}")

    if prediction.shape != target.shape or prediction.shape != cell_segmentation.shape:
        raise ValueError(
            f"Input shape mismatch: pred {prediction.shape} vs target {target.shape} "
            f"vs cell_segmentation {cell_segmentation.shape}"
        )

    # max projection along z-axis to get 2D image for feature extraction, since deep learning model is 2D
    prediction = _minmax_norm(np.max(prediction, axis=0))
    target = _minmax_norm(np.max(target, axis=0))

    pred_features = []
    target_features = []

    for idx in np.unique(cell_segmentation):
        if idx == 0:
            continue  # skip background

        cell_mask_2d = np.any(cell_segmentation == idx, axis=0)  # project 3D mask to 2D
        yx_coords = np.argwhere(cell_mask_2d)
        if len(yx_coords) == 0:
            continue

        com_y, com_x = np.mean(yx_coords, axis=0).astype(int)
        half_patch = patch_size // 2

        y_start, y_end = com_y - half_patch, com_y + half_patch
        x_start, x_end = com_x - half_patch, com_x + half_patch

        pad_y_before = max(0, -y_start)
        pad_y_after = max(0, y_end - prediction.shape[0])
        pad_x_before = max(0, -x_start)
        pad_x_after = max(0, x_end - prediction.shape[1])

        y_slice = slice(max(0, y_start), min(prediction.shape[0], y_end))
        x_slice = slice(max(0, x_start), min(prediction.shape[1], x_end))

        prediction_cell = (prediction * cell_mask_2d)[y_slice, x_slice]
        target_cell = (target * cell_mask_2d)[y_slice, x_slice]

        if pad_y_before or pad_y_after or pad_x_before or pad_x_after:
            pad = ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after))
            prediction_cell = np.pad(prediction_cell, pad, mode="constant")
            target_cell = np.pad(target_cell, pad, mode="constant")

        pred_feature = feature_extractor.extract_features(prediction_cell).detach().cpu().numpy().reshape(-1)
        target_feature = feature_extractor.extract_features(target_cell).detach().cpu().numpy().reshape(-1)

        if pred_feature.shape != target_feature.shape:
            raise ValueError(f"Feature shape mismatch: pred {pred_feature.shape} vs target {target_feature.shape}")

        pred_features.append(pred_feature)
        target_features.append(target_feature)

    if not pred_features:
        return {
            f"{feature_extractor_name}_Median_Cosine_Similarity": float("nan"),
            f"{feature_extractor_name}_FID": float("nan"),
            f"{feature_extractor_name}_KID": float("nan"),
        }

    return _pairwise_feature_metrics(
        np.stack(pred_features, axis=0),
        np.stack(target_features, axis=0),
        feature_extractor_name,
    )


def compute_feature_metrics(
    prediction,
    target,
    cell_segmentation,
    dinov3_feature_extractor,
    dynaclr_feature_extractor,
    spacing,
    patch_size,
):
    """Compute CP, DINOv3, and DynaCLR feature similarity metrics."""
    metrics = {}
    metrics.update(cp_feature_similarity(prediction, target, cell_segmentation, spacing))
    metrics.update(
        deep_feature_similarity(
            prediction,
            target,
            dinov3_feature_extractor,
            cell_segmentation,
            patch_size,
            feature_extractor_name="DINOv3",
        )
    )
    metrics.update(
        deep_feature_similarity(
            prediction,
            target,
            dynaclr_feature_extractor,
            cell_segmentation,
            patch_size,
            feature_extractor_name="DynaCLR",
        )
    )
    return metrics
