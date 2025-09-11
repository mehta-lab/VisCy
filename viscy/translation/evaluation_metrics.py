"""Metrics for model evaluation."""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics.regression import compute_ssim_and_cs
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from skimage.measure import label, regionprops
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def VOI_metric(target: NDArray, prediction: NDArray) -> list[float]:
    """
    Variation of information metric.

    Reports overlap between predicted and ground truth mask.

    Parameters
    ----------
    target : NDArray
        Ground truth mask.
    prediction : NDArray
        Model inferred FL image cellpose mask.

    Returns
    -------
    list[float]
        VI for image masks.
    """
    # cellpose segmentation of predicted image: outputs labl mask
    pred_bin = prediction > 0
    target_bin = target > 0

    # convert to binary mask
    im_targ_mask = target_bin > 0
    im_pred_mask = pred_bin > 0

    # compute entropy from pred_mask
    marg_pred = np.histogramdd(np.ravel(im_pred_mask), bins=256)[0] / im_pred_mask.size
    marg_pred = list(filter(lambda p: p > 0, np.ravel(marg_pred)))
    entropy_pred = -np.sum(np.multiply(marg_pred, np.log2(marg_pred)))

    # compute entropy from target_mask
    marg_targ = np.histogramdd(np.ravel(im_targ_mask), bins=256)[0] / im_targ_mask.size
    marg_targ = list(filter(lambda p: p > 0, np.ravel(marg_targ)))
    entropy_targ = -np.sum(np.multiply(marg_targ, np.log2(marg_targ)))

    # intersection entropy
    im_intersection = np.logical_and(im_pred_mask, im_targ_mask)
    im_inters_informed = im_intersection * im_targ_mask * im_pred_mask

    marg_intr = (
        np.histogramdd(np.ravel(im_inters_informed), bins=256)[0]
        / im_inters_informed.size
    )
    marg_intr = list(filter(lambda p: p > 0, np.ravel(marg_intr)))
    entropy_intr = -np.sum(np.multiply(marg_intr, np.log2(marg_intr)))

    # variation of entropy/information
    VI = entropy_pred + entropy_targ - (2 * entropy_intr)

    return [VI]


def POD_metric(
    target_bin: NDArray, pred_bin: NDArray
) -> tuple[float, float, float, int, int]:
    """
    Probability of detection metric for object matching.

    Parameters
    ----------
    target_bin : NDArray
        Binary ground truth mask.
    pred_bin : NDArray
        Binary predicted mask.

    Returns
    -------
    tuple[float, float, float, int, int]
        POD and various detection statistics.
        - POD: Probability of detection
        - FAR: False alarm rate
        - PCD: Probability of correct detection
        - n_targObj: Number of target objects
        - n_predObj: Number of predicted objects
    """
    # pred_bin = cpmask_array(prediction)

    # relabel mask for ordered labelling across images for efficient LAP mapping
    props_pred = regionprops(label(pred_bin))
    props_targ = regionprops(label(target_bin))

    # construct empty cost matrix based on the number of objects being mapped
    n_predObj = len(props_pred)
    n_targObj = len(props_targ)
    dim_cost = max(n_predObj, n_targObj)

    # calculate cost based on proximity of centroid b/w objects
    cost_matrix = np.zeros((dim_cost, dim_cost))
    a = 0
    b = 0
    lab_targ = []  # enumerate the labels from labelled ground truth mask
    lab_pred = []  # enumerate the labels from labelled predicted image mask
    lab_targ_major_axis = []  # store the major axis of target masks
    for props_t in props_targ:
        y_t, x_t = props_t.centroid
        lab_targ.append(props_t.label)
        lab_targ_major_axis.append(props_t.axis_major_length)
        for props_p in props_pred:
            y_p, x_p = props_p.centroid
            lab_pred.append(props_p.label)
            # using centroid distance as measure for mapping
            cost_matrix[a, b] = np.sqrt(((y_t - y_p) ** 2) + ((x_t - x_p) ** 2))
            b = b + 1
        a = a + 1
        b = 0

    distance_threshold = np.mean(lab_targ_major_axis) / 2

    # minimize cost matrix of objects
    rids, cids = linear_sum_assignment(cost_matrix)

    # filter out rid and cid pairs that exceed distance threshold
    matching_targ = []
    matching_pred = []
    for rid, cid in zip(rids, cids):
        if cost_matrix[rid, cid] <= distance_threshold:
            matching_targ.append(rid)
            matching_pred.append(cid)

    # probability of detection
    POD = len(matching_targ) / len(props_targ)

    # probability of false alarm
    FAR = (len(props_pred) - len(matching_pred)) / len(props_pred)

    # probability of correct detection
    PCD = len(matching_targ) / len(props_targ)

    return (POD, FAR, PCD, len(props_targ), len(props_pred))


def compute_3d_dice_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: float = 1e-8,
    threshold: float = 0.5,
    aggregate: bool = True,
) -> torch.Tensor:
    """Compute 3D Dice similarity coefficient.

    Parameters
    ----------
    y_true : torch.Tensor
        True labels.
    y_pred : torch.Tensor
        Predicted labels.
    eps : float, optional
        Epsilon to avoid division by zero. Defaults to 1e-8.
    threshold : float, optional
        Threshold for binarization. Defaults to 0.5.
    aggregate : bool, optional
        Whether to aggregate the dice score. Defaults to True.

    Returns
    -------
    torch.Tensor
        Dice score.
    """
    y_pred_thresholded = (y_pred > threshold).float()
    intersection = torch.sum(y_true * y_pred_thresholded, dim=(-3, -2, -1))
    total = torch.sum(y_true + y_pred_thresholded, dim=(-3, -2, -1))
    dice = (2.0 * intersection + eps) / (total + eps)
    if aggregate:
        return torch.mean(dice)
    return dice


def compute_jaccard_index(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Compute Jaccard index (IoU).

    Parameters
    ----------
    y_true : torch.Tensor
        True labels.
    y_pred : torch.Tensor
        Predicted labels.
    threshold : float, optional
        Threshold for binarization. Defaults to 0.5.

    Returns
    -------
    torch.Tensor
        Jaccard index.
    """
    y_pred_thresholded = y_pred > threshold
    intersection = torch.sum(y_true & y_pred_thresholded, dim=(-3, -2, -1))
    union = torch.sum(y_true | y_pred_thresholded, dim=(-3, -2, -1))
    return torch.mean(intersection.float() / union.float())


def compute_pearson_correlation_coefficient(
    y_true: torch.Tensor, y_pred: torch.Tensor, dim: Sequence[int] | None = None
) -> torch.Tensor:
    """Compute Pearson correlation coefficient.

    Parameters
    ----------
    y_true : torch.Tensor
        True labels.
    y_pred : torch.Tensor
        Predicted labels.
    dim : Sequence[int] | None, optional
        Dimensions to compute the Pearson correlation coefficient. Defaults to None.

    Returns
    -------
    torch.Tensor
        Pearson correlation coefficient.
    """
    if dim is None:
        # default to spatial dimensions
        dim = (-3, -2, -1)
    y_true_centered = y_true - torch.mean(y_true, dim=dim, keepdim=True)
    y_pred_centered = y_pred - torch.mean(y_pred, dim=dim, keepdim=True)
    numerator = torch.sum(y_true_centered * y_pred_centered, dim=dim)
    # compute stds
    y_true_std = torch.sqrt(torch.sum(y_true_centered**2, dim=dim))
    y_pred_std = torch.sqrt(torch.sum(y_pred_centered**2, dim=dim))
    denominator = y_true_std * y_pred_std
    # torch.full_like makes the entire tensor have the same value,
    # so we have to use torch.full instead
    small_correlation = torch.abs(denominator) < 1e-8
    pcc = torch.where(
        small_correlation, torch.zeros_like(numerator), numerator / denominator
    )
    return torch.mean(pcc)


class MeanAveragePrecisionNuclei(MeanAveragePrecision):
    """Mean Average Precision for nuclei detection.

    Parameters
    ----------
    min_area : int, optional
        Minimum area of nuclei to be considered. Defaults to 20.
    iou_threshold : float, optional
        IoU threshold for matching. Defaults to 0.5.

    Returns
    -------
    torch.Tensor
        Mean average precision score.
    """

    def __init__(self, min_area: int = 20, iou_threshold: float = 0.5) -> None:
        super().__init__(iou_thresholds=[iou_threshold])
        self.min_area = min_area

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute mean average precision for nuclei detection.

        Parameters
        ----------
        prediction : torch.Tensor
            Predicted nuclei segmentation masks.
        target : torch.Tensor
            Ground truth nuclei segmentation masks.

        Returns
        -------
        torch.Tensor
            Mean average precision score.
        """
        prediction_labels = label(prediction > 0.5)
        target_labels = label(target > 0.5)
        device = prediction.device
        preds = []
        targets = []
        for i, (pred_img, target_img) in enumerate(
            zip(prediction_labels, target_labels)
        ):
            pred_props = regionprops(pred_img)
            # binary mask for each instance
            pred_masks = torch.zeros(
                len(pred_props), *pred_img.shape, dtype=torch.bool, device=device
            )
            pred_labels = torch.zeros(len(pred_props), dtype=torch.long, device=device)
            pred_scores = torch.ones(len(pred_props), dtype=torch.float, device=device)
            for j, prop in enumerate(pred_props):
                if prop.area < self.min_area:
                    continue
                pred_masks[j, pred_img == prop.label] = True
                pred_labels[j] = 1  # class 1 for nuclei

            target_props = regionprops(target_img)
            target_masks = torch.zeros(
                len(target_props), *target_img.shape, dtype=torch.bool, device=device
            )
            target_labels = torch.zeros(
                len(target_props), dtype=torch.long, device=device
            )
            for j, prop in enumerate(target_props):
                if prop.area < self.min_area:
                    continue
                target_masks[j, target_img == prop.label] = True
                target_labels[j] = 1

            preds.append(
                {
                    "masks": pred_masks,
                    "labels": pred_labels,
                    "scores": pred_scores,
                }
            )
            targets.append({"masks": target_masks, "labels": target_labels})
        return super().__call__(preds, targets)


def ssim_loss_25d(
    preds: torch.Tensor,
    target: torch.Tensor,
    in_plane_window_size: tuple[int, int] = (11, 11),
    return_contrast_sensitivity: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-scale SSIM loss function for 2.5D volumes (3D with small depth).

    Uses uniform kernel (windows), depth-dimension window size equals to depth size.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted batch (B, C, D, W, H).
    target : torch.Tensor
        Target batch.
    in_plane_window_size : tuple[int, int], optional
        Kernel width and height, by default (11, 11).
    return_contrast_sensitivity : bool, optional
        Whether to return contrast sensitivity, by default False.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        SSIM for the batch, optionally with contrast sensitivity.
    """
    if preds.ndim != 5:
        raise ValueError(
            f"Expected preds to have 5 dimensions (B, C, D, W, H), got {preds.ndim}"
        )
    if preds.shape != target.shape:
        raise ValueError(
            f"Expected preds and target to have the same shape, "
            f"got {preds.shape} and {target.shape}"
        )

    B, C, D, H, W = preds.shape
    # Compute SSIM for each channel and each depth slice
    ssim_per_channel = []
    cs_per_channel = []

    for c in range(C):
        # Window size for depth dimension is the depth size
        window_size = (*in_plane_window_size, D)
        ssim, cs = compute_ssim_and_cs(
            preds[:, c, :, :, :], target[:, c, :, :, :], window_size
        )
        ssim_per_channel.append(ssim)
        if return_contrast_sensitivity:
            cs_per_channel.append(cs)

    # Average across channels
    ssim_result = torch.mean(torch.stack(ssim_per_channel))

    if return_contrast_sensitivity:
        cs_result = torch.mean(torch.stack(cs_per_channel))
        return ssim_result, cs_result

    return ssim_result


def ms_ssim_25d(
    preds: torch.Tensor,
    target: torch.Tensor,
    in_plane_window_size: tuple[int, int] = (11, 11),
    clamp: bool = False,
    betas: Sequence[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
) -> torch.Tensor:
    """
    Multi-scale SSIM for 2.5D volumes (3D with small depth).

    Uses uniform kernel (windows), depth-dimension window size equals to depth size.
    Depth dimension is not downsampled.

    Adapted from torchmetrics@99d6d9d6ac4eb1b3398241df558604e70521e6b0
    Original license:
        Copyright The Lightning team, http://www.apache.org/licenses/LICENSE-2.0

    Parameters
    ----------
    preds : torch.Tensor
        Predicted images.
    target : torch.Tensor
        Target images.
    in_plane_window_size : tuple[int, int], optional
        Kernel width and height, defaults to (11, 11).
    clamp : bool, optional
        Clamp to [1e-6, 1] for training stability when used in loss,
        defaults to False.
    betas : Sequence[float], optional
        Exponents of each resolution,
        defaults to (0.0448, 0.2856, 0.3001, 0.2363, 0.1333).

    Returns
    -------
    torch.Tensor
        Multi-scale SSIM.
    """
    base_min = 1e-4
    mcs_list = []
    ssim_list = []

    B, C, D, H, W = preds.shape

    for c in range(C):
        # Window size for depth dimension is the depth size
        window_size = (*in_plane_window_size, D)

        pred_c = preds[:, c]
        target_c = target[:, c]

        for level in range(len(betas)):
            if level > 0:
                # Downsample only in spatial dimensions, not depth
                pred_c = F.avg_pool2d(pred_c.view(-1, H, W), kernel_size=2).view(
                    B, D, H // 2, W // 2
                )
                target_c = F.avg_pool2d(target_c.view(-1, H, W), kernel_size=2).view(
                    B, D, H // 2, W // 2
                )
                H, W = H // 2, W // 2

            ssim, cs = compute_ssim_and_cs(pred_c, target_c, window_size)

            if level == len(betas) - 1:
                ssim_list.append(ssim)
            else:
                mcs_list.append(cs)

    # Compute the final ms-ssim score
    mcs_tensor = torch.stack(mcs_list)
    ssim_tensor = torch.stack(ssim_list)

    # Apply betas weighting
    betas_tensor = torch.tensor(betas, device=preds.device, dtype=preds.dtype)

    # For numerical stability
    if clamp:
        mcs_tensor = torch.clamp(mcs_tensor, base_min, 1)
        ssim_tensor = torch.clamp(ssim_tensor, base_min, 1)

    # Compute weighted geometric mean
    ms_ssim_val = torch.prod(mcs_tensor ** betas_tensor[:-1]) * (
        ssim_tensor ** betas_tensor[-1]
    )

    return torch.mean(ms_ssim_val)
