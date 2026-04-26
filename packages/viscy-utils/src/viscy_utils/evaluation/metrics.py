"""Metrics for model evaluation"""

from math import prod
from typing import Sequence, Union
from warnings import warn

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from skimage.measure import label, regionprops
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import masks_to_boxes


def VOI_metric(target, prediction):
    """variation of information metric
    Reports overlap between predicted and ground truth mask
    : param np.array target: ground truth mask
    : param np.array prediction: model infered FL image cellpose mask
    : return float VI: VI for image masks
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

    marg_intr = np.histogramdd(np.ravel(im_inters_informed), bins=256)[0] / im_inters_informed.size
    marg_intr = list(filter(lambda p: p > 0, np.ravel(marg_intr)))
    entropy_intr = -np.sum(np.multiply(marg_intr, np.log2(marg_intr)))

    # variation of entropy/information
    VI = entropy_pred + entropy_targ - (2 * entropy_intr)

    return [VI]


def POD_metric(target_bin, pred_bin):
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

    true_positives = len(matching_pred)
    false_positives = n_predObj - len(matching_pred)
    false_negatives = n_targObj - len(matching_targ)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall / (precision + recall))

    return [
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1_score,
    ]


def labels_to_masks(labels: torch.ShortTensor) -> torch.BoolTensor:
    """Convert integer labels to a stack of boolean masks.

    :param torch.ShortTensor labels: 2D labels where each value is an object
        (0 is background)
    :return torch.BoolTensor: Boolean masks of shape (objects, H, W)
    """
    if labels.ndim != 2:
        raise ValueError(f"Labels must be 2D, got shape {labels.shape}.")
    segments = torch.unique(labels)
    n_instances = segments.numel() - 1
    masks = torch.zeros((n_instances, *labels.shape), dtype=torch.bool, device=labels.device)
    # TODO: optimize this?
    for s, segment in enumerate(segments):
        # start from label value 1, i.e. skip background label
        masks[s - 1] = labels == segment
    return masks


def labels_to_detection(labels: torch.ShortTensor) -> dict[str, torch.Tensor]:
    """Convert integer labels to a torchvision/torchmetrics detection dictionary.

    :param torch.ShortTensor labels: 2D labels where each value is an object
        (0 is background)
    :return dict[str, torch.Tensor]: detection boxes, scores, labels, and masks
    """
    masks = labels_to_masks(labels)
    boxes = masks_to_boxes(masks)
    return {
        "boxes": boxes,
        # dummy confidence scores
        "scores": torch.ones((boxes.shape[0],), dtype=torch.float32, device=boxes.device),
        # dummy class labels
        "labels": torch.zeros((boxes.shape[0],), dtype=torch.uint8, device=boxes.device),
        "masks": masks,
    }


def mean_average_precision(
    pred_labels: torch.ShortTensor, target_labels: torch.ShortTensor, **kwargs
) -> dict[str, torch.Tensor]:
    """Compute the mAP metric for instance segmentation.

    :param torch.ShortTensor pred_labels: 2D integer prediction labels
    :param torch.ShortTensor target_labels: 2D integer prediction labels
    :param dict **kwargs: keyword arguments passed to
        :py:class:`torchmetrics.detection.MeanAveragePrecision`
    :return dict[str, torch.Tensor]: COCO-style metrics
    """
    defaults = dict(iou_type="segm", box_format="xyxy", max_detection_thresholds=[1, 100, 10000])
    if not kwargs:
        kwargs = {}
    map_metric = MeanAveragePrecision(**(defaults | kwargs))
    map_metric.update([labels_to_detection(pred_labels)], [labels_to_detection(target_labels)])
    return map_metric.compute()


def _compute_ssim_and_cs_bf16(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    kernel_size: Sequence[int],
    data_range: Union[float, torch.Tensor] = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SSIM and contrast-sensitivity with bf16 convolutions.

    Replaces monai's ``compute_ssim_and_cs`` (which unconditionally casts both
    inputs to fp32 internally) with a precision-aware variant that runs the
    five Gaussian-mean convolutions in bf16 and promotes only the variance
    subtractions and C1/C2-guarded divisions to fp32. Squared products
    (``y * y``, ``y_pred * y_pred``, ``y_pred * y``) are computed in fp32
    before casting to bf16 for the conv input, which preserves precision on
    the precision-sensitive squaring step at the cost of one extra cast per
    squared term.

    The conv accumulator is fp32 on CUDA tensor cores (sm >= 80) and on CPU,
    so only the conv outputs (returned in bf16) lose precision relative to
    monai's all-fp32 path.

    Specialized to 3D uniform kernels — matches the single call site in
    :func:`ssim_25d`. Gaussian-kernel, ``kernel_sigma``, and ``spatial_dims``
    parameters from monai's signature are intentionally dropped.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted batch with shape ``(B, C, D, H, W)``.
    y : torch.Tensor
        Target batch with the same shape as ``y_pred``.
    kernel_size : Sequence[int]
        Uniform 3D window size ``(D, H, W)``.
    data_range : float or torch.Tensor, optional
        Data range of the inputs; used to compute the C1, C2 stability
        constants. Defaults to ``1.0``.
    k1 : float, optional
        Luminance stability constant. Defaults to ``0.01``.
    k2 : float, optional
        Contrast stability constant. Defaults to ``0.03``.

    Returns
    -------
    ssim : torch.Tensor
        Per-pixel SSIM map in fp32, shape ``(B, C, *reduced_spatial)``.
    cs : torch.Tensor
        Per-pixel contrast-sensitivity map in fp32, same shape as ``ssim``.
    """
    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y must have same shape, got {y_pred.shape} and {y.shape}.")

    num_channels = y_pred.size(1)

    # Build uniform kernel in fp32 then cast to bf16 once.
    kernel_fp32 = torch.ones((num_channels, 1, *kernel_size), device=y_pred.device, dtype=torch.float32) / float(
        prod(kernel_size)
    )
    kernel_bf = kernel_fp32.to(torch.bfloat16)

    # Squared products in fp32 to preserve squaring precision; cast to bf16
    # for the conv input. The simple (non-squared) inputs are cast inline at
    # the conv site to avoid holding redundant bf16 copies of y / y_pred.
    y_pred_fp32 = y_pred.float()
    y_fp32 = y.float()
    y_pred_sq_bf = (y_pred_fp32 * y_pred_fp32).to(torch.bfloat16)
    y_sq_bf = (y_fp32 * y_fp32).to(torch.bfloat16)
    y_pred_y_bf = (y_pred_fp32 * y_fp32).to(torch.bfloat16)

    mu_x = F.conv3d(y_pred_fp32.to(torch.bfloat16), kernel_bf, groups=num_channels).float()
    mu_y = F.conv3d(y_fp32.to(torch.bfloat16), kernel_bf, groups=num_channels).float()
    mu_xx = F.conv3d(y_pred_sq_bf, kernel_bf, groups=num_channels).float()
    mu_yy = F.conv3d(y_sq_bf, kernel_bf, groups=num_channels).float()
    mu_xy = F.conv3d(y_pred_y_bf, kernel_bf, groups=num_channels).float()

    # Stability constants in fp32 (data_range may be a 0-dim tensor; the
    # multiplication promotes to fp32 since k1/k2 are Python floats).
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    sigma_x = mu_xx - mu_x * mu_x
    sigma_y = mu_yy - mu_y * mu_y
    sigma_xy = mu_xy - mu_x * mu_y

    contrast_sensitivity = (2 * sigma_xy + c2) / (sigma_x + sigma_y + c2)
    ssim_full = ((2 * mu_x * mu_y + c1) / (mu_x * mu_x + mu_y * mu_y + c1)) * contrast_sensitivity

    return ssim_full, contrast_sensitivity


def ssim_25d(
    preds: torch.Tensor,
    target: torch.Tensor,
    in_plane_window_size: tuple[int, int] = (11, 11),
    return_contrast_sensitivity: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Multi-scale SSIM loss function for 2.5D volumes (3D with small depth).
    Uses uniform kernel (windows), depth-dimension window size equals to depth size.

    :param torch.Tensor preds: predicted batch (B, C, D, W, H)
    :param torch.Tensor target: target batch
    :param tuple[int, int] in_plane_window_size: kernel width and height,
        by default (11, 11)
    :param bool return_contrast_sensitivity: whether to return contrast sensitivity
    :return torch.Tensor: SSIM for the batch
    :return Optional[torch.Tensor]: contrast sensitivity
    """
    if preds.ndim != 5:
        raise ValueError(f"Input shape must be (B, C, D, W, H), got input shape {preds.shape}")
    depth = preds.shape[2]
    if depth > 15:
        warn(f"Input depth {depth} is potentially too large for 2.5D SSIM.")
    ssim_img, cs_img = _compute_ssim_and_cs_bf16(
        preds,
        target,
        kernel_size=(depth, *in_plane_window_size),
        data_range=target.max(),
    )
    # aggregate to one scalar per batch
    ssim = ssim_img.view(ssim_img.shape[0], -1).mean(1)
    if return_contrast_sensitivity:
        return ssim, cs_img.view(cs_img.shape[0], -1).mean(1)
    else:
        return ssim


def ms_ssim_25d(
    preds: torch.Tensor,
    target: torch.Tensor,
    in_plane_window_size: tuple[int, int] = (11, 11),
    clamp: bool = False,
    betas: Sequence[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
) -> torch.Tensor:
    """Multi-scale SSIM for 2.5D volumes (3D with small depth).
    Uses uniform kernel (windows), depth-dimension window size equals to depth size.
    Depth dimension is not downsampled.

    Adapted from torchmetrics@99d6d9d6ac4eb1b3398241df558604e70521e6b0
    Original license:
        Copyright The Lightning team, http://www.apache.org/licenses/LICENSE-2.0

    :param torch.Tensor preds: predicted images
    :param torch.Tensor target: target images
    :param tuple[int, int] in_plane_window_size: kernel width and height,
        defaults to (11, 11)
    :param bool clamp: clamp to [1e-6, 1] for training stability when used in loss,
        defaults to False
    :param Sequence[float] betas: exponents of each resolution,
        defaults to (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    :return torch.Tensor: multi-scale SSIM
    """
    base_min = 1e-4
    mcs_list = []
    for _ in range(len(betas)):
        ssim, contrast_sensitivity = ssim_25d(preds, target, in_plane_window_size, return_contrast_sensitivity=True)
        if clamp:
            contrast_sensitivity = contrast_sensitivity.clamp(min=base_min)
        mcs_list.append(contrast_sensitivity)
        # do not downsample along depth
        preds = F.avg_pool3d(preds, (1, 2, 2))
        target = F.avg_pool3d(target, (1, 2, 2))
    if clamp:
        ssim = ssim.clamp(min=base_min)
    mcs_list[-1] = ssim
    mcs_stack = torch.stack(mcs_list)
    betas = torch.tensor(betas, device=mcs_stack.device).view(-1, 1)
    mcs_weighted = mcs_stack**betas
    return torch.prod(mcs_weighted, axis=0).mean()
