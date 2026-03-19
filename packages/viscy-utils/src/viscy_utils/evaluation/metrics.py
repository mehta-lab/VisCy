"""Metrics for model evaluation"""

from typing import Sequence, Union
from warnings import warn

import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics.regression import compute_ssim_and_cs
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

    marg_intr = (
        np.histogramdd(np.ravel(im_inters_informed), bins=256)[0]
        / im_inters_informed.size
    )
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
    masks = torch.zeros(
        (n_instances, *labels.shape), dtype=torch.bool, device=labels.device
    )
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
        "scores": torch.ones(
            (boxes.shape[0],), dtype=torch.float32, device=boxes.device
        ),
        # dummy class labels
        "labels": torch.zeros(
            (boxes.shape[0],), dtype=torch.uint8, device=boxes.device
        ),
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
    defaults = dict(
        iou_type="segm", box_format="xyxy", max_detection_thresholds=[1, 100, 10000]
    )
    if not kwargs:
        kwargs = {}
    map_metric = MeanAveragePrecision(**(defaults | kwargs))
    map_metric.update(
        [labels_to_detection(pred_labels)], [labels_to_detection(target_labels)]
    )
    return map_metric.compute()


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
        raise ValueError(
            f"Input shape must be (B, C, D, W, H), got input shape {preds.shape}"
        )
    depth = preds.shape[2]
    if depth > 15:
        warn(f"Input depth {depth} is potentially too large for 2.5D SSIM.")
    ssim_img, cs_img = compute_ssim_and_cs(
        preds,
        target,
        3,
        kernel_sigma=None,
        kernel_size=(depth, *in_plane_window_size),
        data_range=target.max(),
        kernel_type="uniform",
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
        ssim, contrast_sensitivity = ssim_25d(
            preds, target, in_plane_window_size, return_contrast_sensitivity=True
        )
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
