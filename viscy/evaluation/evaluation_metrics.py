"""Metrics for model evaluation"""
import numpy as np
import torch
from lapsolver import solve_dense
from skimage.measure import label, regionprops
from torchmetrics.detection import MeanAveragePrecision
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

    # an uneven number of targ and pred yields zero entry row / column
    # to make the unbalanced assignment problem balanced.
    # The zero entries (=no realobjects)
    # are set to nan to prevent them of being matched.
    cost_matrix[cost_matrix == 0.0] = np.nan

    # LAPsolver for minimizing cost matrix of objects
    rids, cids = solve_dense(cost_matrix)

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
    masks = torch.zeros(
        (labels.max(), *labels.shape), dtype=torch.bool, device=labels.device
    )
    # TODO: optimize this?
    for segment in range(labels.max()):
        # start from label value 1, i.e. skip background label
        masks[segment] = labels == (segment + 1)
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
    map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="segm", **kwargs)
    map_metric.update(
        [labels_to_detection(pred_labels)], [labels_to_detection(target_labels)]
    )
    return map_metric.compute()
