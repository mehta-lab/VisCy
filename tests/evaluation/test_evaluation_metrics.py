import numpy as np
import pytest
import torch
from skimage import data, measure

from viscy.evaluation.evaluation_metrics import (
    POD_metric,
    VOI_metric,
    labels_to_detection,
    labels_to_masks,
    mean_average_precision,
)


@pytest.fixture(scope="session")
def labels_numpy() -> tuple[np.ndarray]:
    return (
        measure.label(image).astype(np.int16)
        for image in (data.binary_blobs(), data.horse() == 0)
    )


@pytest.fixture(scope="session")
def labels_tensor(labels_numpy) -> tuple[torch.ShortTensor]:
    return (torch.from_numpy(labels) for labels in labels_numpy)


def _is_within_unit(metric: float):
    return 1 > metric > 0


def test_VOI_metric(labels_numpy):
    """Variation of information: smaller is better."""
    for labels in labels_numpy:
        assert VOI_metric(labels, labels)[0] == 0
        assert VOI_metric(labels, np.zeros_like(labels))[0] > 0.9


def test_POD_metric(labels_numpy):
    """Test POD_metric()"""
    # FIXME: remove skip after the tested function is fixed
    for labels in labels_numpy:
        (
            true_positives,
            false_positives,
            false_negatives,
            precision,
            recall,
            f1_score,
        ) = POD_metric(labels, labels)
        assert true_positives == labels.max()
        assert precision == recall == f1_score == 1
        assert false_negatives == 0
        assert false_positives == 0
        wrong_labels = np.copy(labels)
        wrong_labels[wrong_labels == 1] = 0
        wrong_labels[0, 0] == 1
        (
            true_positives,
            false_positives,
            false_negatives,
            precision,
            recall,
            f1_score,
        ) = POD_metric(labels, wrong_labels)
        for i, metric in enumerate((precision, recall, f1_score)):
            assert _is_within_unit(metric), i
        assert true_positives < labels.max()
        assert false_negatives > 0
        assert false_positives > 0


def test_labels_to_masks(labels_tensor: torch.ShortTensor):
    for labels in labels_tensor:
        masks = labels_to_masks(labels)
        assert masks.shape == (int(labels.max()), *labels.shape)
        assert masks.dtype == torch.bool
        assert torch.equal(masks[0], labels == 1)
        with pytest.raises(ValueError):
            _ = labels_to_masks(torch.randint(0, 5, (3, 32, 32), dtype=torch.short))


def test_labels_to_detection(labels_tensor: torch.ShortTensor):
    for labels in labels_tensor:
        num_boxes = int(labels.max())
        detection = labels_to_detection(labels)
        assert set(detection) == {"boxes", "scores", "labels", "masks"}
        assert detection["boxes"].shape == (num_boxes, 4)
        assert detection["scores"].shape == (num_boxes,)
        assert detection["labels"].shape == (num_boxes,)
        assert detection["masks"].device == labels.device


def test_mean_average_precision(labels_tensor: torch.ShortTensor):
    for labels in labels_tensor:
        coco_metrics = mean_average_precision(labels, labels)
        assert coco_metrics["map"] == 1
        assert _is_within_unit(coco_metrics["mar_1"])
        assert coco_metrics["mar_10"] == 1
