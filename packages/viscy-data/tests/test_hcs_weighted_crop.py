"""Integration tests for BatchedRandWeightedCropd with HCSDataModule pipeline."""

import torch
from monai.transforms import Compose

from viscy_data.foreground_masks import ForegroundMaskSupport
from viscy_transforms import BatchedCenterSpatialCropd, BatchedRandWeightedCropd


def test_weighted_crop_fg_mask_coalignment():
    """Verify patch_spatial_transforms injects fg_mask keys into weighted crop."""
    weighted_crop = BatchedRandWeightedCropd(
        keys=["source", "target"],
        w_key="target",
        spatial_size=[8, 32, 32],
    )
    center_crop = BatchedCenterSpatialCropd(
        keys=["source", "target"],
        roi_size=[8, 16, 16],
    )
    transforms = [weighted_crop, center_crop]

    # Patch spatial transforms to include fg_mask — same as hcs.py:139
    ForegroundMaskSupport.patch_spatial_transforms(transforms, target_keys=("target",), mask_keys=("fg_mask",))

    # Verify both transforms now include fg_mask in their keys
    assert "fg_mask" in weighted_crop.keys
    assert "fg_mask" in center_crop.keys
    assert weighted_crop.allow_missing_keys is True

    # Apply the pipeline with fg_mask present
    data = {
        "source": torch.rand(2, 1, 8, 64, 64),
        "target": torch.rand(2, 1, 8, 64, 64),
        "fg_mask": torch.ones(2, 1, 8, 64, 64),
    }
    pipeline = Compose(transforms)
    output = pipeline(data)

    # All keys should have the same spatial dimensions
    assert output["source"].shape == (2, 1, 8, 16, 16)
    assert output["target"].shape == (2, 1, 8, 16, 16)
    assert output["fg_mask"].shape == (2, 1, 8, 16, 16)


def test_weighted_crop_batch_shapes():
    """Verify end-to-end batch shapes through weighted crop + center crop pipeline."""
    data = {
        "source": torch.rand(2, 1, 8, 512, 512),
        "target": torch.rand(2, 1, 8, 512, 512),
    }
    pipeline = Compose(
        [
            BatchedRandWeightedCropd(
                keys=["source", "target"],
                w_key="target",
                spatial_size=[8, 384, 384],
            ),
            BatchedCenterSpatialCropd(
                keys=["source", "target"],
                roi_size=[8, 256, 256],
            ),
        ]
    )
    output = pipeline(data)
    assert output["source"].shape == (2, 1, 8, 256, 256)
    assert output["target"].shape == (2, 1, 8, 256, 256)
