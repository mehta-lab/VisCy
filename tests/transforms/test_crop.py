import pytest
import torch
from monai.transforms import Compose

from viscy.transforms._crop import BatchedRandSpatialCrop, BatchedRandSpatialCropd


@pytest.mark.parametrize("compose", [True, False])
def test_batched_rand_spatial_crop(compose):
    """Test BatchedRandSpatialCrop with 3D data using gather-based implementation."""
    batch_size = 2
    channels = 1
    shape = (batch_size, channels, 16, 16, 16)
    roi_size = [8, 8, 8]
    data = torch.rand(shape)
    transform = BatchedRandSpatialCrop(roi_size=roi_size, random_center=True)
    if compose:
        transform = Compose([transform])
    output = transform(data)
    assert output.shape == (
        batch_size,
        channels,
        roi_size[0],
        roi_size[1],
        roi_size[2],
    )
    assert torch.all(output >= 0) and torch.all(output <= 1)


def test_batched_rand_spatial_crop_2d_error():
    """Test that 2D data raises appropriate error."""
    data_2d = torch.rand((4, 2, 32, 32))
    transform = BatchedRandSpatialCrop(roi_size=(16, 16), random_center=True)
    with pytest.raises(ValueError, match="only supports 3D data"):
        transform(data_2d)


def test_batched_rand_spatial_crop_randomize_control():
    """Test randomize parameter behavior."""
    shape = (2, 3, 16, 16, 16)
    roi_size = [8, 8, 8]
    data = torch.rand(shape)
    transform = BatchedRandSpatialCrop(roi_size=roi_size, random_center=True)
    with pytest.raises(IndexError):
        transform(data, randomize=False)
    output_1 = transform(data, randomize=True)
    output_2 = transform(data, randomize=False)
    output_3 = transform(data, randomize=True)
    assert torch.equal(output_1, output_2)
    assert not torch.equal(output_1, output_3)


def test_batched_rand_spatial_crop_random_size_error():
    """Test that random_size parameter raises ValueError."""
    with pytest.raises(
        ValueError, match="Batched transform does not support random size"
    ):
        BatchedRandSpatialCrop(
            roi_size=[8, 8, 8], max_roi_size=[16, 16, 16], random_size=True
        )


def test_batched_rand_spatial_cropd():
    """Test dict version of BatchedRandSpatialCrop."""
    batch_size = 2
    channels = 1
    shape = (batch_size, channels, 16, 16, 16)
    roi_size = [8, 8, 8]
    
    data = {
        "image": torch.rand(shape),
        "mask": torch.randint(0, 2, shape),
    }
    
    transform = BatchedRandSpatialCropd(
        keys=["image", "mask"], 
        roi_size=roi_size, 
        random_center=True
    )
    
    output = transform(data)
    
    expected_shape = (batch_size, channels, roi_size[0], roi_size[1], roi_size[2])
    assert output["image"].shape == expected_shape
    assert output["mask"].shape == expected_shape
    assert torch.all(output["image"] >= 0) and torch.all(output["image"] <= 1)
    assert torch.all((output["mask"] == 0) | (output["mask"] == 1))


def test_batched_rand_spatial_cropd_missing_keys():
    """Test dict version with allow_missing_keys=True."""
    batch_size = 2
    channels = 1
    shape = (batch_size, channels, 16, 16, 16)
    roi_size = [8, 8, 8]
    
    data = {"image": torch.rand(shape)}
    
    transform = BatchedRandSpatialCropd(
        keys=["image", "nonexistent"], 
        roi_size=roi_size, 
        allow_missing_keys=True
    )
    
    output = transform(data)
    expected_shape = (batch_size, channels, roi_size[0], roi_size[1], roi_size[2])
    assert output["image"].shape == expected_shape
    assert "nonexistent" not in output
