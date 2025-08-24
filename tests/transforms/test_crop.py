import pytest
import torch
from monai.transforms import Compose

from viscy.transforms._crop import BatchedRandSpatialCrop


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
