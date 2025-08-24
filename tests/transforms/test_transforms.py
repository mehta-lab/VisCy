import math

import pytest
import torch
from monai.transforms import Compose

from viscy.transforms._transforms import (
    BatchedScaleIntensityRangePercentiles,
    BatchedRandSpatialCrop,
)
from viscy.transforms._decollate import Decollate


@pytest.mark.parametrize("channel_wise", [True, False])
def test_batched_scale_intensity_range_percentiles(channel_wise):
    batch_size = 3
    channels = 2
    shape = (batch_size, channels, 8, 16, 16)
    broadcast_shape = (batch_size, channels, 1, 1, 1)
    scale = torch.rand(broadcast_shape) * 10
    shift = torch.rand(broadcast_shape) - 0.5
    data = torch.rand(shape) * scale + shift

    batched_transform = BatchedScaleIntensityRangePercentiles(
        lower=50.0,
        upper=99.0,
        b_min=0.0,
        b_max=1.0,
        clip=False,
        relative=False,
        channel_wise=channel_wise,
    )
    batched_output = batched_transform(data)
    assert batched_output.shape == data.shape
    if channel_wise:
        reshaped = batched_output.view(batch_size, channels, -1)
    else:
        reshaped = batched_output.view(batch_size, -1)
    mid, high = torch.quantile(reshaped, torch.tensor([0.5, 0.99]), dim=-1)
    assert torch.allclose(mid, torch.zeros_like(mid), atol=1e-6)
    assert torch.allclose(high, torch.ones_like(high), atol=1e-6)


@pytest.mark.parametrize("use_compose", [False, True])
def test_batched_rand_spatial_crop(use_compose):
    batch_size = 4
    channels = 2
    # Test 2D case
    shape_2d = (batch_size, channels, 32, 32)
    roi_size_2d = [16, 16]

    data_2d = torch.rand(shape_2d)

    # Create transform (random_size not supported in batched mode)
    transform_2d = BatchedRandSpatialCrop(roi_size=roi_size_2d, random_center=True)

    # Test with or without Compose
    if use_compose:
        composed_transform = Compose([transform_2d])
        output = composed_transform(data_2d)
    else:
        output = transform_2d(data_2d)

    # Basic assertions
    assert output.shape[0] == batch_size  # Same batch size
    assert output.shape[1] == channels  # Same number of channels
    assert all(
        s <= orig for s, orig in zip(output.shape[2:], shape_2d[2:])
    )  # Cropped dimensions

    # Fixed size should match roi_size exactly
    assert output.shape[2:] == tuple(roi_size_2d)


def test_batched_rand_spatial_crop_randomize_control():
    """Test randomize parameter behavior."""
    batch_size = 2
    channels = 1
    shape = (batch_size, channels, 16, 16, 16)
    roi_size = [8, 8, 8]
    data = torch.rand(shape)

    transform = BatchedRandSpatialCrop(roi_size=roi_size, random_center=True)

    # Test without randomization (should fail without prior randomization)
    try:
        transform(data, randomize=False)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass  # Expected - no batch parameters generated yet

    # Test with randomization first, then without
    output1 = transform(data, randomize=True)  # This generates parameters
    output2 = transform(data, randomize=False)  # This reuses parameters

    # Should produce same output when randomize=False
    assert torch.equal(output1, output2)


def test_batched_rand_spatial_crop_with_compose():
    """Test that BatchedRandSpatialCrop works correctly with MONAI Compose."""
    batch_size = 2
    channels = 1
    shape = (batch_size, channels, 16, 16, 16)
    roi_size = [8, 8, 8]

    torch.manual_seed(123)
    data = torch.rand(shape)

    transform = BatchedRandSpatialCrop(roi_size=roi_size, random_center=True)
    # First run to generate parameters
    transform(data, randomize=True)

    compose = Compose([transform])
    transform_output = transform(data, randomize=False)
    compose_output = compose(data)
    # Note: compose will call with randomize=True by default, so outputs may differ
    # Just check that both produce valid outputs
    assert transform_output.shape == compose_output.shape
    assert transform_output.shape[2:] == tuple(roi_size)


def test_decollate():
    """Test Decollate transform for unbatching tensors."""
    batch_size = 3
    channels = 2
    height, width = 8, 8

    # Create batched tensor
    batched_data = torch.rand(batch_size, channels, height, width)

    # Apply decollate transform
    transform = Decollate()
    result = transform(batched_data)

    # Should return list of individual tensors
    assert isinstance(result, list)
    assert len(result) == batch_size

    # Each item should have shape (channels, height, width)
    for i, item in enumerate(result):
        assert item.shape == (channels, height, width)
        assert torch.equal(item, batched_data[i])


def test_decollate_single_item():
    """Test Decollate with single item batch."""
    channels = 1
    depth, height, width = 4, 8, 8

    # Single item batch
    data = torch.rand(1, channels, depth, height, width)

    transform = Decollate()
    result = transform(data)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].shape == (channels, depth, height, width)
    assert torch.equal(result[0], data[0])


def test_batched_rand_spatial_crop_random_size_error():
    """Test that random_size parameter raises ValueError."""
    with pytest.raises(
        ValueError, match="Batched transform does not support random size"
    ):
        BatchedRandSpatialCrop(
            roi_size=[8, 8, 8], max_roi_size=[16, 16, 16], random_size=True
        )


def test_batched_rand_spatial_crop_3d():
    """Test BatchedRandSpatialCrop with 3D data."""
    batch_size = 2
    channels = 1
    shape = (batch_size, channels, 16, 16, 16)
    roi_size = [8, 8, 8]

    data = torch.rand(shape)

    transform = BatchedRandSpatialCrop(roi_size=roi_size, random_center=True)
    output = transform(data)

    assert output.shape[0] == batch_size
    assert output.shape[1] == channels
    assert output.shape[2:] == tuple(roi_size)
