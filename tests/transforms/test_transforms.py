import pytest
import torch

from viscy.transforms._decollate import Decollate
from viscy.transforms._transforms import (
    BatchedScaleIntensityRangePercentiles,
)


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
