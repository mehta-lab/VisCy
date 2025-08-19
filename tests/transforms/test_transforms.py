import math

import pytest
import torch

from viscy.transforms._transforms import BatchedScaleIntensityRangePercentiles


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
