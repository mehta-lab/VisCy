import math

import pytest
import torch

from viscy.transforms._transforms import BatchedScaleIntensityRangePercentiles


@pytest.mark.parametrize("channel_wise", [True, False])
def test_batched_scale_intensity_range_percentiles(channel_wise):
    # Create deterministic data using linspace
    shape = (2, 2, 8, 16, 16)
    base_data = torch.linspace(0.0, 1.0, math.prod(shape)).reshape(shape)
    # Apply fixed scaling and shift factors (2 batches, 2 channels)
    scaling_factors = torch.tensor([0.5, 1.0, 2.0, 10.0]).reshape(2, 2, 1, 1, 1)
    shift_factors = torch.tensor([-2.0, -0.2, 0.1, 8.0]).reshape(2, 2, 1, 1, 1)
    data = base_data * scaling_factors + shift_factors

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
        reshaped = batched_output.view(2, 2, -1)
    else:
        reshaped = batched_output.view(2, -1)
    mid, high = torch.quantile(reshaped, torch.tensor([0.5, 0.99]), dim=-1)
    assert torch.allclose(mid, torch.zeros_like(mid), atol=1e-6)
    assert torch.allclose(high, torch.ones_like(high), atol=1e-6)
