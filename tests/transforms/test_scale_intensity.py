import pytest
import torch
from monai.transforms import RandScaleIntensity

from viscy.transforms import BatchedRandScaleIntensity, BatchedRandScaleIntensityd


@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("factors", [0.1, 0.5, (0.1, 0.3), (-0.2, 0.4)])
@pytest.mark.parametrize("channel_wise", [False, True])
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_batched_rand_scale_intensity(device, prob, factors, channel_wise):
    img = torch.ones(16, 3, 4, 8, 8, device=device)
    transform = BatchedRandScaleIntensity(
        prob=prob, factors=factors, channel_wise=channel_wise
    )
    out = transform(img)

    assert out.shape == img.shape
    assert out.device == img.device
    changed = (out != img).any(dim=tuple(range(1, img.ndim)))

    if prob == 0.0:
        assert not changed.any(), "No samples should be changed when prob=0.0"
        assert torch.equal(out, img)
    elif prob == 1.0:
        assert changed.all(), "All samples should be changed when prob=1.0"
        if isinstance(factors, tuple):
            min_factor, max_factor = factors
        else:
            min_factor, max_factor = -abs(factors), abs(factors)

        scale_factors = out / img
        assert (
            (scale_factors >= (1 + min_factor)) & (scale_factors <= (1 + max_factor))
        ).all()
    elif prob == 0.5:
        assert changed.any(), "Some samples should be changed when prob=0.5"
        assert not changed.all(), "Not all samples should be changed when prob=0.5"


@pytest.mark.parametrize("factors", [0.2, (0.1, 0.5)])
def test_batched_rand_scale_intensity_reproducible(factors):
    """Test that the same random state produces the same results."""
    img = torch.ones(4, 2, 8, 8)
    transform = BatchedRandScaleIntensity(prob=1.0, factors=factors)

    transform.randomize(img)
    out1 = transform(img, randomize=False)
    out2 = transform(img, randomize=False)

    assert torch.equal(out1, out2), "Same random state should produce identical results"


def test_batched_rand_scale_intensity_dict():
    """Test dictionary version of the transform."""
    img = torch.ones(8, 2, 4, 8, 8)
    data = {"source": img.clone(), "target": img.clone()}
    transform = BatchedRandScaleIntensityd(
        keys=["source", "target"], prob=1.0, factors=0.3
    )
    out = transform(data)

    assert torch.equal(out["source"], out["target"])
    assert out["source"].shape == img.shape
    assert out["target"].shape == img.shape


def test_batched_rand_scale_intensity_channel_wise():
    """Test that channel-wise scaling applies different factors to each channel."""
    img = torch.ones(2, 3, 4, 4)
    transform = BatchedRandScaleIntensity(prob=1.0, factors=0.5, channel_wise=True)
    out = transform(img)

    assert out.shape == img.shape

    for batch_idx in range(img.shape[0]):
        channel_means = out[batch_idx].mean(dim=(1, 2))
        assert not torch.allclose(channel_means, channel_means[0], atol=1e-6)


def test_batched_rand_scale_intensity_edge_cases():
    """Test edge cases and special parameter combinations."""
    img = torch.ones(4, 2, 8, 8)

    # Test with zero factors (should not change the image)
    transform = BatchedRandScaleIntensity(prob=1.0, factors=0.0)
    out = transform(img)
    assert torch.equal(out, img), "Zero factors should not change the image"

    transform = BatchedRandScaleIntensity(prob=1.0, factors=(-0.5, -0.1))
    out = transform(img)
    assert (out < img).all(), "Negative factors should reduce intensity"

    transform = BatchedRandScaleIntensity(prob=1.0, factors=(0.1, 0.5))
    out = transform(img)
    assert (out > img).all(), "Positive factors should increase intensity"


@pytest.mark.parametrize("factor_value", [-0.3, -0.1, 0.0, 0.2, 0.5])
def test_batched_scale_intensity_vs_monai(factor_value):
    """Test that batched transform produces same results as individual MONAI transforms."""
    batch_size = 4
    img_batch = torch.rand(batch_size, 2, 8, 8) + 0.1

    batched_transform = BatchedRandScaleIntensity(
        prob=1.0, factors=(factor_value, factor_value), channel_wise=False
    )

    batched_result = batched_transform(img_batch)

    monai_results = []
    for i in range(batch_size):
        sample = img_batch[i]
        monai_transform = RandScaleIntensity(
            factors=(factor_value, factor_value), prob=1.0
        )
        individual_result = monai_transform(sample)
        monai_results.append(individual_result)

    monai_batch_result = torch.stack(monai_results)

    assert torch.allclose(batched_result, monai_batch_result, atol=1e-6, rtol=1e-5)
