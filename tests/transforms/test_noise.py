import pytest
import torch
from monai.transforms import Compose

from viscy.transforms import BatchedRandGaussianNoise, BatchedRandGaussianNoised


@pytest.mark.parametrize("ndim", [4, 5])
@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
@pytest.mark.parametrize("compose", [True, False])
def test_batched_gaussian_noise(device, ndim, prob, compose):
    img = torch.zeros([16] + [2] * (ndim - 1), device=device)
    transform = BatchedRandGaussianNoise(prob=prob, mean=0.0, std=1.0)
    if compose:
        transform = Compose([transform])
    result = transform(img)
    assert result.shape == img.shape
    changed = (result != 0).sum(dim=list(range(1, result.ndim))) > 0
    if prob == 1.0:
        assert changed.all()
    elif prob == 0.5:
        assert changed.any()
        assert not changed.all()
    elif prob == 0.0:
        assert not changed.any()
    assert result.device == img.device
    if not compose:
        repeat = transform(img, randomize=False)
        assert torch.equal(result, repeat)


@pytest.mark.parametrize("mean", [0.0, 3.0])
@pytest.mark.parametrize("std", [2.0, 4.0])
@pytest.mark.parametrize("sample_std", [True, False])
def test_batched_gaussian_noise_statistics(mean, std, sample_std):
    img = torch.zeros(64, 8, 8, 8, 8)
    transform = BatchedRandGaussianNoise(
        prob=1.0, mean=mean, std=std, sample_std=sample_std
    )
    result = transform(img)
    assert (result.mean() - mean).abs() < 0.5
    expected_std = std / 2.0 if sample_std else std
    assert (result.std() - expected_std).abs() < 0.6


def test_batched_gaussian_noise_dict():
    img = torch.zeros([16, 1, 4, 8, 8])
    data = {"a": img, "b": img.clone()}
    transform = BatchedRandGaussianNoised(keys=["a", "b"], prob=1.0, mean=0.0, std=1.0)
    result = transform(data)
    assert (result["a"] != img).all()
    assert torch.equal(result["a"], result["b"])
