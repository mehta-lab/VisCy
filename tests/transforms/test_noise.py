import pytest
import torch

from viscy.transforms import BatchedRandGaussianNoise


@pytest.mark.parametrize("ndim", [4, 5])
@pytest.mark.parametrize("prob", [0.0, 1.0])
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_batched_gaussian_noise(device, ndim, prob):
    img = torch.zeros([8] + [2] * (ndim - 1), device=device)
    transform = BatchedRandGaussianNoise(prob=prob, mean=0.0, std=1.0)
    out = transform(img)
    assert out.shape == img.shape
    changed = out != 0
    if prob > 0:
        assert changed.all()
    else:
        assert not changed.any()
    assert out.device == img.device
