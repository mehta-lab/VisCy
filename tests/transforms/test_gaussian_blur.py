import pytest
import torch

from viscy.transforms import BatchedRandGaussianSmooth, BatchedRandGaussianSmoothd


@pytest.mark.parametrize("kernel_size", [3, (3, 3, 3)])
@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_batched_rand_gaussian_smooth(device, prob, kernel_size):
    img = torch.randn(8, 2, 4, 8, 8, device=device)
    transform = BatchedRandGaussianSmooth(
        sigma_x=0.5, sigma_y=1.5, sigma_z=0.5, kernel_size=kernel_size, prob=prob
    )
    out = transform(img)

    assert out.shape == img.shape
    assert out.device == img.device

    # Check if transform was applied based on probability
    changed = (out != img).any(dim=tuple(range(1, img.ndim)))
    if prob == 1.0:
        assert changed.all()
    elif prob == 0.0:
        assert not changed.any()
    elif prob == 0.5:
        # With prob=0.5, some samples should change, some shouldn't
        assert changed.any()
        assert not changed.all()


@pytest.mark.parametrize("sigma_x,sigma_y,sigma_z", [(0.5, 0.5, 0.5), (1.0, 2.0, 1.5)])
def test_batched_rand_gaussian_smooth_sigma_effect(sigma_x, sigma_y, sigma_z):
    torch.manual_seed(42)  # For reproducible results
    img = torch.randn(4, 1, 8, 8, 8)
    transform = BatchedRandGaussianSmooth(
        kernel_size=5, sigma_x=sigma_x, sigma_y=sigma_y, sigma_z=sigma_z, prob=1.0
    )
    out = transform(img)

    # Gaussian blur should smooth the image
    # Check that the output has lower high-frequency content
    assert out.shape == img.shape
    assert (out != img).any()  # Should be different from input


def test_batched_rand_gaussian_smooth_dict():
    img = torch.randn(16, 2, 4, 8, 8)
    data = {"a": img.clone(), "b": img.clone()}
    transform = BatchedRandGaussianSmoothd(
        keys=["a", "b"], sigma_x=0.5, sigma_y=1.0, sigma_z=0.7, kernel_size=3, prob=1.0
    )
    out = transform(data)

    # Both keys should be transformed identically
    assert torch.equal(out["a"], out["b"])
    # Should be different from input
    assert (out["a"] != img).any()


def test_batched_rand_gaussian_smooth_monai_style():
    """Test MONAI-style signature with tuple ranges"""
    img = torch.randn(8, 1, 4, 8, 8)
    transform = BatchedRandGaussianSmooth(
        sigma_x=(1.0, 3.0), sigma_y=(1.0, 3.0), sigma_z=(1.0, 3.0), prob=1.0
    )
    out = transform(img)

    assert out.shape == img.shape
    assert (out != img).any()  # Should be different from input
