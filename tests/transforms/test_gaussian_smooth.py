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
        sigma_x=0.5, sigma_y=(1.5, 2.0), sigma_z=0.5, kernel_size=kernel_size, prob=prob
    )
    out = transform(img)

    assert out.shape == img.shape
    assert out.device == img.device

    changed = (out != img).any(dim=tuple(range(1, img.ndim)))
    if prob == 1.0:
        assert changed.all()
    elif prob == 0.0:
        assert not changed.any()
    elif prob == 0.5:
        assert changed.any()
        assert not changed.all()


def test_batched_rand_gaussian_smooth_sigma():
    img = torch.randn(4, 2, 8, 8, 8)
    stages = [img]
    for sigma_x, sigma_y, sigma_z in [
        (0.5, 0.5, 0.5),
        ((0.5, 1.0), (0.5, 2.0), (0.5, 1.5)),
        (1.0, 2.0, 1.5),
    ]:
        transform = BatchedRandGaussianSmooth(
            kernel_size=5, sigma_x=sigma_x, sigma_y=sigma_y, sigma_z=sigma_z, prob=1.0
        )
        out = transform(img)
        stages.append(out)
    # smoothing should reduce random noise variance
    for i in range(len(stages) - 1):
        assert stages[i].std() > stages[i + 1].std()


def test_batched_rand_gaussian_smooth_dict():
    img = torch.randn(16, 2, 4, 8, 8)
    data = {"a": img.clone(), "b": img.clone()}
    transform = BatchedRandGaussianSmoothd(
        keys=["a", "b"], sigma_x=0.5, sigma_y=1.0, sigma_z=0.7, kernel_size=3, prob=1.0
    )
    out = transform(data)

    # both keys should be transformed identically
    assert torch.equal(out["a"], out["b"])
    assert (out["a"] != img).any()
