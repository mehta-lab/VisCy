import pytest
import torch
from kornia.filters import (
    filter3d,
    get_gaussian_kernel1d,
    get_gaussian_kernel3d,
)
from monai.transforms.intensity.array import GaussianSmooth

from viscy.transforms import BatchedRandGaussianSmooth, BatchedRandGaussianSmoothd
from viscy.transforms._gaussian_smooth import filter3d_separable


@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_batched_rand_gaussian_smooth(device, prob):
    img = torch.randn(8, 2, 4, 8, 8, device=device)
    transform = BatchedRandGaussianSmooth(
        sigma_x=0.5, sigma_y=(1.5, 2.0), sigma_z=0.5, prob=prob
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
    stds = [img.std()]
    for sigma_x, sigma_y, sigma_z in [
        (0.0, 0.5, 0.0),
        (0.5, 0.5, 0.5),
        ((0.5, 1.0), (0.5, 2.0), (0.5, 1.5)),
        (1.0, 2.0, 1.5),
    ]:
        transform = BatchedRandGaussianSmooth(
            sigma_x=sigma_x, sigma_y=sigma_y, sigma_z=sigma_z, prob=1.0
        )
        out = transform(img)
        stds.append(out.std())
    # smoothing should reduce random noise variance
    for i in range(len(stds) - 1):
        assert stds[i] > stds[i + 1]


def test_batched_rand_gaussian_smooth_dict():
    img = torch.randn(16, 2, 4, 8, 8)
    data = {"a": img.clone(), "b": img.clone()}
    transform = BatchedRandGaussianSmoothd(
        keys=["a", "b"], sigma_x=0.5, sigma_y=1.0, sigma_z=0.7, prob=1.0
    )
    out = transform(data)

    # both keys should be transformed identically
    assert torch.equal(out["a"], out["b"])
    assert (out["a"] != img).any()


@pytest.mark.parametrize(
    "sigma,kernel_size",
    [
        (1.0, (3, 3, 3)),
        (1.5, (5, 5, 5)),
        (2.0, (7, 7, 7)),
    ],
)
def test_separable_vs_full_3d_equivalence(sigma, kernel_size):
    """Test that separable filtering produces identical output to full 3D filtering.

    Note: This tests the case where Gaussian kernels are truly separable
    (same sigma for all dimensions). Different sigmas per dimension create
    non-separable kernels where our approach is an approximation.
    """
    batch_size = 2
    sigma_values = torch.tensor([[sigma, sigma, sigma]] * batch_size)
    border_type = "reflect"

    input_tensor = torch.randn(batch_size, 2, 8, 8, 8)

    kernel_3d = get_gaussian_kernel3d(kernel_size, sigma_values)
    output_3d = filter3d(input_tensor, kernel_3d, border_type)

    ksize_z, ksize_y, ksize_x = kernel_size
    kernel_z = get_gaussian_kernel1d(ksize_z, sigma_values[:, 0].view(-1, 1))
    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_values[:, 1].view(-1, 1))
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_values[:, 2].view(-1, 1))

    output_separable = filter3d_separable(
        input_tensor, kernel_z, kernel_y, kernel_x, border_type
    )
    assert torch.allclose(output_3d, output_separable, atol=1e-5)


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5, 2.0])
def test_auto_kernel_size_vs_monai_output(sigma):
    # monai always truncate at 4-sigma
    transform_viscy = BatchedRandGaussianSmooth(
        sigma_x=sigma,
        sigma_y=sigma,
        sigma_z=sigma,
        truncated=4.0,
        prob=1.0,
    )
    transform_monai = GaussianSmooth(sigma=sigma)
    img = torch.randn(1, 1, 16, 16, 16)

    out_viscy = transform_viscy(img)[0]
    out_monai = transform_monai(img[0])

    assert torch.allclose(out_viscy, out_monai, rtol=1e-2)
