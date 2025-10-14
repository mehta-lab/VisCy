import pytest
import torch
from monai.transforms import AdjustContrast, Compose

from viscy.transforms import BatchedRandAdjustContrast, BatchedRandAdjustContrastd


@pytest.mark.parametrize("ndim", [4, 5])
@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
@pytest.mark.parametrize("compose", [True, False])
def test_batched_adjust_contrast(device, ndim, prob, compose):
    img = torch.rand([16] + [2] * (ndim - 1), device=device) + 0.1
    transform = BatchedRandAdjustContrast(prob=prob, gamma=(0.5, 2.0))
    if compose:
        transform = Compose([transform])
    result = transform(img)
    assert result.shape == img.shape
    changed = ~torch.isclose(result, img, atol=1e-6).all(
        dim=list(range(1, result.ndim))
    )
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


@pytest.mark.parametrize("gamma", [0.8, 1.5, (0.5, 2.0)])
@pytest.mark.parametrize("retain_stats", [True, False])
@pytest.mark.parametrize("invert_image", [True, False])
def test_batched_adjust_contrast_options(gamma, retain_stats, invert_image):
    img = torch.rand(8, 1, 8, 8, 8) + 0.1
    original_mean = img.mean()
    original_std = img.std()

    transform = BatchedRandAdjustContrast(
        prob=1.0, gamma=gamma, retain_stats=retain_stats, invert_image=invert_image
    )
    result = transform(img)

    assert result.shape == img.shape

    if retain_stats:
        assert torch.isclose(result.mean(), original_mean, atol=1e-5)
        assert torch.isclose(result.std(), original_std, atol=1e-5)

    if not (isinstance(gamma, (int, float)) and gamma == 1.0):
        assert not torch.equal(result, img)


def test_batched_adjust_contrast_dict():
    img = torch.rand([16, 1, 4, 8, 8]) + 0.1
    data = {"a": img, "b": img.clone()}
    transform = BatchedRandAdjustContrastd(keys=["a", "b"], prob=1.0, gamma=(0.5, 2.0))
    result = transform(data)
    assert not torch.equal(result["a"], img)
    assert torch.equal(result["a"], result["b"])


def test_batched_adjust_contrast_gamma_validation():
    with pytest.raises(ValueError):
        BatchedRandAdjustContrast(gamma=0.0)

    with pytest.raises(ValueError):
        BatchedRandAdjustContrast(gamma=-0.5)

    with pytest.raises(ValueError):
        BatchedRandAdjustContrast(gamma=(0.5, 2.0, 1.0))

    with pytest.raises(ValueError):
        BatchedRandAdjustContrast(gamma=(-0.1, 2.0))

    BatchedRandAdjustContrast(gamma=0.1)
    BatchedRandAdjustContrast(gamma=0.3)
    BatchedRandAdjustContrast(gamma=1.5)
    BatchedRandAdjustContrast(gamma=(0.2, 0.8))
    BatchedRandAdjustContrast(gamma=(0.5, 2.0))


@pytest.mark.parametrize("gamma_value", [0.2, 0.5, 0.8, 1.2, 2.0])
@pytest.mark.parametrize("retain_stats", [True, False])
@pytest.mark.parametrize("invert_image", [True, False])
def test_batched_adjust_contrast_vs_monai(gamma_value, retain_stats, invert_image):
    torch.manual_seed(42)

    batch_size = 4
    img_batch = torch.rand(batch_size, 1, 8, 8, 8) + 0.1

    batched_transform = BatchedRandAdjustContrast(
        prob=1.0,
        gamma=(gamma_value, gamma_value),
        retain_stats=retain_stats,
        invert_image=invert_image,
    )

    torch.manual_seed(42)
    batched_result = batched_transform(img_batch)

    monai_transform = AdjustContrast(
        gamma=gamma_value, retain_stats=retain_stats, invert_image=invert_image
    )

    monai_results = []
    for i in range(batch_size):
        individual_result = monai_transform(img_batch[i])
        monai_results.append(individual_result)

    monai_batch_result = torch.stack(monai_results)

    assert torch.allclose(batched_result, monai_batch_result, atol=1e-6, rtol=1e-5)
