import pytest
import torch

from viscy.transforms import BatchedRandFlip, BatchedRandFlipd


@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("spatial_axes", [[0, 1, 2], [1, 2], [0]])
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_batched_rand_flip(device, prob, spatial_axes):
    img = (
        torch.arange(32 * 2 * 2 * 2 * 2, device=device).reshape(32, 2, 2, 2, 2).float()
    )
    transform = BatchedRandFlip(prob=prob, spatial_axes=spatial_axes)
    out = transform(img)

    assert out.shape == img.shape
    changed = (out != img).any(dim=tuple(range(1, img.ndim)))
    if prob == 1.0:
        assert changed.all()
        assert torch.equal(img, out.flip(dims=[ax + 2 for ax in spatial_axes]))
    elif prob == 0.5:
        assert changed.any()
        assert not changed.all()
    elif prob == 0.0:
        assert not changed.any()
    assert out.device == img.device


def test_batched_rand_flip_dict():
    img = torch.arange(16 * 2 * 4 * 8 * 8).reshape(16, 2, 4, 8, 8).float()
    data = {"a": img.clone(), "b": img.clone()}
    transform = BatchedRandFlipd(keys=["a", "b"], prob=1.0, spatial_axes=[0, 1, 2])
    out = transform(data)
    assert torch.equal(out["a"], out["b"])
