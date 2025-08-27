import pytest
import torch

from viscy.transforms import BatchedRandFlipd


@pytest.mark.parametrize("prob", [0.0, 1.0])
@pytest.mark.parametrize("spatial_axis", [[0, 1, 2], [1, 2], [0]])
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_batched_rand_flipd(device, prob, spatial_axis):
    img = torch.arange(16 * 2 * 4 * 8 * 8, device=device).reshape(16, 2, 4, 8, 8).float()
    data = {"image": img}
    transform = BatchedRandFlipd(keys=["image"], prob=prob, spatial_axis=spatial_axis)
    out = transform(data)
    
    assert out["image"].shape == img.shape
    changed = (out["image"] != img).any(dim=tuple(range(1, img.ndim)))
    
    if prob == 1.0:
        assert changed.all()
    elif prob == 0.0:
        assert not changed.any()
    
    assert out["image"].device == img.device