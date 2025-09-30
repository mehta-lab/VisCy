import torch

from viscy.transforms._zoom import BatchedZoom, BatchedZoomd


def test_batched_zoom():
    """Test BatchedZoom transform."""
    batch_size = 2
    channels = 3
    depth, height, width = 8, 16, 16
    data = torch.rand(batch_size, channels, depth, height, width)

    transform = BatchedZoom(scale_factor=0.5, mode="area")
    result = transform(data)

    expected_shape = (batch_size, channels, depth // 2, height // 2, width // 2)
    assert result.shape == expected_shape
    assert torch.allclose(result.mean(), data.mean(), rtol=1e-3)


def test_batched_zoomd():
    """Test BatchedZoomd dictionary transform."""
    batch_size = 2
    channels = 1
    depth, height, width = 4, 8, 8
    data = {
        "image": torch.rand(batch_size, channels, depth, height, width),
        "label": torch.rand(batch_size, channels, depth, height, width),
    }

    transform = BatchedZoomd(keys=["image", "label"], scale_factor=2.0, mode="nearest")
    result = transform(data)

    expected_shape = (batch_size, channels, depth * 2, height * 2, width * 2)
    assert result["image"].shape == expected_shape
    assert result["label"].shape == expected_shape


def test_batched_zoom_roundtrip():
    """Test roundtrip zoom (2x then 0.5x) returns close to original."""
    batch_size = 4
    channels = 3
    depth, height, width = 4, 8, 8
    data = torch.rand(batch_size, channels, depth, height, width)

    zoom_in = BatchedZoom(scale_factor=2.0, mode="nearest")
    zoom_out = BatchedZoom(scale_factor=0.5, mode="nearest")

    zoomed_in = zoom_in(data)
    zoomed_out = zoom_out(zoomed_in)

    assert torch.all(data == zoomed_out)
