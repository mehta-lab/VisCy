import pytest
import torch
from monai.transforms import Compose

from viscy.transforms._crop import (
    BatchedCenterSpatialCrop,
    BatchedCenterSpatialCropd,
    BatchedRandSpatialCrop,
    BatchedRandSpatialCropd,
)


@pytest.mark.parametrize("compose", [True, False])
def test_batched_rand_spatial_crop(compose):
    """Test BatchedRandSpatialCrop with 3D data using gather-based implementation."""
    batch_size = 2
    channels = 1
    shape = (batch_size, channels, 16, 16, 16)
    roi_size = [8, 8, 8]
    data = torch.rand(shape)
    transform = BatchedRandSpatialCrop(roi_size=roi_size, random_center=True)
    if compose:
        transform = Compose([transform])
    output = transform(data)
    assert output.shape == (
        batch_size,
        channels,
        roi_size[0],
        roi_size[1],
        roi_size[2],
    )
    assert torch.all(output >= 0) and torch.all(output <= 1)


def test_batched_rand_spatial_crop_2d_error():
    """Test that 2D data raises appropriate error."""
    data_2d = torch.rand((4, 2, 32, 32))
    transform = BatchedRandSpatialCrop(roi_size=(16, 16), random_center=True)
    with pytest.raises(ValueError, match="only supports 3D data"):
        transform(data_2d)


def test_batched_rand_spatial_crop_randomize_control():
    """Test randomize parameter behavior."""
    shape = (2, 3, 16, 16, 16)
    roi_size = [8, 8, 8]
    data = torch.rand(shape)
    transform = BatchedRandSpatialCrop(roi_size=roi_size, random_center=True)
    with pytest.raises(IndexError):
        transform(data, randomize=False)
    output_1 = transform(data, randomize=True)
    output_2 = transform(data, randomize=False)
    output_3 = transform(data, randomize=True)
    assert torch.equal(output_1, output_2)
    assert not torch.equal(output_1, output_3)


def test_batched_rand_spatial_crop_random_size_error():
    """Test that random_size parameter raises ValueError."""
    with pytest.raises(
        ValueError, match="Batched transform does not support random size"
    ):
        BatchedRandSpatialCrop(
            roi_size=[8, 8, 8], max_roi_size=[16, 16, 16], random_size=True
        )


def test_batched_rand_spatial_cropd():
    """Test dict version of BatchedRandSpatialCrop."""
    batch_size = 2
    channels = 1
    shape = (batch_size, channels, 16, 16, 16)
    roi_size = [8, 8, 8]

    data = {
        "image": torch.rand(shape),
        "mask": torch.randint(0, 2, shape),
    }

    transform = BatchedRandSpatialCropd(
        keys=["image", "mask"], roi_size=roi_size, random_center=True
    )

    output = transform(data)

    expected_shape = (batch_size, channels, roi_size[0], roi_size[1], roi_size[2])
    assert output["image"].shape == expected_shape
    assert output["mask"].shape == expected_shape
    assert torch.all(output["image"] >= 0) and torch.all(output["image"] <= 1)
    assert torch.all((output["mask"] == 0) | (output["mask"] == 1))


def test_batched_rand_spatial_cropd_missing_keys():
    """Test dict version with allow_missing_keys=True."""
    batch_size = 2
    channels = 1
    shape = (batch_size, channels, 16, 16, 16)
    roi_size = [8, 8, 8]

    data = {"image": torch.rand(shape)}

    transform = BatchedRandSpatialCropd(
        keys=["image", "nonexistent"], roi_size=roi_size, allow_missing_keys=True
    )

    output = transform(data)
    expected_shape = (batch_size, channels, roi_size[0], roi_size[1], roi_size[2])
    assert output["image"].shape == expected_shape
    assert "nonexistent" not in output


@pytest.mark.parametrize("compose", [True, False])
def test_batched_center_spatial_crop(compose):
    """Test BatchedCenterSpatialCrop with 3D data."""
    batch_size = 3
    channels = 2
    shape = (batch_size, channels, 128, 128, 64)
    roi_size = [64, 64, 32]
    data = torch.rand(shape)
    transform = BatchedCenterSpatialCrop(roi_size=roi_size)
    if compose:
        transform = Compose([transform])
    output = transform(data)

    expected_shape = (batch_size, channels, roi_size[0], roi_size[1], roi_size[2])
    assert output.shape == expected_shape
    assert torch.all(output >= 0) and torch.all(output <= 1)


def test_batched_center_spatial_crop_single_vs_batch():
    """Test that BatchedCenterSpatialCrop matches single image behavior."""
    from monai.transforms.croppad.array import CenterSpatialCrop

    roi_size = [64, 64, 32]
    single_crop = CenterSpatialCrop(roi_size=roi_size)
    batch_crop = BatchedCenterSpatialCrop(roi_size=roi_size)

    # Create test data
    single_img = torch.rand(2, 128, 128, 64)  # C, H, W, D
    batch_img = single_img.unsqueeze(0)  # 1, C, H, W, D

    single_result = single_crop(single_img)
    batch_result = batch_crop(batch_img)

    assert single_result.shape == batch_result[0].shape
    assert torch.allclose(single_result, batch_result[0])


def test_batched_center_spatial_crop_deterministic():
    """Test that center cropping is deterministic."""
    batch_size = 2
    channels = 1
    shape = (batch_size, channels, 128, 128, 64)
    roi_size = [64, 64, 32]

    data = torch.rand(shape)
    transform = BatchedCenterSpatialCrop(roi_size=roi_size)

    output1 = transform(data)
    output2 = transform(data)

    assert torch.equal(output1, output2)


def test_batched_center_spatial_crop_edge_cases():
    """Test edge cases for BatchedCenterSpatialCrop."""
    batch_size = 2
    channels = 1
    shape = (batch_size, channels, 64, 64, 32)
    data = torch.rand(shape)

    # Test ROI larger than image (should not crop)
    transform_large = BatchedCenterSpatialCrop(roi_size=[128, 128, 64])
    output_large = transform_large(data)
    assert output_large.shape == shape

    # Test ROI with zero/negative values
    transform_neg = BatchedCenterSpatialCrop(roi_size=[-1, 0, 16])
    output_neg = transform_neg(data)
    expected_shape = (batch_size, channels, 64, 64, 16)  # Only depth should be cropped
    assert output_neg.shape == expected_shape

    # Test exact same size (should not crop)
    transform_same = BatchedCenterSpatialCrop(roi_size=[64, 64, 32])
    output_same = transform_same(data)
    assert output_same.shape == shape


def test_batched_center_spatial_crop_different_roi_formats():
    """Test different ROI size formats."""
    batch_size = 2
    channels = 1
    shape = (batch_size, channels, 128, 128, 64)
    data = torch.rand(shape)

    # Test with int
    transform_int = BatchedCenterSpatialCrop(roi_size=64)
    output_int = transform_int(data)
    assert output_int.shape == (batch_size, channels, 64, 64, 64)

    # Test with list
    transform_list = BatchedCenterSpatialCrop(roi_size=[64, 96, 48])
    output_list = transform_list(data)
    assert output_list.shape == (batch_size, channels, 64, 96, 48)


def test_batched_center_spatial_cropd():
    """Test dict version of BatchedCenterSpatialCrop."""
    batch_size = 2
    channels = 1
    shape = (batch_size, channels, 128, 128, 64)
    roi_size = [64, 64, 32]

    data = {
        "image": torch.rand(shape),
        "mask": torch.randint(0, 2, shape, dtype=torch.float32),
        "other": "unchanged",
    }

    transform = BatchedCenterSpatialCropd(keys=["image", "mask"], roi_size=roi_size)
    output = transform(data)

    expected_shape = (batch_size, channels, roi_size[0], roi_size[1], roi_size[2])
    assert output["image"].shape == expected_shape
    assert output["mask"].shape == expected_shape
    assert output["other"] == "unchanged"
    assert torch.all(output["image"] >= 0) and torch.all(output["image"] <= 1)
    assert torch.all((output["mask"] == 0) | (output["mask"] == 1))


def test_batched_center_spatial_cropd_missing_keys():
    """Test dict version with allow_missing_keys=True."""
    batch_size = 2
    channels = 1
    shape = (batch_size, channels, 128, 128, 64)
    roi_size = [64, 64, 32]

    data = {"image": torch.rand(shape)}

    transform = BatchedCenterSpatialCropd(
        keys=["image", "nonexistent"], roi_size=roi_size, allow_missing_keys=True
    )

    output = transform(data)
    expected_shape = (batch_size, channels, roi_size[0], roi_size[1], roi_size[2])
    assert output["image"].shape == expected_shape
    assert "nonexistent" not in output


def test_batched_center_spatial_cropd_consistency():
    """Test that dictionary transform applies consistent crops across keys."""
    batch_size = 2
    channels = 1
    shape = (batch_size, channels, 128, 128, 64)
    roi_size = [64, 64, 32]

    # Create identical data for two keys
    base_data = torch.rand(shape)
    data = {
        "key1": base_data.clone(),
        "key2": base_data.clone(),
    }

    transform = BatchedCenterSpatialCropd(keys=["key1", "key2"], roi_size=roi_size)
    output = transform(data)

    # Results should be identical since input was identical and cropping is deterministic
    assert torch.equal(output["key1"], output["key2"])


def test_batched_center_spatial_crop_2d():
    """Test BatchedCenterSpatialCrop with 2D data."""
    batch_size = 2
    channels = 1
    shape = (batch_size, channels, 128, 128)
    roi_size = [64, 64]
    data = torch.rand(shape)

    transform = BatchedCenterSpatialCrop(roi_size=roi_size)
    output = transform(data)

    expected_shape = (batch_size, channels, roi_size[0], roi_size[1])
    assert output.shape == expected_shape


def test_batched_center_spatial_crop_vs_monai_loop():
    """Test BatchedCenterSpatialCrop against MONAI's CenterSpatialCrop in a loop with multiple configurations."""
    from monai.transforms.croppad.array import CenterSpatialCrop

    # Test configurations
    test_configs = [
        # (batch_size, channels, H, W, D, roi_size)
        (1, 1, 64, 64, 32, [32, 32, 16]),
        (3, 2, 128, 128, 64, [64, 64, 32]),
        (2, 1, 96, 96, 48, [48, 48, 24]),
        (4, 3, 160, 160, 80, [80, 80, 40]),
        (1, 1, 100, 100, 50, [50, 50, 25]),
        # Edge cases
        (2, 1, 64, 64, 32, [128, 128, 64]),  # ROI larger than image
        (1, 1, 64, 64, 32, [64, 64, 32]),  # ROI same as image
        (2, 2, 64, 64, 32, [32, 96, 16]),  # Mixed ROI sizes
        (1, 1, 100, 100, 50, [33, 33, 17]),  # Odd ROI sizes
        (2, 1, 127, 127, 63, [64, 64, 32]),  # Odd image sizes
    ]

    for batch_size, channels, H, W, D, roi_size in test_configs:
        # Create test data with different seeds for variety
        torch.manual_seed(hash((batch_size, channels, H, W, D)) % 10000)
        batch_data = torch.rand(batch_size, channels, H, W, D)

        # MONAI's single-image transform
        monai_transform = CenterSpatialCrop(roi_size=roi_size)

        # Our batched transform
        batch_transform = BatchedCenterSpatialCrop(roi_size=roi_size)

        # Apply batched transform
        batch_result = batch_transform(batch_data)

        # Apply MONAI transform to each image individually and compare
        for i in range(batch_size):
            single_img = batch_data[i]  # Shape: (C, H, W, D)
            monai_result = monai_transform(single_img)
            batch_single_result = batch_result[i]

            # Check shapes match
            assert monai_result.shape == batch_single_result.shape, (
                f"Shape mismatch for config {(batch_size, channels, H, W, D, roi_size)} "
                f"at batch index {i}: MONAI={monai_result.shape}, "
                f"Batch={batch_single_result.shape}"
            )

            # Check values match exactly
            assert torch.allclose(
                monai_result, batch_single_result, rtol=1e-6, atol=1e-6
            ), (
                f"Value mismatch for config {(batch_size, channels, H, W, D, roi_size)} "
                f"at batch index {i}: max diff = {torch.max(torch.abs(monai_result - batch_single_result))}"
            )


def test_batched_center_spatial_crop_vs_monai_2d_loop():
    """Test 2D cases against MONAI's behavior in a loop."""
    from monai.transforms.croppad.array import CenterSpatialCrop

    test_configs_2d = [
        # (batch_size, channels, H, W, roi_size)
        (1, 1, 64, 64, [32, 32]),
        (3, 2, 128, 128, [64, 64]),
        (2, 1, 96, 96, [48, 48]),
        (4, 1, 100, 100, [50, 50]),
        (1, 3, 80, 120, [40, 60]),  # Non-square
        (2, 1, 127, 127, [64, 64]),  # Odd sizes
    ]

    for batch_size, channels, H, W, roi_size in test_configs_2d:
        # Create test data
        torch.manual_seed(hash((batch_size, channels, H, W)) % 10000)
        batch_data = torch.rand(batch_size, channels, H, W)

        # MONAI's single-image transform
        monai_transform = CenterSpatialCrop(roi_size=roi_size)

        # Our batched transform
        batch_transform = BatchedCenterSpatialCrop(roi_size=roi_size)

        # Apply transforms
        batch_result = batch_transform(batch_data)

        # Compare each image in batch
        for i in range(batch_size):
            single_img = batch_data[i]  # Shape: (C, H, W)
            monai_result = monai_transform(single_img)
            batch_single_result = batch_result[i]

            assert monai_result.shape == batch_single_result.shape, (
                f"2D Shape mismatch for config {(batch_size, channels, H, W, roi_size)} at batch {i}"
            )
            assert torch.allclose(
                monai_result, batch_single_result, rtol=1e-6, atol=1e-6
            ), (
                f"2D Value mismatch for config {(batch_size, channels, H, W, roi_size)} at batch {i}"
            )
