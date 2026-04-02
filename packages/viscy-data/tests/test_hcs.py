import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
from imageio import imwrite
from iohub import open_ome_zarr
from monai.transforms import Compose, RandAdjustContrastd, RandAffined, RandFlipd, RandSpatialCropSamplesd
from pytest import TempPathFactory, fixture, mark, raises

from viscy_data import HCSDataModule
from viscy_data.sliding_window import SlidingWindowDataset
from viscy_transforms import BatchedRandFlipd, RandSpatialCropd


@mark.parametrize("multi_sample_augmentation", [True, False])
def test_datamodule_setup_fit(preprocessed_hcs_dataset, multi_sample_augmentation):
    data_path = preprocessed_hcs_dataset
    z_window_size = 5
    channel_split = 2
    split_ratio = 0.8
    batch_size = 4
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
        img = next(dataset.positions())[1]["0"]
        fov_yx = (img.height, img.width)
    if multi_sample_augmentation:
        yx_patch_size = [128, 96]
        transforms = [
            RandSpatialCropSamplesd(
                keys=channel_names,
                roi_size=[z_window_size, *yx_patch_size],
                num_samples=2,
            )
        ]
    else:
        # No augmentation: yx_patch_size matches FOV so no crop needed
        yx_patch_size = list(fov_yx)
        transforms = []
    dm = HCSDataModule(
        data_path=data_path,
        source_channel=channel_names[:channel_split],
        target_channel=channel_names[channel_split:],
        z_window_size=z_window_size,
        batch_size=batch_size,
        num_workers=0,
        augmentations=transforms,
        target_2d=False,
        split_ratio=split_ratio,
        yx_patch_size=yx_patch_size,
    )
    dm.setup(stage="fit")
    for batch in dm.train_dataloader():
        assert batch["source"].shape == (
            batch_size,
            channel_split,
            z_window_size,
            *yx_patch_size,
        )
        assert batch["target"].shape == (
            batch_size,
            len(channel_names) - channel_split,
            z_window_size,
            *yx_patch_size,
        )


def test_on_after_batch_transfer_shape_mismatch_raises(preprocessed_hcs_dataset):
    """Shape mismatch between source and yx_patch_size raises ValueError."""
    data_path = preprocessed_hcs_dataset
    z_window_size = 5
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
    dm = HCSDataModule(
        data_path=data_path,
        source_channel=channel_names[:2],
        target_channel=channel_names[2:],
        z_window_size=z_window_size,
        batch_size=2,
        num_workers=0,
        yx_patch_size=[128, 96],
    )
    dm.setup(stage="fit")
    dm.trainer = MagicMock(training=True, validating=False)
    batch = {
        "source": torch.randn(2, 2, z_window_size, 256, 256),
        "target": torch.randn(2, 2, z_window_size, 256, 256),
    }
    with raises(ValueError, match="does not match expected"):
        dm.on_after_batch_transfer(batch, 0)


def test_on_after_batch_transfer_passthrough(preprocessed_hcs_dataset):
    """Correct-sized batch passes through without error."""
    data_path = preprocessed_hcs_dataset
    z_window_size = 5
    yx_patch_size = [128, 96]
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
    dm = HCSDataModule(
        data_path=data_path,
        source_channel=channel_names[:2],
        target_channel=channel_names[2:],
        z_window_size=z_window_size,
        batch_size=2,
        num_workers=0,
        yx_patch_size=yx_patch_size,
    )
    dm.setup(stage="fit")
    dm.trainer = MagicMock(training=False, validating=True)
    batch = {
        "source": torch.randn(2, 2, z_window_size, *yx_patch_size),
        "target": torch.randn(2, 2, z_window_size, *yx_patch_size),
    }
    result = dm.on_after_batch_transfer(batch, 0)
    assert result["source"].shape == (2, 2, z_window_size, *yx_patch_size)
    assert result["target"].shape == (2, 2, z_window_size, *yx_patch_size)


def test_on_after_batch_transfer_target_2d(preprocessed_hcs_dataset):
    """target_2d slices target Z to 1."""
    data_path = preprocessed_hcs_dataset
    z_window_size = 5
    yx_patch_size = [128, 96]
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
    dm = HCSDataModule(
        data_path=data_path,
        source_channel=channel_names[:2],
        target_channel=channel_names[2:],
        z_window_size=z_window_size,
        batch_size=2,
        num_workers=0,
        yx_patch_size=yx_patch_size,
        target_2d=True,
    )
    dm.setup(stage="fit")
    dm.trainer = MagicMock(training=True, validating=False)
    batch = {
        "source": torch.randn(2, 2, z_window_size, *yx_patch_size),
        "target": torch.randn(2, 2, z_window_size, *yx_patch_size),
    }
    result = dm.on_after_batch_transfer(batch, 0)
    assert result["source"].shape == (2, 2, z_window_size, *yx_patch_size)
    assert result["target"].shape == (2, 2, 1, *yx_patch_size)


def test_datamodule_caching(preprocessed_hcs_dataset):
    """Test that prepare_data caches the dataset to a local directory."""
    data_path = preprocessed_hcs_dataset
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
    dm = HCSDataModule(
        data_path=data_path,
        source_channel=channel_names[:2],
        target_channel=channel_names[2:],
        z_window_size=5,
        batch_size=2,
        num_workers=0,
        caching=True,
    )
    cache_path = dm.cache_path
    if cache_path.exists():
        shutil.rmtree(cache_path)
    try:
        assert not cache_path.exists()
        dm.prepare_data()
        assert cache_path.exists()
        with open_ome_zarr(cache_path) as cached:
            assert len(list(cached.positions())) == len(list(open_ome_zarr(data_path).positions()))
        # Second call should skip (cache exists)
        dm.prepare_data()
        # Setup should use cached path
        dm.setup(stage="fit")
        for batch in dm.train_dataloader():
            assert batch["source"].shape[1] == 2
            break
    finally:
        if cache_path.exists():
            shutil.rmtree(cache_path)


@mark.parametrize("z_window_size", [1, 5])
def test_datamodule_setup_predict(preprocessed_hcs_dataset, z_window_size):
    data_path = preprocessed_hcs_dataset
    channel_split = 2
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
        img = next(dataset.positions())[1][0]
        total_p = len(list(dataset.positions()))
    dm = HCSDataModule(
        data_path=data_path,
        source_channel=channel_names[:channel_split],
        target_channel=channel_names[channel_split:],
        z_window_size=z_window_size,
        target_2d=bool(z_window_size == 1),
        batch_size=2,
        num_workers=0,
    )
    dm.setup(stage="predict")
    dataset = dm.predict_dataset
    assert len(dataset) == total_p * 2 * (img.slices - z_window_size + 1)
    assert dataset[0]["source"].shape == (
        channel_split,
        z_window_size,
        img.height,
        img.width,
    )


def test_datamodule_setup_test_no_masks(preprocessed_hcs_dataset):
    """setup('test') without ground_truth_masks creates a SlidingWindowDataset."""
    data_path = preprocessed_hcs_dataset
    z_window_size = 5
    channel_split = 2
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
        total_p = len(list(dataset.positions()))
        img = next(dataset.positions())[1]["0"]
    dm = HCSDataModule(
        data_path=data_path,
        source_channel=channel_names[:channel_split],
        target_channel=channel_names[channel_split:],
        z_window_size=z_window_size,
        batch_size=1,
        num_workers=0,
    )
    dm.setup(stage="test")
    assert hasattr(dm, "test_dataset")
    expected_len = total_p * img.frames * (img.slices - z_window_size + 1)
    assert len(dm.test_dataset) == expected_len
    sample = dm.test_dataset[0]
    assert "source" in sample
    assert "target" in sample
    assert sample["source"].shape[0] == channel_split
    assert sample["source"].shape[1] == z_window_size


@fixture(scope="function")
def mask_dir(tmp_path_factory: TempPathFactory, preprocessed_hcs_dataset) -> Path:
    """Create a directory with fake ground truth mask PNGs matching the dataset."""
    masks_path = tmp_path_factory.mktemp("gt_masks")
    rng = np.random.default_rng(0)
    with open_ome_zarr(preprocessed_hcs_dataset) as plate:
        for fov_name, fov in plate.positions():
            # Extract the FOV number (last path component)
            fov_num = int(fov_name.split("/")[-1])
            img = fov["0"]
            z_center = img.slices // 2
            mask = rng.integers(0, 10, size=(img.height, img.width), dtype=np.int16)
            fname = f"channel_p{fov_num:03d}_z{z_center}_cp_masks.png"
            imwrite(masks_path / fname, mask.astype(np.uint16))
    return masks_path


def test_datamodule_setup_test_with_masks(preprocessed_hcs_dataset, mask_dir):
    """setup('test') with ground_truth_masks creates a MaskTestDataset."""
    data_path = preprocessed_hcs_dataset
    z_window_size = 5
    channel_split = 2
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
    dm = HCSDataModule(
        data_path=data_path,
        source_channel=channel_names[:channel_split],
        target_channel=channel_names[channel_split:],
        z_window_size=z_window_size,
        batch_size=1,
        num_workers=0,
        ground_truth_masks=mask_dir,
    )
    dm.setup(stage="test")
    assert hasattr(dm, "test_dataset")
    assert len(dm.test_dataset) > 0
    # At least some samples should have "labels" when the mask key matches
    found_labels = False
    for i in range(min(len(dm.test_dataset), 50)):
        sample = dm.test_dataset[i]
        assert "source" in sample
        if "labels" in sample:
            found_labels = True
            break
    assert found_labels, "No sample had 'labels' key from ground truth masks"


@fixture(scope="function")
def hcs_with_fg_mask(tmp_path):
    """HCS dataset with precomputed fg_mask array."""
    dataset_path = tmp_path / "fg_mask.zarr"
    ch_names = ["Phase", "Fluorescence"]
    rng = np.random.default_rng(42)
    with open_ome_zarr(dataset_path, layout="hcs", mode="w", channel_names=ch_names) as dataset:
        for row in ("A",):
            for col in ("1",):
                for fov in ("0", "1"):
                    pos = dataset.create_position(row, col, fov)
                    img = rng.random((1, len(ch_names), 8, 32, 32)).astype(np.float32)
                    pos.create_image("0", img, chunks=(1, 1, 1, 32, 32))
                    mask = np.zeros_like(img, dtype=np.uint8)
                    mask[:, 1] = (img[:, 1] > 0.5).astype(np.uint8)
                    pos.create_image("fg_mask", mask, chunks=(1, 1, 1, 32, 32))
        norm = {ch: {"fov_statistics": {"mean": 0.5, "std": 0.29}} for ch in ch_names}
        for _, fov in dataset.positions():
            fov.zattrs["normalization"] = norm
    return dataset_path


def test_sliding_window_with_fg_mask(hcs_with_fg_mask):
    """fg_mask is present in samples when fg_mask_key is set."""
    with open_ome_zarr(hcs_with_fg_mask, mode="r") as ds:
        positions = [pos for _, pos in ds.positions()]
        dataset = SlidingWindowDataset(
            positions=positions,
            channels={"source": ["Phase"], "target": ["Fluorescence"]},
            z_window_size=4,
            fg_mask_key="fg_mask",
        )
        sample = dataset[0]
    assert "fg_mask" in sample
    fg_mask = sample["fg_mask"]
    assert fg_mask.shape[0] == 1  # C_target = 1 (Fluorescence)
    assert fg_mask.shape[1] == 4  # Z = z_window_size
    assert set(fg_mask.unique().tolist()).issubset({0.0, 1.0})


def test_sliding_window_without_fg_mask(hcs_with_fg_mask):
    """fg_mask is NOT in samples when fg_mask_key is None (default)."""
    with open_ome_zarr(hcs_with_fg_mask, mode="r") as ds:
        positions = [pos for _, pos in ds.positions()]
        dataset = SlidingWindowDataset(
            positions=positions,
            channels={"source": ["Phase"], "target": ["Fluorescence"]},
            z_window_size=4,
        )
        sample = dataset[0]
    assert "fg_mask" not in sample


def test_fg_mask_key_missing_errors(tmp_path):
    """FileNotFoundError when fg_mask_key is set but array is absent."""
    dataset_path = tmp_path / "no_mask.zarr"
    ch_names = ["Phase", "Fluorescence"]
    rng = np.random.default_rng(0)
    with open_ome_zarr(dataset_path, layout="hcs", mode="w", channel_names=ch_names) as ds:
        pos = ds.create_position("A", "1", "0")
        pos.create_image("0", rng.random((1, 2, 8, 32, 32)).astype(np.float32), chunks=(1, 1, 1, 32, 32))

    with open_ome_zarr(dataset_path, mode="r") as ds:
        positions = [pos for _, pos in ds.positions()]
        with raises(FileNotFoundError, match="fg_mask"):
            SlidingWindowDataset(
                positions=positions,
                channels={"source": ["Phase"], "target": ["Fluorescence"]},
                z_window_size=4,
                fg_mask_key="fg_mask",
            )


# ---------------------------------------------------------------------------
# fg_mask spatial co-alignment tests
# ---------------------------------------------------------------------------


def test_fg_mask_keys_injected_into_spatial_not_intensity():
    """Spatial augmentations get mask keys; intensity augmentations do not."""
    spatial = RandAffined(keys=["Phase", "Fluorescence"], prob=0.5, rotate_range=[0.1])
    intensity = RandAdjustContrastd(keys=["Phase", "Fluorescence"], prob=0.5)
    HCSDataModule._inject_mask_keys(
        [spatial, intensity],
        ("Fluorescence",),
        ("__fg_mask_Fluorescence",),
    )
    assert "__fg_mask_Fluorescence" in spatial.keys
    assert "__fg_mask_Fluorescence" not in intensity.keys
    assert spatial.allow_missing_keys is True
    # Idempotent — calling again should not duplicate keys
    HCSDataModule._inject_mask_keys(
        [spatial],
        ("Fluorescence",),
        ("__fg_mask_Fluorescence",),
    )
    assert spatial.keys.count("__fg_mask_Fluorescence") == 1


def test_fg_mask_aligned_after_cpu_spatial_augmentation():
    """fg_mask stays pixel-aligned with target after RandFlipd(prob=1)."""
    H, W = 16, 16
    target = torch.zeros(1, 1, H, W)
    target[:, :, : H // 2, :] = 1.0
    mask = (target > 0).float()

    flip = RandFlipd(keys=["target"], prob=1.0, spatial_axis=-2)
    HCSDataModule._inject_mask_keys([flip], ("target",), ("__fg_mask",))
    result = Compose([flip])({"target": target, "__fg_mask": mask})

    assert torch.equal((result["target"] > 0).float(), result["__fg_mask"]), (
        "fg_mask is not spatially aligned with target after CPU augmentation"
    )


def test_fg_mask_aligned_after_gpu_spatial_augmentation():
    """fg_mask stays pixel-aligned with target after BatchedRandFlipd(prob=1)."""
    B, C, D, H, W = 2, 1, 4, 16, 16
    target = torch.zeros(B, C, D, H, W)
    target[:, :, :, : H // 2, :] = 1.0
    mask = (target > 0).float()

    flip = BatchedRandFlipd(keys=["target"], prob=1.0, spatial_axes=[1])
    HCSDataModule._inject_mask_keys([flip], ("target",), ("fg_mask",))
    result = flip({"target": target, "fg_mask": mask})

    assert torch.equal((result["target"] > 0).float(), result["fg_mask"]), (
        "fg_mask is not spatially aligned with target after GPU augmentation"
    )


# ---------------------------------------------------------------------------
# CPU-side RandSpatialCropd co-alignment test
# ---------------------------------------------------------------------------


def test_fg_mask_aligned_after_cpu_rand_spatial_crop():
    """fg_mask stays pixel-aligned with target after RandSpatialCropd."""
    H, W = 32, 32
    target = torch.zeros(1, 1, H, W)
    target[:, :, : H // 2, :] = 1.0
    mask = (target > 0).float()

    crop = RandSpatialCropd(keys=["target"], roi_size=[1, 16, 16])
    HCSDataModule._inject_mask_keys([crop], ("target",), ("__fg_mask",))
    result = Compose([crop])({"target": target, "__fg_mask": mask})

    assert torch.equal((result["target"] > 0).float(), result["__fg_mask"]), (
        "fg_mask is not spatially aligned with target after RandSpatialCropd"
    )


# ---------------------------------------------------------------------------
# Partial zarr read (crop_at_read) tests
# ---------------------------------------------------------------------------


def test_partial_zarr_read(hcs_with_fg_mask):
    """yx_patch_size on SlidingWindowDataset produces correct output shapes."""
    yx_patch_size = (16, 16)
    z_window_size = 4
    with open_ome_zarr(hcs_with_fg_mask, mode="r") as ds:
        positions = [pos for _, pos in ds.positions()]
        dataset = SlidingWindowDataset(
            positions=positions,
            channels={"source": ["Phase"], "target": ["Fluorescence"]},
            z_window_size=z_window_size,
            yx_patch_size=yx_patch_size,
        )
        sample = dataset[0]
    assert sample["source"].shape == (1, z_window_size, *yx_patch_size)
    assert sample["target"].shape == (1, z_window_size, *yx_patch_size)


def test_partial_zarr_read_with_fg_mask(hcs_with_fg_mask):
    """yx_patch_size reads fg_mask from the same spatial region."""
    yx_patch_size = (16, 16)
    z_window_size = 4
    with open_ome_zarr(hcs_with_fg_mask, mode="r") as ds:
        positions = [pos for _, pos in ds.positions()]
        dataset = SlidingWindowDataset(
            positions=positions,
            channels={"source": ["Phase"], "target": ["Fluorescence"]},
            z_window_size=z_window_size,
            fg_mask_key="fg_mask",
            yx_patch_size=yx_patch_size,
        )
        sample = dataset[0]
    assert "fg_mask" in sample
    assert sample["fg_mask"].shape == (1, z_window_size, *yx_patch_size)
    assert set(sample["fg_mask"].unique().tolist()).issubset({0.0, 1.0})


def test_crop_at_read_datamodule(hcs_with_fg_mask):
    """HCSDataModule with crop_at_read=True produces correct batch shapes."""
    yx_patch_size = [16, 16]
    z_window_size = 4
    dm = HCSDataModule(
        data_path=hcs_with_fg_mask,
        source_channel="Phase",
        target_channel="Fluorescence",
        z_window_size=z_window_size,
        batch_size=2,
        num_workers=0,
        yx_patch_size=yx_patch_size,
        split_ratio=0.5,
        crop_at_read=True,
        fg_mask_key="fg_mask",
    )
    dm.setup(stage="fit")
    dm.trainer = MagicMock(training=True, validating=False)
    for batch in dm.train_dataloader():
        assert batch["source"].shape == (2, 1, z_window_size, *yx_patch_size)
        assert batch["target"].shape == (2, 1, z_window_size, *yx_patch_size)
        assert batch["fg_mask"].shape == (2, 1, z_window_size, *yx_patch_size)
        result = dm.on_after_batch_transfer(batch, 0)
        assert result["source"].shape == batch["source"].shape
        break
