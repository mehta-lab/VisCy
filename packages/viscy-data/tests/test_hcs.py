import shutil
from pathlib import Path

import numpy as np
from imageio import imwrite
from iohub import open_ome_zarr
from monai.transforms import RandSpatialCropSamplesd
from pytest import TempPathFactory, fixture, mark, raises

from viscy_data import HCSDataModule
from viscy_data.hcs import SlidingWindowDataset


@mark.parametrize("multi_sample_augmentation", [True, False])
def test_datamodule_setup_fit(preprocessed_hcs_dataset, multi_sample_augmentation):
    data_path = preprocessed_hcs_dataset
    z_window_size = 5
    channel_split = 2
    split_ratio = 0.8
    yx_patch_size = [128, 96]
    batch_size = 4
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
    if multi_sample_augmentation:
        transforms = [
            RandSpatialCropSamplesd(
                keys=channel_names,
                roi_size=[z_window_size, *yx_patch_size],
                num_samples=2,
            )
        ]
    else:
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
