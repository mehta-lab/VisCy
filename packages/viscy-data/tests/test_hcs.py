from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
from imageio import imwrite
from iohub import open_ome_zarr
from lightning.pytorch.trainer.states import TrainerFn
from monai.transforms import Compose, RandAdjustContrastd, RandAffined, RandFlipd, RandSpatialCropSamplesd
from pytest import TempPathFactory, fixture, importorskip, mark, raises

from viscy_data import HCSDataModule
from viscy_data.sliding_window import SlidingWindowDataset
from viscy_transforms import BatchedRandFlipd, NormalizeSampled, RandSpatialCropd


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


def _fov_names(dataset_path: Path) -> list[str]:
    with open_ome_zarr(dataset_path) as plate:
        return [name for name, _ in plate.positions()]


def test_fov_name_filters_applied(preprocessed_hcs_dataset):
    """include_fov_names keeps only listed positions; exclude_fov_names drops them."""
    data_path = preprocessed_hcs_dataset
    all_fovs = _fov_names(data_path)
    with open_ome_zarr(data_path) as ds:
        channel_names = ds.channel_names

    kept = all_fovs[:4]
    dm = HCSDataModule(
        data_path=data_path,
        source_channel=channel_names[:1],
        target_channel=channel_names[1:2],
        z_window_size=5,
        batch_size=1,
        num_workers=0,
        split_ratio=0.5,
        yx_patch_size=[16, 16],
        include_fov_names=kept,
    )
    dm.setup(stage="fit")
    selected = {str(p.zgroup.path) for p in dm.train_dataset.positions} | {
        str(p.zgroup.path) for p in dm.val_dataset.positions
    }
    assert all(any(name.endswith(f) for name in selected) for f in kept)
    assert len(selected) == len(kept)

    dropped = all_fovs[:2]
    dm = HCSDataModule(
        data_path=data_path,
        source_channel=channel_names[:1],
        target_channel=channel_names[1:2],
        z_window_size=5,
        batch_size=1,
        num_workers=0,
        split_ratio=0.5,
        yx_patch_size=[16, 16],
        exclude_fov_names=dropped,
    )
    dm.setup(stage="fit")
    n_positions = len(dm.train_dataset.positions) + len(dm.val_dataset.positions)
    assert n_positions == len(all_fovs) - len(dropped)


def test_fov_name_filters_raise_when_empty(preprocessed_hcs_dataset):
    """Filtering to zero positions raises a clear ValueError."""
    data_path = preprocessed_hcs_dataset
    with open_ome_zarr(data_path) as ds:
        channel_names = ds.channel_names
    dm = HCSDataModule(
        data_path=data_path,
        source_channel=channel_names[:1],
        target_channel=channel_names[1:2],
        z_window_size=5,
        batch_size=1,
        num_workers=0,
        yx_patch_size=[16, 16],
        include_fov_names=["does/not/exist"],
    )
    with raises(ValueError, match="No positions"):
        dm.setup(stage="fit")


def test_fov_name_filters_with_mmap_preload(preprocessed_hcs_dataset, tmp_path):
    """include/exclude_fov_names is honored during mmap_preload staging.

    The mmap buffer is sized and indexed by the filtered position list, so
    the buffer must contain only the kept FOVs — otherwise _setup_fit's
    filtered orig_positions would alias the wrong entries.
    """
    importorskip("tensordict")
    data_path = preprocessed_hcs_dataset
    all_fovs = _fov_names(data_path)
    with open_ome_zarr(data_path) as ds:
        channel_names = ds.channel_names
    dropped = all_fovs[:2]
    dm = HCSDataModule(
        data_path=data_path,
        source_channel=channel_names[:1],
        target_channel=channel_names[1:2],
        z_window_size=5,
        batch_size=1,
        num_workers=0,
        yx_patch_size=[16, 16],
        split_ratio=0.5,
        mmap_preload=True,
        scratch_dir=tmp_path,
        exclude_fov_names=dropped,
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    selected = {str(p.zgroup.path) for p in dm.train_dataset.positions} | {
        str(p.zgroup.path) for p in dm.val_dataset.positions
    }
    assert len(selected) == len(all_fovs) - len(dropped)
    assert not any(name.endswith(d) for name in selected for d in dropped)
    for batch in dm.train_dataloader():
        assert batch["source"].shape[1] == 1
        assert batch["target"].shape[1] == 1
        break
    assert (dm._mmap_cache_dir / ".done").exists()


def test_fov_name_filters_with_mmap_preload_cache_fingerprint(preprocessed_hcs_dataset, tmp_path):
    """Different FOV filters must produce different mmap cache dirs.

    Sharing a cache across filter configs would alias wrong FOVs because
    the buffer layout is filter-dependent.
    """
    importorskip("tensordict")
    data_path = preprocessed_hcs_dataset
    all_fovs = _fov_names(data_path)
    with open_ome_zarr(data_path) as ds:
        channel_names = ds.channel_names
    base_kwargs = dict(
        data_path=data_path,
        source_channel=channel_names[:1],
        target_channel=channel_names[1:2],
        z_window_size=5,
        batch_size=1,
        num_workers=0,
        yx_patch_size=[16, 16],
        mmap_preload=True,
        scratch_dir=tmp_path,
    )
    dm_no_filter = HCSDataModule(**base_kwargs)
    dm_exclude = HCSDataModule(exclude_fov_names=all_fovs[:1], **base_kwargs)
    dm_include = HCSDataModule(include_fov_names=all_fovs[:2], **base_kwargs)
    dirs = {dm_no_filter._mmap_cache_dir, dm_exclude._mmap_cache_dir, dm_include._mmap_cache_dir}
    assert len(dirs) == 3


def _bad_filter_dm(data_path: Path, scratch_dir: Path) -> HCSDataModule:
    with open_ome_zarr(data_path) as ds:
        channel_names = ds.channel_names
    return HCSDataModule(
        data_path=data_path,
        source_channel=channel_names[:1],
        target_channel=channel_names[1:2],
        z_window_size=5,
        batch_size=1,
        num_workers=0,
        yx_patch_size=[16, 16],
        mmap_preload=True,
        scratch_dir=scratch_dir,
        include_fov_names=["does/not/exist"],
    )


def test_mmap_preload_raises_for_fit_with_bad_filter(preprocessed_hcs_dataset, tmp_path):
    """Fit-stage prepare_data() keeps the strict empty-filter signal.

    Guards the stage gate: if a broken helper ever returned False for
    FITTING, prepare_data() would silently skip and a ValueError on
    config typos would only surface later in setup('fit').
    """
    importorskip("tensordict")
    dm = _bad_filter_dm(preprocessed_hcs_dataset, tmp_path)
    dm.trainer = MagicMock(state=MagicMock(fn=TrainerFn.FITTING))
    with raises(ValueError, match="No positions left"):
        dm.prepare_data()
    assert not (dm._mmap_cache_dir / ".done").exists()


def test_mmap_preload_skipped_for_predict_with_bad_filter(preprocessed_hcs_dataset, tmp_path):
    """Predict-only runs don't consume the mmap buffer, so prepare_data() skips.

    Before this change, a bad include_fov_names filter would kill a
    predict-only job even though _setup_predict ignores FOV filters
    entirely and never opens the mmap cache.
    """
    importorskip("tensordict")
    dm = _bad_filter_dm(preprocessed_hcs_dataset, tmp_path)
    dm.trainer = MagicMock(state=MagicMock(fn=TrainerFn.PREDICTING))
    dm.prepare_data()
    assert not (dm._mmap_cache_dir / ".done").exists()


def test_mmap_preload_skipped_for_test_with_bad_filter(preprocessed_hcs_dataset, tmp_path):
    """Test-only runs also skip mmap staging; mirror of predict."""
    importorskip("tensordict")
    dm = _bad_filter_dm(preprocessed_hcs_dataset, tmp_path)
    dm.trainer = MagicMock(state=MagicMock(fn=TrainerFn.TESTING))
    dm.prepare_data()
    assert not (dm._mmap_cache_dir / ".done").exists()


def test_setup_fit_raises_when_mmap_cache_missing(preprocessed_hcs_dataset, tmp_path):
    """setup('fit') without a prior prepare_data() raises an actionable error.

    Previously this failed inside MemoryMappedTensor.from_filename with
    an opaque FileNotFoundError / RuntimeError. The pre-check in
    _setup_fit now surfaces the missing-cache case with a clear hint.
    """
    importorskip("tensordict")
    with open_ome_zarr(preprocessed_hcs_dataset) as ds:
        channel_names = ds.channel_names
    dm = HCSDataModule(
        data_path=preprocessed_hcs_dataset,
        source_channel=channel_names[:1],
        target_channel=channel_names[1:2],
        z_window_size=5,
        batch_size=1,
        num_workers=0,
        yx_patch_size=[16, 16],
        split_ratio=0.5,
        mmap_preload=True,
        scratch_dir=tmp_path,
    )
    with raises(RuntimeError, match="cache not ready"):
        dm.setup(stage="fit")


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
# timepoint_statistics normalization
# ---------------------------------------------------------------------------


def test_resolve_timepoint_norm_meta_flattens_requested_index():
    """Helper flattens timepoint_statistics to the t-th entry; other levels pass through."""
    meta = {
        "Phase": {
            "fov_statistics": {"mean": torch.tensor(0.5), "std": torch.tensor(0.1)},
            "timepoint_statistics": {
                "0": {"mean": torch.tensor(10.0), "std": torch.tensor(1.0)},
                "1": {"mean": torch.tensor(1000.0), "std": torch.tensor(100.0)},
            },
        },
    }
    resolved = SlidingWindowDataset._resolve_timepoint_norm_meta(meta, t=1)
    assert resolved["Phase"]["fov_statistics"]["mean"].item() == 0.5
    assert resolved["Phase"]["timepoint_statistics"]["mean"].item() == 1000.0
    assert resolved["Phase"]["timepoint_statistics"]["std"].item() == 100.0
    assert SlidingWindowDataset._resolve_timepoint_norm_meta(None, t=0) is None


@fixture(scope="function")
def hcs_with_timepoint_stats(tmp_path):
    """Two-timepoint HCS with per-timepoint normalization stats written to zattrs."""
    dataset_path = tmp_path / "tp_stats.zarr"
    ch_names = ["Phase", "Fluor"]
    num_t = 2
    rng = np.random.default_rng(0)
    with open_ome_zarr(dataset_path, layout="hcs", mode="w", channel_names=ch_names) as ds:
        for fov_name in ("0", "1"):
            pos = ds.create_position("A", "1", fov_name)
            img = rng.random((num_t, len(ch_names), 4, 16, 16)).astype(np.float32)
            pos.create_image("0", img, chunks=(1, 1, 1, 16, 16))
    # Per-timepoint stats chosen to be very different so mis-selection is obvious.
    tp_stats = {
        "0": {"mean": 10.0, "std": 1.0},
        "1": {"mean": 1000.0, "std": 100.0},
    }
    norm = {ch: {"timepoint_statistics": tp_stats} for ch in ch_names}
    with open_ome_zarr(dataset_path, mode="r+") as ds:
        for _, fov in ds.positions():
            fov.zattrs["normalization"] = norm
    return dataset_path


def test_sliding_window_timepoint_statistics_normalize(hcs_with_timepoint_stats):
    """NormalizeSampled(level='timepoint_statistics') uses the sample's timepoint stats."""
    z_window = 4
    tp_stats = {0: (10.0, 1.0), 1: (1000.0, 100.0)}
    with open_ome_zarr(hcs_with_timepoint_stats, mode="r") as ds:
        positions = [pos for _, pos in ds.positions()]
        normalized = SlidingWindowDataset(
            positions=positions,
            channels={"source": ["Phase"], "target": ["Fluor"]},
            z_window_size=z_window,
            transform=NormalizeSampled(keys=["Phase"], level="timepoint_statistics"),
        )
        raw = SlidingWindowDataset(
            positions=positions,
            channels={"source": ["Phase"], "target": ["Fluor"]},
            z_window_size=z_window,
        )
        assert len(normalized) == 4  # 2 FOVs * 2 timepoints * 1 z-window
        seen_t = set()
        for idx in range(len(normalized)):
            norm_sample = normalized[idx]
            raw_sample = raw[idx]
            t = norm_sample["index"][1]
            seen_t.add(t)
            mean, std = tp_stats[t]
            expected = (raw_sample["source"] - mean) / (std + 1e-8)
            assert torch.allclose(norm_sample["source"], expected, atol=1e-6), (
                f"Timepoint {t} (idx={idx}) normalization mismatch: "
                f"expected ({mean}, {std}), got source range "
                f"[{norm_sample['source'].min():.3f}, {norm_sample['source'].max():.3f}]"
            )
        assert seen_t == {0, 1}, f"Test did not exercise both timepoints (saw {seen_t})"


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
# Preloaded FOV (mmap) tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# val_gpu_augmentations tests
# ---------------------------------------------------------------------------


def test_val_gpu_augmentations_applied_during_validation(preprocessed_hcs_dataset):
    """val_gpu_augmentations crops validation batches via on_after_batch_transfer."""
    from viscy_transforms import BatchedCenterSpatialCropd

    data_path = preprocessed_hcs_dataset
    z_window_size = 5
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
    crop_yx = [128, 128]
    dm = HCSDataModule(
        data_path=data_path,
        source_channel=channel_names[:2],
        target_channel=channel_names[2:],
        z_window_size=z_window_size,
        batch_size=2,
        num_workers=0,
        yx_patch_size=[256, 256],
        val_gpu_augmentations=[
            BatchedCenterSpatialCropd(keys=["source", "target"], roi_size=[z_window_size, *crop_yx]),
        ],
    )
    dm.setup(stage="fit")
    dm.trainer = MagicMock(training=False, validating=True, sanity_checking=False)
    batch = {
        "source": torch.randn(2, 2, z_window_size, 256, 256),
        "target": torch.randn(2, 2, z_window_size, 256, 256),
    }
    result = dm.on_after_batch_transfer(batch, 0)
    assert result["source"].shape == (2, 2, z_window_size, *crop_yx)
    assert result["target"].shape == (2, 2, z_window_size, *crop_yx)


def test_val_gpu_augmentations_applied_during_sanity_checking(preprocessed_hcs_dataset):
    """val_gpu_augmentations also fires when trainer.sanity_checking is True."""
    from viscy_transforms import BatchedCenterSpatialCropd

    data_path = preprocessed_hcs_dataset
    z_window_size = 5
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
    crop_yx = [128, 128]
    dm = HCSDataModule(
        data_path=data_path,
        source_channel=channel_names[:2],
        target_channel=channel_names[2:],
        z_window_size=z_window_size,
        batch_size=2,
        num_workers=0,
        yx_patch_size=[256, 256],
        val_gpu_augmentations=[
            BatchedCenterSpatialCropd(keys=["source", "target"], roi_size=[z_window_size, *crop_yx]),
        ],
    )
    dm.setup(stage="fit")
    dm.trainer = MagicMock(training=False, validating=False, sanity_checking=True)
    batch = {
        "source": torch.randn(2, 2, z_window_size, 256, 256),
        "target": torch.randn(2, 2, z_window_size, 256, 256),
    }
    result = dm.on_after_batch_transfer(batch, 0)
    assert result["source"].shape == (2, 2, z_window_size, *crop_yx)
    assert result["target"].shape == (2, 2, z_window_size, *crop_yx)


def test_sliding_window_preloaded(hcs_with_fg_mask):
    """preloaded_fovs bypasses zarr reads and returns correct shapes."""
    z_window_size = 4
    with open_ome_zarr(hcs_with_fg_mask, mode="r") as ds:
        positions = [pos for _, pos in ds.positions()]
        # Build plain tensors shaped (T, C, Z, Y, X) — channels: source + target
        arr0 = positions[0]["0"]
        T, C, Z, Y, X = arr0.frames, 2, arr0.slices, arr0.height, arr0.width
        preloaded = [torch.rand(T, C, Z, Y, X) for _ in positions]
        dataset = SlidingWindowDataset(
            positions=positions,
            channels={"source": ["Phase"], "target": ["Fluorescence"]},
            z_window_size=z_window_size,
            preloaded_fovs=preloaded,
        )
        sample = dataset[0]
    assert sample["source"].shape == (1, z_window_size, Y, X)
    assert sample["target"].shape == (1, z_window_size, Y, X)


def test_sliding_window_preloaded_returns_copy(hcs_with_fg_mask):
    """preloaded path returns a clone, not a view into the preloaded buffer."""
    z_window_size = 4
    with open_ome_zarr(hcs_with_fg_mask, mode="r") as ds:
        positions = [pos for _, pos in ds.positions()]
        arr0 = positions[0]["0"]
        T, C, Z, Y, X = arr0.frames, 2, arr0.slices, arr0.height, arr0.width
        preloaded = [torch.rand(T, C, Z, Y, X) for _ in positions]
        dataset = SlidingWindowDataset(
            positions=positions,
            channels={"source": ["Phase"], "target": ["Fluorescence"]},
            z_window_size=z_window_size,
            preloaded_fovs=preloaded,
        )
        sample = dataset[0]
        original_source = sample["source"].clone()
        # Mutate the returned tensor
        sample["source"].fill_(999.0)
        # Fetch the same index again — buffer should be unchanged
        sample2 = dataset[0]
    assert torch.equal(sample2["source"], original_source)


def test_mmap_preload_roundtrip(hcs_with_fg_mask, tmp_path):
    """prepare_data() + setup() + dataloader roundtrip with mmap_preload=True."""
    importorskip("tensordict")
    z_window_size = 4
    yx_patch_size = [32, 32]
    dm = HCSDataModule(
        data_path=hcs_with_fg_mask,
        source_channel="Phase",
        target_channel="Fluorescence",
        z_window_size=z_window_size,
        batch_size=2,
        num_workers=0,
        yx_patch_size=yx_patch_size,
        split_ratio=0.5,
        mmap_preload=True,
        scratch_dir=tmp_path,
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    for batch in dm.train_dataloader():
        assert batch["source"].shape[1] == 1  # 1 source channel
        assert batch["target"].shape[1] == 1  # 1 target channel
        break
    # .done marker must exist after prepare_data
    assert (dm._mmap_cache_dir / ".done").exists()


def test_mmap_preload_skips_when_done(hcs_with_fg_mask, tmp_path):
    """prepare_data() is idempotent: skips mmap preload if .done marker exists."""
    importorskip("tensordict")
    dm = HCSDataModule(
        data_path=hcs_with_fg_mask,
        source_channel="Phase",
        target_channel="Fluorescence",
        z_window_size=4,
        batch_size=2,
        num_workers=0,
        mmap_preload=True,
        scratch_dir=tmp_path,
    )
    dm.prepare_data()
    mmap_file = dm._mmap_cache_dir / "data.mmap"
    mtime_after_first = mmap_file.stat().st_mtime
    # Second call must not rewrite the buffer
    dm.prepare_data()
    assert mmap_file.stat().st_mtime == mtime_after_first


def test_mmap_preload_recovers_from_partial_cache(hcs_with_fg_mask, tmp_path):
    """prepare_data() cleans up and rebuilds if a previous run was killed mid-write."""
    importorskip("tensordict")
    dm = HCSDataModule(
        data_path=hcs_with_fg_mask,
        source_channel="Phase",
        target_channel="Fluorescence",
        z_window_size=4,
        batch_size=2,
        num_workers=0,
        mmap_preload=True,
        scratch_dir=tmp_path,
    )
    # Simulate a killed mmap preload: create the cache dir with a partial .mmap but no .done
    cache_dir = dm._mmap_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "data.mmap").write_bytes(b"partial garbage")
    assert not (cache_dir / ".done").exists()
    # prepare_data should clean up and succeed
    dm.prepare_data()
    assert (cache_dir / ".done").exists()
    # Verify the rebuilt buffer works end-to-end
    dm.setup(stage="fit")
    for batch in dm.train_dataloader():
        assert batch["source"].shape[1] == 1
        break


def test_mmap_preload_multi_process_sharing(hcs_with_fg_mask, tmp_path):
    """Both parent and child processes can open the mmap buffer after prepare_data."""
    import multiprocessing

    importorskip("tensordict")
    from tensordict.memmap import MemoryMappedTensor

    dm = HCSDataModule(
        data_path=hcs_with_fg_mask,
        source_channel="Phase",
        target_channel="Fluorescence",
        z_window_size=4,
        batch_size=2,
        num_workers=0,
        mmap_preload=True,
        scratch_dir=tmp_path,
    )
    dm.prepare_data()
    cache_dir = dm._mmap_cache_dir

    result_queue = multiprocessing.Queue()

    def _child(cache_dir, result_queue):
        try:
            from iohub import open_ome_zarr

            with open_ome_zarr(hcs_with_fg_mask, mode="r") as ds:
                positions = [pos for _, pos in ds.positions()]
            arr_shape = positions[0]["0"].shape
            T, C = arr_shape[0], 2
            shape = (len(positions) * T, C, *arr_shape[2:])
            buf = MemoryMappedTensor.from_filename(cache_dir / "data.mmap", dtype=torch.float32, shape=shape)
            result_queue.put(("ok", tuple(buf.shape)))
        except Exception as e:
            result_queue.put(("err", str(e)))

    proc = multiprocessing.Process(target=_child, args=(cache_dir, result_queue))
    proc.start()
    proc.join(timeout=30)
    status, value = result_queue.get_nowait()
    assert status == "ok", f"Child process failed: {value}"
    with open_ome_zarr(hcs_with_fg_mask, mode="r") as ds:
        positions = [pos for _, pos in ds.positions()]
    arr0 = positions[0]["0"]
    expected_n = len(positions) * arr0.frames
    assert value[0] == expected_n


def test_mmap_preload_preserves_native_dtype_and_casts_on_read(tmp_path):
    """mmap_preload stores native uint16 data and casts sampled patches to float32."""
    importorskip("tensordict")
    from tensordict.memmap import MemoryMappedTensor

    dataset_path = tmp_path / "uint16_preload.zarr"
    ch_names = ["Phase", "Fluorescence"]
    rng = np.random.default_rng(7)
    with open_ome_zarr(dataset_path, layout="hcs", mode="w", channel_names=ch_names) as ds:
        for fov in ("0", "1"):
            pos = ds.create_position("A", "1", fov)
            img = rng.integers(0, 4096, size=(1, len(ch_names), 8, 32, 32), dtype=np.uint16)
            pos.create_image("0", img, chunks=(1, 1, 1, 32, 32))

    dm = HCSDataModule(
        data_path=dataset_path,
        source_channel="Phase",
        target_channel="Fluorescence",
        z_window_size=4,
        batch_size=2,
        num_workers=0,
        yx_patch_size=[32, 32],
        split_ratio=0.5,
        mmap_preload=True,
        scratch_dir=tmp_path,
    )
    dm.prepare_data()

    with open_ome_zarr(dataset_path, mode="r") as ds:
        positions = [pos for _, pos in ds.positions()]
        arr0 = positions[0]["0"]
        shape = (len(positions) * arr0.frames, len(ch_names), arr0.slices, arr0.height, arr0.width)
        preload_buf = MemoryMappedTensor.from_filename(
            dm._mmap_cache_dir / "data.mmap",
            dtype=torch.uint16,
            shape=shape,
        )
        assert preload_buf.dtype == torch.uint16

    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch["source"].dtype == torch.float32
    assert batch["target"].dtype == torch.float32


def test_mmap_preload_fg_mask_preserves_native_dtype(hcs_with_fg_mask, tmp_path):
    """fg_mask.mmap preserves native uint8 dtype; sampled masks cast to float32."""
    importorskip("tensordict")
    from tensordict.memmap import MemoryMappedTensor

    dm = HCSDataModule(
        data_path=hcs_with_fg_mask,
        source_channel="Phase",
        target_channel="Fluorescence",
        fg_mask_key="fg_mask",
        z_window_size=4,
        batch_size=2,
        num_workers=0,
        yx_patch_size=[32, 32],
        split_ratio=0.5,
        mmap_preload=True,
        scratch_dir=tmp_path,
    )
    dm.prepare_data()

    with open_ome_zarr(hcs_with_fg_mask, mode="r") as ds:
        positions = [pos for _, pos in ds.positions()]
        mask_arr0 = positions[0]["fg_mask"]
        mask_shape = (len(positions) * mask_arr0.frames, 1, mask_arr0.slices, mask_arr0.height, mask_arr0.width)
        mask_buf = MemoryMappedTensor.from_filename(
            dm._mmap_cache_dir / "fg_mask.mmap",
            dtype=torch.uint8,
            shape=mask_shape,
        )
        assert mask_buf.dtype == torch.uint8

    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch["fg_mask"].dtype == torch.float32


@fixture(scope="function")
def hcs_heterogeneous_t(tmp_path):
    """HCS dataset where positions have different T along axis 0.

    Mirrors the layout of the a549 mantis_v1 pooled stores
    (``SEC61B_all.zarr``, ``TOMM20_all.zarr``) where each FOV is one
    condition's time-lapse and conditions have different frame counts.
    """
    dataset_path = tmp_path / "het_t.zarr"
    ch_names = ["Phase", "Fluorescence"]
    rng = np.random.default_rng(42)
    # Three positions with T=2, T=3, T=4 — heterogeneous.
    fov_t = [("0", 2), ("1", 3), ("2", 4)]
    with open_ome_zarr(dataset_path, layout="hcs", mode="w", channel_names=ch_names) as dataset:
        for fov, t in fov_t:
            pos = dataset.create_position("A", "1", fov)
            img = rng.random((t, len(ch_names), 8, 32, 32)).astype(np.float32)
            pos.create_image("0", img, chunks=(1, 1, 1, 32, 32))
        norm = {ch: {"fov_statistics": {"mean": 0.5, "std": 0.29}} for ch in ch_names}
        for _, pos in dataset.positions():
            pos.zattrs["normalization"] = norm
    return dataset_path, fov_t


def test_mmap_preload_heterogeneous_t(hcs_heterogeneous_t, tmp_path):
    """prepare_data() + setup() + dataloader works when T varies per FOV.

    Regression guard for the bug that hit a549 mantis SEC61B/TOMM20:
    the buffer was sized as ``len(positions) * T_pos0`` and indexed by
    ``i * T``, which crashed at the first FOV with a different T. The
    fix uses a per-FOV cumulative offsets table; this test exercises
    that path end-to-end (write, re-open, slice into per-FOV views,
    sliding-window iteration).
    """
    importorskip("tensordict")
    from tensordict.memmap import MemoryMappedTensor

    data_path, fov_t = hcs_heterogeneous_t
    expected_total_T = sum(t for _, t in fov_t)
    dm = HCSDataModule(
        data_path=data_path,
        source_channel="Phase",
        target_channel="Fluorescence",
        z_window_size=4,
        batch_size=1,
        num_workers=0,
        yx_patch_size=[16, 16],
        split_ratio=2 / 3,  # 2 FOVs train, 1 FOV val
        mmap_preload=True,
        scratch_dir=tmp_path,
    )
    dm.prepare_data()

    # On-disk buffer must be sized at sum(T_i), not n * T_pos0.
    with open_ome_zarr(data_path) as plate:
        positions = [p for _, p in plate.positions()]
    arr0 = positions[0]["0"]
    buf = MemoryMappedTensor.from_filename(
        dm._mmap_cache_dir / "data.mmap",
        dtype=torch.float32,
        shape=(expected_total_T, 2, arr0.slices, arr0.height, arr0.width),
    )
    assert buf.shape[0] == expected_total_T

    # Per-FOV slabs in the buffer must equal each FOV's full-channel oindex read.
    # Use the same helper the production code uses, so test divergence on a
    # later refactor is impossible.
    offsets = HCSDataModule._fov_t_offsets(positions, "0")
    ch_idx = [positions[0].get_channel_index(c) for c in ("Phase", "Fluorescence")]
    for i, pos in enumerate(positions):
        staged = np.asarray(buf[offsets[i] : offsets[i + 1]])
        expected = np.asarray(pos["0"].oindex[:, ch_idx, :])
        assert staged.shape == expected.shape
        np.testing.assert_array_equal(staged, expected)

    # End-to-end: setup + iterate the train loader without crashing.
    dm.setup(stage="fit")
    saw_batch = False
    for batch in dm.train_dataloader():
        assert batch["source"].shape[1] == 1
        assert batch["target"].shape[1] == 1
        saw_batch = True
        break
    assert saw_batch


def test_mmap_preload_matches_oindex(preprocessed_hcs_dataset, tmp_path):
    """Staged buffer is byte-equal to ``arr.oindex[:, ch_idx, :]`` per FOV.

    Regression guard for the prepare_data read pattern: the production loop
    selects channels by per-channel basic indexing + numpy concat instead
    of zarr's ``oindex`` (which materializes O(prod(out_shape)) coordinate
    arrays in CoordinateIndexer). This test asserts the two produce
    identical bytes — runs against both v2 (unsharded) and v3 (sharded)
    via the parameterized fixture.
    """
    importorskip("tensordict")
    from tensordict.memmap import MemoryMappedTensor

    # Pick non-adjacent source/target channel indices so ch_idx is a
    # genuine list (not a contiguous slice) — exactly the case oindex
    # was being used for.
    dm = HCSDataModule(
        data_path=preprocessed_hcs_dataset,
        source_channel="Phase",
        target_channel="GFP",
        z_window_size=4,
        batch_size=1,
        num_workers=0,
        mmap_preload=True,
        scratch_dir=tmp_path,
    )
    dm.prepare_data()

    with open_ome_zarr(preprocessed_hcs_dataset) as plate:
        positions = [p for _, p in plate.positions()]
        ch_idx = [positions[0].get_channel_index(c) for c in ("Phase", "GFP")]
        arr0 = positions[0]["0"]
        T = arr0.frames
        total_shape = (len(positions) * T, len(ch_idx), arr0.slices, arr0.height, arr0.width)
        buf = MemoryMappedTensor.from_filename(
            dm._mmap_cache_dir / "data.mmap",
            dtype=torch.float32,
            shape=total_shape,
        )
        for i, pos in enumerate(positions):
            staged = np.asarray(buf[i * T : (i + 1) * T])
            expected = np.asarray(pos["0"].oindex[:, ch_idx, :])
            assert staged.shape == expected.shape
            np.testing.assert_array_equal(staged, expected)
