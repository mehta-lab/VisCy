"""Smoke tests for MmappedDataset and MmappedDataModule."""

from __future__ import annotations

import pytest

td = pytest.importorskip("tensordict", reason="tensordict is required for mmap tests")

import torch  # noqa: E402
from iohub import open_ome_zarr  # noqa: E402
from pytest import raises  # noqa: E402
from tensordict.memmap import MemoryMappedTensor  # noqa: E402
from torch.multiprocessing import Manager  # noqa: E402

from viscy_data import MmappedDataModule, MmappedDataset  # noqa: E402


def _make_buffer(positions, channels, array_key="0"):
    """Create a MemoryMappedTensor buffer matching the dataset shape."""
    total_frames = sum(p[array_key].frames for p in positions)
    # Shape: (total_frames, num_channels, *spatial_dims)
    spatial = positions[0][array_key].shape[2:]
    shape = (total_frames, len(channels), *spatial)
    return MemoryMappedTensor.empty(shape, dtype=torch.float32)


def test_mmapped_dataset_length(preprocessed_hcs_dataset):
    """Dataset length equals total timepoints across all positions."""
    with open_ome_zarr(preprocessed_hcs_dataset) as plate:
        positions = [p for _, p in plate.positions()]
        channel_names = plate.channel_names[:2]
        expected_len = sum(p["0"].frames for p in positions)

    cache_map = Manager().dict()
    buffer = _make_buffer(positions, channel_names)
    dataset = MmappedDataset(
        positions=positions,
        channel_names=channel_names,
        cache_map=cache_map,
        buffer=buffer,
    )
    assert len(dataset) == expected_len


def test_mmapped_dataset_getitem(preprocessed_hcs_dataset):
    """Dataset __getitem__ returns a list of dicts with per-channel tensors."""
    with open_ome_zarr(preprocessed_hcs_dataset) as plate:
        positions = [p for _, p in plate.positions()]
        channel_names = plate.channel_names[:2]

    cache_map = Manager().dict()
    buffer = _make_buffer(positions, channel_names)
    dataset = MmappedDataset(
        positions=positions,
        channel_names=channel_names,
        cache_map=cache_map,
        buffer=buffer,
    )
    sample = dataset[0]
    assert isinstance(sample, list)
    sample_dict = sample[0]
    for ch in channel_names:
        assert ch in sample_dict
        assert isinstance(sample_dict[ch], torch.Tensor)


def test_mmapped_dataset_caching(preprocessed_hcs_dataset):
    """After first access, cache_map entry should be truthy."""
    with open_ome_zarr(preprocessed_hcs_dataset) as plate:
        positions = [p for _, p in plate.positions()]
        channel_names = plate.channel_names[:1]

    cache_map = Manager().dict()
    buffer = _make_buffer(positions, channel_names)
    dataset = MmappedDataset(
        positions=positions,
        channel_names=channel_names,
        cache_map=cache_map,
        buffer=buffer,
    )
    # Before access
    assert cache_map[0] is None
    _ = dataset[0]
    # After access, should be marked as cached
    assert cache_map[0] is not None


def test_mmapped_datamodule_setup_fit(preprocessed_hcs_dataset):
    """MmappedDataModule.setup('fit') creates train and val datasets."""
    with open_ome_zarr(preprocessed_hcs_dataset) as plate:
        channel_names = plate.channel_names[:2]

    dm = MmappedDataModule(
        data_path=preprocessed_hcs_dataset,
        channels=channel_names,
        batch_size=2,
        num_workers=0,
        split_ratio=0.8,
        preprocess_transforms=[],
        train_cpu_transforms=[],
        val_cpu_transforms=[],
        train_gpu_transforms=[],
        val_gpu_transforms=[],
        pin_memory=False,
    )
    dm.setup(stage="fit")
    assert hasattr(dm, "train_dataset")
    assert hasattr(dm, "val_dataset")
    assert len(dm.train_dataset) > 0
    assert len(dm.val_dataset) > 0
    assert isinstance(dm.train_dataset, MmappedDataset)


def test_mmapped_datamodule_unsupported_stage(preprocessed_hcs_dataset):
    """MmappedDataModule.setup with unsupported stage raises NotImplementedError."""
    with open_ome_zarr(preprocessed_hcs_dataset) as plate:
        channel_names = plate.channel_names[:2]

    dm = MmappedDataModule(
        data_path=preprocessed_hcs_dataset,
        channels=channel_names,
        batch_size=2,
        num_workers=0,
        split_ratio=0.8,
        preprocess_transforms=[],
        train_cpu_transforms=[],
        val_cpu_transforms=[],
        train_gpu_transforms=[],
        val_gpu_transforms=[],
        pin_memory=False,
    )
    with raises(NotImplementedError):
        dm.setup(stage="predict")
