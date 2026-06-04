"""Smoke tests for CellDivisionTripletDataset and CellDivisionTripletDataModule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pytest import TempPathFactory, fixture, raises

from viscy_data import CellDivisionTripletDataModule, CellDivisionTripletDataset


@fixture(scope="function")
def npy_data_dir(tmp_path_factory: TempPathFactory) -> Path:
    """Create a temporary directory with several .npy cell-division track files.

    Each file has shape (T, C, Y, X) with 2 channels (bf, h2b) like the real data.
    """
    data_dir = tmp_path_factory.mktemp("cell_div_npy")
    rng = np.random.default_rng(42)
    for i in range(6):
        # Vary timepoints so filtering logic is exercised
        t = rng.integers(3, 8)
        arr = rng.random((t, 2, 32, 32)).astype(np.float32)
        np.save(data_dir / f"track_{i}.npy", arr)
    return data_dir


# -- Dataset-level tests --


def test_dataset_getitem_fit(npy_data_dir):
    """__getitem__ returns anchor/positive/negative in fit mode."""
    npy_files = sorted(npy_data_dir.glob("*.npy"))
    ds = CellDivisionTripletDataset(
        data_paths=npy_files,
        channel_names=["bf", "h2b"],
        fit=True,
        return_negative=True,
        time_interval="any",
    )
    assert len(ds) > 0
    sample = ds[0]
    assert "anchor" in sample
    assert "positive" in sample
    assert "negative" in sample
    # Default 3D output: (C, 1, Y, X)
    assert sample["anchor"].ndim == 4
    assert sample["anchor"].shape[0] == 2  # two channels
    assert sample["anchor"].shape[1] == 1  # depth dim


def test_dataset_getitem_predict(npy_data_dir):
    """In predict mode (fit=False) only anchor and index are returned."""
    npy_files = sorted(npy_data_dir.glob("*.npy"))
    ds = CellDivisionTripletDataset(
        data_paths=npy_files,
        channel_names=["bf"],
        fit=False,
        time_interval=1,
    )
    assert len(ds) > 0
    sample = ds[0]
    assert "anchor" in sample
    assert "index" in sample
    assert "positive" not in sample


def test_dataset_output_2d(npy_data_dir):
    """output_2d=True produces (C, Y, X) tensors."""
    npy_files = sorted(npy_data_dir.glob("*.npy"))
    ds = CellDivisionTripletDataset(
        data_paths=npy_files,
        channel_names=["bf"],
        fit=True,
        return_negative=True,
        time_interval="any",
        output_2d=True,
    )
    sample = ds[0]
    assert sample["anchor"].ndim == 3  # (C, Y, X)


def test_dataset_time_interval_filter(npy_data_dir):
    """Non-'any' time_interval should filter valid anchors."""
    npy_files = sorted(npy_data_dir.glob("*.npy"))
    ds_any = CellDivisionTripletDataset(
        data_paths=npy_files,
        channel_names=["bf"],
        fit=True,
        time_interval="any",
    )
    ds_int = CellDivisionTripletDataset(
        data_paths=npy_files,
        channel_names=["bf"],
        fit=True,
        time_interval=2,
    )
    # With a concrete interval, fewer anchors should be valid
    assert len(ds_int) < len(ds_any)


def test_dataset_no_return_negative(npy_data_dir):
    """return_negative=False omits the negative key."""
    npy_files = sorted(npy_data_dir.glob("*.npy"))
    ds = CellDivisionTripletDataset(
        data_paths=npy_files,
        channel_names=["bf"],
        fit=True,
        return_negative=False,
        time_interval="any",
    )
    sample = ds[0]
    assert "positive" in sample
    assert "negative" not in sample


def test_dataset_channel_mapping(npy_data_dir):
    """Channel name aliases map to correct indices."""
    npy_files = sorted(npy_data_dir.glob("*.npy"))
    ds = CellDivisionTripletDataset(
        data_paths=npy_files,
        channel_names=["h2b"],
        fit=False,
        time_interval="any",
    )
    # Should select only channel index 1
    assert ds.channel_indices == [1]
    sample = ds[0]
    assert sample["anchor"].shape[0] == 1  # single channel


def test_dataset_invalid_channel_raises(npy_data_dir):
    """Unknown channel name that is not an integer should raise ValueError."""
    npy_files = sorted(npy_data_dir.glob("*.npy"))
    with raises(ValueError, match="not found in CHANNEL_MAPPING"):
        CellDivisionTripletDataset(
            data_paths=npy_files,
            channel_names=["nonexistent_channel"],
            fit=False,
        )


# -- DataModule-level tests --


def test_datamodule_setup_fit(npy_data_dir):
    """DataModule.setup('fit') creates train and val datasets."""
    dm = CellDivisionTripletDataModule(
        data_path=str(npy_data_dir),
        source_channel=["bf", "h2b"],
        split_ratio=0.5,
        batch_size=2,
        num_workers=0,
        time_interval="any",
        return_negative=True,
    )
    dm.setup(stage="fit")
    assert hasattr(dm, "train_dataset")
    assert hasattr(dm, "val_dataset")
    assert len(dm.train_dataset) > 0
    assert len(dm.val_dataset) > 0
    # Verify datasets are CellDivisionTripletDataset instances
    assert isinstance(dm.train_dataset, CellDivisionTripletDataset)
    assert isinstance(dm.val_dataset, CellDivisionTripletDataset)


def test_datamodule_setup_predict(npy_data_dir):
    """DataModule.setup('predict') creates predict dataset from all files."""
    dm = CellDivisionTripletDataModule(
        data_path=str(npy_data_dir),
        source_channel=["bf"],
        batch_size=2,
        num_workers=0,
    )
    dm.setup(stage="predict")
    assert hasattr(dm, "predict_dataset")
    assert len(dm.predict_dataset) > 0


def test_datamodule_setup_test_raises(npy_data_dir):
    """DataModule.setup('test') raises NotImplementedError."""
    dm = CellDivisionTripletDataModule(
        data_path=str(npy_data_dir),
        source_channel=["bf"],
        batch_size=2,
        num_workers=0,
    )
    with raises(NotImplementedError):
        dm.setup(stage="test")


def test_datamodule_empty_dir_raises(tmp_path_factory: TempPathFactory):
    """Constructing DataModule with no npy files should raise ValueError."""
    empty_dir = tmp_path_factory.mktemp("empty")
    with raises(ValueError, match="No .npy files found"):
        CellDivisionTripletDataModule(
            data_path=str(empty_dir),
            source_channel=["bf"],
            batch_size=2,
            num_workers=0,
        )
