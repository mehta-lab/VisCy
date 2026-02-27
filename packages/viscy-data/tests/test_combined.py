"""Tests for CombinedDataModule and ConcatDataModule."""

import pytest
from iohub import open_ome_zarr

from viscy_data import CombinedDataModule, CombineMode, ConcatDataModule, HCSDataModule


def _make_dm(data_path, batch_size=4, num_workers=0):
    with open_ome_zarr(data_path) as dataset:
        ch = dataset.channel_names
    return HCSDataModule(
        data_path=data_path,
        source_channel=ch[:2],
        target_channel=ch[2:],
        z_window_size=5,
        batch_size=batch_size,
        num_workers=num_workers,
        split_ratio=0.5,
        yx_patch_size=(128, 96),
    )


def test_combined_datamodule_fit(preprocessed_hcs_dataset):
    """CombinedDataModule sets up and iterates in fit stage."""
    dm1 = _make_dm(preprocessed_hcs_dataset)
    dm2 = _make_dm(preprocessed_hcs_dataset)
    combined = CombinedDataModule(
        data_modules=[dm1, dm2],
        train_mode=CombineMode.MAX_SIZE_CYCLE,
    )
    combined.setup(stage="fit")
    train_dl = combined.train_dataloader()
    for batch_data in train_dl:
        # CombinedLoader with max_size_cycle returns (batch_list, batch_idx, dataloader_idx)
        batch_list = batch_data[0]
        assert isinstance(batch_list, (list, tuple))
        assert len(batch_list) == 2
        break

    val_dl = combined.val_dataloader()
    for batch_data in val_dl:
        # CombinedLoader with sequential mode returns (batch, batch_idx, dataloader_idx)
        batch = batch_data[0]
        assert isinstance(batch, dict)
        assert "source" in batch
        break


def test_combined_datamodule_combine_modes():
    """CombineMode enum maps to valid string values."""
    assert CombineMode.MIN_SIZE.value == "min_size"
    assert CombineMode.MAX_SIZE_CYCLE.value == "max_size_cycle"
    assert CombineMode.MAX_SIZE.value == "max_size"
    assert CombineMode.SEQUENTIAL.value == "sequential"


def test_concat_datamodule_fit(preprocessed_hcs_dataset):
    """ConcatDataModule concatenates datasets and produces correct batch shapes."""
    dm1 = _make_dm(preprocessed_hcs_dataset)
    dm2 = _make_dm(preprocessed_hcs_dataset)
    concat = ConcatDataModule(data_modules=[dm1, dm2])
    concat.setup(stage="fit")

    # Concatenated dataset should have combined length
    assert len(concat.train_dataset) == len(dm1.train_dataset) + len(dm2.train_dataset)
    assert len(concat.val_dataset) == len(dm1.val_dataset) + len(dm2.val_dataset)

    for batch in concat.train_dataloader():
        assert "source" in batch
        assert batch["source"].shape[1] == 2  # 2 source channels
        break

    for batch in concat.val_dataloader():
        assert "source" in batch
        break


def test_concat_datamodule_inconsistent_batch_size(preprocessed_hcs_dataset):
    """ConcatDataModule raises on inconsistent batch sizes."""
    dm1 = _make_dm(preprocessed_hcs_dataset, batch_size=4)
    dm2 = _make_dm(preprocessed_hcs_dataset, batch_size=8)
    with pytest.raises(ValueError, match="Inconsistent batch size"):
        ConcatDataModule(data_modules=[dm1, dm2])


def test_concat_datamodule_inconsistent_num_workers(preprocessed_hcs_dataset):
    """ConcatDataModule raises on inconsistent num_workers."""
    dm1 = _make_dm(preprocessed_hcs_dataset, num_workers=0)
    dm2 = _make_dm(preprocessed_hcs_dataset, num_workers=2)
    with pytest.raises(ValueError, match="Inconsistent number of workers"):
        ConcatDataModule(data_modules=[dm1, dm2])


def test_concat_datamodule_only_fit_supported(preprocessed_hcs_dataset):
    """ConcatDataModule raises NotImplementedError for non-fit stages."""
    dm1 = _make_dm(preprocessed_hcs_dataset)
    dm2 = _make_dm(preprocessed_hcs_dataset)
    concat = ConcatDataModule(data_modules=[dm1, dm2])
    with pytest.raises(NotImplementedError):
        concat.setup(stage="predict")
