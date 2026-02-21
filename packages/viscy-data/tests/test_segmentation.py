import pytest
import torch
from iohub import open_ome_zarr

from viscy_data import SegmentationDataModule, SegmentationDataset


def test_segmentation_dataset_length(segmentation_hcs_pair):
    pred_path, target_path = segmentation_hcs_pair
    pred_plate = open_ome_zarr(pred_path)
    target_plate = open_ome_zarr(target_path)
    ds = SegmentationDataset(
        pred_dataset=pred_plate,
        target_dataset=target_plate,
        pred_channel="Pred",
        target_channel="Target",
        pred_z_slice=0,
        target_z_slice=0,
    )
    num_positions = len(list(target_plate.positions()))
    num_timepoints = 2  # default from _build_hcs
    assert len(ds) == num_positions * num_timepoints


def test_segmentation_dataset_getitem(segmentation_hcs_pair):
    pred_path, target_path = segmentation_hcs_pair
    pred_plate = open_ome_zarr(pred_path)
    target_plate = open_ome_zarr(target_path)
    ds = SegmentationDataset(
        pred_dataset=pred_plate,
        target_dataset=target_plate,
        pred_channel="Pred",
        target_channel="Target",
        pred_z_slice=0,
        target_z_slice=0,
    )
    sample = ds[0]
    assert "pred" in sample
    assert "target" in sample
    assert "position_idx" in sample
    assert "time_idx" in sample
    assert sample["pred"].dtype == torch.int16
    assert sample["target"].dtype == torch.int16


def test_segmentation_dataset_z_slice(segmentation_hcs_pair):
    pred_path, target_path = segmentation_hcs_pair
    pred_plate = open_ome_zarr(pred_path)
    target_plate = open_ome_zarr(target_path)
    ds = SegmentationDataset(
        pred_dataset=pred_plate,
        target_dataset=target_plate,
        pred_channel="Pred",
        target_channel="Target",
        pred_z_slice=slice(0, 2),
        target_z_slice=slice(0, 2),
    )
    sample = ds[0]
    assert sample["pred"].shape[0] == 2
    assert sample["target"].shape[0] == 2


def test_segmentation_datamodule_setup_test(segmentation_hcs_pair):
    pred_path, target_path = segmentation_hcs_pair
    dm = SegmentationDataModule(
        pred_dataset=pred_path,
        target_dataset=target_path,
        pred_channel="Pred",
        target_channel="Target",
        pred_z_slice=0,
        target_z_slice=0,
        batch_size=4,
        num_workers=0,
    )
    dm.setup("test")
    batch = next(iter(dm.test_dataloader()))
    assert "pred" in batch
    assert "target" in batch
    assert batch["pred"].shape[0] <= 4


def test_segmentation_unsupported_stage(segmentation_hcs_pair):
    pred_path, target_path = segmentation_hcs_pair
    dm = SegmentationDataModule(
        pred_dataset=pred_path,
        target_dataset=target_path,
        pred_channel="Pred",
        target_channel="Target",
        pred_z_slice=0,
        target_z_slice=0,
        batch_size=4,
        num_workers=0,
    )
    with pytest.raises(NotImplementedError):
        dm.setup("fit")
