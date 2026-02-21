import pytest
from iohub import open_ome_zarr

from viscy_data import ClassificationDataModule, ClassificationDataset


def test_classification_dataset_getitem(classification_hcs_dataset):
    dataset_path, ann_path = classification_hcs_dataset
    import pandas as pd

    plate = open_ome_zarr(dataset_path)
    annotation = pd.read_csv(ann_path)
    ds = ClassificationDataset(
        plate=plate,
        annotation=annotation,
        channel_name="Phase",
        z_range=(0, 4),
        transform=None,
        initial_yx_patch_size=(16, 16),
    )
    assert len(ds) > 0
    img, label = ds[0]
    assert img.ndim == 4  # 1, Z, Y, X
    assert img.shape[0] == 1
    assert img.shape[1] == 4  # z_range
    assert label.shape == (1,)


def test_classification_dataset_with_indices(classification_hcs_dataset):
    dataset_path, ann_path = classification_hcs_dataset
    import pandas as pd

    plate = open_ome_zarr(dataset_path)
    annotation = pd.read_csv(ann_path)
    ds = ClassificationDataset(
        plate=plate,
        annotation=annotation,
        channel_name="Phase",
        z_range=(0, 4),
        transform=None,
        initial_yx_patch_size=(16, 16),
        return_indices=True,
    )
    result = ds[0]
    assert len(result) == 3
    img, label, indices = result
    assert isinstance(indices, dict)


def test_classification_datamodule_setup_fit(classification_hcs_dataset):
    dataset_path, ann_path = classification_hcs_dataset
    with open_ome_zarr(dataset_path) as plate:
        all_fovs = [name for name, _ in plate.positions()]
    val_fovs = all_fovs[:4]
    dm = ClassificationDataModule(
        image_path=dataset_path,
        annotation_path=ann_path,
        val_fovs=val_fovs,
        channel_name="Phase",
        z_range=(0, 4),
        train_exclude_timepoints=[],
        train_transforms=None,
        val_transforms=None,
        initial_yx_patch_size=(16, 16),
        batch_size=4,
        num_workers=0,
    )
    dm.setup("fit")
    assert len(dm.train_dataset) > 0
    assert len(dm.val_dataset) > 0


def test_classification_datamodule_setup_predict(classification_hcs_dataset):
    dataset_path, ann_path = classification_hcs_dataset
    with open_ome_zarr(dataset_path) as plate:
        all_fovs = [name for name, _ in plate.positions()]
    dm = ClassificationDataModule(
        image_path=dataset_path,
        annotation_path=ann_path,
        val_fovs=all_fovs[:4],
        channel_name="Phase",
        z_range=(0, 4),
        train_exclude_timepoints=[],
        train_transforms=None,
        val_transforms=None,
        initial_yx_patch_size=(16, 16),
        batch_size=4,
        num_workers=0,
    )
    dm.setup("predict")
    assert len(dm.predict_dataset) > 0


def test_classification_datamodule_exclude_timepoints(classification_hcs_dataset):
    dataset_path, ann_path = classification_hcs_dataset
    with open_ome_zarr(dataset_path) as plate:
        all_fovs = [name for name, _ in plate.positions()]
    val_fovs = all_fovs[:4]
    # First without exclusion
    dm1 = ClassificationDataModule(
        image_path=dataset_path,
        annotation_path=ann_path,
        val_fovs=val_fovs,
        channel_name="Phase",
        z_range=(0, 4),
        train_exclude_timepoints=[],
        train_transforms=None,
        val_transforms=None,
        initial_yx_patch_size=(16, 16),
        batch_size=4,
        num_workers=0,
    )
    dm1.setup("fit")
    full_len = len(dm1.train_dataset)
    # Now exclude timepoint 0
    dm2 = ClassificationDataModule(
        image_path=dataset_path,
        annotation_path=ann_path,
        val_fovs=val_fovs,
        channel_name="Phase",
        z_range=(0, 4),
        train_exclude_timepoints=[0],
        train_transforms=None,
        val_transforms=None,
        initial_yx_patch_size=(16, 16),
        batch_size=4,
        num_workers=0,
    )
    dm2.setup("fit")
    assert len(dm2.train_dataset) < full_len


def test_classification_unsupported_stage(classification_hcs_dataset):
    dataset_path, ann_path = classification_hcs_dataset
    with open_ome_zarr(dataset_path) as plate:
        all_fovs = [name for name, _ in plate.positions()]
    dm = ClassificationDataModule(
        image_path=dataset_path,
        annotation_path=ann_path,
        val_fovs=all_fovs[:4],
        channel_name="Phase",
        z_range=(0, 4),
        train_exclude_timepoints=[],
        train_transforms=None,
        val_transforms=None,
        initial_yx_patch_size=(16, 16),
        batch_size=4,
        num_workers=0,
    )
    with pytest.raises(ValueError, match="Unknown stage"):
        dm.setup("unknown")
