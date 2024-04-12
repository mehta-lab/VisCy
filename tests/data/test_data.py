from pathlib import Path

from iohub import open_ome_zarr
from monai.transforms import RandSpatialCropSamplesd
from pytest import mark

from viscy.data.hcs import HCSDataModule
from viscy.light.trainer import VSTrainer


@mark.parametrize("default_channels", [True, False])
def test_preprocess(small_hcs_dataset: Path, default_channels: bool):
    data_path = small_hcs_dataset
    if default_channels:
        channel_names = -1
    else:
        with open_ome_zarr(data_path) as dataset:
            channel_names = dataset.channel_names
    trainer = VSTrainer(accelerator="cpu")
    trainer.preprocess(data_path, channel_names=channel_names, num_workers=2)
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
        for channel in channel_names:
            assert "dataset_statistics" in dataset.zattrs["normalization"][channel]
        for _, fov in dataset.positions():
            norm_metadata = fov.zattrs["normalization"]
            for channel in channel_names:
                assert channel in norm_metadata
                assert "dataset_statistics" in norm_metadata[channel]
                assert "fov_statistics" in norm_metadata[channel]


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
        architecture="3D",
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


def test_datamodule_setup_predict(preprocessed_hcs_dataset):
    data_path = preprocessed_hcs_dataset
    z_window_size = 5
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
