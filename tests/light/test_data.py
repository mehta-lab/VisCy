from pathlib import Path

import torch
from iohub import open_ome_zarr
from pytest import mark

from viscy.light.data import HCSDataModule
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


def test_datamodule_predict_scales(preprocessed_hcs_dataset):
    data_path = preprocessed_hcs_dataset
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names

    def get_normalized_stack(predict_scale_source):
        factor = 1 if predict_scale_source is None else predict_scale_source
        dm = HCSDataModule(
            data_path=data_path,
            source_channel=channel_names[:2],
            target_channel=channel_names[2:],
            z_window_size=5,
            batch_size=2,
            num_workers=0,
            predict_scale_source=predict_scale_source,
            normalize_source=True,
        )
        dm.setup(stage="predict")
        return dm.predict_dataset[0]["source"] / factor

    assert torch.allclose(get_normalized_stack(None), get_normalized_stack(2))
