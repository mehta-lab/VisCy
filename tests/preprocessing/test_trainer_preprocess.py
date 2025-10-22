from pathlib import Path

from iohub import open_ome_zarr
from pytest import mark

from viscy.trainer import VisCyTrainer


@mark.parametrize("default_channels", [True, False])
def test_preprocess(small_hcs_dataset: Path, default_channels: bool):
    data_path = small_hcs_dataset
    if default_channels:
        channel_names = -1
    else:
        with open_ome_zarr(data_path) as dataset:
            channel_names = dataset.channel_names
    trainer = VisCyTrainer(accelerator="cpu")
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
