import pytest
from iohub import open_ome_zarr

from viscy.data.hcs import HCSDataModule
from viscy.trainer import VisCyTrainer
from viscy.translation.engine import VSUNet
from viscy.translation.predict_writer import HCSPredictionWriter, _pad_shape


def test_pad_shape():
    assert _pad_shape((2, 3), 3) == (1, 2, 3)
    assert _pad_shape((4, 5), 4) == (1, 1, 4, 5)
    full_shape = tuple(range(1, 6))
    assert _pad_shape(full_shape, 5) == full_shape


@pytest.mark.parametrize("array_key", ["0", "1"])
def test_predict_writer(preprocessed_hcs_dataset, tmp_path, array_key):
    z_window_size = 5
    data_path = preprocessed_hcs_dataset
    channel_split = 2
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
        expected_shape = list(next(dataset.positions())[1][array_key].shape)
        expected_shape[1] = len(channel_names) - channel_split
        expected_shape = tuple(expected_shape)
    dm = HCSDataModule(
        data_path=data_path,
        source_channel=channel_names[:channel_split],
        target_channel=channel_names[channel_split:],
        z_window_size=z_window_size,
        target_2d=bool(z_window_size == 1),
        batch_size=2,
        num_workers=0,
        array_key=array_key,
    )

    model = VSUNet(
        architecture="fcmae",
        model_config=dict(
            in_channels=channel_split,
            out_channels=len(channel_names) - channel_split,
            encoder_blocks=[2, 2, 2, 2],
            dims=[4, 8, 16, 32],
            decoder_conv_blocks=2,
            stem_kernel_size=[z_window_size, 4, 4],
            in_stack_depth=z_window_size,
            pretraining=False,
        ),
    )

    output_path = tmp_path / "predictions.zarr"
    prediction_writer = HCSPredictionWriter(
        output_store=str(output_path), write_input=False
    )

    trainer = VisCyTrainer(
        logger=False,
        callbacks=[prediction_writer],
        fast_dev_run=False,
        default_root_dir=tmp_path,
    )

    trainer.predict(model, datamodule=dm)
    assert output_path.exists()
    with open_ome_zarr(output_path) as result:
        for _, pos in result.positions():
            assert pos[array_key][:].any()
            assert pos[array_key].shape == expected_shape
