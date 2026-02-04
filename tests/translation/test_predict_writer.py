import numpy as np
import pytest
import torch
from iohub import open_ome_zarr

from viscy.data.hcs import HCSDataModule
from viscy.trainer import VisCyTrainer
from viscy.translation.engine import AugmentedPredictionVSUNet, VSUNet
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


def test_blend_in_consistency():
    """Verify _blend_in produces identical results for torch and numpy inputs."""
    from viscy.translation.engine import _blend_in

    depth = 5
    shape_4d = (2, depth, 8, 8)  # C, Z, Y, X (numpy from HCSPredictionWriter)

    np.random.seed(42)
    old_np = np.random.rand(*shape_4d).astype(np.float32)
    new_np = np.random.rand(*shape_4d).astype(np.float32)
    old_torch = torch.from_numpy(old_np).unsqueeze(0)  # Add batch dim
    new_torch = torch.from_numpy(new_np).unsqueeze(0)

    z_slice = slice(2, 2 + depth)

    result_np = _blend_in(old_np, new_np, z_slice)
    result_torch = _blend_in(old_torch, new_torch, z_slice)

    np.testing.assert_allclose(
        result_np, result_torch.squeeze(0).numpy(), rtol=1e-5, atol=1e-5
    )


def test_predict_sliding_windows_output_shape(preprocessed_hcs_dataset):
    """Verify predict_sliding_windows produces correct output shape."""
    z_window_size = 5
    data_path = preprocessed_hcs_dataset
    channel_split = 2

    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
        first_pos = next(dataset.positions())[1]
        source_data = first_pos["0"][0:1, :channel_split]
        source_tensor = torch.from_numpy(source_data).float()

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

    vs = AugmentedPredictionVSUNet(model=model.model)

    with torch.inference_mode():
        output = vs.predict_sliding_windows(
            source_tensor,
            out_channel=len(channel_names) - channel_split,
            step=1,
        )

    expected_shape = (1, len(channel_names) - channel_split, *source_tensor.shape[2:])
    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, got {output.shape}"
    )
