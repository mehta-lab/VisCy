# %% Imports and paths
from pathlib import Path

from iohub import open_ome_zarr
from viscy.data.hcs import HCSDataModule

# Viscy classes for the trainer and model
from viscy.translation.engine import VSUNet
from viscy.translation.predict_writer import HCSPredictionWriter
from viscy.trainer import VisCyTrainer
from viscy.transforms import NormalizeSampled
from tempfile import TemporaryDirectory
import numpy as np
import os


# %%
def predict_neuromast(
    array_in: np.ndarray,
    model_ckpt_path,
    phase_channel_name="Phase3D",
    BATCH_SIZE=5,
    NUM_WORKERS=16,
) -> np.ndarray:
    """
    Predict the membrane and nuclei channels from the input phase channel array.
    Parameters
    ----------
    array_in : np.ndarray
        TCZYX input array with the phase channel
    model_ckpt_path : str
        Path to the model checkpoint
    phase_channel_name : str, optional
        Name of the phase channel, by default 'Phase3D'
    BATCH_SIZE : int, optional
        Batch size for prediction, by default 5
    NUM_WORKERS : int, optional
        Number of workers for data loading and calculating normalization, by default 16

    Returns
    -------
    np.ndarray
        TCZYX array with the predicted membrane and nuclei channels

    """

    # Create a temporary directory to store the input data
    tmp_input_dir = TemporaryDirectory()
    input_store = os.path.join(tmp_input_dir.name, ".zarr")
    tmp_output_dir = TemporaryDirectory()
    output_store = os.path.join(tmp_output_dir.name, "output.zarr")

    print("Tmp input store path", input_store)

    # Creating dummy zarr
    with open_ome_zarr(
        input_store, layout="hcs", mode="a", channel_names=["Phase3D"]
    ) as dataset:
        position = dataset.create_position("0", "0", "0")
        position.create_image("0", array_in)

    # VisCy Traine instantation
    trainer = VisCyTrainer(
        accelerator="gpu",
        callbacks=[HCSPredictionWriter(output_store)],
    )
    # Preprocess the data to get the normalization values
    trainer.preprocess(
        data_path=input_store,
        channel_names=[phase_channel_name],
        num_workers=NUM_WORKERS,
    )

    # Normalization transform
    normalizations = [
        NormalizeSampled(
            [phase_channel_name],
            level="fov_statistics",
            subtrahend="median",
            divisor="iqr",
        )
    ]

    # Setup the data module.
    data_module = HCSDataModule(
        data_path=input_store,
        source_channel=phase_channel_name,
        target_channel=["Membrane", "Nuclei"],
        z_window_size=21,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        architecture="UNeXt2",
        normalizations=normalizations,
    )
    data_module.prepare_data()
    data_module.setup(stage="predict")

    # Setup the model.
    # Dictionary that specifies key parameters of the model.
    config_VSNeuromast = {
        "in_channels": 1,
        "out_channels": 2,
        "in_stack_depth": 21,
        "backbone": "convnextv2_tiny",
        "stem_kernel_size": (7, 4, 4),
        "decoder_mode": "pixelshuffle",
        "head_expansion_ratio": 4,
        "head_pool": True,
    }

    model_VSNeuromast = VSUNet.load_from_checkpoint(
        model_ckpt_path, architecture="UNeXt2", model_config=config_VSNeuromast
    )
    model_VSNeuromast.eval()

    # Start the predictions
    trainer.predict(
        model=model_VSNeuromast,
        datamodule=data_module,
        return_predictions=False,
    )
    output_dataset = open_ome_zarr(output_store, mode="r")
    output_array = output_dataset["0/0/0/0"][:]

    # Cleanup
    tmp_input_dir.cleanup()
    tmp_output_dir.cleanup()

    return output_array


# %%
if __name__ == "__main__":

    input_data_path = "/hpc/projects/comp.micro/virtual_staining/datasets/tmp/20230803_fish2_60x_1_cropped_zyx_resampled_clipped_2.zarr"
    model_ckpt_path = "/hpc/projects/comp.micro/virtual_staining/datasets/public/VS_models/VSNeuromast/epoch=44-step=1215.ckpt"

    # Reduce the batch size if encountering out-of-memory errors
    BATCH_SIZE = 12
    array_in = open_ome_zarr(input_data_path, mode="r")["0/0/0/0"][:, 0:1]
    phase_channel_name = "Phase3D"
    NUM_WORKERS = 16

    output_array = predict_neuromast(
        array_in, model_ckpt_path, phase_channel_name, BATCH_SIZE, NUM_WORKERS
    )
    
    import napari 
    v = napari.Viewer()
    v.add_image(array_in)
    v.add_image(output_array)
    import pdb; pdb.set_trace()