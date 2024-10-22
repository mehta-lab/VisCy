# %% Imports and paths
from pathlib import Path

from iohub import open_ome_zarr
from viscy.data.hcs import HCSDataModule
from recOrder.io.utils import yaml_to_model
from recOrder.cli.settings import ReconstructionSettings
from recOrder.cli import apply_inverse_models

from torch.nn.functional import mse_loss

# Viscy classes for the trainer and model
from viscy.translation.engine import VSUNet
from viscy.translation.predict_writer import HCSPredictionWriter
from viscy.trainer import VisCyTrainer
from viscy.transforms import NormalizeSampled
from tempfile import TemporaryDirectory
import numpy as np
import os
import torch


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
        devices=1,
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


def reconstruct(
    array_in: np.ndarray,
    regularization_parameter: float,
    recon_config_path,
    transfer_function_path,
):
    settings = yaml_to_model(recon_config_path, ReconstructionSettings)
    tf_dataset = open_ome_zarr(transfer_function_path, mode="r")

    settings.phase.apply_inverse.regularization_strength = regularization_parameter

    czyx_data = torch.tensor(array_in[0])  # single timepoint for now
    czyx_recon = apply_inverse_models.phase(
        czyx_data, recon_dim=3, settings_phase=settings.phase, transfer_function_dataset=tf_dataset
    )

    return np.array(czyx_recon[None])


# %%
if __name__ == "__main__":

    # input paths
    # input_data_path = "/hpc/projects/comp.micro/virtual_staining/datasets/tmp/20230803_fish2_60x_1_cropped_zyx_resampled_clipped_2.zarr"
    input_data_path = "/hpc/projects/comp.micro/virtual_staining/datasets/training/neuromast/20230801_20230803_datasets/20230803_fish2_60x_concat_resampled_cropped_zyx.zarr"
    model_ckpt_path = "/hpc/projects/comp.micro/virtual_staining/datasets/public/VS_models/VSNeuromast/epoch=44-step=1215.ckpt"
    recon_config_path = "/hpc/projects/comp.micro/zebrafish/20240109_6dpf_she_h2b_gfp_cldnb_mcherry/1-reconstruction/phase.yml"
    transfer_function_path = "/hpc/projects/comp.micro/joint-restoration/neuromast-isim/1-regularization-sweep/tf.zarr"

    # output paths
    base_output_path = Path(
        "/hpc/projects/comp.micro/joint-restoration/neuromast-isim/1-regularization-sweep/"
    )
    output_data_path = base_output_path / "output.zarr"
    results_csv = base_output_path / "output.csv"

    # input misc
    BATCH_SIZE = 12  # Reduce the batch size if encountering out-of-memory errors
    NUM_WORKERS = 16
    PHASE_IDX, NUCLEI_IDX, MEMBRANE_IDX, BRIGHTFIELD_IDX = 0, 1, 2, 3
    regularization_list = np.logspace(-10, 10, 100)

    # load input data
    input_dataset = open_ome_zarr(input_data_path, mode="r")
    # phase_array = input_dataset["0/0/0/0"][:, PHASE_IDX:PHASE_IDX + 1]
    nuclei_array = input_dataset["0/0/0/0"][:, NUCLEI_IDX : NUCLEI_IDX + 1]
    membrane_array = input_dataset["0/0/0/0"][:, MEMBRANE_IDX : MEMBRANE_IDX + 1]
    brightfield_array = input_dataset["0/0/0/0"][
        :, BRIGHTFIELD_IDX : BRIGHTFIELD_IDX + 1
    ]
    Z, Y, X = brightfield_array.shape[-3:]

    # prepare outputs
    output_dataset = open_ome_zarr(
        output_data_path,
        layout="hcs",
        mode="w",
        channel_names=["phase", "membrane", "nuclei"],
    )
    output_position = output_dataset.create_position("0", "0", "0")
    output_position.create_zeros(
        "0", (len(regularization_list), 3, Z, Y, X), dtype="float32"
    )
    with open(results_csv, mode="w") as file:
        file.write("reg,nuclei_loss,membrane_loss\n")

    # main loop
    for i, regularization_parameter in enumerate(regularization_list):
        print("Reconstructing reg = ", regularization_parameter)
        phase_array = reconstruct(
            brightfield_array,
            regularization_parameter,
            recon_config_path,
            transfer_function_path,
        )

        output_array = predict_neuromast(
            phase_array, model_ckpt_path, "Phase3D", BATCH_SIZE, NUM_WORKERS
        )
        vs_membrane_array = output_array[:, 0:1]
        vs_nuclei_array = output_array[:, 1:2]

        mse_nuclei = ((vs_nuclei_array - nuclei_array) ** 2).mean()
        mse_membrane = ((vs_membrane_array - membrane_array) ** 2).mean()

        output_position["0"][i, 0] = phase_array[0, 0]
        output_position["0"][i, 1] = vs_membrane_array[0, 0]
        output_position["0"][i, 2] = vs_nuclei_array[0, 0]

        line = f"{regularization_parameter:.2e},{mse_nuclei:.6e},{mse_membrane:.6e}\n"
        print(line)
        with open(results_csv, mode="a") as file:
            file.write(line)
