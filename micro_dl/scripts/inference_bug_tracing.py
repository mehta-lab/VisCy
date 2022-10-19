# %%
import argparse
import datetime
import os
import torch
import yaml
import sys

sys.path.insert(0, "/home/christian.foley/virtual_staining/code_revisions/microDL")

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.inference.image_inference as image_inf
import micro_dl.torch_unet.utils.inference as torch_inference_utils
import micro_dl.utils.train_utils as train_utils


def check_save_folder(inference_config, preprocess_config):
    """
    Helper method to ensure that save folder exists.
    If no save folder specified in inference_config, force saving in data
    directory with dynamic name and timestamp.

    :param pd.dataframe inference_config: inference config file (not) containing save_folder_name
    :param pd.dataframe preprocess_config: preprocessing config file containing input_dir
    """

    if "save_folder_name" not in inference_config:
        assert "input_dir" in preprocess_config, (
            "Error in autosaving: 'input_dir'" "unspecified in preprocess config"
        )
        now = (
            str(datetime.datetime.now())
            .replace(" ", "_")
            .replace(":", "_")
            .replace("-", "_")[:-10]
        )
        save_dir = os.path.join(preprocess_config["input_dir"], f"../prediction_{now}")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        preprocess_config["save_folder_name"] = save_dir
        print(
            f"No save folder specified in inference config: automatically saving predictions in : \n\t{save_dir}"
        )


# %%
config = "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2022_09_27_A549_NuclStain/ptTest_Soorya_Christian/torch_config_25D_A549Nucl.yml"
torch_config = aux_utils.read_config(config)

# Get GPU ID and memory fraction
gpu_id, gpu_mem_frac = 0, 0.95
device = torch.device(gpu_id)

# read configuration parameters and metadata
preprocess_config = aux_utils.read_config(torch_config["preprocess_config_path"])
train_config = aux_utils.read_config(torch_config["train_config_path"])
inference_config = aux_utils.read_config(torch_config["inference_config_path"])

network_config = torch_config["model"]

# if no save_folder_name specified, automatically incur saving in data folder
check_save_folder(inference_config, preprocess_config)

# instantiate and prep TorchPredictor interfacing object
torch_predictor = torch_inference_utils.TorchPredictor(
    network_config=network_config, device=device
)
torch_predictor.load_model_torch()

# instantiate ImagePredictor object and run inference
image_predictor = image_inf.ImagePredictor(
    train_config, inference_config, torch_predictor, preprocess_config=preprocess_config
)
image_predictor.run_prediction()
# %%
