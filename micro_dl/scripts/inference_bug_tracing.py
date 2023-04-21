#%%
import argparse
import datetime
import os
import torch
import yaml
import sys

sys.path.insert(0, "/home/christian.foley/virtual_staining/workspaces/microDL")

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.inference.image_inference as image_inf
import micro_dl.inference.inference as torch_inference_utils
import micro_dl.utils.train_utils as train_utils

import micro_dl.cli.torch_inference_script as inference_script


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
        now = aux_utils.get_timestamp()
        save_dir = os.path.join(preprocess_config["input_dir"], f"../prediction_{now}")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        preprocess_config["save_folder_name"] = save_dir
        print(
            f"No save folder specified in inference config: automatically saving predictions in : \n\t{save_dir}"
        )


#%%
__name__ == "__main__"
if __name__ == "__main__":
    config_test = (
        "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/"
        "2019_02_15_KidneyTissue_full_dataset/11_02_2022_parameter_tuning/25D_Unet/"
        "z5_ret_actin/"
        "Stack5_fltr16_256_do20_otus_masked_MAE_1chan_ret_actin_pix_iqr_norm_tf10_pt20_torch_config.yml"
    )
    config = (
        "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/"
        "config_files/2022_09_27_A549_NuclStain/ptTest_Soorya_Christian/"
        "torch_config_25D_A549Nucl.yml"
    )
    config_2D = (
        "/hpc/projects/compmicro/projects/virtualstaining/"
        "torch_microDL/config_files/2022_09_27_A549_NuclStain/"
        "ptTest_Soorya_Christian/torch_config_2D_A549Nucl.yml"
    )
    config_recent = (
        "/hpc/projects/CompMicro/projects/"
        "virtualstaining/torch_microDL/config_files/"
        "2022_11_01_VeroMemNuclStain/gunpowder_testing_12_13/"
        "torch_config_Soorya_VeroA549_25D_mem.yml"
    )
    config_soorya_issue190 = (
        "/hpc/projects/CompMicro/projects/"
        "virtualstaining/torch_microDL/config_files/"
        "2022_11_01_VeroMemNuclStain/gunpowder_testing_12_13/"
        "torch_config_Soorya_VeroA549_25D_nucl.yml"
    )

    inference_script.main(config_soorya_issue190, gpu=0, gpu_mem_frac=0.1)

# %%
