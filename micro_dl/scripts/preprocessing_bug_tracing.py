# %%
import argparse
import yaml
import os
import sys
import torch

sys.path.insert(0, "/home/christian.foley/virtual_staining/code_revisions/microDL")

from micro_dl.cli.preprocess_script import *

# %%
if __name__ == "__main__":
    config = "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2022_09_27_A549_NuclStain/ptTest_Soorya_Christian/torch_config_25D_A549Nucl.yml"
    config_2D = "/hpc/projects/compmicro/projects/virtualstaining/torch_microDL/config_files/2022_09_27_A549_NuclStain/10_12_2022_15_22/torch_config_2D.yml"
    torch_config = aux_utils.read_config(config_2D)
    preprocess_config = aux_utils.read_config(torch_config["preprocess_config_path"])
    preprocess_config, runtime = pre_process(preprocess_config)
    # type(test)
    # save_config(preprocess_config, runtime)

# %%
