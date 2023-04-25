# %%
import os
import sys

sys.path.insert(0, "/home/christian.foley/virtual_staining/workspaces/microDL")
import micro_dl.cli.torch_inference_script as inference_script

if __name__ == "__main__":
    config_new_models = (
        "/home/christian.foley/virtual_staining/workspaces/config_files/"
        "inference_config_processed_hek_no_perturb_256_256.yml"
    )

    inference_script.main(config_new_models, gpu=0, gpu_mem_frac=0.1)

# %%
