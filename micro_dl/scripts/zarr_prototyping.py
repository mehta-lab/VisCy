#%%
import sys

sys.path.insert(0, "/home/christian.foley/virtual_staining/workspaces/microDL")

import zarr
import numpy as np
import zarr.core as core
import os
from pathlib import Path
import matplotlib.pyplot as plt

import micro_dl.utils.io_utils as io_utils
from micro_dl.preprocessing.generate_masks import MaskProcessor
from micro_dl.utils.meta_utils import generate_normalization_metadata

# zarr_dir = (
#     "/home/christian.foley/virtual_staining/data_visualization/"
#     "A549PhaseFLDeconvolution_63X_pos0.zarr"
# )
# zarr_dir = (
#     "/hpc/projects/CompMicro/projects/virtualstaining/"
#     "torch_microDL/data/2022_03_31_GOLGA2_nuc_mem_LF_63x_04NA_HEK/"
#     "test_no_pertubation.zarr"MiRaLd
# )
# zarr_dir = "/hpc/projects/CompMicro/rawdata/hummingbird/Janie/2022_03_15_orgs_nuc_mem_63x_04NA/all_21_3.zarr"
# zarr_dir = "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/data/2022_11_01_VeroMemNuclStain/output.zarr"
# zarr_dir = "/home/christian.foley/virtual_staining/data_visualization/A549PhaseFLDeconvolution_63X_pos0.zarr"
zarr_dir = "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/data/2022_11_01_VeroMemNuclStain/output.zarr"

reader = io_utils.ZarrReader(zarrfile=zarr_dir)
modifier = io_utils.HCSZarrModifier(zarr_file=zarr_dir, enable_creation=True)

# %%
print("\nCalculating normalization statistics:")
generate_normalization_metadata(zarr_dir=zarr_dir, channel_ids=-1, num_workers=1)
modifier.get_position_meta(0)

# %%
print("\nGenerating masks and calculating foreground fraction:")
mask_generator = MaskProcessor(
    zarr_dir=zarr_dir,
    channel_ids=[1, 2],
    time_ids=-1,
    pos_ids=-1,
    num_workers=4,
    mask_type="otsu",
    output_channel_index=4,
)
mask_generator.generate_masks()
print(modifier.get_position_meta(0))
print(modifier.channel_names)
# %%
print("\nGenerating masks and calculating foreground fraction:")
mask_generator = MaskProcessor(
    zarr_dir=zarr_dir,
    channel_ids=[1],
    time_ids=-1,
    pos_ids=-1,
    num_workers=4,
    mask_type="unimodal",
    output_channel_index=3,
)
mask_generator.generate_masks()
print(modifier.get_position_meta(0))
print(modifier.channel_names)

#%%
import micro_dl.cli.preprocess_script as preprocess_script

preprocess_config = {
    "zarr_dir": "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/data/2022_11_01_VeroMemNuclStain/output.zarr",
    "preprocessing": {
        "normalize": {
            "num_workers": 4,
            "channel_ids": -1,
            "block_size": 32,
            "scheme": "fov",
        },
        "masks": {
            "channel_ids": -1,
            "time_ids": -1,
            "slice_ids": -1,
            "num_workers": 4,
            "thresholding_type": "unimodal",
            "output_channel": None,
            "structure_element_radius": 5,
        },
    },
}
preprocess_script.pre_process(preprocess_config=preprocess_config)

# %%
