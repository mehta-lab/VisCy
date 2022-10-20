#%%
import numpy as np
import time
import torch
import sys

sys.path.insert(0, "/home/christian.foley/virtual_staining/microDL")
import micro_dl.torch_unet.utils.io as io


def main():
    # create test data
    data = np.random.random((5, 5, 5, 128, 128))
    data_tensor = torch.Tensor(data)
    print(data_tensor.shape)

    # run feature logger on test data
    io.log_feature_map(
        data_tensor,
        "/home/christian.foley/virtual_staining/example_log/",
        dim_names=["batch", "channels"],
        spatial_dims=3,
        vmax=None,
    )


if __name__ == "main":
    start = time.time()
    try:
        main()
    except Exception as e:
        print(f"Error after {time.time() - start} seconds:\n{e}")
