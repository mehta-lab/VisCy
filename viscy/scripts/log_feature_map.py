# %%
import time

import numpy as np
import torch

from viscy.utils import cli_utils


def main():
    # create test data
    data = np.random.random((5, 5, 5, 128, 128))
    data_tensor = torch.Tensor(data)
    print(data_tensor.shape)

    # run feature logger on test data
    cli_utils.log_feature_map(
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
