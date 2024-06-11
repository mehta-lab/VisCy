from enum import Enum
from typing import Sequence
from pytorch_lightning import LightningDataModule

from augment import get_augmentations
import numpy as np
from viscy.transforms import NormalizeSampled

from viscy.data.hcs import HCSDataModule
from viscy.data.combined import CombinedDataModule


class CombineMode(Enum):
    MIN_SIZE = "min_size"
    MAX_SIZE_CYCLE = "max_size_cycle"
    MAX_SIZE = "max_size"
    SEQUENTIAL = "sequential"


def data_kwargs(ch: list[str], noise_std: float):
    return dict(
        source_channel=ch,
        target_channel=ch,
        z_window_size=5,
        split_ratio=0.8,
        num_workers=16,
        architecture="3D",
        yx_patch_size=[384, 384],
        augmentations=get_augmentations(ch, ch, noise_std=noise_std),
    )


data_kwargs_phase = data_kwargs(["Phase3D"], noise_std=3.0)
data_kwargs_phase["normalizations"] = [
    NormalizeSampled("Phase3D", "dataset_statistics", "median", "iqr")
]
neuromast_data = HCSDataModule(
    data_path="/hpc/projects/comp.micro/virtual_staining/datasets/training/neuromast/202401_mix_datasets_training_cropped_clipped.zarr",
    batch_size=8,
    **data_kwargs_phase,
)

data_kwargs_aics_bf = data_kwargs(["Brightfield"], noise_std=3.0)
data_kwargs_aics_bf["normalizations"] = [
    NormalizeSampled("Brightfield", "fov_statistics", "mean", "std")
]
aics_bf_data = HCSDataModule(
    data_path="/hpc/projects/comp.micro/virtual_staining/datasets/training/aics-hipsc_sequential/4_1_fluor.zarr",
    batch_size=8,
    **data_kwargs_aics_bf,
)

combined_module = CombinedDataModule(
    [neuromast_data, aics_bf_data],
    train_mode="max_size_cycle",
)
# Accessing the enum value as a string for comparison or print statements
print(combined_module.val_mode.value)  # This will print "sequential"

# Comparing with a string
if combined_module.val_mode.value == "sequential":
    print("Validation mode is sequential.")
