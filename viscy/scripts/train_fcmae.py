# %%
from lightning.pytorch.loggers import TensorBoardLogger
from torch import set_float32_matmul_precision

from viscy.data.hcs import HCSDataModule
from viscy.light.engine import FcmaeUNet
from viscy.light.trainer import VSTrainer
from viscy.transforms import (
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandWeightedCropd,
)

# %%
model = FcmaeUNet(
    architecture="fcmae",
    model_config=dict(in_channels=1),
    train_mask_ratio=0.6,
)

# %%
ch = "reconstructed-labelfree"

data = HCSDataModule(
    data_path="/hpc/projects/comp.micro/virtual_staining/datasets/training/raw-and-reconstructed.zarr",
    source_channel=ch,
    target_channel=ch,
    z_window_size=5,
    batch_size=64,
    num_workers=12,
    architecture="3D",
    augmentations=[
        RandWeightedCropd(ch, ch, spatial_size=[-1, 512, 512], num_samples=2),
        RandAffined(
            ch,
            prob=0.5,
            rotate_range=[3.14, 0.0, 0.0],
            shear_range=[0.0, 0.05, 0.05],
            scale_range=[0.2, 0.3, 0.3],
        ),
        RandAdjustContrastd(ch, prob=0.3, gamma=[0.75, 1.5]),
        RandScaleIntensityd(ch, prob=0.3, factors=0.5),
        RandGaussianNoised(ch, prob=0.5, mean=0.0, std=5.0),
        RandGaussianSmoothd(
            ch, prob=0.5, sigma_z=[0.25, 1.5], sigma_y=[0.25, 1.5], sigma_x=[0.25, 1.5]
        ),
    ],
)


# %%
set_float32_matmul_precision("high")

trainer = VSTrainer(
    fast_dev_run=False,
    max_epochs=50,
    logger=TensorBoardLogger(
        save_dir="/hpc/mydata/ziwen.liu/fcmae", version="test_0", log_graph=False
    ),
)
trainer.fit(model, data)

# %%
