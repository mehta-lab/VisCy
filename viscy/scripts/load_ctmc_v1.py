# %%
from pathlib import Path

import matplotlib.pyplot as plt
from monai.transforms import (
    CenterSpatialCropd,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
)
from tqdm import tqdm

from viscy.data.ctmc_v1 import CTMCv1DataModule

# %%
channel = "DIC"
data_path = Path("/hpc/reference/imaging/ctmc")

normalize_transform = NormalizeIntensityd(keys=[channel], channel_wise=True)
crop_transform = CenterSpatialCropd(keys=[channel], roi_size=[1, 224, 224])

data = CTMCv1DataModule(
    train_data_path=data_path / "CTMCV1_test.zarr",
    val_data_path=data_path / "CTMCV1_train.zarr",
    train_transforms=[
        normalize_transform,
        RandAffined(
            keys=[channel],
            rotate_range=[3.14, 0.0, 0.0],
            scale_range=[0.0, [-0.6, 0.1], [-0.6, 0.1]],
            prob=0.8,
            padding_mode="zeros",
        ),
        RandFlipd(keys=[channel], prob=0.5, spatial_axis=(1, 2)),
        RandAdjustContrastd(keys=[channel], prob=0.5, gamma=(0.8, 1.2)),
        RandScaleIntensityd(keys=[channel], factors=0.3, prob=0.5),
        RandGaussianNoised(keys=[channel], prob=0.5, mean=0.0, std=0.2),
        RandGaussianSmoothd(
            keys=[channel],
            sigma_x=(0.05, 0.3),
            sigma_y=(0.05, 0.3),
            sigma_z=(0.05, 0.0),
            prob=0.5,
        ),
        crop_transform,
    ],
    val_transforms=[normalize_transform, crop_transform],
    batch_size=32,
    num_workers=0,
    channel_name=channel,
)

# %%
data.setup("fit")
dmt = data.train_dataloader()
dmv = data.val_dataloader()

# %%
for batch in tqdm(dmt):
    img = batch["source"]
    img[:, :, :, 32:64, 32:64] = 0
    f, ax = plt.subplots(5, 5, figsize=(15, 15))
    for sample, a in zip(img, ax.flatten()):
        a.imshow(sample[0, 0].cpu().numpy(), cmap="gray", vmin=-5, vmax=5)
        a.axis("off")
    f.tight_layout()
    break

# %%
for batch in tqdm(dmv):
    img = batch["source"]
    f, ax = plt.subplots(5, 5, figsize=(15, 15))
    for sample, a in zip(img, ax.flatten()):
        a.imshow(sample[0, 0].cpu().numpy(), cmap="gray", vmin=-5, vmax=5)
        a.axis("off")
    f.tight_layout()
    break


# %%
