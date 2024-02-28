# %%
from pathlib import Path

import matplotlib.pyplot as plt
from monai.transforms import (
    CenterSpatialCropd,
    NormalizeIntensityd,
    RandAffined,
    RandScaleIntensityd,
)
from tqdm import tqdm

from viscy.data.ctmc_v1 import CTMCv1DataModule

# %%
data_path = Path("")

normalize_transform = NormalizeIntensityd(keys=["DIC"], channel_wise=True)
crop_transform = CenterSpatialCropd(keys=["DIC"], roi_size=[1, 256, 256])

data = CTMCv1DataModule(
    train_data_path=data_path / "CTMCV1_test.zarr",
    val_data_path=data_path / "CTMCV1_train.zarr",
    train_transforms=[
        normalize_transform,
        RandAffined(
            keys=["DIC"],
            rotate_range=[3.14, 0.0, 0.0],
            shear_range=[0.0, 0.3, 0.3],
            scale_range=[0.0, 0.3, 0.3],
            prob=0.8,
        ),
        RandScaleIntensityd(keys=["DIC"], factors=0.3, prob=0.5),
        crop_transform,
    ],
    val_transforms=[normalize_transform, crop_transform],
    batch_size=4,
    num_workers=0,
    channel_name="DIC",
)

# %%
data.setup("fit")
dmt = data.train_dataloader()
dmv = data.val_dataloader()

# %%
for batch in tqdm(dmt):
    img = batch["source"]
    f, ax = plt.subplots(4, 4, figsize=(12, 12))
    for sample, a in zip(img, ax.flatten()):
        a.imshow(sample[0, 0].cpu().numpy(), cmap="gray", vmin=-5, vmax=5)
        a.axis("off")
    f.tight_layout()
    break

# %%
for batch in tqdm(dmv):
    img = batch["source"]
    f, ax = plt.subplots(4, 4, figsize=(12, 12))
    for sample, a in zip(img, ax.flatten()):
        a.imshow(sample[0, 0].cpu().numpy(), cmap="gray", vmin=-5, vmax=5)
        a.axis("off")
    f.tight_layout()
    break


# %%
