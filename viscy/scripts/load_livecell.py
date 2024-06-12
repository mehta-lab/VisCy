# %%
from pathlib import Path

import matplotlib.pyplot as plt
from monai.transforms import (
    CenterSpatialCrop,
    NormalizeIntensity,
    RandAdjustContrast,
    RandAffine,
    RandFlip,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandScaleIntensity,
    RandSpatialCrop,
)
from tqdm import tqdm

from viscy.data.livecell import LiveCellDataModule

# %%
data_path = Path("/hpc/reference/imaging/livecell")

normalize_transform = NormalizeIntensity(channel_wise=True)
crop_transform = CenterSpatialCrop(roi_size=[1, 224, 224])

data = LiveCellDataModule(
    train_val_images=data_path / "images" / "livecell_train_val_images",
    train_annotations=data_path
    / "annotations"
    / "livecell_coco_train_images_only.json",
    val_annotations=data_path / "annotations" / "livecell_coco_val_images_only.json",
    train_transforms=[
        normalize_transform,
        RandSpatialCrop(roi_size=[1, 384, 384]),
        RandAffine(
            rotate_range=[3.14, 0.0, 0.0],
            scale_range=[0.0, [-0.2, 0.8], [-0.2, 0.8]],
            prob=0.8,
            padding_mode="zeros",
        ),
        RandFlip(prob=0.5, spatial_axis=(1, 2)),
        RandAdjustContrast(prob=0.5, gamma=(0.8, 1.2)),
        RandScaleIntensity(factors=0.3, prob=0.5),
        RandGaussianNoise(prob=0.5, mean=0.0, std=0.3),
        RandGaussianSmooth(
            sigma_x=(0.05, 0.3),
            sigma_y=(0.05, 0.3),
            sigma_z=(0.05, 0.0),
            prob=0.5,
        ),
        crop_transform,
    ],
    val_transforms=[normalize_transform, crop_transform],
    batch_size=16,
    num_workers=0,
)

# %%
data.setup("fit")
dmt = data.train_dataloader()
dmv = data.val_dataloader()

# %%
for batch in tqdm(dmt):
    img = batch["target"]
    img[:, :, :, 32:64, 32:64] = 0
    f, ax = plt.subplots(4, 4, figsize=(15, 15))
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
