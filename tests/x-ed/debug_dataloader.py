# %%
import matplotlib.pyplot as plt

from viscy.data.hcs import HCSDataModule
from augment import get_augmentations
import numpy as np
from viscy.transforms import NormalizeSampled


# %%
def data_kwargs(ch: list[str], noise_std: float):
    return dict(
        source_channel=ch[0],
        target_channel=ch[1],
        z_window_size=7,
        split_ratio=0.8,
        num_workers=0,  # TODO: MP for debugging
        architecture="3D",
        yx_patch_size=[384, 384],
        augmentations=get_augmentations(ch, ch, noise_std=noise_std),
        normalizations=[
            NormalizeSampled(
                [ch[0]], level="fov_statistics", subtrahend="median", divisor="iqr"
            ),
            NormalizeSampled(
                [ch[1]], level="dataset_statistics", subtrahend="mean", divisor="std"
            ),
        ],
    )


data_path1 = "/hpc/projects/comp.micro/zebrafish/20240126_3dpf_she_h2b_cldnb_mcherry/1-reconstruction/fish1_60x_2_multipos_1.zarr"
data_path2 = "/hpc/projects/comp.micro/virtual_staining/datasets/training/raw-and-reconstructed.zarr"
bf_data = HCSDataModule(
    data_path=data_path1,
    batch_size=16,
    **data_kwargs(["Phase3D", "GFP_Density3D"], noise_std=2e-4),
)
# bf_data = HCSDataModule(
#     data_path=data_path2,
#     batch_size=16,
#     **data_kwargs(["raw-labelfree", "reconstructed-nucleus"], noise_std=2e-4),
# )

# %%
bf_data.setup("fit")
dl = bf_data.train_dataloader()

# %%
f, ax = plt.subplots(4, 4, figsize=(8, 8))

for batch in dl:
    for i, a in enumerate(ax.ravel()):
        a.imshow(batch["source"][i, 0, 2], cmap="gray")
    break

for a in ax.flatten():
    a.axis("off")

plt.tight_layout()
plt.show()

# %%
batch["source"].min(), batch["source"].max(), batch["source"].std()
# %%

f, ax = plt.subplots(4, 4, figsize=(8, 8))

for batch in dl:
    for i, a in enumerate(ax.ravel()):
        a.imshow(batch["target"][i, 0, 2], cmap="gray")
    break

for a in ax.flatten():
    a.axis("off")

plt.tight_layout()
plt.show()

batch["target"].min(), batch["target"].max(), batch["target"].std()
# %%
from iohub import open_ome_zarr
from pathlib import Path

data_path1 = Path(data_path1)
with open_ome_zarr(data_path1 / "0" / "0" / "0") as dataset:
    T, C, Z, Y, X = dataset.data.shape
    img = dataset[0][
        0,
        0,
        slice(Z // 2 - 2, Z // 2 + 2),
        slice(Y // 2 - 142, Y // 2 + 142),
        slice(X // 2 - 142, X // 2 + 142),
    ]
    mean = dataset.zattrs["normalization"]["Phase3D"]["fov_statistics"]["mean"]
    std = dataset.zattrs["normalization"]["Phase3D"]["fov_statistics"]["std"]
    median = dataset.zattrs["normalization"]["Phase3D"]["fov_statistics"]["median"]
    iqr = dataset.zattrs["normalization"]["Phase3D"]["fov_statistics"]["iqr"]

    # normalize this fov
    norm_img = (img - mean) / std
    norm_img_median = (img - median) / iqr
    print(mean, std, median, iqr)
    print(norm_img.min(), norm_img.max(), norm_img.std())
    print(norm_img_median.min(), norm_img_median.max(), norm_img_median.std())

# %%
