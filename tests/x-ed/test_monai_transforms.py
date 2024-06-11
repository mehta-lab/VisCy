# %%
from monai.transforms import (
    Compose,
    Rotate90d,
    EnsureChannelFirstd,
    ScaleIntensityd,
    EnsureTyped,
    Invertd,
)
from iohub import open_ome_zarr
from natsort import natsorted
import glob
from pathlib import Path
from mantis.cli.utils import (
    copy_n_paste_czyx,
    create_empty_hcs_zarr,
    process_single_position_v2,
)
import numpy as np
import torch
import matplotlib.pyplot as plt
from viscy.light.engine import VSUNet
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
# Load dataset
dataset = open_ome_zarr(
    "/hpc/projects/comp.micro/mantis/2024_05_24_A549_Fig1_redo/5-register/a549_tomm20_lysotracker_w2_1_lf_recon_no_decon.zarr/0/FOV0/0"
)
Z, Y, X = dataset.data.shape[-3:]

depth = 5
center_index = 30
crop = 384 * 3
# Normalize phase
norm_meta = dataset.zattrs["normalization"]["Phase3D"]["fov_statistics"]
T, C, Z, Y, X = dataset.data.shape
channel_names = dataset.channel_names
c_idx = channel_names.index("Phase3D")

img = dataset["0"][
    0:1,
    c_idx : c_idx + 1,
    center_index - depth // 2 : center_index + depth // 2 + 1,
    Y // 2 - crop // 2 : Y // 2 + crop // 2,
    X // 2 - crop // 2 : X // 2 + crop // 2,
]
phase = img[:, 0:1]
phase = (phase - norm_meta["median"]) / norm_meta["iqr"]
plt.imshow(phase[0, 0, depth // 2], cmap="gray")

# Load model
model_ckpt = "/hpc/projects/comp.micro/mantis/2023_11_01_OpenCell_infection/5.1-VS_training/lightning_logs/20231130-120039/checkpoints/epoch=145-step=114902.ckpt"
model = VSUNet.load_from_checkpoint(
    model_ckpt,
    architecture="2.2D",
    model_config=dict(
        in_channels=1,
        out_channels=2,
        in_stack_depth=5,
        backbone="convnextv2_tiny",
        pretrained=False,
        stem_kernel_size=[5, 4, 4],
        decoder_mode="pixelshuffle",
        decoder_conv_blocks=2,
        head_pool=True,
        head_expansion_ratio=4,
        drop_path_rate=0.0,
    ),
)
model.eval()

# Create transformations
rotate_90 = Rotate90d(keys=["image"], k=1, spatial_axes=(1, 2))
rotate_180 = Rotate90d(keys=["image"], k=2, spatial_axes=(1, 2))
rotate_270 = Rotate90d(keys=["image"], k=3, spatial_axes=(1, 2))


# Inversion transformations
invert_90 = Invertd(
    keys="pred", orig_keys="phase", transform=rotate_90, nearest_interp=False
)
invert_180 = Invertd(
    keys="pred", orig_keys="phase", transform=rotate_180, nearest_interp=False
)
invert_270 = Invertd(
    keys="pred", orig_keys="phase", transform=rotate_270, nearest_interp=False
)

# %%


def plot_rotated_images(phase_tensor):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original
    axes[0].imshow(phase_tensor[0, depth // 2].cpu().numpy(), cmap="gray")
    axes[0].set_title("Original")

    # 90 degrees
    phase_90 = rotate_90({"image": phase_tensor})["image"]
    axes[1].imshow(phase_90[0, depth // 2].cpu().numpy(), cmap="gray")
    axes[1].set_title("90 Degrees")

    # 180 degrees
    phase_180 = rotate_180({"image": phase_tensor})["image"]
    axes[2].imshow(phase_180[0, depth // 2].cpu().numpy(), cmap="gray")
    axes[2].set_title("180 Degrees")

    # 270 degrees
    phase_270 = rotate_270({"image": phase_tensor})["image"]
    axes[3].imshow(phase_270[0, depth // 2].cpu().numpy(), cmap="gray")
    axes[3].set_title("270 Degrees")

    plt.show()


def predict(phase_tensor, model):
    with torch.inference_mode():
        pred = model(phase_tensor).cpu().numpy()
    return pred


# %%
# Apply test time augmentations
# def apply_tta(phase_tensor, model):

with torch.inference_mode():
    phase_tensor = torch.from_numpy(phase).to(model.device)
    plot_rotated_images(phase_tensor[0])  # Plot rotated images before inference

    phase_tensor = phase_tensor[0]
    predictions = []
    phase_dict = {"image": phase_tensor}
    for rotate in [None, rotate_90, rotate_180, rotate_270]:
        if rotate is not None:
            phase_aug = rotate(phase_dict)["image"]
        else:
            phase_aug = phase_tensor
        phase_aug = phase_aug.unsqueeze(0)
        print(phase_dict["image"].meta["image_transforms"])
        # print(f"Phase shape: {phase_aug.shape}")
        with torch.inference_mode():
            phase_dict["pred"] = model(phase_aug)

            if rotate is not None:
                invert_rotate = Invertd(
                    keys="pred",  # invert the `pred` data field, also support multiple fields
                    transform=rotate,
                    orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
                    # then invert `pred` based on this information. we can use same info
                    # for multiple fields, also support different orig_keys for different fields
                    nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                    # to ensure a smooth output, then execute `AsDiscreted` transform
                    to_tensor=True,  # convert to PyTorch Tensor after inverting
                )
                phase_dict["pred"] = invert_rotate(phase_dict)
            pred = phase_dict["pred"]
            predictions.append(pred)
            # plot pair of images (raw,rotated, unrotated)
            fig, axes = plt.subplots(1, 4, figsize=(10, 5))
            axes[0].imshow(phase_tensor[0, depth // 2].cpu().numpy(), cmap="gray")
            axes[0].set_title("Original")
            axes[1].imshow(phase_aug[0, 0, depth // 2].cpu().numpy(), cmap="gray")
            axes[1].set_title("Rotated")
            axes[2].imshow(pred[0, 0, depth // 2].cpu().numpy(), cmap="gray")
            axes[2].set_title("Prediction")
            plt.show()

    pred_shape = predictions[0].shape
    for pred in predictions:
        assert pred.shape == pred_shape, "Predictions have different shapes!"

    # return predictions


# %%
# # Apply test time augmentations
# def apply_tta(phase_tensor, model):
#     predictions = []
#     for rotate, invert in zip(
#         [None, rotate_90, rotate_180, rotate_270],
#         [None, rotate_270, rotate_180, rotate_90],
#     ):

#         if rotate is not None:
#             phase_aug = rotate({"image": phase_tensor})["image"]
#         else:
#             phase_aug = phase_tensor

#         phase_aug = phase_aug.unsqueeze(0)
#         # print(f"Phase shape: {phase_aug.shape}")
#         with torch.inference_mode():
#             pred = model(phase_aug).cpu().numpy()
#             if invert is not None:
#                 pred = invert({"image": pred[0]})[
#                     "image"
#                 ]  # Reapply rotation for inversion
#                 pred = pred.unsqueeze(0)
#             else:
#                 pred = torch.from_numpy(pred)
#             print(f"Prediction shape: {pred.shape}")
#             predictions.append(pred.numpy())

#         # plot pair of images (raw,rotated, unrotated)
#         fig, axes = plt.subplots(1, 3, figsize=(10, 5))
#         axes[0].imshow(phase_tensor[0, depth // 2].cpu().numpy(), cmap="gray")
#         axes[0].set_title("Original")
#         axes[1].imshow(phase_aug[0, 0, depth // 2].cpu().numpy(), cmap="gray")
#         axes[1].set_title("Rotated")
#         axes[2].imshow(pred[0, 0, depth // 2].cpu().numpy(), cmap="gray")
#         axes[2].set_title("Prediction")

#         plt.show()

#     pred_shape = predictions[0].shape
#     for pred in predictions:
#         assert pred.shape == pred_shape, "Predictions have different shapes!"

#     return predictions


# %%
# Run TTA and visualize the results
with torch.inference_mode():
    phase_tensor = torch.from_numpy(phase).to(model.device)
    plot_rotated_images(phase_tensor[0])  # Plot rotated images before inference
    predictions = apply_tta(phase_tensor[0], model)
    non_altered_pred = predict(phase_tensor, model)
    # %%
    # Show prediction with and without the TTA
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(non_altered_pred[0, 0, depth // 2], cmap="gray")
    axes[0].set_title("Non-Altered Prediction")
    pred_avg = np.mean(predictions, axis=0)
    axes[1].imshow(pred_avg[0, 0, depth // 2], cmap="gray")
    plt.title("Averaged Prediction")
    plt.show()

# %%
import os
import napari

os.environ["DISPLAY"] = ":1005"
viewer = napari.Viewer()
# %%
viewer.add_image(phase, name="Phase", blending="additive")
viewer.add_image(
    non_altered_pred,
    name="Non-Altered Prediction",
    blending="additive",
    colormap="green",
)
viewer.add_image(
    pred_avg, name="Averaged Prediction", blending="additive", colormap="magenta"
)

# %%
