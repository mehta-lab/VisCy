
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchview
import torchvision
from iohub import open_ome_zarr
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger

# pytorch lightning wrapper for Tensorboard.
from tensorboard import notebook  # for viewing tensorboard in notebook
from torch.utils.tensorboard import SummaryWriter  # for logging to tensorboard

# HCSDataModule makes it easy to load data during training.
from viscy.light.data import HCSDataModule

# Trainer class and UNet.
from viscy.light.engine import VSTrainer, VSUNet

log_dir = Path("/home/ankitr/MBL-Project/Logs")


test_data_path = Path(
    "/home/ankitr/MBL-Project/Data/cropped_dataset_v3.zarr"
).expanduser()
model_version = "phase2fluor"
save_dir = Path(log_dir, "test")
ckpt_path = Path(
    r"/home/ankitr/MBL-Project/Logs/phase2fluor/lightning_logs/version_8/checkpoints/epoch=49-step=22400.ckpt"
)  # prefix the string with 'r' to avoid the need for escape characters.


# The entire training loop is contained in this cell.
GPU_ID = 0
BATCH_SIZE = 1
YX_PATCH_SIZE = (512, 512)

# Apply reflection padding
def i_pad(test_data, img, x_start, x_end, y_start, y_end):
    extra_patch = test_data.yx_patch_size[0] // 2

    x_start = x_start - extra_patch
    x_end = x_end + extra_patch
    y_start = y_start - extra_patch
    y_end = y_end + extra_patch

    pad_x_start = 0 if x_start >= 0 else - x_start
    pad_x_end = 0 if x_end <= img.shape[-1] else (x_end - img.shape[-1])
    pad_y_start = 0 if y_start >= 0 else - y_start
    pad_y_end = 0 if y_end <= img.shape[-2] else (y_end - img.shape[-2])

    x_start = 0 if x_start < 0 else x_start
    x_end = x_end if x_end <= img.shape[-1] else img.shape[-1]
    y_start = 0 if y_start < 0 else y_start
    y_end = y_end if y_end <= img.shape[-2] else img.shape[-2]

    crop_img = img[:, :, y_start:y_end, x_start:x_end]

    mirror = torch.nn.ReflectionPad2d((pad_x_start, pad_x_end, pad_y_start, pad_y_end))

    padded_img = mirror(crop_img)
    # print(padded_img.shape)

    return padded_img

# A funky way to create patches for the prediction stage
def funky_patches(test_data):

    x_dim = test_data.predict_dataset[0]['source'].shape[-1] // test_data.yx_patch_size[0]
    y_dim = test_data.predict_dataset[0]['source'].shape[-2] // test_data.yx_patch_size[1]

    patches = []
    patch_idx = []

    for index in range(len(test_data.predict_dataset)//10):
        print(f"{index}\t{test_data.predict_dataset[index]['index']}")
        print(f"{index}\t{test_data.predict_dataset[index]['source'].shape}")

        img = test_data.predict_dataset[index]['source']

        for x_cut in range(1, x_dim+1):
            for y_cut in range(1, y_dim+1):
                patch_idx.append(test_data.predict_dataset[index]['index'] + (x_cut, y_cut))
                # print(patch_idx)
                x_start = (x_cut-1) * test_data.yx_patch_size[0]
                x_end = (x_cut) * test_data.yx_patch_size[0]
                y_start = (y_cut-1) * test_data.yx_patch_size[1]
                y_end = (y_cut) * test_data.yx_patch_size[1]

                # print(x_cut, y_cut)
                # print(x_start, x_end, y_start, y_end)
                patches.append(test_data.i_pad(img, x_start, x_end, y_start, y_end))

    print("Funky patches have been generated")

    return patches, patch_idx


# Dictionary that specifies key parameters of the model.
phase2fluor_config = {
    # "architecture": "21D_AR",
    "num_filters": [24, 48, 96, 192, 384],
    "in_channels": 1,
    "out_channels": 2,
    "in_stack_depth": 5,
    "residual": True,
    "dropout": 0.1,  # dropout randomly turns off weights to avoid overfitting of the model to data.
    "task": "reg",  # reg = regression task.
}

phase2fluor_model = VSUNet(
    "2.5D",
    model_config=phase2fluor_config.copy(),
    # batch_size=BATCH_SIZE,
    loss_function=torch.nn.functional.l1_loss,
    schedule="WarmupCosine",
    log_num_samples=10,  # Number of samples from each batch to log to tensorboard.
    example_input_yx_shape=YX_PATCH_SIZE,
)

test_data = HCSDataModule(
    test_data_path, 
    source_channel="Phase3D",
    target_channel=["GFP EX488 EM525-45", "mCherry EX561 EM600-37"],
    z_window_size=5,
    batch_size=BATCH_SIZE,
    num_workers=8,
    architecture="21D_AR",
    yx_patch_size=(512,512),
    normalize_source=True,
    tiled_prediction=True,
)

test_data.setup("predict")
trainer = VSTrainer(
    accelerator="gpu",
    devices=[GPU_ID],
    logger=CSVLogger(save_dir=save_dir, version=model_version),
)

print(test_data)

patches, patch_idx = funky_patches(test_data)


def get_center_crop(img):
    start = img.shape[-1]//4
    end = (img.shape[-1]*3)//4

    return img[:,:,:,start:end,start:end]

predicted_imgs = []

for i in patches:
    with torch.no_grad():
        i = i[None]
        pred_img = phase2fluor_model(i).cpu().numpy()
        pred_img = get_center_crop(pred_img)
        predicted_imgs.append(pred_img)
        print(len(predicted_imgs), pred_img.shape)

# trainer.test(
#     phase2fluor_model,
#     datamodule=test_data,
#     ckpt_path=ckpt_path,
# )



# # read metrics and plot
# metrics = pd.read_csv(Path(save_dir, "lightning_logs", model_version, "metrics.csv"))
# metrics.boxplot(
#     column=[
#         "test_metrics/r2_step",
#         "test_metrics/pearson_step",
#         "test_metrics/SSIM_step",
#     ],
#     rot=30,
# )