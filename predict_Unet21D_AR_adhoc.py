
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
    r"/home/ankitr/MBL-Project/Logs/phase2fluor/lightning_logs/version_2/checkpoints/epoch=49-step=16800.ckpt"
)  # prefix the string with 'r' to avoid the need for escape characters.




# The entire training loop is contained in this cell.
GPU_ID = 0
BATCH_SIZE = 4
YX_PATCH_SIZE = (512, 512)


# Dictionary that specifies key parameters of the model.
phase2fluor_config = {
    # "architecture": "2.5D",
    "num_filters": [24, 48, 96, 192, 384],
    "in_channels": 1,
    "out_channels": 2,
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
    architecture="2.5D",
)

test_data.setup("test")
trainer = VSTrainer(
    accelerator="gpu",
    devices=[GPU_ID],
    logger=CSVLogger(save_dir=save_dir, version=model_version),
)

trainer.test(
    phase2fluor_model,
    datamodule=test_data,
    ckpt_path=ckpt_path,
)



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