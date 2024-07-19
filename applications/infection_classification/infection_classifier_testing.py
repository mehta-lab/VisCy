# %%
import lightning.pytorch as pl
from applications.infection_classification.classify_infection_2D import (
    SemanticSegUNet2D,
)
from pytorch_lightning.loggers import TensorBoardLogger

from viscy.data.hcs import HCSDataModule
from viscy.transforms import NormalizeSampled

# %% test the model on the test set
test_datapath = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_04_25_BJ5a_DENV_TimeCourse/4-human_annotation/test_data.zarr"

data_module = HCSDataModule(
    data_path=test_datapath,
    source_channel=["TXR_Density3D", "Phase3D"],
    target_channel=["Inf_mask"],
    split_ratio=0.7,
    z_window_size=1,
    architecture="2D",
    num_workers=1,
    batch_size=1,
    normalizations=[
        NormalizeSampled(
            keys=["TXR_Density3D", "Phase3D"],
            level="fov_statistics",
            subtrahend="median",
            divisor="iqr",
        )
    ],
)

data_module.setup(stage="test")

# %% create trainer and input

logger = TensorBoardLogger(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_04_25_BJ5a_DENV_TimeCourse/5-infection_classifier/0-model_training/",
    name="logs",
)

trainer = pl.Trainer(
    logger=logger,
    default_root_dir="/hpc/projects/intracellular_dashboard/viral-sensor/2024_04_25_BJ5a_DENV_TimeCourse/5-infection_classifier/0-model_training/logs",
    log_every_n_steps=1,
    devices=1,  # Set the number of GPUs to use. This avoids run-time exception from distributed training when the node has multiple GPUs
)

model = SemanticSegUNet2D(
    in_channels=2,
    out_channels=3,
    ckpt_path="/hpc/projects/intracellular_dashboard/viral-sensor/2024_04_25_BJ5a_DENV_TimeCourse/5-infection_classifier/0-model_training/logs/checkpoint_epoch=206.ckpt",
)

# %% test the model

trainer.test(model=model, datamodule=data_module)

# %%
