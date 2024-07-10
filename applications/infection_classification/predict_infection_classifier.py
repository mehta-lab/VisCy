# %%

import lightning.pytorch as pl
from applications.infection_classification.classify_infection_2D import (
    SemanticSegUNet2D,
)

from viscy.data.hcs import HCSDataModule
from viscy.light.predict_writer import HCSPredictionWriter
from viscy.transforms import NormalizeSampled

# %% # %% write the predictions to a zarr file

pred_datapath = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/A549_63X/2024_02_04_A549_DENV_ZIKV_timelapse/0-train_test_data/2024_02_04_A549_DENV_ZIKV_timelapse_test_2D.zarr"

data_module = HCSDataModule(
    data_path=pred_datapath,
    source_channel=["RFP", "Phase3D"],
    target_channel=["Inf_mask"],
    split_ratio=0.7,
    z_window_size=1,
    architecture="2D",
    num_workers=1,
    batch_size=1,
    normalizations=[
        NormalizeSampled(
            keys=["RFP", "Phase3D"],
            level="fov_statistics",
            subtrahend="median",
            divisor="iqr",
        )
    ],
)

data_module.setup(stage="predict")

model = SemanticSegUNet2D(
    in_channels=2,
    out_channels=3,
    ckpt_path="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/A549_63X/2024_02_04_A549_DENV_ZIKV_timelapse/1-model_train/logs/version_0/checkpoints/epoch=199-step=800.ckpt",
)

# %% perform prediction

output_path = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/A549_63X/2024_02_04_A549_DENV_ZIKV_timelapse/2-predict_infection/2024_02_04_A549_DENV_ZIKV_timelapse_pred_2D_new.zarr"

trainer = pl.Trainer(
    default_root_dir="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/A549_63X/2024_02_04_A549_DENV_ZIKV_timelapse/1-model_train/logs",
    callbacks=[HCSPredictionWriter(output_path, write_input=False)],
    devices=1,  # Set the number of GPUs to use. This avoids run-time exception from distributed training when the node has multiple GPUs
)

trainer.predict(
    model=model,
    datamodule=data_module,
    return_predictions=False,
)

# %%
