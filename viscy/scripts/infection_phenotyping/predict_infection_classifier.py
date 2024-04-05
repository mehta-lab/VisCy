# %%

from viscy.light.predict_writer import HCSPredictionWriter
from viscy.data.hcs import HCSDataModule
import lightning.pytorch as pl
from viscy.scripts.infection_phenotyping.classify_infection import SemanticSegUNet2D
from viscy.transforms import NormalizeSampled

# %% # %% write the predictions to a zarr file

pred_datapath = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/Exp_2024_02_13_DENV_3infMarked_test.zarr"

data_module = HCSDataModule(
    data_path=pred_datapath,
    source_channel=['Sensor','Phase'],
    target_channel=['Inf_mask'],
    split_ratio=0.8,
    z_window_size=1,
    architecture="2D",
    num_workers=1,
    batch_size=1,
    normalizations=[
        NormalizeSampled(
            keys=["Phase", "Sensor"],
            level="fov_statistics",
            subtrahend="median",
            divisor="iqr",
        )
    ],
)
data_module.prepare_data()
data_module.setup(stage="predict")

model = SemanticSegUNet2D(
    in_channels=2,
    out_channels=3,
    ckpt_path="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/sensorInf_phenotyping/logs_wPhase/version_34/checkpoints/epoch=99-step=300.ckpt",
)

# %% perform prediction

output_path = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/pred/Exp_2024_02_13_DENV_3infMarked_pred_SP.zarr"

trainer = pl.Trainer(
    default_root_dir="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/sensorInf_phenotyping/logs_wPhase",
    callbacks=[HCSPredictionWriter(output_path, write_input=False)],
    devices=1,  # Set the number of GPUs to use. This avoids run-time exception from distributed training when the node has multiple GPUs
)

trainer.predict(
    model=model,
    datamodule=data_module,
    return_predictions=True,
)

# %%
