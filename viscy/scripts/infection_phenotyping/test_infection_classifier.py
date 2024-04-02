# %%
from viscy.data.hcs import HCSDataModule
import lightning.pytorch as pl
from viscy.light.predict_writer import HCSPredictionWriter
from viscy.scripts.infection_phenotyping.classify_infection import SemanticSegUNet2D
from pytorch_lightning.loggers import TensorBoardLogger
from viscy.transforms import NormalizeSampled

# %% test the model on the test set
test_datapath = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/Exp_2024_02_13_DENV_3infMarked_test.zarr"

data_module = HCSDataModule(
    data_path=test_datapath,
    source_channel=['Sensor','Phase'],
    target_channel=['Inf_mask'],
    split_ratio=0.8,
    z_window_size=1,
    architecture="2D",
    num_workers=0,
    batch_size=1,
    normalizations=[
        NormalizeSampled(
            keys=["Sensor", "Phase"],
            level="fov_statistics",
            subtrahend="median",
            divisor="iqr",
        )
    ],
)

data_module.setup(stage="test")

# %% create trainer and input

logger = TensorBoardLogger(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/sensorInf_phenotyping/",
    name="logs_wPhase",
)

trainer = pl.Trainer(
    logger=logger,
    default_root_dir="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/sensorInf_phenotyping/logs_wPhase",
    log_every_n_steps=1,
    devices=1,  # Set the number of GPUs to use. This avoids run-time exception from distributed training when the node has multiple GPUs
)

model = SemanticSegUNet2D(
    in_channels=2,
    out_channels=3,
    ckpt_path="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/sensorInf_phenotyping/logs_wPhase/version_74/checkpoints/epoch=99-step=300.ckpt",
)

trainer.test(model=model, datamodule=data_module)

# %% predict the test set

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
