# %%
import numpy as np
from viscy.data.hcs import HCSDataModule
from viscy.transforms import NormalizeSampled
from viscy.unet.networks.Unet2D import Unet2d
from viscy.data.hcs import Sample
import lightning.pytorch as pl
import torch

from viscy.light.predict_writer import HCSPredictionWriter
from monai.transforms import DivisiblePad

# %% test the model on the test set
test_datapath = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/Exp_2024_02_13_DENV_3infMarked_test.zarr"

data_module = HCSDataModule(
    test_datapath,
    source_channel=["Sensor", "Phase"],
    target_channel=[],
    split_ratio=0.8,
    z_window_size=1,
    architecture="2D",
    num_workers=1,
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

# Prepare the data
data_module.prepare_data()

data_module.setup(stage="predict")
test_dm = data_module.test_dataloader()
sample = next(iter(test_dm))

# %%
class LightningUNet(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        ckpt_path,
    ):
        super(LightningUNet, self).__init__()
        self.unet_model = Unet2d(in_channels=in_channels, out_channels=out_channels)
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
            state_dict.pop("loss_function.weight", None)  # Remove the unexpected key
            self.load_state_dict(state_dict)  # loading only weights

    def forward(self, x):
        return self.unet_model(x)

    def predict_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0):
        source = self._predict_pad(batch["source"])
        pred_class = self.forward(source)
        pred_int = torch.argmax(pred_class, dim=4, keepdim=True)
        return self._predict_pad.inverse(pred_int)

    def on_predict_start(self):
        """Pad the input shape to be divisible by the downsampling factor.
        The inverse of this transform crops the prediction to original shape.
        """
        down_factor = 2**self.unet_model.num_blocks
        self._predict_pad = DivisiblePad((0, 0, down_factor, down_factor))


# %% create trainer and input

output_path = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/pred/Exp_2024_02_13_DENV_3infMarked_pred.zarr"

trainer = pl.Trainer(
    default_root_dir="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/sensorInf_phenotyping/logs_wPhase",
    callbacks=[HCSPredictionWriter(output_path, write_input=True)],
)
model = LightningUNet(
    in_channels=2,
    out_channels=3,
    ckpt_path="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/sensorInf_phenotyping/logs_wPhase/version_34/checkpoints/epoch=99-step=300.ckpt",
)

trainer.predict(
    model=model,
    datamodule=data_module,
    return_predictions=True,
)

# %% test the model on the test set and write to zarr store