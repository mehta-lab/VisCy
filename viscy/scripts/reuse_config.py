"""Script showing how to parse a config file in python with LightningCLI."""

# %%
from typing import Optional

import torch
from matplotlib import pyplot as plt

from viscy.cli.cli import HCSDataModule, VSLightningCLI, VSTrainer, VSUNet


class PlaceholderPredictTrainer(VSTrainer):
    def predict(self, ckpt_path: Optional[str] = None, *args, **kwargs):
        self.ckpt_path = ckpt_path


# %%
stage = "predict"

args = [stage, "--config", "predict.yml"]


# %%
cli = VSLightningCLI(
    model_class=VSUNet,
    trainer_class=PlaceholderPredictTrainer,
    datamodule_class=HCSDataModule,
    save_config_callback=None,
    args=args,
)

# %%
dm = cli.datamodule
dm.setup(stage)
dataloader = dm.predict_dataloader()

model = cli.model
model.load_state_dict(torch.load(cli.trainer.ckpt_path)["state_dict"])
model.on_predict_start()

# %%
for batch in dataloader:
    break

with torch.inference_mode():
    prediction = model.predict_step(batch, 0)  # %%

plt.imshow(prediction[0, 0, 2].cpu().numpy())

# %%
