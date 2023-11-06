"""Script showing how to parse a config file in python with LightningCLI."""

# %%
from typing import Optional

from viscy.cli.cli import HCSDataModule, VSLightningCLI, VSTrainer, VSUNet


class PlaceholderPredictTrainer(VSTrainer):
    def predict(self, ckpt_path: Optional[str] = None, *args, **kwargs):
        pass


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

dl = dm.predict_dataloader()

# %%
for batch in dl:
    print(batch)
    break
# %%
