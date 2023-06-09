import warnings
from datetime import datetime

import torch
from jsonargparse import lazy_instance
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

from viscy.light.data import HCSDataModule
from viscy.light.engine import PhaseToNuc25D, VSTrainer


class VSLightningCLI(LightningCLI):
    """Extending lightning CLI arguments and defualts."""

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        subcommands = LightningCLI.subcommands()
        subcommands["export"] = {"model", "dataloaders", "datamodule"}
        return subcommands

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments(
            "data.yx_patch_size", "model.example_input_yx_shape"
        )
        parser.link_arguments(
            "trainer.default_root_dir", "trainer.logger.init_args.save_dir"
        )
        parser.link_arguments(
            "model.model_config.architecture", "data.architecture"
        )
        parser.set_defaults(
            {
                "trainer.logger": lazy_instance(
                    TensorBoardLogger,
                    save_dir="",
                    version=datetime.now().strftime(r"%Y%m%d-%H%M%S"),
                    log_graph=True,
                ),
                "trainer.callbacks": [
                    {
                        "class_path": "viscy.light.prediction_writer.HCSPredictionWriter",
                    }
                ],
            }
        )


def main():
    torch.set_float32_matmul_precision("high")
    # TODO: remove this after MONAI 1.2 release
    # https://github.com/Project-MONAI/MONAI/pull/6105
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="TypedStorage"
        )
        _ = VSLightningCLI(
            model_class=PhaseToNuc25D,
            datamodule_class=HCSDataModule,
            trainer_class=VSTrainer,
        )
