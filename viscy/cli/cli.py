import logging
import os
from datetime import datetime

import torch
from jsonargparse import lazy_instance
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

from viscy.light.data import HCSDataModule
from viscy.light.engine import VSUNet
from viscy.light.trainer import VSTrainer


class VSLightningCLI(LightningCLI):
    """Extending lightning CLI arguments and defualts."""

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        subcommands = LightningCLI.subcommands()
        subcommands["preprocess"] = {"trainer", "model", "dataloaders", "datamodule"}
        subcommands["export"] = {"model", "dataloaders", "datamodule"}
        return subcommands

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.yx_patch_size", "model.example_input_yx_shape")
        parser.link_arguments("model.architecture", "data.architecture")
        parser.set_defaults(
            {
                "trainer.logger": lazy_instance(
                    TensorBoardLogger,
                    save_dir="",
                    version=datetime.now().strftime(r"%Y%m%d-%H%M%S"),
                    log_graph=True,
                )
            }
        )


def main():
    """Main Lightning CLI entry point."""
    log_level = os.getenv("VISCY_LOG_LEVEL", logging.INFO)
    logging.getLogger("lightning.pytorch").setLevel((log_level))
    torch.set_float32_matmul_precision("high")
    _ = VSLightningCLI(
        model_class=VSUNet,
        datamodule_class=HCSDataModule,
        trainer_class=VSTrainer,
    )


if __name__ == "__main__":
    main()
