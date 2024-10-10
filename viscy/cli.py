import logging
import os
import sys
from datetime import datetime

import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

from viscy.trainer import VisCyTrainer


class VisCyCLI(LightningCLI):
    """Extending lightning CLI arguments and defualts."""

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        subcommands = LightningCLI.subcommands()
        subcommands["preprocess"] = {"model", "dataloaders", "datamodule"}
        subcommands["export"] = {"model", "dataloaders", "datamodule"}
        return subcommands

    def add_arguments_to_parser(self, parser) -> None:
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


def setup_environment() -> None:
    """
    Set log level and TF32 precision.
    """
    log_level = os.getenv("VISCY_LOG_LEVEL", logging.INFO)
    logging.getLogger("lightning.pytorch").setLevel(log_level)
    torch.set_float32_matmul_precision("high")


def run_cli(
    cli_class: type[LightningCLI],
    model_class: type[LightningModule],
    datamodule_class: type[LightningDataModule],
    trainer_class: type[VisCyTrainer],
    **cli_kwargs,
) -> None:
    """
    Main Lightning CLI entry point.
    Parse log level and set TF32 precision.

    Parameters
    ----------
    cli_class : type[LightningCLI]
        Lightning CLI class
    model_class : type[LightningModule]
        Lightning module class. Not used in preprocess mode.
    datamodule_class : type[LightningDataModule]
        Lightning datamodule class
    trainer_class : type[VisCyTrainer]
        Lightning trainer class
    """
    setup_environment()
    seed = True
    if "preprocess" in sys.argv:
        seed = False
        model_class = LightningModule
        datamodule_class = None
    _ = cli_class(
        model_class=model_class,
        datamodule_class=datamodule_class,
        trainer_class=trainer_class,
        seed_everything_default=seed,
        **cli_kwargs,
    )


def main() -> None:
    """Main CLI with subclass mode enabled."""
    run_cli(
        cli_class=VisCyCLI,
        model_class=LightningModule,
        datamodule_class=LightningDataModule,
        trainer_class=VisCyTrainer,
        subclass_mode_data=True,
        subclass_mode_model=True,
    )
