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
        subcommand_base_args = {"model"}
        subcommands["preprocess"] = subcommand_base_args
        subcommands["export"] = subcommand_base_args
        subcommands["precompute"] = subcommand_base_args
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


def _setup_environment() -> None:
    """Set log level and TF32 precision."""
    log_level = os.getenv("VISCY_LOG_LEVEL", logging.INFO)
    logging.getLogger("lightning.pytorch").setLevel(log_level)
    torch.set_float32_matmul_precision("high")


def main() -> None:
    """
    Main Lightning CLI entry point.
    Parse log level and set TF32 precision.
    Set default random seed to 42.
    """
    _setup_environment()
    require_model = {"preprocess", "precompute"}.isdisjoint(sys.argv)
    require_data = {"preprocess", "precompute", "export"}.isdisjoint(sys.argv)
    _ = VisCyCLI(
        model_class=LightningModule,
        datamodule_class=LightningDataModule if require_data else None,
        trainer_class=VisCyTrainer,
        seed_everything_default=42,
        subclass_mode_model=require_model,
        subclass_mode_data=require_data,
        parser_kwargs={
            "description": "Computer vision models for single-cell phenotyping."
        },
    )
