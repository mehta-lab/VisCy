"""VisCy Lightning CLI with custom defaults."""

import logging
import os
import sys
from datetime import datetime

import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

from viscy_utils.trainer import VisCyTrainer


class VisCyCLI(LightningCLI):
    """Extending lightning CLI arguments and defaults."""

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """Define custom subcommands."""
        subcommands = LightningCLI.subcommands()
        subcommand_base_args = {"model"}
        subcommands["preprocess"] = subcommand_base_args
        subcommands["export"] = subcommand_base_args
        subcommands["precompute"] = subcommand_base_args
        subcommands["convert_to_anndata"] = subcommand_base_args
        return subcommands

    def add_arguments_to_parser(self, parser) -> None:
        """Set default logger and progress bar."""
        defaults = {
            "trainer.logger": lazy_instance(
                TensorBoardLogger,
                save_dir="",
                version=datetime.now().strftime(r"%Y%m%d-%H%M%S"),
                log_graph=True,
            ),
        }
        if not sys.stdout.isatty():
            defaults["trainer.callbacks"] = [lazy_instance(TQDMProgressBar, refresh_rate=10, leave=True)]
        parser.set_defaults(defaults)

    def _parse_ckpt_path(self) -> None:
        try:
            return super()._parse_ckpt_path()
        except SystemExit:
            # FIXME: https://github.com/Lightning-AI/pytorch-lightning/issues/21255
            return None


def _setup_environment() -> None:
    """Set log level and TF32 precision."""
    log_level = os.getenv("VISCY_LOG_LEVEL", logging.INFO)
    logging.getLogger("lightning.pytorch").setLevel(log_level)
    torch.set_float32_matmul_precision("high")


def main() -> None:
    """Run the Lightning CLI entry point.

    Parse log level and set TF32 precision.
    Set default random seed to 42.
    """
    _setup_environment()
    require_model = {
        "preprocess",
        "precompute",
        "convert_to_anndata",
    }.isdisjoint(sys.argv)
    require_data = {
        "preprocess",
        "precompute",
        "export",
        "convert_to_anndata",
    }.isdisjoint(sys.argv)
    _ = VisCyCLI(
        model_class=LightningModule,
        datamodule_class=LightningDataModule if require_data else None,
        trainer_class=VisCyTrainer,
        seed_everything_default=42,
        subclass_mode_model=require_model,
        subclass_mode_data=require_data,
        parser_kwargs={"description": "Computer vision models for single-cell phenotyping."},
    )


if __name__ == "__main__":
    main()
