import logging
import os
from datetime import datetime

import torch
from jsonargparse import lazy_instance
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

from viscy.data.triplet import TripletDataModule
from viscy.light.engine import ContrastiveModule


class ContrastiveLightningCLI(LightningCLI):
    """Lightning CLI with default logger."""

    def add_arguments_to_parser(self, parser):
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
    logging.getLogger("lightning.pytorch").setLevel(log_level)
    torch.set_float32_matmul_precision("high")
    _ = ContrastiveLightningCLI(
        model_class=ContrastiveModule, datamodule_class=TripletDataModule
    )


if __name__ == "__main__":
    main()
