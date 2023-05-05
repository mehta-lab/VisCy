import warnings
from datetime import datetime
import os
import torch
from jsonargparse import lazy_instance
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger
from numcodecs import blosc

from micro_dl.light.data import HCSDataModule
from micro_dl.light.engine import PhaseToNuc25D


class VSLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # https://pytorch-lightning.readthedocs.io/en/1.6.0/api/pytorch_lightning.utilities.cli.html#pytorch_lightning.utilities.cli.LightningCLI.add_arguments_to_parser
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("trainer.default_root_dir","trainer.logger.init_args.save_dir")
        parser.add_argument("--architecture", type=str, default="2.5D")
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
    # https://zarr.readthedocs.io/en/stable/tutorial.html#configuring-blosc
    blosc.use_threads = False
    torch.set_float32_matmul_precision("high")
    # TODO: remove this after MONAI 1.2 release
    # https://github.com/Project-MONAI/MONAI/pull/6105
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage")
        _ = VSLightningCLI(
            PhaseToNuc25D,
            HCSDataModule,
        )


if __name__ == "__main__":
    main()
