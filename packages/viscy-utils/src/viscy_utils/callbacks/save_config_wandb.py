"""Save resolved Lightning config to W&B files."""

from __future__ import annotations

import logging
from pathlib import Path

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.loggers import WandbLogger

logger = logging.getLogger(__name__)


class SaveConfigToWandb(Callback):
    """Upload the resolved config.yaml to W&B so it appears in the Files tab.

    Lightning's SaveConfigCallback writes config.yaml to ``trainer.log_dir``,
    but WandbLogger does not sync arbitrary files from that directory.
    This callback copies it into the W&B run's files directory on fit start.
    """

    def setup(self, trainer: Trainer, pl_module, stage: str) -> None:
        """Copy config.yaml to W&B run files on fit start."""
        if stage != "fit":
            return
        wandb_logger = None
        for lg in trainer.loggers:
            if isinstance(lg, WandbLogger):
                wandb_logger = lg
                break
        if wandb_logger is None:
            return
        config_path = Path(trainer.log_dir) / "config.yaml"
        if not config_path.exists():
            logger.debug("No config.yaml found at %s, skipping W&B upload.", config_path)
            return
        run = wandb_logger.experiment
        run.save(str(config_path), base_path=str(config_path.parent), policy="now")
        logger.info("Uploaded %s to W&B run %s.", config_path, run.id)
