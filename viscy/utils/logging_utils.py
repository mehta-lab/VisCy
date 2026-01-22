"""Utilities for logging hyperparameters."""

from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf

from viscy.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Log hyperparameters to all loggers.

    This function extracts hyperparameters from config and logs them to all
    available loggers.

    Parameters
    ----------
    object_dict : dict
        Dictionary containing:
        - "cfg": DictConfig with configuration
        - "model": LightningModule
        - "trainer": Trainer
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # Save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # Send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
