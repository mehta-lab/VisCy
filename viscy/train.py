"""Main training script for VisCy using Hydra configuration.

Based on lightning-hydra-template for production-ready ML workflows.
"""

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from viscy.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train the model using Lightning Trainer.

    This function:
    1. Sets random seed for reproducibility
    2. Instantiates datamodule, model, callbacks, loggers, and trainer
    3. Logs hyperparameters
    4. Runs training and/or testing
    5. Returns metrics for optimization

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        Tuple containing metrics dict and object dict.
    """
    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters...")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training...")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing...")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning(
                "Best checkpoint not found! Using current weights for testing..."
            )
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best checkpoint path:\n{ckpt_path}")

    test_metrics = trainer.callback_metrics

    # Merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    Optional[float]
        Optimized metric value for hyperparameter optimization sweeps.
        Returns None if no optimization metric is specified.
    """
    # Apply extra utilities (warnings, tags, config printing)
    extras(cfg)

    # Train the model
    metric_dict, _ = train(cfg)

    # Return optimized metric for hyperparameter sweeps
    metric_name = cfg.get("optimized_metric")
    if metric_name and metric_name in metric_dict:
        return get_metric_value(metric_dict=metric_dict, metric_name=metric_name)

    return None


if __name__ == "__main__":
    main()
