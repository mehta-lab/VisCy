"""Utilities for instantiating objects from Hydra configs."""

from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from viscy.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    Parameters
    ----------
    callbacks_cfg : DictConfig
        A DictConfig object containing callback configurations.

    Returns
    -------
    List[Callback]
        A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping...")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    Parameters
    ----------
    logger_cfg : DictConfig
        A DictConfig object containing logger configurations.

    Returns
    -------
    List[Logger]
        A list of instantiated loggers.
    """
    loggers: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return loggers

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers
