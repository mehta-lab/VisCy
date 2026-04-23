"""Shared optimizer/scheduler factories for VisCy LightningModules."""

import torch
from lightning.pytorch import LightningModule
from monai.optimizers import WarmupCosineSchedule
from torch import nn
from torch.optim.lr_scheduler import ConstantLR


def configure_adamw_scheduler(
    module: LightningModule,
    model: nn.Module,
    lr: float,
    schedule: str,
    warmup_steps: int = 3,
    warmup_multiplier: float = 1e-3,
) -> tuple[list, list]:
    """Build an AdamW optimizer with a WarmupCosine or Constant LR schedule.

    Parameters
    ----------
    module : LightningModule
        The LightningModule whose ``trainer`` supplies
        ``estimated_stepping_batches`` (WarmupCosine) or ``max_epochs``
        (Constant).
    model : nn.Module
        The network whose parameters are optimized.
    lr : float
        Learning rate.
    schedule : {"WarmupCosine", "Constant"}
        Learning rate scheduler.
    warmup_steps : int, optional
        WarmupCosine only: number of steps to linearly ramp the LR from
        ``lr * warmup_multiplier`` up to ``lr``. Ignored for Constant.
    warmup_multiplier : float, optional
        WarmupCosine only: initial LR multiplier at step 0 (final LR at
        ``warmup_steps`` is ``lr``). Ignored for Constant.

    Returns
    -------
    tuple[list, list]
        ``([optimizer], [scheduler_config])`` as expected by
        ``LightningModule.configure_optimizers``.

    Raises
    ------
    ValueError
        If ``schedule`` is not ``"WarmupCosine"`` or ``"Constant"``.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if schedule == "WarmupCosine":
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=warmup_steps,
            t_total=module.trainer.estimated_stepping_batches,
            warmup_multiplier=warmup_multiplier,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    if schedule == "Constant":
        scheduler = ConstantLR(optimizer, factor=1, total_iters=module.trainer.max_epochs)
        return [optimizer], [scheduler]
    raise ValueError(f"Unknown schedule {schedule!r}, expected 'WarmupCosine' or 'Constant'")
