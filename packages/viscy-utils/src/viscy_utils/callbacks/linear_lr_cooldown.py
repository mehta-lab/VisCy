"""Linear learning-rate cooldown over a fixed range of optimizer steps."""

from lightning.pytorch import Callback, LightningModule, Trainer
from torch.optim.lr_scheduler import ConstantLR


class LinearLRCooldown(Callback):
    """Ramp every param group's LR linearly from its initial value to a final fraction.

    Before each training batch, sets ``lr = initial_lr * factor`` with
    ``factor`` equal to 1 before ``start_step``, decreasing linearly to
    ``final_factor`` at ``end_step``, and ``final_factor`` afterwards. The
    factor depends only on ``trainer.global_step``, so the callback holds no
    state and a resumed fit continues the ramp where the checkpoint left it.

    Only valid on top of a constant schedule: the module must have exactly one
    optimizer whose scheduler is ``ConstantLR`` with ``factor=1`` (the
    ``"Constant"`` schedule of
    :func:`viscy_utils.optimizers.configure_adamw_scheduler`), which never
    changes the LR. Any other scheduler would overwrite or compound the
    cooldown. List this callback before ``LearningRateMonitor`` so the logged
    LR is the applied one.

    Parameters
    ----------
    start_step : int
        Global step at which the cooldown begins (factor 1 at this step).
    end_step : int
        Global step at which the factor reaches ``final_factor``.
    final_factor : float, default 0.0
        LR multiplier at and after ``end_step``, in ``[0, 1]``.
    """

    def __init__(self, start_step: int, end_step: int, final_factor: float = 0.0) -> None:
        if not 0 <= start_step < end_step:
            raise ValueError(f"Need 0 <= start_step < end_step, got start_step={start_step}, end_step={end_step}")
        if not 0.0 <= final_factor <= 1.0:
            raise ValueError(f"final_factor must be in [0, 1], got {final_factor}")
        self.start_step = start_step
        self.end_step = end_step
        self.final_factor = final_factor

    def factor(self, step: int) -> float:
        """Return the LR multiplier for a global step.

        Parameters
        ----------
        step : int
            Global optimizer step.

        Returns
        -------
        float
            LR multiplier.
        """
        if step <= self.start_step:
            return 1.0
        if step >= self.end_step:
            return self.final_factor
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        return 1.0 - (1.0 - self.final_factor) * progress

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Check that the LR is otherwise constant, so the cooldown is the only LR change."""
        if len(trainer.optimizers) != 1:
            raise ValueError(f"LinearLRCooldown needs exactly one optimizer, got {len(trainer.optimizers)}")
        configs = trainer.lr_scheduler_configs
        if len(configs) != 1:
            raise ValueError(f"LinearLRCooldown needs exactly one ConstantLR scheduler, got {len(configs)} schedulers")
        scheduler = configs[0].scheduler
        if not isinstance(scheduler, ConstantLR) or scheduler.factor != 1:
            raise ValueError(
                "LinearLRCooldown needs the constant schedule (ConstantLR with factor=1, schedule='Constant'); "
                f"got {type(scheduler).__name__}, which would overwrite or compound the cooldown"
            )

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int) -> None:
        """Set each param group's LR for the step about to run."""
        factor = self.factor(trainer.global_step)
        for group in trainer.optimizers[0].param_groups:
            if "initial_lr" not in group:
                raise KeyError(
                    "LinearLRCooldown needs param_group['initial_lr'], which torch LR schedulers set; "
                    "the optimizer has no scheduler attached"
                )
            group["lr"] = group["initial_lr"] * factor
