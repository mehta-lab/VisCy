"""Fail loud when a checkpoint monitor stops carrying signal.

``ModelCheckpoint(monitor=..., save_top_k=k)`` only writes a checkpoint while
``save_top_k`` is unfilled or when the monitored metric improves. A monitor that
goes non-finite or freezes therefore *silently* starves checkpointing: the first
``k`` epochs save, and nothing after that — including ``last.ckpt``, which
Lightning writes on the same cadence. A 200-epoch run can complete with
``Trainer.fit stopped: max_epochs=200 reached``, exit code 0, and no weights
newer than epoch ``k``.

This callback converts that silent waste into an early crash, in keeping with
the repo's prefer-raising philosophy. It observes only ``trainer.callback_metrics``
and never touches checkpoint selection, so adding it cannot change which
checkpoint a run considers "best".
"""

from collections import deque

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

__all__ = ["MonitorHealthCheck", "MonitorHealthError"]


class MonitorHealthError(RuntimeError):
    """Raised when a checkpoint monitor is non-finite or frozen."""


class MonitorHealthCheck(Callback):
    """Raise when the checkpoint monitor goes non-finite or stops varying.

    Parameters
    ----------
    monitor : str
        Metric key to watch, matching the ``monitor`` of the run's monitored
        ``ModelCheckpoint`` (e.g. ``loss/validate``).
    patience : int
        Number of consecutive validation epochs that must share one identical
        value before the run is declared frozen. Must be >= 2.
    check_finite : bool
        Raise as soon as the monitored value is NaN or infinite.

    Raises
    ------
    MonitorHealthError
        When the monitored metric is non-finite, or has been bit-identical for
        ``patience`` consecutive validation epochs.

    Notes
    -----
    Sanity-check validation is skipped: it runs before training and would
    contribute a value from untrained weights.

    A missing monitor key is *not* an error here. Lightning's own
    ``ModelCheckpoint`` already raises for that case, and the metric can be
    legitimately absent on the first epochs of some schedules.
    """

    def __init__(self, monitor: str = "loss/validate", patience: int = 10, check_finite: bool = True) -> None:
        if patience < 2:
            raise ValueError(f"patience must be >= 2 to detect a frozen metric, got {patience}")
        self.monitor = monitor
        self.patience = patience
        self.check_finite = check_finite
        self._recent: deque[float] = deque(maxlen=patience)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Inspect metrics after the module's ``on_validation_epoch_end`` logs them."""
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        scalar = value.item() if isinstance(value, torch.Tensor) else float(value)

        if self.check_finite and not torch.isfinite(torch.tensor(scalar)):
            raise MonitorHealthError(
                f"checkpoint monitor {self.monitor!r} is non-finite ({scalar}) at epoch {trainer.current_epoch}. "
                "ModelCheckpoint cannot rank a non-finite metric, so no further checkpoint will be written "
                "and this run would finish with no usable weights."
            )

        self._recent.append(scalar)
        if len(self._recent) == self.patience and len(set(self._recent)) == 1:
            raise MonitorHealthError(
                f"checkpoint monitor {self.monitor!r} has been identical ({scalar}) for {self.patience} "
                f"consecutive validation epochs, through epoch {trainer.current_epoch}. A metric that does not "
                "respond to the weights starves ModelCheckpoint: save_top_k fills once and nothing is saved "
                "afterwards, so the run would end with no weights newer than the first few epochs. "
                "Check that the validation loss actually depends on the model output."
            )

    def state_dict(self) -> dict:
        """Persist the recent-value window so a resumed run keeps its history."""
        return {"recent": list(self._recent)}

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the recent-value window from a checkpoint."""
        self._recent = deque(state_dict.get("recent", []), maxlen=self.patience)
