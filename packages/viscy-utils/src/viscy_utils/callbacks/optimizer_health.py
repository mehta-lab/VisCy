"""Fail loud when the optimizer stops making progress at the step level.

Two failure modes hide from every loss-based monitor, and one of them cost 24
GPU-days of FCMAE-2D fits that ran to ``epoch 199/200`` with the weights still at
their initialization:

**Under ``16-mixed``, a NaN *gradient* is silent.** ``GradScaler`` inspects the
gradients, skips the optimizer step when any is non-finite, and halves its scale.
The forward pass — and therefore the logged loss — stays perfectly finite. From
``2**16`` it takes ~166 halvings to reach exactly ``0.0``, after which *every*
step is skipped forever: the weights never move again, and the run still exits 0.
The trigger there was a ``sqrt`` at 0 (infinite derivative) inside an SSIM
Cauchy-Schwarz bound, but any NaN-producing backward does the same thing.

**Under ``bf16-mixed`` there is no scaler at all**, so a NaN gradient instead
lands in the weights and the loss goes non-finite on the next forward. That is
loud in a log, but nothing *stops* the run, so it burns the rest of its wall
allocation.

This callback covers both: it watches the scaler's scale where there is one, and
the training loss everywhere. It only reads state — it never rescales, skips, or
touches optimization — so enabling it cannot change a healthy run's numerics.

Related but distinct: :class:`~viscy_utils.callbacks.MonitorHealthCheck` watches
the *checkpoint monitor* once per validation epoch, so it catches the same freeze
only after ``patience`` epochs and cannot say why. This one trips within tens of
steps and names the mechanism.
"""

from collections.abc import Mapping

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

__all__ = ["OptimizerHealthCheck", "OptimizerHealthError"]


class OptimizerHealthError(RuntimeError):
    """Raised when the optimizer can no longer be making progress."""


class OptimizerHealthCheck(Callback):
    """Raise when the AMP scale collapses or the training loss goes non-finite.

    Parameters
    ----------
    min_scale : float
        Raise once ``GradScaler``'s scale falls below this value. fp16 AMP
        normally operates between ``2**10`` and ``2**16``; below ``1.0`` the
        scaler is shrinking gradients rather than protecting them, which no
        healthy run needs. Ignored when the run has no scaler.
    max_consecutive_skips : int
        Raise after this many consecutive scale *reductions* with no growth in
        between. Each reduction is one skipped optimizer step. A handful during
        AMP warmup is normal and expected; a couple of dozen in a row is not.
    max_consecutive_nonfinite : int
        Raise after this many consecutive training batches whose loss is NaN or
        infinite. Small but non-zero so a single transient overflow does not kill
        a run.

    Raises
    ------
    OptimizerHealthError
        When the scale falls below ``min_scale``, when ``max_consecutive_skips``
        steps are skipped in a row, or when the loss is non-finite for
        ``max_consecutive_nonfinite`` batches in a row.

    Notes
    -----
    Skips are counted only when the scale actually *changes*, so gradient
    accumulation — which fires ``on_train_batch_end`` on batches where no
    optimizer step happened, leaving the scale flat — cannot reset the counter
    and mask a genuine run of skipped steps.

    The loss arm reads the value ``training_step`` returned rather than a logged
    metric, so it also covers modules that do not log a training loss. Manual-
    optimization modules that return ``None`` (e.g. the GAN engine) are skipped
    by the loss arm; their scaler arm still applies.
    """

    def __init__(
        self,
        min_scale: float = 1.0,
        max_consecutive_skips: int = 25,
        max_consecutive_nonfinite: int = 5,
    ) -> None:
        if max_consecutive_skips < 1:
            raise ValueError(f"max_consecutive_skips must be >= 1, got {max_consecutive_skips}")
        if max_consecutive_nonfinite < 1:
            raise ValueError(f"max_consecutive_nonfinite must be >= 1, got {max_consecutive_nonfinite}")
        self.min_scale = min_scale
        self.max_consecutive_skips = max_consecutive_skips
        self.max_consecutive_nonfinite = max_consecutive_nonfinite
        self._prev_scale: float | None = None
        self._consecutive_skips = 0
        self._consecutive_nonfinite = 0

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int) -> None:
        """Check the batch loss, then the AMP scale, after every training batch."""
        self._check_loss(outputs, trainer)
        self._check_scale(trainer)

    def _check_loss(self, outputs, trainer: Trainer) -> None:
        """Count consecutive non-finite training losses and raise past the limit."""
        loss = outputs.get("loss") if isinstance(outputs, Mapping) else outputs
        if not isinstance(loss, torch.Tensor):
            return
        if torch.isfinite(loss).all():
            self._consecutive_nonfinite = 0
            return
        self._consecutive_nonfinite += 1
        if self._consecutive_nonfinite >= self.max_consecutive_nonfinite:
            raise OptimizerHealthError(
                f"training loss has been non-finite for {self._consecutive_nonfinite} consecutive batches "
                f"(step {trainer.global_step}, last value {loss.detach().float().mean().item()}). Without a "
                "GradScaler to skip the step -- bf16-mixed and 32-true have none -- a NaN gradient lands in "
                "the weights and every later forward is non-finite, so this run cannot recover."
            )

    def _check_scale(self, trainer: Trainer) -> None:
        """Track the ``GradScaler`` scale and raise on collapse or sustained skips."""
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        if scaler is None:
            return
        scale = float(scaler.get_scale())
        prev = self._prev_scale
        self._prev_scale = scale
        if prev is not None and scale != prev:
            # A reduction means GradScaler found a non-finite gradient and skipped
            # the step; growth means a run of successful steps cleared the counter.
            self._consecutive_skips = self._consecutive_skips + 1 if scale < prev else 0

        if scale < self.min_scale:
            raise OptimizerHealthError(
                f"GradScaler scale collapsed to {scale} (floor {self.min_scale}) at step {trainer.global_step}. "
                "The scale only halves when a gradient is non-finite, so the backward pass is producing NaN "
                "or inf while the forward stays finite. At scale 0 every optimizer step is skipped forever "
                "and the weights stay at their last value for the rest of the run."
            )
        if self._consecutive_skips >= self.max_consecutive_skips:
            raise OptimizerHealthError(
                f"GradScaler halved its scale {self._consecutive_skips} times in a row without growing "
                f"(now {scale}, step {trainer.global_step}). Every one of those optimizer steps was skipped "
                "for a non-finite gradient, so the weights are not moving even though the loss is finite."
            )

    def state_dict(self) -> dict:
        """Persist the counters so a resumed run does not restart mid-collapse."""
        return {
            "prev_scale": self._prev_scale,
            "consecutive_skips": self._consecutive_skips,
            "consecutive_nonfinite": self._consecutive_nonfinite,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the counters from a checkpoint."""
        self._prev_scale = state_dict.get("prev_scale")
        self._consecutive_skips = state_dict.get("consecutive_skips", 0)
        self._consecutive_nonfinite = state_dict.get("consecutive_nonfinite", 0)
