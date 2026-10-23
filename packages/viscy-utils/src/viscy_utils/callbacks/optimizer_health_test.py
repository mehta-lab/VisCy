"""Tests for the step-level optimizer guardrail.

Runs a real ``Trainer`` throughout: the failure being guarded against is produced
by the interaction of a NaN backward with Lightning's precision plugin, so a
mocked scaler would prove nothing. The NaN-gradient module reproduces the actual
mechanism that froze the 24 FCMAE-2D fits — a ``sqrt`` evaluated at exactly 0,
whose derivative is infinite, multiplied by a zero incoming gradient — rather
than asserting on a hand-set scale.
"""

import pytest
import torch
from jsonargparse import Namespace
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from viscy_utils.callbacks import OptimizerHealthCheck, OptimizerHealthError
from viscy_utils.cli import _inject_optimizer_health_guard

_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fp16 AMP has no GradScaler on CPU -- Lightning silently downgrades 16-mixed to bf16-mixed",
)


class _HealthyModule(LightningModule):
    """Ordinary regression module with finite gradients everywhere."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class _NaNGradientModule(_HealthyModule):
    """Finite forward, NaN backward — the FCMAE-2D freeze, minimally reproduced.

    ``sqrt`` at 0 has an infinite derivative, and the incoming gradient of the
    zeroed product is 0, so autograd evaluates ``0 * inf = nan`` for the whole
    term while its forward contribution is exactly 0. The loss therefore stays
    finite and only the gradient is poisoned.
    """

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.layer(x)
        bound = torch.sqrt(pred * torch.zeros_like(pred))
        loss = torch.nn.functional.mse_loss(pred, y) + bound.mean()
        assert torch.isfinite(loss).all(), "forward must stay finite for this test to mean anything"
        return loss


class _NaNLossModule(_HealthyModule):
    """Loss itself goes non-finite, as it does under bf16 with no scaler."""

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y) * float("nan")


def _loader() -> DataLoader:
    return DataLoader(TensorDataset(torch.randn(32, 2), torch.randn(32, 1)), batch_size=4)


def _run(module: LightningModule, tmp_path, precision: str, accelerator: str, **guard_kwargs) -> Trainer:
    trainer = Trainer(
        max_epochs=4,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        callbacks=[OptimizerHealthCheck(**guard_kwargs)],
        default_root_dir=tmp_path,
    )
    trainer.fit(module, _loader())
    return trainer


@_needs_cuda
def test_nan_gradient_under_fp16_raises_while_the_loss_stays_finite(tmp_path):
    """The silent case: finite loss, skipped steps, scale walking to zero."""
    with pytest.raises(OptimizerHealthError, match="halved its scale|scale collapsed"):
        _run(
            _NaNGradientModule(),
            tmp_path,
            precision="16-mixed",
            accelerator="gpu",
            max_consecutive_skips=3,
        )


@_needs_cuda
def test_healthy_fp16_run_completes_untouched(tmp_path):
    """A normal fp16 run must not trip the guard, and must actually move weights."""
    module = _HealthyModule()
    before = module.layer.weight.detach().clone()
    trainer = _run(module, tmp_path, precision="16-mixed", accelerator="gpu", max_consecutive_skips=3)
    assert trainer.current_epoch == 4
    assert not torch.allclose(before, module.layer.weight.detach().cpu())


def test_non_finite_loss_raises_without_a_scaler(tmp_path):
    """bf16 has no GradScaler, so the loss arm is the only thing that can catch it."""
    with pytest.raises(OptimizerHealthError, match="non-finite for"):
        _run(
            _NaNLossModule(),
            tmp_path,
            precision="bf16-mixed",
            accelerator="cpu",
            max_consecutive_nonfinite=2,
        )


def test_healthy_bf16_run_completes_untouched(tmp_path):
    """The precision the 2D FCMAE fits actually use must train to completion."""
    module = _HealthyModule()
    before = module.layer.weight.detach().clone()
    trainer = _run(module, tmp_path, precision="bf16-mixed", accelerator="cpu", max_consecutive_nonfinite=2)
    assert trainer.current_epoch == 4
    assert not torch.allclose(before, module.layer.weight.detach())


def test_transient_non_finite_loss_is_tolerated(tmp_path):
    """One overflow must not kill a run — only a sustained streak may.

    The NaN is added as a *constant*, so its gradient contribution is zero and the
    weights stay healthy. Multiplying the loss by NaN instead would send NaN into
    the weights and every later batch would be non-finite too — which is precisely
    why bf16 needs this guard, and which made an earlier version of this test fail
    for the right reason.
    """

    class _OneNaNModule(_HealthyModule):
        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = torch.nn.functional.mse_loss(self.layer(x), y)
            if self.global_step == 1:
                return loss + torch.tensor(float("nan"))
            return loss

    module = _OneNaNModule()
    trainer = _run(module, tmp_path, precision="bf16-mixed", accelerator="cpu", max_consecutive_nonfinite=3)
    assert trainer.current_epoch == 4
    assert torch.isfinite(module.layer.weight).all()


def test_accumulation_flat_scale_does_not_reset_the_skip_counter():
    """Batches with no optimizer step leave the scale flat; that must not clear skips.

    Under ``accumulate_grad_batches>1`` the scale only changes on real steps, so a
    counter that reset on every unchanged sample would never reach its limit and
    the guard would be dead exactly where the bug lives.
    """
    guard = OptimizerHealthCheck(max_consecutive_skips=3, min_scale=0.0)

    class _FakeScaler:
        def __init__(self) -> None:
            self.scale = 65536.0

        def get_scale(self) -> float:
            return self.scale

    scaler = _FakeScaler()
    trainer = Namespace(precision_plugin=Namespace(scaler=scaler), global_step=0)

    guard._check_scale(trainer)  # first observation, nothing to compare against
    for _ in range(2):
        scaler.scale /= 2  # a skipped step
        guard._check_scale(trainer)
        guard._check_scale(trainer)  # accumulation batch: scale unchanged
    assert guard._consecutive_skips == 2

    scaler.scale /= 2
    with pytest.raises(OptimizerHealthError, match="halved its scale 3 times"):
        guard._check_scale(trainer)


def test_scale_growth_clears_the_skip_counter():
    """A successful growth interval means the run recovered; forget the earlier skips."""
    guard = OptimizerHealthCheck(max_consecutive_skips=3, min_scale=0.0)

    class _FakeScaler:
        def __init__(self) -> None:
            self.scale = 1024.0

        def get_scale(self) -> float:
            return self.scale

    scaler = _FakeScaler()
    trainer = Namespace(precision_plugin=Namespace(scaler=scaler), global_step=0)
    guard._check_scale(trainer)
    scaler.scale /= 2
    guard._check_scale(trainer)
    assert guard._consecutive_skips == 1
    scaler.scale *= 2
    guard._check_scale(trainer)
    assert guard._consecutive_skips == 0


def test_guard_is_injected_into_a_fit_config():
    """``fit`` configs get the guard even though leaves override ``trainer.callbacks``."""
    config = Namespace(fit=Namespace(trainer=Namespace(callbacks=[])))
    _inject_optimizer_health_guard(config, "fit")
    paths = [callback.class_path for callback in config.fit.trainer.callbacks]
    assert paths == ["viscy_utils.callbacks.OptimizerHealthCheck"]


def test_guard_injection_is_idempotent():
    """Re-running the injector (e.g. a resumed resolved config) must not duplicate it."""
    config = Namespace(fit=Namespace(trainer=Namespace(callbacks=[])))
    _inject_optimizer_health_guard(config, "fit")
    _inject_optimizer_health_guard(config, "fit")
    assert len(config.fit.trainer.callbacks) == 1


@pytest.mark.parametrize("subcommand", ["predict", "validate", "test", None])
def test_guard_is_not_injected_outside_fit(subcommand):
    """Only training can stall this way; predict/validate configs stay untouched."""
    config = Namespace(**{str(subcommand): Namespace(trainer=Namespace(callbacks=[]))})
    _inject_optimizer_health_guard(config, subcommand)
    root = config.get(str(subcommand))
    assert root.trainer.callbacks == []
