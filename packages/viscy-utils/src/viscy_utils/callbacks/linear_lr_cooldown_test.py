"""Tests for the linear LR cooldown callback.

Runs a real ``Trainer`` with the shared ``configure_adamw_scheduler`` so the
epoch-interval ``ConstantLR`` steps between epochs exactly as in a Constant-
schedule fit; the module records the LR its optimizer step actually uses.
"""

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from viscy_utils.callbacks import LinearLRCooldown
from viscy_utils.optimizers import configure_adamw_scheduler

LR = 0.1
START, END, FINAL = 4, 12, 0.25
MAX_STEPS = 16


class _RecordingModule(LightningModule):
    """Tiny regression module that records the LR applied at each global step."""

    def __init__(self, schedule: str = "Constant") -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)
        self.schedule = schedule
        self.applied_lr: dict[int, float] = {}

    def training_step(self, batch, batch_idx):
        self.applied_lr[self.trainer.global_step] = self.trainer.optimizers[0].param_groups[0]["lr"]
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def configure_optimizers(self):
        return configure_adamw_scheduler(self, self.layer, LR, self.schedule, warmup_steps=2)


def _loader() -> DataLoader:
    # 4 batches per epoch, so the ramp crosses several epoch-end scheduler steps.
    return DataLoader(TensorDataset(torch.randn(4, 2), torch.randn(4, 1)), batch_size=1)


def _trainer(max_steps: int) -> Trainer:
    return Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[LinearLRCooldown(START, END, FINAL)],
    )


def _expected(step: int) -> float:
    if step <= START:
        return LR
    if step >= END:
        return LR * FINAL
    return LR * (1 - (1 - FINAL) * (step - START) / (END - START))


def test_cooldown_applies_the_linear_ramp():
    """Applied LR is initial_lr up to start, linear to its final fraction at end, constant after."""
    module = _RecordingModule()
    _trainer(MAX_STEPS).fit(module, _loader())
    assert sorted(module.applied_lr) == list(range(MAX_STEPS))
    mid = (START + END) // 2
    assert module.applied_lr[START - 1] == pytest.approx(LR)
    assert module.applied_lr[START] == pytest.approx(LR)
    assert module.applied_lr[mid] == pytest.approx(LR * (1 + FINAL) / 2)
    assert module.applied_lr[END] == pytest.approx(LR * FINAL)
    assert module.applied_lr[MAX_STEPS - 1] == pytest.approx(LR * FINAL)
    for step, lr in module.applied_lr.items():
        assert lr == pytest.approx(_expected(step)), step


def test_cooldown_continues_after_resume_from_mid_ramp(tmp_path):
    """A fit resumed mid-cooldown continues the ramp from the restored global step."""
    first = _RecordingModule()
    resume_step = 7
    trainer = _trainer(resume_step)
    trainer.fit(first, _loader())
    ckpt = tmp_path / "mid.ckpt"
    trainer.save_checkpoint(ckpt)

    resumed = _RecordingModule()
    _trainer(MAX_STEPS).fit(resumed, _loader(), ckpt_path=ckpt)
    # The resumed fit starts at the checkpoint's global step, not at 0, and
    # the ramp picks up from there rather than restarting.
    assert sorted(resumed.applied_lr) == list(range(resume_step, MAX_STEPS))
    for step, lr in resumed.applied_lr.items():
        assert lr == pytest.approx(_expected(step)), step
    assert resumed.applied_lr[resume_step] < LR


def test_cooldown_rejects_a_stepwise_scheduler():
    """A step-wise scheduler would overwrite the cooldown, so fit refuses to start."""
    with pytest.raises(ValueError, match="ConstantLR"):
        _trainer(MAX_STEPS).fit(_RecordingModule(schedule="WarmupCosine"), _loader())


@pytest.mark.parametrize(
    ("start", "end", "final"),
    [(5, 5, 0.0), (6, 5, 0.0), (-1, 5, 0.0), (0, 5, -0.1), (0, 5, 1.5)],
)
def test_cooldown_rejects_invalid_arguments(start, end, final):
    """Out-of-range steps or final_factor raise at construction."""
    with pytest.raises(ValueError):
        LinearLRCooldown(start, end, final)
