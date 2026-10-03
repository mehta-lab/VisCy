"""Tests for the checkpoint-monitor guardrails.

Runs a real ``Trainer`` so the interaction with Lightning's ``ModelCheckpoint``
is exercised, not mocked — the failure this guards against (top-k fills, then
nothing is saved) only appears through the real callback.
"""

import pytest
import torch
from jsonargparse import Namespace
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from viscy_utils.callbacks import MonitorHealthCheck, MonitorHealthError
from viscy_utils.cli import _inject_checkpoint_guardrails


class _ValLossModule(LightningModule):
    """Minimal module that logs a scripted ``loss/validate`` sequence."""

    def __init__(self, val_values: list[float]):
        super().__init__()
        self.val_values = val_values
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def on_validation_epoch_end(self):
        # Index defensively: sanity-check validation also fires this hook.
        idx = min(self.current_epoch, len(self.val_values) - 1)
        self.log("loss/validate", torch.tensor(self.val_values[idx]))

    def validation_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _loaders() -> tuple[DataLoader, DataLoader]:
    ds = TensorDataset(torch.randn(8, 2), torch.randn(8, 1))
    return DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4)


def _run(val_values: list[float], tmp_path, patience: int = 3, max_epochs: int = 8) -> Trainer:
    train_dl, val_dl = _loaders()
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[MonitorHealthCheck(monitor="loss/validate", patience=patience)],
        default_root_dir=tmp_path,
    )
    trainer.fit(_ValLossModule(val_values), train_dl, val_dl)
    return trainer


def test_frozen_monitor_raises(tmp_path):
    """A monitor that never changes must raise rather than train silently."""
    with pytest.raises(MonitorHealthError, match="identical"):
        _run([1.024275] * 3, tmp_path, patience=3, max_epochs=3)


def test_non_finite_monitor_raises(tmp_path):
    """A NaN monitor must raise immediately."""
    with pytest.raises(MonitorHealthError, match="non-finite"):
        _run([0.9, float("nan")], tmp_path, patience=3, max_epochs=2)


def test_improving_monitor_does_not_raise(tmp_path):
    """A normally-varying monitor must train to completion untouched."""
    trainer = _run([0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45], tmp_path, patience=3)
    assert trainer.current_epoch == 8
    health = next(callback for callback in trainer.callbacks if isinstance(callback, MonitorHealthCheck))
    assert health.state_dict()["recent"] == pytest.approx([0.55, 0.5, 0.45])


def test_brief_plateau_below_patience_does_not_raise(tmp_path):
    """A plateau shorter than ``patience`` is normal training, not a dead metric."""
    trainer = _run([0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.55, 0.5], tmp_path, patience=3)
    assert trainer.current_epoch == 8


def test_frozen_monitor_starves_real_model_checkpoint(tmp_path):
    """Reproduce the underlying failure: a frozen monitor stops all saving.

    Without the guardrail a constant monitor fills ``save_top_k`` and then writes
    nothing more — including ``last.ckpt``. This is the regression the injected
    unmonitored checkpoint exists to prevent.
    """
    train_dl, val_dl = _loaders()
    ckpt_dir = tmp_path / "checkpoints"
    monitored = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="loss/validate",
        filename="epoch={epoch}-step={step}",
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_top_k=2,
        save_last=True,
    )
    trainer = Trainer(
        max_epochs=8,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[monitored],
        default_root_dir=tmp_path,
    )
    trainer.fit(_ValLossModule([1.024275] * 8), train_dl, val_dl)

    saved = sorted(p.name for p in ckpt_dir.glob("epoch=*.ckpt"))
    # save_top_k=2 fills at epochs 0 and 1; the constant metric never improves after.
    assert len(saved) == 2, saved
    assert saved == ["epoch=0-step=2.ckpt", "epoch=1-step=4.ckpt"], saved
    # 8 epochs ran, yet nothing newer than epoch 1 exists.
    assert trainer.current_epoch == 8


def test_injection_adds_guardrails_and_preserves_selection():
    """Injection must add the safety net without touching selection semantics."""
    monitored = Namespace(
        class_path="lightning.pytorch.callbacks.ModelCheckpoint",
        init_args=Namespace(
            monitor="loss/validate",
            mode="min",
            filename="epoch={epoch}-step={step}-loss={loss/validate:.3f}",
            save_top_k=5,
            every_n_epochs=1,
            save_last=True,
            dirpath="/models/run/checkpoints",
        ),
    )
    config = Namespace(fit=Namespace(trainer=Namespace(callbacks=[monitored])))
    _inject_checkpoint_guardrails(config, "fit")
    callbacks = config["fit"]["trainer"]["callbacks"]

    assert len(callbacks) == 3
    # Selection semantics of the monitored callback are untouched...
    assert monitored["init_args"]["monitor"] == "loss/validate"
    assert monitored["init_args"]["mode"] == "min"
    assert monitored["init_args"]["save_top_k"] == 5
    assert monitored["init_args"]["filename"] == "epoch={epoch}-step={step}-loss={loss/validate:.3f}"
    # ...except that it no longer owns `last.ckpt`.
    assert monitored["init_args"]["save_last"] is False

    latest = callbacks[1]
    assert latest["class_path"] == "lightning.pytorch.callbacks.ModelCheckpoint"
    assert latest["init_args"]["monitor"] is None
    assert latest["init_args"]["save_last"] is True
    assert latest["init_args"]["save_top_k"] == 1
    assert latest["init_args"]["dirpath"] == "/models/run/checkpoints"

    health = callbacks[2]
    assert health["class_path"] == "viscy_utils.callbacks.MonitorHealthCheck"
    assert health["init_args"]["monitor"] == "loss/validate"


def test_injection_is_idempotent():
    """Re-running injection must not stack duplicate callbacks."""
    monitored = Namespace(
        class_path="lightning.pytorch.callbacks.ModelCheckpoint",
        init_args=Namespace(monitor="loss/validate", save_top_k=5, save_last=True, dirpath="/d"),
    )
    config = Namespace(fit=Namespace(trainer=Namespace(callbacks=[monitored])))
    _inject_checkpoint_guardrails(config, "fit")
    _inject_checkpoint_guardrails(config, "fit")
    assert len(config["fit"]["trainer"]["callbacks"]) == 3


def test_injection_skips_non_fit_subcommands():
    """predict/test/validate write no checkpoints, so nothing is injected."""
    monitored = Namespace(
        class_path="lightning.pytorch.callbacks.ModelCheckpoint",
        init_args=Namespace(monitor="loss/validate", save_last=True, dirpath="/d"),
    )
    config = Namespace(predict=Namespace(trainer=Namespace(callbacks=[monitored])))
    _inject_checkpoint_guardrails(config, "predict")
    assert len(config["predict"]["trainer"]["callbacks"]) == 1
    assert monitored["init_args"]["save_last"] is True


def test_injection_noop_without_monitored_checkpoint():
    """A config with no monitored checkpoint gets no health check to attach to."""
    unmonitored = Namespace(
        class_path="lightning.pytorch.callbacks.ModelCheckpoint",
        init_args=Namespace(monitor=None, save_last=True, dirpath="/d"),
    )
    config = Namespace(fit=Namespace(trainer=Namespace(callbacks=[unmonitored])))
    _inject_checkpoint_guardrails(config, "fit")
    assert len(config["fit"]["trainer"]["callbacks"]) == 1


def test_patience_below_two_rejected():
    """patience<2 cannot distinguish a frozen metric from a single sample."""
    with pytest.raises(ValueError, match="patience must be >= 2"):
        MonitorHealthCheck(patience=1)


def test_injection_inherits_the_leaf_write_cadence():
    """The latest-weights callback must not force ``every_n_epochs: 1``.

    A leaf that sets ``every_n_epochs`` is making a deliberate write-amplification
    decision. The Phase 17 arms run ~1470 epochs with a 424 MB checkpoint and set
    cadence 10 for exactly that reason; forcing 1 here rewrote both ``latest-*``
    and ``last.ckpt`` every epoch (~1.25 TB of NFS traffic per arm) and silently
    overrode the leaf.
    """
    monitored = Namespace(
        class_path="lightning.pytorch.callbacks.ModelCheckpoint",
        init_args=Namespace(
            monitor="loss/validate",
            save_top_k=4,
            every_n_epochs=10,
            save_last=True,
            dirpath="/models/run/checkpoints",
        ),
    )
    config = Namespace(fit=Namespace(trainer=Namespace(callbacks=[monitored])))
    _inject_checkpoint_guardrails(config, "fit")

    latest = config["fit"]["trainer"]["callbacks"][1]
    assert latest["init_args"]["every_n_epochs"] == 10
    # The monitored callback's own cadence is still its own.
    assert monitored["init_args"]["every_n_epochs"] == 10


def test_injection_defaults_cadence_to_every_epoch():
    """A leaf that sets no cadence keeps the original every-epoch behaviour."""
    monitored = Namespace(
        class_path="lightning.pytorch.callbacks.ModelCheckpoint",
        init_args=Namespace(monitor="loss/validate", save_top_k=4, save_last=True, dirpath="/d"),
    )
    config = Namespace(fit=Namespace(trainer=Namespace(callbacks=[monitored])))
    _inject_checkpoint_guardrails(config, "fit")

    assert config["fit"]["trainer"]["callbacks"][1]["init_args"]["every_n_epochs"] == 1


def test_inherited_cadence_still_saves_when_monitor_is_frozen(tmp_path):
    """End-to-end: cadence 4 writes latest weights without writing every epoch.

    Runs the real Trainer with a frozen monitor, which starves the monitored
    callback after ``save_top_k`` fills. The unmonitored callback must still
    produce weights from the run's final epochs -- the guardrail's whole purpose --
    while writing 2 times over 8 epochs rather than 8.
    """
    train_dl, val_dl = _loaders()
    ckpt_dir = tmp_path / "checkpoints"
    monitored = Namespace(
        class_path="lightning.pytorch.callbacks.ModelCheckpoint",
        init_args=Namespace(
            monitor="loss/validate",
            filename="epoch={epoch}-step={step}",
            auto_insert_metric_name=False,
            save_top_k=2,
            every_n_epochs=4,
            save_last=True,
            dirpath=str(ckpt_dir),
        ),
    )
    config = Namespace(fit=Namespace(trainer=Namespace(callbacks=[monitored])))
    _inject_checkpoint_guardrails(config, "fit")
    specs = config["fit"]["trainer"]["callbacks"]

    latest_cb = ModelCheckpoint(**{k: v for k, v in specs[1]["init_args"].items()})
    monitored_cb = ModelCheckpoint(**{k: v for k, v in specs[0]["init_args"].items()})
    trainer = Trainer(
        max_epochs=8,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[monitored_cb, latest_cb],
        default_root_dir=tmp_path,
    )
    trainer.fit(_ValLossModule([1.024275] * 8), train_dl, val_dl)

    # Lightning fires every_n_epochs when (epoch + 1) % n == 0, so this writes at
    # epochs 3 and 7 only -- 2 of 8 epochs -- and save_top_k=1 keeps just the last.
    latest = sorted(p.name for p in ckpt_dir.glob("latest-*.ckpt"))
    assert latest == ["latest-epoch=7-step=16.ckpt"], latest
    assert (ckpt_dir / "last.ckpt").is_file()
    # The point of the guardrail: weights exist from the END of the run (epoch 7 is
    # the last of 8, zero-indexed), not just from where the frozen monitor stopped
    # improving -- the monitored callback's top-k filled at epochs 3 and never moved.
    assert torch.load(ckpt_dir / "last.ckpt", map_location="cpu", weights_only=False)["epoch"] == 7
