"""End-to-end training, whole-track metrics, and config composition."""

from pathlib import Path

import numpy as np
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from dynaformer.engine import DynaformerModule
from dynaformer.evaluation import TrackTrajectoryMetrics, channel_readout, track_trajectory_metrics
from dynaformer.models import RoPETransformerEncoder, SpatioTemporalModel, UNeXt2FrameEncoder, UNeXt2ImageDecoder
from viscy_models.components.heads import RegressionHead
from viscy_utils.compose import load_composed_config

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
D_MODEL = 32


def build_module(track_spec, **overrides) -> DynaformerModule:
    """Build a small end-to-end model matching the synthetic dataset."""
    patch = track_spec["final_patch"]
    n_target = len(track_spec["target_channels"])
    encoder = UNeXt2FrameEncoder(
        in_channels=1,
        in_stack_depth=1,
        backbone="convnextv2_atto",
        stem_kernel_size=(1, 4, 4),
        yx_patch_size=(patch, patch),
        token_dim=D_MODEL,
        token_channels=8,
    )
    model = SpatioTemporalModel(
        encoder=encoder,
        temporal=RoPETransformerEncoder(d_model=D_MODEL, n_heads=4, n_layers=1),
        decoder=UNeXt2ImageDecoder(
            encoder.feature_channels,
            out_channels=n_target,
            out_stack_depth=1,
            stem_stride=encoder.stem_stride,
            cond_dim=D_MODEL,
        ),
        heads={"scalar": RegressionHead("scalar", "scalar_target", in_dims=D_MODEL, out_dims=n_target)},
    )
    return DynaformerModule(model=model, **overrides)


def test_fit_runs_end_to_end(make_datamodule, track_spec, tmp_path):
    """A training step runs through the real data module on CPU."""
    module = build_module(track_spec)
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=str(tmp_path)),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=make_datamodule())
    assert trainer.state.finished
    assert "loss/train" in trainer.callback_metrics


def test_discriminative_learning_rates_reach_the_optimizer(make_datamodule, track_spec, tmp_path):
    """The encoder gets its own learning rate, everything else gets the base one.

    A pretrained backbone fine-tuned at the head learning rate loses the features
    that justify using it, so this split has to survive into the optimizer.
    """
    module = build_module(track_spec, lr=3e-4, encoder_lr=1e-5, weight_decay=0.05)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu", logger=False, enable_progress_bar=False)
    trainer.fit(module, datamodule=make_datamodule())

    optimizer = trainer.optimizers[0]
    rates = {group["lr"] for group in optimizer.param_groups}
    assert rates == {3e-4, 1e-5}
    encoder_ids = {id(p) for p in module.model.encoder.parameters()}
    for group in optimizer.param_groups:
        expected = 1e-5 if id(group["params"][0]) in encoder_ids else 3e-4
        assert group["lr"] == expected
        assert all((id(p) in encoder_ids) == (group["lr"] == 1e-5) for p in group["params"])
    # No weight decay on biases and normalization scales.
    for group in optimizer.param_groups:
        assert (group["weight_decay"] == 0.0) == (group["params"][0].ndim <= 1)


def test_track_metrics_callback_logs_both_readouts(make_datamodule, track_spec, tmp_path):
    """Whole-track inference produces image and head trajectory metrics."""
    module = build_module(track_spec)
    callback = TrackTrajectoryMetrics(
        channel_names=track_spec["target_channels"],
        readout="ratio",
        max_tracks=2,
        max_frames=8,
        min_frames=4,
    )
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=str(tmp_path)),
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[callback],
    )
    trainer.fit(module, datamodule=make_datamodule())
    for key in ("val/track_pearson/image", "val/track_pearson/scalar", "val/track_dominance/image"):
        assert key in trainer.callback_metrics


def test_track_metrics_require_normalization_statistics(track_spec):
    """Denormalization is mandatory: the readout is nonlinear."""
    callback = TrackTrajectoryMetrics(channel_names=track_spec["target_channels"])
    with pytest.raises(ValueError, match="norm_meta"):
        callback._to_values(torch.rand(1, 4, 2), None)


def test_channel_readout_and_metrics():
    """A perfectly predicted trajectory scores 1; an inverted one scores -1."""
    values = np.stack([np.linspace(1.0, 9.0, 8), np.linspace(9.0, 1.0, 8)], axis=1)
    truth = channel_readout(values, "ratio", (0, 1))
    assert np.allclose(truth, values[:, 0] / values.sum(axis=1))
    assert channel_readout(values, "identity", (1,)).tolist() == values[:, 1].tolist()

    perfect = track_trajectory_metrics([truth], [truth], [values], [values])
    assert perfect["pearson"] == pytest.approx(1.0)
    assert perfect["ccc"] == pytest.approx(1.0)
    assert perfect["mae"] == pytest.approx(0.0)
    assert perfect["dominance"] == pytest.approx(1.0)

    inverted = track_trajectory_metrics([truth], [truth[::-1].copy()])
    assert inverted["pearson"] == pytest.approx(-1.0)


def test_constant_trajectories_do_not_poison_the_average():
    """A track with no variation yields NaN and is averaged away, not counted as zero."""
    flat = np.full(8, 0.5)
    ramp = np.linspace(0.0, 1.0, 8)
    metrics = track_trajectory_metrics([flat, ramp], [flat, ramp])
    assert metrics["n_tracks"] == 2
    assert metrics["pearson"] == pytest.approx(1.0)


@pytest.mark.parametrize("leaf", ["fucci/fit_unext2.yml", "fucci/fit_morphem_tcond.yml", "fucci/predict.yml"])
def test_example_configs_compose(leaf):
    """Every example leaf resolves its base recipes into a complete config."""
    config = load_composed_config(CONFIG_ROOT / "examples" / leaf)
    assert config["model"]["class_path"] == "dynaformer.engine.DynaformerModule"
    assert config["data"]["class_path"] == "viscy_data.track_window.TrackWindowDataModule"
    assert "trainer" in config


def test_fit_configs_select_checkpoints_on_trajectory_correlation():
    """Checkpoint selection watches the trajectory metric, not the validation loss.

    The image loss rewards per-frame reconstruction, which a model can win while
    producing a trajectory that jitters between frames.
    """
    for leaf in ("fucci/fit_unext2.yml", "fucci/fit_morphem_tcond.yml"):
        config = load_composed_config(CONFIG_ROOT / "examples" / leaf)
        checkpoints = [
            callback
            for callback in config["trainer"]["callbacks"]
            if callback["class_path"].endswith("ModelCheckpoint")
        ]
        assert len(checkpoints) == 1
        assert checkpoints[0]["init_args"]["monitor"] == "val/track_pearson/image"
        assert checkpoints[0]["init_args"]["mode"] == "max"
        metrics = [c for c in config["trainer"]["callbacks"] if c["class_path"].endswith("TrackTrajectoryMetrics")]
        assert len(metrics) == 1, f"{leaf} monitors a metric no callback produces"
