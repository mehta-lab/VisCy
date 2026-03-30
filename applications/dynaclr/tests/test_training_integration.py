"""Training integration tests for DynaCLR ContrastiveModule."""

from pathlib import Path

import pytest
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_metric_learning.losses import NTXentLoss
from torch import nn

from dynaclr.engine import ContrastiveModule


def test_contrastive_fast_dev_run(tmp_path, _SimpleEncoder, _SyntheticTripletDataModule, synth_dims):
    seed_everything(42)
    module = ContrastiveModule(
        encoder=_SimpleEncoder(),
        loss_function=nn.TripletMarginLoss(margin=0.5),
        lr=1e-3,
        example_input_array_shape=(1, synth_dims["c"], synth_dims["d"], synth_dims["h"], synth_dims["w"]),
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=_SyntheticTripletDataModule())
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def test_contrastive_ntxent_fast_dev_run(tmp_path, _SimpleEncoder, _SyntheticTripletDataModule, synth_dims):
    seed_everything(42)
    module = ContrastiveModule(
        encoder=_SimpleEncoder(),
        loss_function=NTXentLoss(),
        lr=1e-3,
        example_input_array_shape=(1, synth_dims["c"], synth_dims["d"], synth_dims["h"], synth_dims["w"]),
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=_SyntheticTripletDataModule())
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


@pytest.mark.parametrize(
    "config_name,config_subdir",
    [("fit.yml", "training"), ("predict.yml", "prediction")],
)
def test_config_class_paths_resolve(config_name, config_subdir, _extract_class_paths, _resolve_class_path):
    configs_dir = Path(__file__).parents[1] / "configs" / config_subdir
    config_path = configs_dir / config_name
    assert config_path.exists(), f"Config file not found: {config_path}"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    class_paths = _extract_class_paths(config)
    assert len(class_paths) > 0, f"No class_path entries found in {config_name}"

    for cp in class_paths:
        cls = _resolve_class_path(cp)
        assert cls is not None, f"Failed to resolve class_path: {cp}"
