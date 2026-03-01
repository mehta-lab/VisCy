"""Training integration tests for DynaCLR ContrastiveModule."""

import importlib
from pathlib import Path

import pytest
import yaml
from conftest import SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W, SimpleEncoder, SyntheticTripletDataModule
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_metric_learning.losses import NTXentLoss
from torch import nn

from dynaclr.engine import ContrastiveModule


def test_contrastive_fast_dev_run(tmp_path):
    seed_everything(42)
    module = ContrastiveModule(
        encoder=SimpleEncoder(),
        loss_function=nn.TripletMarginLoss(margin=0.5),
        lr=1e-3,
        example_input_array_shape=(1, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=SyntheticTripletDataModule())
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def test_contrastive_ntxent_fast_dev_run(tmp_path):
    seed_everything(42)
    module = ContrastiveModule(
        encoder=SimpleEncoder(),
        loss_function=NTXentLoss(),
        lr=1e-3,
        example_input_array_shape=(1, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=SyntheticTripletDataModule())
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def _extract_class_paths(obj):
    """Recursively extract all class_path values from a parsed YAML dict."""
    paths = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "class_path" and isinstance(value, str):
                paths.append(value)
            else:
                paths.extend(_extract_class_paths(value))
    elif isinstance(obj, list):
        for item in obj:
            paths.extend(_extract_class_paths(item))
    return paths


def _resolve_class_path(class_path: str):
    """Resolve a dotted class_path to the actual class object."""
    parts = class_path.rsplit(".", 1)
    module_path, class_name = parts[0], parts[1]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


@pytest.mark.parametrize(
    "config_name",
    ["fit.yml", "predict.yml"],
)
def test_config_class_paths_resolve(config_name):
    configs_dir = Path(__file__).parents[1] / "examples" / "configs"
    config_path = configs_dir / config_name
    assert config_path.exists(), f"Config file not found: {config_path}"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    class_paths = _extract_class_paths(config)
    assert len(class_paths) > 0, f"No class_path entries found in {config_name}"

    for cp in class_paths:
        cls = _resolve_class_path(cp)
        assert cls is not None, f"Failed to resolve class_path: {cp}"
