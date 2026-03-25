"""Training integration tests for CellDiffE2E.

Validates that the forward+backward pass works for CellDiffE2E
using ``fast_dev_run=True`` (1 batch of train + val).

Synthetic tests use lightweight random data and always run on CPU.
"""

import importlib
from pathlib import Path

import pytest
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from celldiff_e2e.engine import CellDiffE2E

from .conftest import SyntheticE2EDataModule, tiny_model_config


def test_celldiff_e2e_mse_fast_dev_run(tmp_path):
    """CellDiffE2E + MSELoss + WarmupCosine schedule trains for 1 batch."""
    seed_everything(42)
    module = CellDiffE2E(model_config=tiny_model_config, schedule="WarmupCosine")
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=SyntheticE2EDataModule())
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def test_celldiff_e2e_constant_schedule_fast_dev_run(tmp_path):
    """CellDiffE2E + MSELoss + Constant schedule trains for 1 batch."""
    seed_everything(42)
    module = CellDiffE2E(model_config=tiny_model_config, schedule="Constant")
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=SyntheticE2EDataModule())
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
    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


@pytest.mark.parametrize("config_name", ["fit_unetvit3d.yml", "predict_unetvit3d.yml"])
def test_config_class_paths_resolve(config_name):
    """All class_path entries in example configs resolve to importable classes."""
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
