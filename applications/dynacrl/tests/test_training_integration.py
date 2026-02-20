"""Training integration tests for DynaCLR ContrastiveModule."""

import importlib
from pathlib import Path

import pytest
import torch
import yaml
from lightning.pytorch import LightningDataModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_metric_learning.losses import NTXentLoss
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from dynacrl.engine import ContrastiveModule
from viscy_data._typing import TripletSample

C, D, H, W = 1, 1, 4, 4
FLAT_DIM = C * D * H * W


class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(FLAT_DIM, 64)
        self.proj = nn.Linear(64, 32)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.flatten(1)
        features = self.fc(x)
        projections = self.proj(features)
        return features, projections


class SyntheticTripletDataset(Dataset):
    def __init__(self, size: int = 4):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> TripletSample:
        return {
            "anchor": torch.randn(C, D, H, W),
            "positive": torch.randn(C, D, H, W),
            "negative": torch.randn(C, D, H, W),
            "index": {"fov_name": f"fov_{idx}", "id": idx},
        }


class SyntheticTripletDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 2, num_samples: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            SyntheticTripletDataset(self.num_samples),
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            SyntheticTripletDataset(self.num_samples),
            batch_size=self.batch_size,
        )


def test_contrastive_fast_dev_run(tmp_path):
    encoder = SimpleEncoder()
    module = ContrastiveModule(
        encoder=encoder,
        loss_function=nn.TripletMarginLoss(margin=0.5),
        lr=1e-3,
        example_input_array_shape=(1, C, D, H, W),
    )
    datamodule = SyntheticTripletDataModule()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=datamodule)
    assert trainer.state.finished is True
    assert trainer.state.status == "finished"


def test_contrastive_ntxent_fast_dev_run(tmp_path):
    encoder = SimpleEncoder()
    module = ContrastiveModule(
        encoder=encoder,
        loss_function=NTXentLoss(),
        lr=1e-3,
        example_input_array_shape=(1, C, D, H, W),
    )
    datamodule = SyntheticTripletDataModule()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=datamodule)
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
