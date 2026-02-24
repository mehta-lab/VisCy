"""End-to-end integration tests for multi-experiment DynaCLR training.

Validates that MultiExperimentDataModule + ContrastiveModule + NTXentHCL
work together in a real Lightning training loop with synthetic data
from 2 experiments having different channel sets (GFP vs RFP).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, nn

from dynaclr.engine import ContrastiveModule
from dynaclr.experiment import ExperimentConfig
from dynaclr.loss import NTXentHCL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMG_H = 64
_IMG_W = 64
_N_T = 10
_N_Z = 1
_N_TRACKS = 5

# SimpleEncoder input dimensions: C=2 source channels, Z=1, Y=24, X=24
_C = 2
_Z = 1
_Y = 24
_X = 24
_FLAT_DIM = _C * _Z * _Y * _X


# ---------------------------------------------------------------------------
# SimpleEncoder
# ---------------------------------------------------------------------------


class SimpleEncoder(nn.Module):
    """Minimal encoder for integration testing.

    Input: (B, 2, 1, 24, 24) -> flatten -> fc(64) -> proj(32).
    Output: (features, projections) tuple.
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(_FLAT_DIM, 64)
        self.proj = nn.Linear(64, 32)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.flatten(1)
        features = self.fc(x)
        projections = self.proj(features)
        return features, projections


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_tracks_csv(
    path: Path,
    n_tracks: int = _N_TRACKS,
    n_t: int = _N_T,
) -> None:
    """Write a tracking CSV with standard columns."""
    rows = []
    for tid in range(n_tracks):
        for t in range(n_t):
            rows.append(
                {
                    "track_id": tid,
                    "t": t,
                    "id": tid * n_t + t,
                    "parent_track_id": float("nan"),
                    "parent_id": float("nan"),
                    "z": 0,
                    "y": 32.0,
                    "x": 32.0,
                }
            )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _create_experiment(
    tmp_path: Path,
    name: str,
    channel_names: list[str],
    source_channel: list[str],
    wells: list[tuple[str, str]],
    condition_wells: dict[str, list[str]],
    n_tracks: int = _N_TRACKS,
    n_t: int = _N_T,
    interval_minutes: float = 30.0,
    start_hpi: float = 0.0,
) -> ExperimentConfig:
    """Create a mini HCS OME-Zarr store, tracking CSVs, and return config."""
    from iohub.ngff import open_ome_zarr

    zarr_path = tmp_path / f"{name}.zarr"
    tracks_root = tmp_path / f"tracks_{name}"
    n_ch = len(channel_names)
    rng = np.random.default_rng(42)

    with open_ome_zarr(
        zarr_path, layout="hcs", mode="w", channel_names=channel_names
    ) as plate:
        for row, col in wells:
            pos = plate.create_position(row, col, "0")
            arr = pos.create_zeros(
                "0",
                shape=(n_t, n_ch, _N_Z, _IMG_H, _IMG_W),
                dtype=np.float32,
            )
            arr[:] = rng.standard_normal(arr.shape).astype(np.float32)
            fov_name = f"{row}/{col}/0"
            csv_path = tracks_root / fov_name / "tracks.csv"
            _make_tracks_csv(csv_path, n_tracks=n_tracks, n_t=n_t)

    return ExperimentConfig(
        name=name,
        data_path=str(zarr_path),
        tracks_path=str(tracks_root),
        channel_names=channel_names,
        source_channel=source_channel,
        condition_wells=condition_wells,
        interval_minutes=interval_minutes,
        start_hpi=start_hpi,
    )


def _write_experiments_yaml(
    tmp_path: Path, configs: list[ExperimentConfig]
) -> Path:
    """Write experiments YAML from a list of ExperimentConfig objects."""
    yaml_path = tmp_path / "experiments.yaml"
    data = {
        "experiments": [
            {
                "name": c.name,
                "data_path": c.data_path,
                "tracks_path": c.tracks_path,
                "channel_names": c.channel_names,
                "source_channel": c.source_channel,
                "condition_wells": c.condition_wells,
                "interval_minutes": c.interval_minutes,
                "start_hpi": c.start_hpi,
            }
            for c in configs
        ]
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f)
    return yaml_path


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


def test_multi_experiment_fast_dev_run(tmp_path):
    """End-to-end: 2 experiments with different channel sets, fast_dev_run."""
    seed_everything(42)

    # Create 2 experiments with DIFFERENT channel sets
    exp_alpha = _create_experiment(
        tmp_path,
        name="exp_alpha",
        channel_names=["Phase3D", "GFP", "Mito"],
        source_channel=["Phase3D", "GFP"],
        wells=[("A", "1")],
        condition_wells={"control": ["A/1"]},
    )
    exp_beta = _create_experiment(
        tmp_path,
        name="exp_beta",
        channel_names=["Phase3D", "RFP", "StressGranules"],
        source_channel=["Phase3D", "RFP"],
        wells=[("B", "1")],
        condition_wells={"control": ["B/1"]},
    )
    yaml_path = _write_experiments_yaml(tmp_path, [exp_alpha, exp_beta])

    from dynaclr.datamodule import MultiExperimentDataModule

    datamodule = MultiExperimentDataModule(
        experiments_yaml=str(yaml_path),
        z_range=(0, 1),
        yx_patch_size=(32, 32),
        final_yx_patch_size=(24, 24),
        val_experiments=["exp_beta"],
        tau_range=(0.5, 2.0),
        batch_size=4,
        num_workers=1,
        experiment_aware=True,
        condition_balanced=False,
        temporal_enrichment=False,
        channel_dropout_channels=[1],
        channel_dropout_prob=0.5,
    )

    encoder = SimpleEncoder()
    module = ContrastiveModule(
        encoder=encoder,
        loss_function=NTXentHCL(temperature=0.07, beta=0.5),
        lr=1e-3,
        example_input_array_shape=(1, _C, _Z, _Y, _X),
    )

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


def test_multi_experiment_fast_dev_run_with_all_sampling_axes(tmp_path):
    """End-to-end: 2 experiments with all sampling axes enabled."""
    seed_everything(42)

    # 2 conditions per experiment, 2 wells each
    exp_alpha = _create_experiment(
        tmp_path,
        name="exp_alpha",
        channel_names=["Phase3D", "GFP", "Mito"],
        source_channel=["Phase3D", "GFP"],
        wells=[("A", "1"), ("A", "2")],
        condition_wells={"uninfected": ["A/1"], "infected": ["A/2"]},
        start_hpi=0.0,
    )
    exp_beta = _create_experiment(
        tmp_path,
        name="exp_beta",
        channel_names=["Phase3D", "RFP", "StressGranules"],
        source_channel=["Phase3D", "RFP"],
        wells=[("B", "1"), ("B", "2")],
        condition_wells={"uninfected": ["B/1"], "infected": ["B/2"]},
        start_hpi=0.0,
    )
    yaml_path = _write_experiments_yaml(tmp_path, [exp_alpha, exp_beta])

    from dynaclr.datamodule import MultiExperimentDataModule

    datamodule = MultiExperimentDataModule(
        experiments_yaml=str(yaml_path),
        z_range=(0, 1),
        yx_patch_size=(32, 32),
        final_yx_patch_size=(24, 24),
        val_experiments=["exp_beta"],
        tau_range=(0.5, 2.0),
        batch_size=4,
        num_workers=1,
        # All sampling axes enabled
        experiment_aware=True,
        condition_balanced=True,
        temporal_enrichment=True,
        temporal_window_hours=2.0,
        temporal_global_fraction=0.3,
        channel_dropout_channels=[1],
        channel_dropout_prob=0.5,
    )

    encoder = SimpleEncoder()
    module = ContrastiveModule(
        encoder=encoder,
        loss_function=NTXentHCL(temperature=0.07, beta=0.5),
        lr=1e-3,
        example_input_array_shape=(1, _C, _Z, _Y, _X),
    )

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


# ---------------------------------------------------------------------------
# Config class_path validation
# ---------------------------------------------------------------------------


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


def test_multi_experiment_config_class_paths_resolve():
    """All class_paths in multi_experiment_fit.yml resolve to importable classes."""
    configs_dir = Path(__file__).parents[1] / "examples" / "configs"
    config_path = configs_dir / "multi_experiment_fit.yml"
    assert config_path.exists(), f"Config file not found: {config_path}"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    class_paths = _extract_class_paths(config)
    assert len(class_paths) > 0, "No class_path entries found in multi_experiment_fit.yml"

    for cp in class_paths:
        cls = _resolve_class_path(cp)
        assert cls is not None, f"Failed to resolve class_path: {cp}"
