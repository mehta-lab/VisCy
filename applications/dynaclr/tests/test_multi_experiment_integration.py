"""End-to-end integration tests for multi-experiment DynaCLR training.

Validates that MultiExperimentDataModule + ContrastiveModule + NTXentHCL
work together in a real Lightning training loop with synthetic data
from 2 experiments having different channel sets (GFP vs RFP).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from dynaclr.engine import ContrastiveModule
from viscy_models.contrastive.loss import NTXentHCL

# ---------------------------------------------------------------------------
# Constants — encoder input matches final_yx_patch_size=(24,24) with 2 channels
# ---------------------------------------------------------------------------

_C = 2
_Z = 1
_Y = 24
_X = 24
_FLAT_DIM = _C * _Z * _Y * _X


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


def test_multi_experiment_fast_dev_run(tmp_path, _create_experiment, _write_collection_yaml, _SimpleEncoder):
    """End-to-end: 2 experiments with different channel sets, fast_dev_run."""
    seed_everything(42)

    # Create 2 experiments with DIFFERENT channel sets
    exp_alpha = _create_experiment(
        tmp_path,
        name="exp_alpha",
        channel_names=["Phase3D", "GFP", "Mito"],
        wells=[("A", "1")],
        perturbation_wells={"control": ["A/1"]},
    )
    exp_beta = _create_experiment(
        tmp_path,
        name="exp_beta",
        channel_names=["Phase3D", "RFP", "StressGranules"],
        wells=[("B", "1")],
        perturbation_wells={"control": ["B/1"]},
    )
    yaml_path = _write_collection_yaml(tmp_path, [exp_alpha, exp_beta])

    from dynaclr.data.datamodule import MultiExperimentDataModule

    datamodule = MultiExperimentDataModule(
        collection_path=str(yaml_path),
        z_window=1,
        yx_patch_size=(32, 32),
        final_yx_patch_size=(24, 24),
        val_experiments=["exp_beta"],
        tau_range=(0.5, 2.0),
        batch_size=4,
        num_workers=1,
        batch_group_by="experiment",
        stratify_by=None,
        temporal_enrichment=False,
        channel_dropout_channels=[1],
        channel_dropout_prob=0.5,
    )

    encoder = _SimpleEncoder(in_dim=_FLAT_DIM)
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


def test_multi_experiment_fast_dev_run_with_parquet(
    tmp_path, _create_experiment, _write_collection_yaml, _SimpleEncoder
):
    """End-to-end: same as test_multi_experiment_fast_dev_run but loading from cell_index parquet."""
    seed_everything(42)

    from dynaclr.data.datamodule import MultiExperimentDataModule
    from viscy_data.cell_index import build_timelapse_cell_index

    exp_alpha = _create_experiment(
        tmp_path,
        name="exp_alpha",
        channel_names=["Phase3D", "GFP", "Mito"],
        wells=[("A", "1")],
        perturbation_wells={"control": ["A/1"]},
    )
    exp_beta = _create_experiment(
        tmp_path,
        name="exp_beta",
        channel_names=["Phase3D", "RFP", "StressGranules"],
        wells=[("B", "1")],
        perturbation_wells={"control": ["B/1"]},
    )
    yaml_path = _write_collection_yaml(tmp_path, [exp_alpha, exp_beta])

    # Build cell index parquet
    parquet_path = tmp_path / "cell_index.parquet"
    build_timelapse_cell_index(yaml_path, parquet_path)

    datamodule = MultiExperimentDataModule(
        collection_path=str(yaml_path),
        z_window=1,
        yx_patch_size=(32, 32),
        final_yx_patch_size=(24, 24),
        val_experiments=["exp_beta"],
        tau_range=(0.5, 2.0),
        batch_size=4,
        num_workers=1,
        batch_group_by="experiment",
        stratify_by=None,
        temporal_enrichment=False,
        channel_dropout_channels=[1],
        channel_dropout_prob=0.5,
        cell_index_path=str(parquet_path),
    )

    encoder = _SimpleEncoder(in_dim=_FLAT_DIM)
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


def test_multi_experiment_fast_dev_run_with_all_sampling_axes(
    tmp_path, _create_experiment, _write_collection_yaml, _SimpleEncoder
):
    """End-to-end: 2 experiments with all sampling axes enabled."""
    seed_everything(42)

    # 2 conditions per experiment, 2 wells each
    exp_alpha = _create_experiment(
        tmp_path,
        name="exp_alpha",
        channel_names=["Phase3D", "GFP", "Mito"],
        wells=[("A", "1"), ("A", "2")],
        perturbation_wells={"uninfected": ["A/1"], "infected": ["A/2"]},
        start_hpi=0.0,
    )
    exp_beta = _create_experiment(
        tmp_path,
        name="exp_beta",
        channel_names=["Phase3D", "RFP", "StressGranules"],
        wells=[("B", "1"), ("B", "2")],
        perturbation_wells={"uninfected": ["B/1"], "infected": ["B/2"]},
        start_hpi=0.0,
    )
    yaml_path = _write_collection_yaml(tmp_path, [exp_alpha, exp_beta])

    from dynaclr.data.datamodule import MultiExperimentDataModule

    datamodule = MultiExperimentDataModule(
        collection_path=str(yaml_path),
        z_window=1,
        yx_patch_size=(32, 32),
        final_yx_patch_size=(24, 24),
        val_experiments=["exp_beta"],
        tau_range=(0.5, 2.0),
        batch_size=4,
        num_workers=1,
        # All sampling axes enabled
        batch_group_by="experiment",
        stratify_by="perturbation",
        temporal_enrichment=True,
        temporal_window_hours=2.0,
        temporal_global_fraction=0.3,
        channel_dropout_channels=[1],
        channel_dropout_prob=0.5,
    )

    encoder = _SimpleEncoder(in_dim=_FLAT_DIM)
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


def test_multi_experiment_config_class_paths_resolve(_extract_class_paths, _resolve_class_path):
    """All class_paths in multi_experiment_fit.yml resolve to importable classes."""
    configs_dir = Path(__file__).parents[1] / "configs" / "training"
    config_path = configs_dir / "multi_experiment_fit.yml"
    assert config_path.exists(), f"Config file not found: {config_path}"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    class_paths = _extract_class_paths(config)
    assert len(class_paths) > 0, "No class_path entries found in multi_experiment_fit.yml"

    for cp in class_paths:
        cls = _resolve_class_path(cp)
        assert cls is not None, f"Failed to resolve class_path: {cp}"
