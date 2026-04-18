"""Sanity tests for benchmark leaf composition.

Each benchmark leaf composes through
:func:`viscy_utils.compose.load_composed_config` without error and
resolves the expected trainer topology keys.

The prior pre-schema equivalence tests against
``tools/LEGACY/examples_configs/`` have been removed: LEGACY was
deleted as part of the topology/trainer-recipe ownership cleanup (see
``recipes/topology/`` and the unified ``recipes/trainer/fit.yml``).
Post-landing validation now lives in the CI-gated tests below plus
``test_submit_benchmark_job.py`` sbatch renders.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from viscy_utils.compose import load_composed_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS = REPO_ROOT / "applications" / "dynacell" / "configs" / "benchmarks" / "virtual_staining"


def _strip_reserved(d: dict) -> dict:
    d.pop("launcher", None)
    d.pop("benchmark", None)
    return d


# (organelle, model_name) for every train leaf under virtual_staining/train/.
TRAIN_LEAVES = [
    ("er", "celldiff"),
    ("er", "fnet3d_paper"),
    ("er", "unetvit3d"),
    ("er", "unext2"),
    ("mito", "celldiff"),
    ("mito", "fnet3d_paper"),
    ("nucleus", "celldiff"),
    ("nucleus", "fnet3d_paper"),
    ("membrane", "celldiff"),
    ("membrane", "fnet3d_paper"),
]

# (organelle,) for every predict leaf.
PREDICT_ORGANELLES = ["er", "mito", "nucleus", "membrane"]


@pytest.mark.parametrize("organelle,model", TRAIN_LEAVES)
def test_train_leaf_composes(organelle: str, model: str) -> None:
    """Train leaf composes cleanly and resolves core trainer keys."""
    leaf = BENCHMARKS / "train" / organelle / "ipsc_confocal" / f"{model}.yml"
    cfg = _strip_reserved(load_composed_config(leaf))
    t = cfg["trainer"]
    assert t["accelerator"] == "gpu"
    assert t["devices"] in (1, 4)
    assert t["num_nodes"] == 1
    # WandbLogger is pinned by the unified fit recipe.
    assert t["logger"]["class_path"] == "lightning.pytorch.loggers.WandbLogger"
    assert t["logger"]["init_args"]["project"] == "dynacell"
    # Precision must be set explicitly by the model overlay.
    assert "precision" in t


@pytest.mark.parametrize("organelle", PREDICT_ORGANELLES)
def test_predict_leaf_composes(organelle: str) -> None:
    """Predict leaf composes cleanly and points at test_cropped data."""
    leaf = BENCHMARKS / "predict" / organelle / "ipsc_confocal" / "celldiff" / "ipsc_confocal.yml"
    cfg = _strip_reserved(load_composed_config(leaf))
    t = cfg["trainer"]
    assert t["accelerator"] == "gpu"
    assert t["devices"] == 1
    data_path = cfg["data"]["init_args"]["data_path"]
    assert "test_cropped" in data_path, f"{organelle}: data_path must point at test_cropped/, got {data_path}"


def test_unext2_train_leaf_inherits_topology_and_logger() -> None:
    """Regression guard for the latent unext2 logger bug fixed by unified fit.yml.

    Pre-refactor, the unext2 benchmark leaf composed fit_4gpu.yml which
    set no ``trainer.logger.class_path``. The leaf only supplied
    ``logger.init_args.{name, save_dir}``, producing a fragile config
    that relied on Lightning's default TensorBoardLogger. After
    unification under fit.yml, WandbLogger is pinned.
    """
    leaf = BENCHMARKS / "train" / "er" / "ipsc_confocal" / "unext2.yml"
    cfg = _strip_reserved(load_composed_config(leaf))
    t = cfg["trainer"]
    assert t["devices"] == 4
    assert t["strategy"] == "ddp"
    assert t["precision"] == "16-mixed"
    assert t["max_epochs"] == 200
    assert t["logger"]["class_path"] == "lightning.pytorch.loggers.WandbLogger"
    assert t["logger"]["init_args"]["project"] == "dynacell"
    assert t["logger"]["init_args"]["name"] == "UNeXt2_iPSC_SEC61B"


def test_fnet3d_paper_leaf_preserves_32true_precision() -> None:
    """FNet3D paper reproduction keeps precision=32-true (not inherited bf16)."""
    leaf = BENCHMARKS / "train" / "er" / "ipsc_confocal" / "fnet3d_paper.yml"
    cfg = _strip_reserved(load_composed_config(leaf))
    assert cfg["trainer"]["precision"] == "32-true"
    assert cfg["trainer"]["max_steps"] == 200000
    assert cfg["trainer"]["devices"] == 1
