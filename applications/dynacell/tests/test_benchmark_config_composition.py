"""Composition sanity tests for benchmark leaves."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from viscy_utils.compose import load_composed_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS = REPO_ROOT / "applications" / "dynacell" / "configs" / "benchmarks" / "virtual_staining"


TRAIN_LEAVES = [
    ("er", "celldiff"),
    ("er", "fnet3d_paper"),
    ("er", "unetvit3d"),
    ("er", "unext2"),
    ("er", "fcmae_vscyto3d_scratch"),
    ("er", "fcmae_vscyto3d_pretrained"),
    ("mito", "celldiff"),
    ("mito", "fnet3d_paper"),
    ("mito", "unetvit3d"),
    ("mito", "fcmae_vscyto3d_scratch"),
    ("mito", "fcmae_vscyto3d_pretrained"),
    ("nucleus", "celldiff"),
    ("nucleus", "fnet3d_paper"),
    ("nucleus", "unetvit3d"),
    ("membrane", "celldiff"),
    ("membrane", "fnet3d_paper"),
    ("membrane", "unetvit3d"),
]

PREDICT_LEAVES = [
    (organelle, model) for organelle in ("er", "mito", "nucleus", "membrane") for model in ("celldiff", "unetvit3d")
]
EVAL_LEAVES = PREDICT_LEAVES


@pytest.mark.parametrize("organelle,model", TRAIN_LEAVES)
def test_train_leaf_composes(organelle: str, model: str) -> None:
    leaf = BENCHMARKS / organelle / "ipsc_confocal" / model / "train.yml"
    cfg = load_composed_config(leaf)
    t = cfg["trainer"]
    assert t["accelerator"] == "gpu"
    assert t["devices"] in (1, 4)
    assert t["num_nodes"] == 1
    assert t["logger"]["class_path"] == "lightning.pytorch.loggers.WandbLogger"
    assert t["logger"]["init_args"]["project"] == "dynacell"
    assert "precision" in t


@pytest.mark.parametrize("organelle,model", PREDICT_LEAVES)
def test_predict_leaf_composes(organelle: str, model: str) -> None:
    leaf = BENCHMARKS / organelle / "ipsc_confocal" / model / "predict" / "ipsc_confocal.yml"
    cfg = load_composed_config(leaf)
    t = cfg["trainer"]
    assert t["accelerator"] == "gpu"
    assert t["devices"] == 1
    data_path = cfg["data"]["init_args"]["data_path"]
    assert "test_cropped" in data_path, f"{organelle}/{model}: data_path must point at test_cropped/, got {data_path}"


@pytest.mark.parametrize("organelle,model", EVAL_LEAVES)
def test_eval_leaf_symlink_resolves(organelle: str, model: str) -> None:
    """Every eval leaf at <org>/<train>/<model>/eval/<predset>.yaml has a
    corresponding symlink under leaf/ so Hydra can resolve leaf=<path>."""
    real = BENCHMARKS / organelle / "ipsc_confocal" / model / "eval" / "ipsc_confocal.yaml"
    link = BENCHMARKS / "leaf" / organelle / "ipsc_confocal" / model / "eval" / "ipsc_confocal.yaml"
    assert real.is_file(), f"missing canonical eval leaf: {real}"
    assert link.is_symlink(), f"missing leaf/ symlink: {link}"
    assert link.resolve() == real.resolve()
    assert link.read_text().startswith("# @package _global_")


def test_unext2_train_leaf_inherits_topology_and_logger() -> None:
    """Regression guard: unified fit.yml pins WandbLogger for a leaf that previously had no class_path."""
    leaf = BENCHMARKS / "er" / "ipsc_confocal" / "unext2" / "train.yml"
    cfg = load_composed_config(leaf)
    t = cfg["trainer"]
    assert t["devices"] == 4
    assert t["strategy"] == "ddp"
    assert t["precision"] == "16-mixed"
    assert t["max_epochs"] == 200
    assert t["logger"]["class_path"] == "lightning.pytorch.loggers.WandbLogger"
    assert t["logger"]["init_args"]["project"] == "dynacell"
    assert t["logger"]["init_args"]["name"] == "UNeXt2_iPSC_SEC61B"


def _strip_run_identity(cfg: dict) -> dict:
    """Remove fields expected to differ between scratch and pretrained leaves.

    Returns a deep-copied config with ``encoder_only``, ``ckpt_path``, and
    all per-leaf identifier/path fields removed. What remains must be
    byte-equal between the scratch and pretrained FCMAE leaves.
    """
    cfg = copy.deepcopy(cfg)
    init = cfg["model"]["init_args"]
    init.pop("encoder_only", None)
    init.pop("ckpt_path", None)
    cfg.pop("benchmark", None)
    cfg.pop("launcher", None)
    logger_init = cfg["trainer"]["logger"]["init_args"]
    logger_init.pop("name", None)
    logger_init.pop("save_dir", None)
    for cb in cfg["trainer"].get("callbacks", []):
        if cb.get("class_path", "").endswith("ModelCheckpoint"):
            cb["init_args"].pop("dirpath", None)
    return cfg


@pytest.mark.parametrize("organelle", ["er", "mito"])
def test_fcmae_pretrained_differs_from_scratch_only_in_encoder_init(organelle: str) -> None:
    """Scientific invariant: pretrained leaf equals scratch leaf modulo init.

    Guards against silent drift in lr / loss / crop / augs / model_config /
    trainer / epochs between the two FCMAE leaves — such drift would
    invalidate the pretrained-vs-scratch comparison.
    """
    scratch_leaf = BENCHMARKS / organelle / "ipsc_confocal" / "fcmae_vscyto3d_scratch" / "train.yml"
    pretrained_leaf = BENCHMARKS / organelle / "ipsc_confocal" / "fcmae_vscyto3d_pretrained" / "train.yml"
    cfg_scratch = load_composed_config(scratch_leaf)
    cfg_pretrained = load_composed_config(pretrained_leaf)

    pt_init = cfg_pretrained["model"]["init_args"]
    assert pt_init.get("encoder_only") is True
    assert pt_init.get("ckpt_path") == ("/hpc/projects/virtual_staining/models/mehta-lab/VSCyto3D/fcmae.ckpt")
    sc_init = cfg_scratch["model"]["init_args"]
    assert not sc_init.get("encoder_only")
    assert sc_init.get("ckpt_path") is None

    assert _strip_run_identity(cfg_scratch) == _strip_run_identity(cfg_pretrained)


@pytest.mark.parametrize("organelle,model", TRAIN_LEAVES)
def test_train_leaf_topology_consistency(organelle: str, model: str) -> None:
    """Regression guard: under SLURM, Lightning's SLURMEnvironment reads
    world_size from SLURM_NTASKS, not from trainer.devices. If ntasks
    mismatches devices, DDP silently runs with world_size=ntasks and
    only that many GPUs train — the rest sit idle. All three must agree:
    ``sbatch.ntasks == sbatch.gpus == sbatch.nodes × trainer.devices``.
    """
    leaf = BENCHMARKS / organelle / "ipsc_confocal" / model / "train.yml"
    cfg = load_composed_config(leaf)
    devices = cfg["trainer"]["devices"]
    sbatch = cfg["launcher"]["sbatch"]
    expected = devices * sbatch.get("nodes", 1)
    assert sbatch["gpus"] == expected, f"{organelle}/{model}: sbatch.gpus={sbatch['gpus']} ≠ {expected}"
    assert sbatch["ntasks"] == expected, f"{organelle}/{model}: sbatch.ntasks={sbatch['ntasks']} ≠ {expected}"


def test_fnet3d_paper_leaf_preserves_32true_precision() -> None:
    """FNet3D paper reproduction keeps precision=32-true (the unified fit recipe defaults to nothing)."""
    leaf = BENCHMARKS / "er" / "ipsc_confocal" / "fnet3d_paper" / "train.yml"
    cfg = load_composed_config(leaf)
    assert cfg["trainer"]["precision"] == "32-true"
    assert cfg["trainer"]["max_steps"] == 200000
    assert cfg["trainer"]["devices"] == 1
