"""Composition sanity tests for benchmark leaves."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

pytest.importorskip("yaml")

from dynacell._compose_hook import _dynacell_ref_resolver  # noqa: E402
from viscy_utils.compose import load_composed_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
assert (REPO_ROOT / "pyproject.toml").exists(), f"REPO_ROOT drift: {REPO_ROOT}"
BENCHMARKS = REPO_ROOT / "applications" / "dynacell" / "configs" / "benchmarks" / "virtual_staining"
FIXTURE_MANIFEST_ROOT = Path(__file__).resolve().parent / "fixtures" / "manifests"


@pytest.fixture(autouse=True)
def _fixture_manifest_root(monkeypatch):
    """Point the resolver at the on-disk fixture manifest for every test."""
    monkeypatch.setenv("DYNACELL_MANIFEST_ROOTS", str(FIXTURE_MANIFEST_ROOT))


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


@pytest.mark.parametrize("organelle,model", TRAIN_LEAVES)
def test_train_leaf_composes(organelle: str, model: str) -> None:
    leaf = BENCHMARKS / organelle / model / "ipsc_confocal" / "train.yml"
    cfg = load_composed_config(leaf)
    t = cfg["trainer"]
    assert t["accelerator"] == "gpu"
    assert t["devices"] in (1, 4)
    assert t["num_nodes"] == 1
    assert t["logger"]["class_path"] == "lightning.pytorch.loggers.WandbLogger"
    assert t["logger"]["init_args"]["project"] == "dynacell"
    assert "precision" in t


@pytest.mark.parametrize("organelle,model", PREDICT_LEAVES)
def test_predict_leaf_composes(organelle: str, model: str, monkeypatch) -> None:
    """ER leaves resolve data_path via dataset_ref; other organelles inherit hardcoded paths."""
    monkeypatch.setattr("sys.argv", ["dynacell", "predict"])
    leaf = BENCHMARKS / organelle / model / "ipsc_confocal" / "predict__ipsc_confocal.yml"
    cfg = load_composed_config(leaf, resolver=_dynacell_ref_resolver)
    t = cfg["trainer"]
    assert t["accelerator"] == "gpu"
    assert t["devices"] == 1
    data_path = cfg["data"]["init_args"]["data_path"]
    assert "test_cropped" in data_path, f"{organelle}/{model}: data_path must point at test_cropped/, got {data_path}"


@pytest.mark.parametrize("organelle,model", PREDICT_LEAVES)
def test_eval_leaf_symlink_resolves(organelle: str, model: str) -> None:
    """Every canonical eval leaf at <org>/<model>/<train_set>/eval__<predict_set>.yaml
    has a corresponding symlink under _internal/leaf/ so Hydra can resolve
    ``leaf=<path>`` from the _internal searchpath."""
    real = BENCHMARKS / organelle / model / "ipsc_confocal" / "eval__ipsc_confocal.yaml"
    link = BENCHMARKS / "_internal" / "leaf" / organelle / model / "ipsc_confocal" / "eval__ipsc_confocal.yaml"
    assert real.is_file(), f"missing canonical eval leaf: {real}"
    assert link.is_symlink(), f"missing symlink: {link}"
    assert link.resolve() == real.resolve()


def test_unext2_train_leaf_inherits_topology_and_logger() -> None:
    """Regression guard: unified fit.yml pins WandbLogger for a leaf that previously had no class_path."""
    leaf = BENCHMARKS / "er" / "unext2" / "ipsc_confocal" / "train.yml"
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
    scratch_leaf = BENCHMARKS / organelle / "fcmae_vscyto3d_scratch" / "ipsc_confocal" / "train.yml"
    pretrained_leaf = BENCHMARKS / organelle / "fcmae_vscyto3d_pretrained" / "ipsc_confocal" / "train.yml"
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
    world_size from SLURM_NTASKS and rejects bare ``--ntasks`` in favor
    of ``--ntasks-per-node``. If ntasks_per_node ≠ devices, DDP silently
    runs with the wrong world_size and only some GPUs train.
    Invariant: ``ntasks_per_node == devices`` and
    ``gpus == nodes × devices``.
    """
    leaf = BENCHMARKS / organelle / model / "ipsc_confocal" / "train.yml"
    cfg = load_composed_config(leaf)
    devices = cfg["trainer"]["devices"]
    sbatch = cfg["launcher"]["sbatch"]
    nodes = sbatch.get("nodes", 1)
    assert sbatch["ntasks_per_node"] == devices, (
        f"{organelle}/{model}: ntasks_per_node={sbatch['ntasks_per_node']} ≠ devices={devices}"
    )
    assert sbatch["gpus"] == nodes * devices, (
        f"{organelle}/{model}: gpus={sbatch['gpus']} ≠ nodes×devices={nodes * devices}"
    )


def test_fnet3d_paper_leaf_preserves_32true_precision() -> None:
    """FNet3D paper reproduction keeps precision=32-true (the unified fit recipe defaults to nothing)."""
    leaf = BENCHMARKS / "er" / "fnet3d_paper" / "ipsc_confocal" / "train.yml"
    cfg = load_composed_config(leaf)
    assert cfg["trainer"]["precision"] == "32-true"
    assert cfg["trainer"]["max_steps"] == 200000
    assert cfg["trainer"]["devices"] == 1


# -- dataset_ref resolver integration tests -------------------------------


def test_migrated_er_train_resolves_to_manifest_paths(monkeypatch) -> None:
    """Full dataset_ref on ER fit leaf splices train store + channels from fixture."""
    monkeypatch.setattr("sys.argv", ["dynacell", "fit"])
    leaf = BENCHMARKS / "er" / "celldiff" / "ipsc_confocal" / "train.yml"
    cfg = load_composed_config(leaf, resolver=_dynacell_ref_resolver)
    ia = cfg["data"]["init_args"]
    assert ia["data_path"].endswith("train/SEC61B.zarr")
    assert ia["source_channel"] == "Phase3D"
    assert ia["target_channel"] == "Structure"
    assert cfg["benchmark"]["spacing"] == [0.29, 0.108, 0.108]


def test_migrated_er_predict_resolves_to_test_store(monkeypatch) -> None:
    """Full dataset_ref on ER predict leaf splices test_cropped store + channels."""
    monkeypatch.setattr("sys.argv", ["dynacell", "predict"])
    leaf = BENCHMARKS / "er" / "celldiff" / "ipsc_confocal" / "predict__ipsc_confocal.yml"
    cfg = load_composed_config(leaf, resolver=_dynacell_ref_resolver)
    ia = cfg["data"]["init_args"]
    assert ia["data_path"].endswith("test_cropped/SEC61B.zarr")
    assert ia["source_channel"] == "Phase3D"
    assert ia["target_channel"] == "Structure"


def test_collision_raises_with_both_paths_in_message(tmp_path, monkeypatch) -> None:
    """Leaf with full dataset_ref + conflicting explicit data_path raises ValueError."""
    monkeypatch.setattr("sys.argv", ["dynacell", "fit"])
    leaf_content = {
        "benchmark": {"dataset_ref": {"dataset": "aics-hipsc", "target": "sec61b"}},
        "data": {
            "class_path": "viscy_data.hcs.HCSDataModule",
            "init_args": {"data_path": "/tmp/some/other/path.zarr"},
        },
    }
    leaf = tmp_path / "collide.yml"
    leaf.write_text(yaml.dump(leaf_content))
    with pytest.raises(ValueError) as exc:
        load_composed_config(leaf, resolver=_dynacell_ref_resolver)
    msg = str(exc.value)
    assert "/tmp/some/other/path.zarr" in msg
    assert "SEC61B.zarr" in msg


@pytest.mark.parametrize("organelle,model", [("mito", "celldiff"), ("mito", "unetvit3d")])
def test_mito_partial_ref_noop_parity(organelle: str, model: str, monkeypatch) -> None:
    """Mito leaves inherit partial dataset_ref; resolver must leave the composed dict unchanged."""
    monkeypatch.setattr("sys.argv", ["dynacell", "fit"])
    leaf = BENCHMARKS / organelle / model / "ipsc_confocal" / "train.yml"
    without = load_composed_config(leaf)
    with_resolver = load_composed_config(leaf, resolver=_dynacell_ref_resolver)
    assert without == with_resolver
    assert with_resolver["data"]["init_args"]["data_path"].endswith("TOMM20.zarr")


@pytest.mark.parametrize("organelle,model", [("mito", "celldiff"), ("mito", "unetvit3d")])
def test_mito_predict_partial_ref_noop_parity(organelle: str, model: str, monkeypatch) -> None:
    """Mito predict leaves also inherit partial dataset_ref from predict_sets; resolver is no-op."""
    monkeypatch.setattr("sys.argv", ["dynacell", "predict"])
    leaf = BENCHMARKS / organelle / model / "ipsc_confocal" / "predict__ipsc_confocal.yml"
    without = load_composed_config(leaf)
    with_resolver = load_composed_config(leaf, resolver=_dynacell_ref_resolver)
    assert without == with_resolver


def test_synthetic_target_only_partial_ref_is_noop(tmp_path) -> None:
    """Target-only dataset_ref (no dataset) composes without touching data fields."""
    leaf_content = {
        "benchmark": {"dataset_ref": {"target": "sec61b"}},
        "data": {"init_args": {"data_path": "/kept.zarr"}},
    }
    leaf = tmp_path / "partial.yml"
    leaf.write_text(yaml.dump(leaf_content))
    cfg = load_composed_config(leaf, resolver=_dynacell_ref_resolver)
    assert cfg["data"]["init_args"]["data_path"] == "/kept.zarr"
