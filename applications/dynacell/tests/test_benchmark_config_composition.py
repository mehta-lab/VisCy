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

# celldiff predict leaves are split per inference method; unetvit3d has a single file.
PREDICT_LEAF_FILES = [
    (organelle, "celldiff", f"predict__ipsc_confocal__{method}.yml")
    for organelle in ("er", "mito", "nucleus", "membrane")
    for method in ("denoise", "iterative", "sliding_window")
] + [(organelle, "unetvit3d", "predict__ipsc_confocal.yml") for organelle in ("er", "mito", "nucleus", "membrane")]


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


@pytest.mark.parametrize("organelle,model,predict_file", PREDICT_LEAF_FILES)
def test_predict_leaf_composes(organelle: str, model: str, predict_file: str, monkeypatch) -> None:
    """Predict leaves compose and point at the test_cropped store."""
    monkeypatch.setattr("sys.argv", ["dynacell", "predict"])
    leaf = BENCHMARKS / organelle / model / "ipsc_confocal" / predict_file
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


# Per-model HCS data hparams that must still compose after splitting
# model_overlays/<model>_fit.yml into model+trainer + data_overlays/<model>_fit.yml.
# Guards against silent key drops or moves between the two halves.
_EXPECTED_DATA_HPARAMS = {
    "celldiff": {"batch_size": 4, "z_window_size": 13, "yx_patch_size": [512, 512], "num_workers": 4},
    "unetvit3d": {"batch_size": 4, "z_window_size": 13, "yx_patch_size": [512, 512], "num_workers": 4},
    "fcmae_vscyto3d_scratch": {"batch_size": 32, "z_window_size": 20, "yx_patch_size": [384, 384], "num_workers": 4},
    "fcmae_vscyto3d_pretrained": {"batch_size": 32, "z_window_size": 20, "yx_patch_size": [384, 384], "num_workers": 4},
    "fnet3d_paper": {"batch_size": 48, "z_window_size": 32, "yx_patch_size": [64, 64], "num_workers": 8},
    "unext2": {"batch_size": 32, "z_window_size": 20, "yx_patch_size": [384, 384], "num_workers": 8},
}


@pytest.mark.parametrize("organelle,model", TRAIN_LEAVES)
def test_data_overlay_split_preserves_hparams(organelle: str, model: str) -> None:
    """Every train leaf still composes its model's expected data hparams.

    After moving ``data.init_args.*`` out of ``model_overlays/<model>_fit.yml``
    into ``data_overlays/<model>_fit.yml``, each single-store train leaf
    must compose to the same (batch_size, z_window_size, yx_patch_size,
    num_workers) as before — otherwise the split silently dropped or
    moved a field.
    """
    leaf = BENCHMARKS / organelle / model / "ipsc_confocal" / "train.yml"
    cfg = load_composed_config(leaf)
    ia = cfg["data"]["init_args"]
    expected = _EXPECTED_DATA_HPARAMS[model]
    for key, value in expected.items():
        assert ia[key] == value, f"{organelle}/{model}: data.init_args.{key} = {ia[key]!r}, expected {value!r}"
    # GPU augmentations must land on every model (the list-replacement path
    # for unext2 and the straight copy for the others). fnet3d_paper has
    # BatchedRandFlipd pair (no val_gpu_augmentations); the others have a
    # longer affine+intensity stack.
    assert ia["gpu_augmentations"], f"{organelle}/{model}: gpu_augmentations missing after split"


# -- dataset_ref resolver integration tests -------------------------------


# Each migrated organelle → (train store suffix, test store suffix, target_channel).
# Every entry here must have a matching block in the fixture manifest at
# tests/fixtures/manifests/aics-hipsc/manifest.yaml.
_MIGRATED_TARGET_INFO = {
    "er": ("train/SEC61B.zarr", "test_cropped/SEC61B.zarr", "Structure"),
    "mito": ("train/TOMM20.zarr", "test_cropped/TOMM20.zarr", "Structure"),
    "nucleus": ("train/cell.zarr", "test_cropped/cell.zarr", "Nuclei"),
    "membrane": ("train/cell.zarr", "test_cropped/cell.zarr", "Membrane"),
}

_MIGRATED_TRAIN_LEAVES = [(o, m) for o, m in TRAIN_LEAVES if o in _MIGRATED_TARGET_INFO]
_MIGRATED_PREDICT_LEAVES = [(o, m) for o, m in PREDICT_LEAVES if o in _MIGRATED_TARGET_INFO]


@pytest.mark.parametrize("organelle,model", _MIGRATED_TRAIN_LEAVES)
def test_migrated_target_train_resolves_to_manifest_paths(organelle: str, model: str, monkeypatch) -> None:
    """Full dataset_ref on a migrated fit leaf splices train store + channels from the fixture manifest."""
    monkeypatch.setattr("sys.argv", ["dynacell", "fit"])
    leaf = BENCHMARKS / organelle / model / "ipsc_confocal" / "train.yml"
    cfg = load_composed_config(leaf, resolver=_dynacell_ref_resolver)
    train_store, _, target_channel = _MIGRATED_TARGET_INFO[organelle]
    ia = cfg["data"]["init_args"]
    assert ia["data_path"].endswith(train_store)
    assert ia["source_channel"] == "Phase3D"
    assert ia["target_channel"] == target_channel


# celldiff predict leaves are split per inference method; unetvit3d has a single
# file. The test store + channel resolution is identical across the three
# celldiff method variants (they differ only in model/predict-method fields),
# so we pick sliding_window as the representative for celldiff.
_MIGRATED_PREDICT_FILES = {
    "celldiff": "predict__ipsc_confocal__sliding_window.yml",
    "unetvit3d": "predict__ipsc_confocal.yml",
}


@pytest.mark.parametrize("organelle,model", _MIGRATED_PREDICT_LEAVES)
def test_migrated_target_predict_resolves_to_test_store(organelle: str, model: str, monkeypatch) -> None:
    """Full dataset_ref on a migrated predict leaf splices the test store + channels."""
    monkeypatch.setattr("sys.argv", ["dynacell", "predict"])
    leaf = BENCHMARKS / organelle / model / "ipsc_confocal" / _MIGRATED_PREDICT_FILES[model]
    cfg = load_composed_config(leaf, resolver=_dynacell_ref_resolver)
    _, test_store, target_channel = _MIGRATED_TARGET_INFO[organelle]
    ia = cfg["data"]["init_args"]
    assert ia["data_path"].endswith(test_store)
    assert ia["source_channel"] == "Phase3D"
    assert ia["target_channel"] == target_channel


# -- Stage 6: A549 cross-eval predict leaves ------------------------------

# The a549_mantis predict leaf was split per treatment condition into
# predict__a549_mantis_{mock,denv,zikv}.yml. Each condition points at the
# condition-pooled test store in mantis_v1/test/. Per-organelle the gene
# slug differs: er → SEC61B, mito → TOMM20, nucleus → H2B, membrane → CAAX.
# The condition flag in the store filename is mock → "_mock", denv →
# "_DENV", zikv → "_ZIKV".
#
# Each cross-eval cell → (organelle, model, condition, gene_slug,
# target_channel, store_filename). dataset_ref.dataset is
# f"a549-mantis-{gene_slug}-{condition}", dataset_ref.target is gene_slug,
# and experiment_id is
# f"{organelle}__ipsc_confocal__{model}__a549_mantis_{gene_slug}_{condition}".
_A549_GENE_INFO = {
    "er": ("sec61b", "SEC61B", "Structure"),
    "mito": ("tomm20", "TOMM20", "Structure"),
    "nucleus": ("h2b", "H2B", "Nuclei"),
    "membrane": ("caax", "CAAX", "Membrane"),
}
# Condition → suffix in the test store filename.
_A549_CONDITION_SUFFIX = {"mock": "_mock", "denv": "_DENV", "zikv": "_ZIKV"}

_A549_PREDICT_EXPECTATIONS = [
    (
        organelle,
        model,
        condition,
        gene_slug,
        target_channel,
        f"mantis_v1/test/{gene_upper}{_A549_CONDITION_SUFFIX[condition]}.ozx",
    )
    for organelle, (gene_slug, gene_upper, target_channel) in _A549_GENE_INFO.items()
    for model in ("celldiff", "unetvit3d")
    for condition in ("mock", "denv", "zikv")
]


@pytest.mark.parametrize(
    "organelle,model,condition,gene_slug,target_channel,gt_test_substring",
    _A549_PREDICT_EXPECTATIONS,
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_a549_predict_leaf_composes(
    organelle: str,
    model: str,
    condition: str,
    gene_slug: str,
    target_channel: str,
    gt_test_substring: str,
    monkeypatch,
) -> None:
    """Stage 6 cross-eval predict leaves compose against a549 manifests.

    Verifies for each cell:
    - data.init_args.data_path resolves to the condition's test store.
    - target_channel is a bare string (not a list) per _compose_hook.py.
    - dataset_ref.{dataset,target} carry through composition (proves the
      gene-keyed predict_set + targets overlays took effect).
    - experiment_id reflects the cross-eval pairing (per condition).
    - sbatch.constraint is unset (None): predict leaves use
      hardware_predict_any_gpu (any GPU), not the H200 pin.
    """
    monkeypatch.setattr("sys.argv", ["dynacell", "predict"])
    leaf = BENCHMARKS / organelle / model / "ipsc_confocal" / f"predict__a549_mantis_{condition}.yml"
    assert leaf.is_file(), f"missing predict leaf: {leaf}"
    cfg = load_composed_config(leaf, resolver=_dynacell_ref_resolver)

    ia = cfg["data"]["init_args"]
    assert ia["data_path"].endswith(gt_test_substring), (
        f"{organelle}/{model}/{condition}: data_path={ia['data_path']!r} does not end with {gt_test_substring!r}"
    )
    assert ia["source_channel"] == "Phase3D"
    assert ia["target_channel"] == target_channel  # bare string, not [list]

    bench = cfg["benchmark"]
    assert bench["dataset_ref"]["dataset"] == f"a549-mantis-{gene_slug}-{condition}"
    assert bench["dataset_ref"]["target"] == gene_slug
    assert bench["experiment_id"] == f"{organelle}__ipsc_confocal__{model}__a549_mantis_{gene_slug}_{condition}"

    # Single-GPU predict topology, any GPU (no vendor constraint pinned).
    assert cfg["launcher"]["sbatch"].get("constraint") is None


def test_manifest_spacing_propagates(monkeypatch) -> None:
    """Resolver exposes manifest spacing via benchmark.spacing on composed fit configs."""
    monkeypatch.setattr("sys.argv", ["dynacell", "fit"])
    leaf = BENCHMARKS / "er" / "celldiff" / "ipsc_confocal" / "train.yml"
    cfg = load_composed_config(leaf, resolver=_dynacell_ref_resolver)
    assert cfg["benchmark"]["spacing"] == [0.29, 0.108, 0.108]


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


# -- joint train leaf (Stage 7) ------------------------------------------


def test_joint_train_leaf_composes() -> None:
    """First joint train leaf — BatchedConcatDataModule wrapping two
    HCSDataModule children (ipsc SEC61B + a549_mantis pooled SEC61B).

    Joint leaves bypass the single-dataset resolver: no benchmark.dataset_ref.
    The data block is authored inline; the base chain composes only model
    + launcher overlays. Topology is single H200 single-GPU — the leaf keeps
    the paper baseline pattern so iPSC-only and joint runs are apples-to-apples.
    """
    leaf = BENCHMARKS / "er" / "celldiff" / "joint_ipsc_confocal_a549_mantis" / "train.yml"
    assert leaf.is_file(), f"joint leaf missing: {leaf}"
    cfg = load_composed_config(leaf)

    # No top-level keys starting with `_` survive composition. The leaf
    # uses `_hcs_init_args:` as a YAML merge anchor; LightningCLI rejects
    # unknown top-level keys, so `load_composed_config` must strip these.
    leaked = [k for k in cfg if k.startswith("_")]
    assert not leaked, f"private anchor keys leaked into composed config: {leaked}"

    # Topology: single GPU on H200 (hardware_h200_single.yml), no DDP override.
    t = cfg["trainer"]
    assert t["accelerator"] == "gpu"
    assert t.get("strategy", "auto") != "ddp"
    assert t["devices"] == 1
    assert t["num_nodes"] == 1
    assert t["precision"] == "bf16-mixed"

    # Joint leaves must not carry dataset_ref (the resolver is scalar).
    assert "dataset_ref" not in cfg.get("benchmark", {})

    # Data: BatchedConcatDataModule with two HCSDataModule children.
    data = cfg["data"]
    assert data["class_path"] == "viscy_data.BatchedConcatDataModule"
    children = data["init_args"]["data_modules"]
    assert len(children) == 2
    for child in children:
        assert child["class_path"] == "viscy_data.hcs.HCSDataModule"
        ia = child["init_args"]
        # Anchor-shared hparams reach each child (bare-string channels).
        assert ia["source_channel"] == "Phase3D"
        assert ia["target_channel"] == "Structure"
        assert ia["z_window_size"] == 13
        # batch_size=2 (not the celldiff_fit.yml default of 4): BatchedConcatDataModule
        # does NOT divide by num_samples (see CLAUDE.md), so the product is the
        # effective per-step sample count.
        assert ia["batch_size"] == 2
        assert ia["yx_patch_size"] == [512, 512]
        assert ia["gpu_augmentations"], "gpu_augmentations missing"
        assert ia["normalizations"], "normalizations missing"
        assert ia["augmentations"], "augmentations missing"

    # Child ordering + paths.
    assert children[0]["init_args"]["data_path"].endswith("ipsc/dataset_v4/train/SEC61B.zarr")
    assert children[1]["init_args"]["data_path"].endswith("a549/mantis_v1/train/SEC61B_all.zarr")

    # Launcher: single GPU matches topology, SLURM invariant holds.
    assert cfg["launcher"]["mode"] == "fit"
    sbatch = cfg["launcher"]["sbatch"]
    nodes = sbatch.get("nodes", 1)
    assert sbatch["ntasks_per_node"] == t["devices"]
    assert sbatch["gpus"] == nodes * t["devices"]


def test_joint_train_smoke_leaf_composes() -> None:
    """Smoke sibling of the joint train leaf — single H200, no DDP, iPSC test48.

    The smoke leaf exists because submit_benchmark_job's --override parser
    cannot index into list elements (e.g. data.init_args.data_modules.0...).
    Pre-swapping data_paths in a sibling leaf is the supported alternative.
    """
    leaf = BENCHMARKS / "er" / "celldiff" / "joint_ipsc_confocal_a549_mantis" / "train_smoke.yml"
    assert leaf.is_file(), f"joint smoke leaf missing: {leaf}"
    cfg = load_composed_config(leaf)

    leaked = [k for k in cfg if k.startswith("_")]
    assert not leaked, f"private anchor keys leaked into composed config: {leaked}"

    # Topology: single GPU, no DDP override (single_gpu.yml from celldiff_fit wins).
    t = cfg["trainer"]
    assert t["accelerator"] == "gpu"
    assert t["devices"] == 1
    assert t.get("strategy", "auto") != "ddp"
    assert t["precision"] == "bf16-mixed"

    # Logger disabled at the leaf level; consumers don't need --override.
    # LearningRateMonitor (recipe default) is dropped because it requires a logger.
    assert t["logger"] is False
    callback_classes = [c["class_path"] for c in t["callbacks"]]
    assert "lightning.pytorch.callbacks.LearningRateMonitor" not in callback_classes

    # Joint leaves bypass dataset_ref.
    assert "dataset_ref" not in cfg.get("benchmark", {})

    # Data: BatchedConcatDataModule, two children, small zarrs.
    data = cfg["data"]
    assert data["class_path"] == "viscy_data.BatchedConcatDataModule"
    children = data["init_args"]["data_modules"]
    assert len(children) == 2
    for child in children:
        assert child["class_path"] == "viscy_data.hcs.HCSDataModule"
        ia = child["init_args"]
        assert ia["source_channel"] == ["Phase3D"]
        assert ia["target_channel"] == ["Structure"]
        assert ia["z_window_size"] == 13
        # batch_size=1 keeps the smoke under a single H200's memory budget;
        # train.yml uses batch_size=4 across 4 GPUs.
        assert ia["batch_size"] == 1
        assert ia["gpu_augmentations"], "gpu_augmentations missing"

    # iPSC child: test48 zarr (smoke-sized). a549 child: pooled SEC61B
    # all-conditions train store (mantis_v1/train/SEC61B_all.zarr).
    assert children[0]["init_args"]["data_path"].endswith("SEC61B_test48.zarr")
    assert children[1]["init_args"]["data_path"].endswith("a549/mantis_v1/train/SEC61B_all.zarr")

    # Launcher: single GPU on H200, smoke-sized wall, SLURM invariant holds.
    assert cfg["launcher"]["mode"] == "fit"
    sbatch = cfg["launcher"]["sbatch"]
    nodes = sbatch.get("nodes", 1)
    assert sbatch["ntasks_per_node"] == t["devices"]
    assert sbatch["gpus"] == nodes * t["devices"]
    assert sbatch.get("constraint") == "h200"
    # Smoke wall is bounded so a smoke job cannot sit on a multi-day allocation.
    assert sbatch["time"] == "00:30:00"


# Lock-in for the H100/H200 restriction baked into hardware_4gpu.yml.
# 4-GPU FCMAE/UNeXt2 leaves train at large spatial patches where a single
# DDP rank needs 30-50 GB, so A40/A6000/L40S (48 GB) leave no headroom;
# A100 is excluded separately due to repeat NCCL coordination hangs on this
# cluster's A100 partition. Every 4-GPU train leaf inherits this constraint
# by default. Leaves needing other cards must explicitly opt out via
# `--override launcher.sbatch.constraint=null`.
_HARDWARE_4GPU_CONSTRAINT = "h100|h200"


def _all_train_leaves() -> list[Path]:
    """All ``train*.yml`` leaves under benchmarks/virtual_staining/ except _internal/."""
    return sorted(p for p in BENCHMARKS.rglob("train*.yml") if "_internal" not in p.parts)


@pytest.mark.parametrize("leaf", _all_train_leaves(), ids=lambda p: str(p.relative_to(BENCHMARKS)))
def test_4gpu_train_leaves_inherit_a100_exclude(leaf: Path) -> None:
    """Every 4-GPU train leaf must inherit the A100-exclude constraint.

    Data-driven: walks every train leaf under virtual_staining/ and
    skips single-GPU leaves; for 4-GPU leaves, asserts the constraint.
    Adding a new 4-GPU leaf without the override picks this up
    automatically.
    """
    cfg = load_composed_config(leaf)
    if cfg["trainer"]["devices"] != 4:
        pytest.skip(f"single-GPU leaf: {leaf.relative_to(BENCHMARKS)}")
    constraint = cfg["launcher"]["sbatch"].get("constraint")
    assert constraint == _HARDWARE_4GPU_CONSTRAINT, (
        f"{leaf.relative_to(BENCHMARKS)}: 4-GPU leaf has constraint={constraint!r}, "
        f"expected {_HARDWARE_4GPU_CONSTRAINT!r}. If this leaf must run on A100, "
        f"override with `--override launcher.sbatch.constraint=null` instead of "
        f"unsetting the profile default."
    )
