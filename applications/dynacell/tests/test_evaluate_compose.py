"""Hydra-side compose integration tests for ``apply_dataset_ref``.

Layer 1
    Parametrized checks that composing the real ``target`` / ``predict_set``
    groups from the repo's ``_internal/`` tree, or a real grouped leaf, and then
    calling :func:`apply_dataset_ref` produces the manifest-derived ``io.*`` and
    ``pixel_metrics.spacing`` values.

Layer 2
    End-to-end wiring checks that the real ``@hydra.main`` entry points
    ``evaluate_model`` and ``precompute_gt`` call the hook before the
    heavy work.

The tests replicate the external-searchpath injection that
``dynacell.__main__._inject_external_configs`` performs in production
CLI calls by passing ``hydra.searchpath`` overrides to ``compose``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf

from dynacell.evaluation._ref_hook import apply_dataset_ref
from dynacell.evaluation.pipeline import _merge_condition, _select_conditions

from ._eval_fixtures import make_hcs_plate

_DYNACELL_ROOT = Path(__file__).resolve().parents[1]
_INTERNAL = _DYNACELL_ROOT / "configs" / "benchmarks" / "virtual_staining" / "_internal"
_SHARED_EVAL = _INTERNAL / "shared" / "eval"
_LEAF_ROOT = _INTERNAL / "leaf"

_EXPECTED_SPACING = [0.29, 0.108, 0.108]

# (organelle, manifest-target-slug, gt_channel, gt_store_suffix, seg_store_suffix, cache_suffix)
_ORGANELLE_EXPECTATIONS: dict[str, tuple[str, str, str, str, str]] = {
    "er": ("er_sec61b", "Structure", "test_cropped/SEC61B.zarr", "SEC61B_segmented_cleaned.zarr", "eval_cache/SEC61B"),
    "mito": (
        "mito_tomm20",
        "Structure",
        "test_cropped/TOMM20.zarr",
        "TOMM20_segmented_cleaned.zarr",
        "eval_cache/TOMM20",
    ),
    "nucleus": (
        "nucleus",
        "Nuclei",
        "test_cropped/cell.zarr",
        "cell_segmented_cleaned.zarr",
        "eval_cache/nucleus",
    ),
    "membrane": (
        "membrane",
        "Membrane",
        "test_cropped/cell.zarr",
        "cell_segmented_cleaned.zarr",
        "eval_cache/membrane",
    ),
}


@pytest.fixture(autouse=True)
def _clear_global_hydra(clear_global_hydra):
    """Inherit the shared Hydra reset fixture from conftest."""


def _searchpath_override() -> str:
    """Build a ``hydra.searchpath=[...]`` override pointing at the external tree.

    Mirrors ``dynacell.__main__._inject_external_configs`` so test compose
    calls see the same ``target/``, ``leaf/``, and
    ``feature_extractor/dynaclr/`` groups that production CLI calls do.
    """
    return f"hydra.searchpath=[file://{_INTERNAL},file://{_SHARED_EVAL}]"


def _compose_eval_cfg(overrides: list[str], config_name: str = "eval") -> DictConfig:
    """Compose an eval or precompute config with the external searchpath injected."""
    with initialize_config_module(config_module="dynacell.evaluation._configs", version_base="1.2"):
        cfg = compose(config_name=config_name, overrides=[*overrides, _searchpath_override()])
    return cfg


def test_default_eval_pins_morphem_to_a_hub_commit() -> None:
    """The morphem group must carry a full commit SHA: the model is trust_remote_code."""
    cfg = _compose_eval_cfg([])
    assert cfg.feature_extractor.morphem.pretrained_model_name == "CaicedoLab/MorphEm"
    assert re.fullmatch(r"[0-9a-f]{40}", cfg.feature_extractor.morphem.revision)


# -- Layer 1: compose + hook produces correct resolved values ---------------


@pytest.mark.parametrize("organelle", _ORGANELLE_EXPECTATIONS)
def test_ipsc_eval_target_composes_and_splices(organelle: str) -> None:
    """Compose a real target group + the iPSC predict set; check manifest splicing."""
    target_slug, gt_channel, gt_suffix, seg_suffix, cache_suffix = _ORGANELLE_EXPECTATIONS[organelle]

    cfg = _compose_eval_cfg([f"target={target_slug}", "predict_set=ipsc_confocal"])

    # The hook is *not* auto-invoked by compose alone.
    apply_dataset_ref(cfg)

    assert str(cfg.io.gt_path).endswith(gt_suffix)
    assert str(cfg.io.cell_segmentation_path).endswith(seg_suffix)
    assert str(cfg.io.gt_cache_dir).endswith(cache_suffix)
    assert cfg.io.gt_channel_name == gt_channel
    assert cfg.io.pred_channel_name == f"{gt_channel}_prediction"
    assert list(cfg.pixel_metrics.spacing) == _EXPECTED_SPACING


def test_nucleus_vs_membrane_share_store_but_differ_elsewhere() -> None:
    """Nucleus and membrane share cell.zarr but split on channel + cache_dir."""
    nuc = _compose_eval_cfg(
        [
            "target=nucleus",
            "predict_set=ipsc_confocal",
            "io.pred_path=/tmp/fake",
            "save.save_dir=/tmp/out",
        ]
    )
    mem = _compose_eval_cfg(
        [
            "target=membrane",
            "predict_set=ipsc_confocal",
            "io.pred_path=/tmp/fake",
            "save.save_dir=/tmp/out",
        ]
    )
    apply_dataset_ref(nuc)
    apply_dataset_ref(mem)

    assert str(nuc.io.gt_path) == str(mem.io.gt_path)
    assert str(nuc.io.gt_path).endswith("test_cropped/cell.zarr")

    assert str(nuc.io.gt_cache_dir) != str(mem.io.gt_cache_dir)
    assert str(nuc.io.gt_cache_dir).endswith("eval_cache/nucleus")
    assert str(mem.io.gt_cache_dir).endswith("eval_cache/membrane")

    assert nuc.io.gt_channel_name == "Nuclei"
    assert mem.io.gt_channel_name == "Membrane"


def test_collision_raises_with_both_paths_in_message() -> None:
    """Full ref + conflicting explicit io.gt_path raises with both paths in message."""
    cfg = OmegaConf.create(
        {
            "benchmark": {"dataset_ref": {"dataset": "aics-hipsc", "target": "sec61b"}},
            "io": {"gt_path": "/other/path.zarr"},
        }
    )
    with pytest.raises(ValueError) as exc:
        apply_dataset_ref(cfg)
    msg = str(exc.value)
    assert "/other/path.zarr" in msg
    assert "SEC61B.zarr" in msg


# -- Layer 2: real entry points invoke the hook ----------------------------


def test_evaluate_model_wires_hook(monkeypatch, tmp_path) -> None:
    """``evaluate_model`` runs ``apply_dataset_ref`` before ``evaluate_predictions``."""
    captured: list[DictConfig] = []

    def _fake_evaluate_predictions(cfg: DictConfig, *, cp_space, prediction_snapshot):
        captured.append(cfg)
        return ([], [], [])

    def _fake_save_metrics(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("dynacell.evaluation.pipeline.evaluate_predictions", _fake_evaluate_predictions)
    monkeypatch.setattr("dynacell.evaluation.pipeline.save_metrics", _fake_save_metrics)

    # Feature metrics off: evaluate_model would otherwise load the CP reference before
    # evaluate_predictions, and this test is about the dataset_ref splice only. The
    # prediction store must exist: evaluate_model fingerprints it before scoring.
    make_hcs_plate(tmp_path / "pred.zarr", "Structure_prediction", seed=0)
    cfg = _compose_eval_cfg(
        [
            "target=er_sec61b",
            "predict_set=ipsc_confocal",
            f"io.pred_path={tmp_path / 'pred.zarr'}",
            f"save.save_dir={tmp_path}",
            "compute_feature_metrics=false",
        ]
    )

    from dynacell.evaluation.pipeline import evaluate_model

    evaluate_model.__wrapped__(cfg)

    assert len(captured) == 1
    spliced = captured[0]
    assert str(spliced.io.gt_path).endswith("test_cropped/SEC61B.zarr")
    assert str(spliced.io.cell_segmentation_path).endswith("SEC61B_segmented_cleaned.zarr")
    assert spliced.io.gt_channel_name == "Structure"
    assert spliced.io.pred_channel_name == "Structure_prediction"
    assert list(spliced.pixel_metrics.spacing) == _EXPECTED_SPACING


def test_precompute_gt_wires_hook(monkeypatch, tmp_path) -> None:
    """``precompute_gt`` runs ``apply_dataset_ref`` before ``precompute_gt_artifacts``."""
    captured: list[DictConfig] = []

    def _fake_precompute_gt_artifacts(cfg: DictConfig) -> None:
        captured.append(cfg)

    monkeypatch.setattr(
        "dynacell.evaluation.precompute_cli.precompute_gt_artifacts",
        _fake_precompute_gt_artifacts,
    )

    cfg = _compose_eval_cfg(
        [
            "target=er_sec61b",
            "predict_set=ipsc_confocal",
            "io.pred_path=/tmp/fake",
            f"save.save_dir={tmp_path}",
        ]
    )

    from dynacell.evaluation.precompute_cli import precompute_gt

    precompute_gt.__wrapped__(cfg)

    assert len(captured) == 1
    spliced = captured[0]
    assert str(spliced.io.gt_path).endswith("test_cropped/SEC61B.zarr")
    assert spliced.io.gt_channel_name == "Structure"
    assert spliced.io.pred_channel_name == "Structure_prediction"
    assert list(spliced.pixel_metrics.spacing) == _EXPECTED_SPACING


# -- A549 cross-eval leaves --------------------------------------------------
#
# A549 conditions resolve via the per-condition predict_set fragment
# ``a549_mantis_<marker>_<cond>``. Grouped conditions also set
# ``dataset_ref.target`` to the gene slug so the manifest target lookup keys on
# the marker (h2b, caax, sec61b, tomm20) rather than the iPSC-side organelle key.

# organelle → (eval target group, gt_channel, marker slug, on-disk gene token)
_A549_EVAL_EXPECTATIONS: dict[str, tuple[str, str, str, str]] = {
    "er": ("er_sec61b", "Structure", "sec61b", "SEC61B"),
    "mito": ("mito_tomm20", "Structure", "tomm20", "TOMM20"),
    "nucleus": ("nucleus", "Nuclei", "h2b", "H2B"),
    "membrane": ("membrane", "Membrane", "caax", "CAAX"),
}
# (yaml condition slug, on-disk condition token)
_A549_CONDITIONS: list[tuple[str, str]] = [("mock", "mock"), ("denv", "DENV"), ("zikv", "ZIKV")]
_A549_MATRIX = [
    (organelle, cond_slug, cond_token)
    for organelle in _A549_EVAL_EXPECTATIONS
    for cond_slug, cond_token in _A549_CONDITIONS
]


@pytest.mark.parametrize("organelle,cond_slug,cond_token", _A549_MATRIX)
def test_a549_eval_target_composes_and_splices(organelle: str, cond_slug: str, cond_token: str) -> None:
    """A549 conditions resolve per-condition paths via the predict_set group.

    For all organelles, the gene-keyed ``dataset_ref.target`` override
    (h2b / caax / sec61b / tomm20) flips the manifest target lookup so
    the resolver pulls the gene-keyed entry from the
    ``a549-mantis-<marker>-<cond>`` manifest.
    """
    target_group, gt_channel, marker, gene_token = _A549_EVAL_EXPECTATIONS[organelle]

    cfg = _compose_eval_cfg(
        [
            f"target={target_group}",
            f"predict_set=a549_mantis_{marker}_{cond_slug}",
            f"benchmark.dataset_ref.target={marker}",
        ]
    )
    apply_dataset_ref(cfg)

    # Nucleus (h2b) + membrane (caax) GT now live in the merged dual store; ER/mito
    # keep their per-marker stores. The suffix reflects the on-disk store stem.
    store_stem = "dual_nucl_memb" if marker in ("caax", "h2b") else gene_token
    gt_suffix = f"{store_stem}_{cond_token}.zarr"
    seg_suffix = f"{store_stem}_{cond_token}_seg_cleaned.zarr"
    cache_suffix = f"eval_cache/{marker}_{cond_slug}"

    assert str(cfg.io.gt_path).endswith(gt_suffix), (
        f"{organelle}/{cond_slug}: cfg.io.gt_path={cfg.io.gt_path} does not end with {gt_suffix}"
    )
    assert cfg.io.gt_channel_name == gt_channel
    assert cfg.io.pred_channel_name == f"{gt_channel}_prediction"
    assert str(cfg.io.cell_segmentation_path).endswith(seg_suffix), (
        f"{organelle}/{cond_slug}: cell_segmentation_path={cfg.io.cell_segmentation_path} "
        f"does not end with {seg_suffix}"
    )
    assert str(cfg.io.gt_cache_dir).endswith(cache_suffix), (
        f"{organelle}/{cond_slug}: gt_cache_dir={cfg.io.gt_cache_dir} does not end with {cache_suffix}"
    )
    # Spacing comes from the manifest. All A549-mantis markers share the same
    # acquisition pixel size (0.174 µm Z, 0.1494 µm XY) — verified against the
    # store scale metadata for caax/h2b/sec61b/tomm20.
    spacing = list(cfg.pixel_metrics.spacing)
    assert spacing == [0.174, 0.1494, 0.1494]


# Every grouped benchmark leaf: it composes via ``leaf=``, and each of its
# conditions resolves against its dataset manifest. A condition missing its
# gene-keyed ``dataset_ref.target`` raises TargetNotFoundError here instead of at
# eval time.
_A549_DATASET = re.compile(r"a549-mantis-(?P<gene>[a-z0-9]+)-(mock|denv|zikv)(-lite)?")
_GROUPED_LEAVES = sorted(p.parent.name for p in (_LEAF_ROOT / "grouped").glob("*/eval_grouped.yaml"))


@pytest.mark.parametrize("bucket", _GROUPED_LEAVES)
def test_every_grouped_leaf_composes(bucket: str) -> None:
    """Each grouped leaf composes, and every condition splices its manifest paths."""
    cfg = _compose_eval_cfg([f"leaf=grouped/{bucket}/eval_grouped"], config_name="eval_grouped")
    selected = _select_conditions(cfg.conditions, cfg.only_conditions)
    assert selected, f"{bucket}: no conditions"
    # The splice reads and writes only these blocks; merging onto them alone keeps
    # 1164 conditions from copying the whole eval schema each.
    splice_base = OmegaConf.masked_copy(cfg, ["benchmark", "io", "pixel_metrics"])
    for name, cond in selected:
        merged = _merge_condition(splice_base, cond)
        apply_dataset_ref(merged)
        ref = merged.benchmark.dataset_ref
        assert ref.dataset, f"{bucket}/{name}: no dataset_ref.dataset"
        assert merged.io.gt_path, f"{bucket}/{name}: io.gt_path not resolved"
        assert len(merged.pixel_metrics.spacing) == 3, f"{bucket}/{name}: spacing not resolved"
        a549 = _A549_DATASET.fullmatch(ref.dataset)
        if a549 is not None:
            # The manifest target is the marker gene, not the iPSC-side organelle key.
            assert ref.target == a549.group("gene"), f"{bucket}/{name}: target {ref.target!r} for {ref.dataset}"
            assert list(merged.pixel_metrics.spacing) == [0.174, 0.1494, 0.1494], f"{bucket}/{name}"


def test_only_conditions_override_composes_on_a_grouped_leaf() -> None:
    """``only_conditions`` is in the grouped schema, so a CLI override needs no ``+``."""
    bucket = "er_ipsc_trained"
    name = "fnet3d__ipsc_trained__a549_mock"
    cfg = _compose_eval_cfg(
        [f"leaf=grouped/{bucket}/eval_grouped", f"only_conditions=[{name}]"], config_name="eval_grouped"
    )
    assert [n for n, _ in _select_conditions(cfg.conditions, cfg.only_conditions)] == [name]
