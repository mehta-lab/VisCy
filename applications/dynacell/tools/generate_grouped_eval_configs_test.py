"""Smoke tests for ``generate_grouped_eval_configs.py``.

Tests fall into three groups:

1. Pure-Python filename grammar (no dependencies).
2. Live-data checks (require the dynacell training tree on disk; marked
   with ``@pytest.mark.requires_data``).
3. Real composition + resolver check per generated leaf (requires the
   composed eval base config + dynacell Hydra search path).

Run all groups::

    uv run pytest applications/dynacell/tools/generate_grouped_eval_configs_test.py -v

Skip data tests::

    uv run pytest applications/dynacell/tools/generate_grouped_eval_configs_test.py \
        -v -m 'not requires_data'
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The tools/ directory is not a Python package; add it to sys.path so the
# generator module is importable by short name.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from generate_grouped_eval_configs import (  # noqa: E402
    _DYNACELL_ROOT,
    _LEAF_OUT_ROOT,
    ParsedZarr,
    _is_ablation_track_zarr,
    benchmark_dataset_ref,
    build_leaf_yaml,
    parse_zarr_name,
    pred_cache_dir_for,
    save_dir_for,
    walk_predictions,
)

# ---------------------------------------------------------------------------
# 1. Grammar dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel, expect",
    [
        (
            "ipsc/predictions/sec61b_fnet3d_paper.zarr",
            ("er", "fnet3d_paper", None, "ipsc_trained", "ipsc", None, False),
        ),
        (
            "ipsc/predictions/sec61b_fnet3d_paper_jointtrained.zarr",
            ("er", "fnet3d_paper", None, "joint", "ipsc", None, False),
        ),
        (
            "ipsc/predictions/sec61b_fnet3d_paper_a549trained.zarr",
            ("er", "fnet3d_paper", None, "a549_trained", "ipsc", None, False),
        ),
        (
            "a549/predictions/sec61b_fnet3d_paper__sec61b_mock.zarr",
            ("er", "fnet3d_paper", None, "ipsc_trained", "a549", "mock", True),
        ),
        (
            "a549/predictions/sec61b_fnet3d_paper_jointtrained_mock.zarr",
            ("er", "fnet3d_paper", None, "joint", "a549", "mock", False),
        ),
        (
            "a549/predictions/sec61b_fnet3d_paper_a549trained_mock.zarr",
            ("er", "fnet3d_paper", None, "a549_trained", "a549", "mock", False),
        ),
        (
            "ipsc/predictions/sec61b_celldiff_iterative.zarr",
            ("er", "celldiff", "iterative", "ipsc_trained", "ipsc", None, False),
        ),
        (
            "ipsc/predictions/memb_celldiff_r2_sliding_window.zarr",
            (
                "membrane",
                "celldiff_r2",
                "sliding_window",
                "ipsc_trained",
                "ipsc",
                None,
                False,
            ),
        ),
        (
            "a549/predictions/tomm20_celldiff_r2_iterative__tomm20_mock.zarr",
            ("mitochondria", "celldiff_r2", "iterative", "ipsc_trained", "a549", "mock", True),
        ),
        (
            "ipsc/joint_predictions/sec61b_celldiff_r2.zarr",
            ("er", "celldiff_r2", None, "joint", "ipsc", None, False),
        ),
        (
            "a549/joint_predictions/memb_celldiff_r2_denv.zarr",
            ("membrane", "celldiff_r2", None, "joint", "a549", "denv", False),
        ),
        (
            "a549/joint_predictions/nucl_fnet3d_paper_jointtrained_mock.zarr",
            ("nucleus", "fnet3d_paper", None, "joint", "a549", "mock", False),
        ),
        (
            "ipsc/predictions/memb_fcmae_vscyto3d_pretrained_jointtrained.zarr",
            ("membrane", "fcmae_vscyto3d_pretrained", None, "joint", "ipsc", None, False),
        ),
        (
            "a549/predictions/memb_fcmae_vscyto3d_scratch_a549trained_zikv.zarr",
            (
                "membrane",
                "fcmae_vscyto3d_scratch",
                None,
                "a549_trained",
                "a549",
                "zikv",
                False,
            ),
        ),
        # pix2pix3d_unetvit: iPSC test (model code-name registration).
        (
            "ipsc/predictions/nucl_pix2pix3d_unetvit.zarr",
            ("nucleus", "pix2pix3d_unetvit", None, "ipsc_trained", "ipsc", None, False),
        ),
        (
            "ipsc/predictions/nucl_pix2pix3d_unetvit_a549trained.zarr",
            ("nucleus", "pix2pix3d_unetvit", None, "a549_trained", "ipsc", None, False),
        ),
        # pix2pix3d_unetvit A549: hybrid legacy `__<gene>_<cond>` where the gene
        # marker differs from the organelle prefix (nucleus->h2b, membrane->caax).
        (
            "a549/joint_predictions/nucl_pix2pix3d_unetvit__h2b_mock.zarr",
            ("nucleus", "pix2pix3d_unetvit", None, "joint", "a549", "mock", True),
        ),
        (
            "a549/predictions/nucl_pix2pix3d_unetvit_a549trained__h2b_zikv.zarr",
            ("nucleus", "pix2pix3d_unetvit", None, "a549_trained", "a549", "zikv", True),
        ),
        (
            "a549/predictions/memb_pix2pix3d_unetvit__caax_denv.zarr",
            ("membrane", "pix2pix3d_unetvit", None, "ipsc_trained", "a549", "denv", True),
        ),
    ],
)
def test_parse_zarr_name(rel: str, expect: tuple) -> None:
    """Grammar dispatch covers all seven canonical patterns + variants."""
    fake_root = Path("/fake/root")
    parsed = parse_zarr_name(fake_root / rel, dynacell_root=fake_root)
    assert (
        parsed.organelle,
        parsed.model,
        parsed.variant,
        parsed.train_set,
        parsed.test_set,
        parsed.condition,
        parsed.is_legacy_form,
    ) == expect


def test_parse_zarr_name_unknown_raises() -> None:
    """Unknown grammar must raise ValueError."""
    fake_root = Path("/fake/root")
    with pytest.raises(ValueError, match="organelle prefix"):
        parse_zarr_name(fake_root / "ipsc/predictions/unknown_org_fnet3d_paper.zarr", dynacell_root=fake_root)


def test_parse_zarr_name_unknown_model_raises() -> None:
    """Unknown model code-name must raise ValueError."""
    fake_root = Path("/fake/root")
    with pytest.raises(ValueError, match="unknown model"):
        parse_zarr_name(fake_root / "ipsc/predictions/sec61b_madeup_model.zarr", dynacell_root=fake_root)


def test_parse_zarr_name_unknown_celldiff_variant_raises() -> None:
    """Unknown CellDiff variant must raise ValueError."""
    fake_root = Path("/fake/root")
    with pytest.raises(ValueError, match="unknown CellDiff variant"):
        parse_zarr_name(fake_root / "ipsc/predictions/sec61b_celldiff_fakevariant.zarr", dynacell_root=fake_root)


def test_parse_zarr_name_legacy_gene_mismatch_raises() -> None:
    """Legacy `__<gene>` must match the organelle's marker (nucleus->h2b), not just parse."""
    fake_root = Path("/fake/root")
    with pytest.raises(ValueError, match="gene mismatch"):
        parse_zarr_name(fake_root / "a549/predictions/nucl_fnet3d_paper__caax_mock.zarr", dynacell_root=fake_root)


# ---------------------------------------------------------------------------
# Registry drift guard (this module's _CODE_TO_PAPER vs the runtime resolver)
# ---------------------------------------------------------------------------


def test_paper_key_maps_agree_on_overlap() -> None:
    """The campaign map and the runtime resolver must never assign DIFFERENT paper keys.

    ``save_paths.PAPER_KEY`` (single-condition submitter + paper aggregation) and
    this module's ``_CODE_TO_PAPER`` (grouped campaign) are separate maps with
    intentionally different *membership* — but where they overlap they must agree,
    or eval outputs land in mismatched dirs. Only documented differences are waived.
    """
    from generate_grouped_eval_configs import _CODE_TO_PAPER

    from dynacell.evaluation.save_paths import PAPER_KEY

    # Documented intentional difference: the grouped campaign keeps `celldiff`
    # literal; the runtime resolver collapses it to `celldiff_iterative`.
    waivers = {"celldiff"}
    disagreements = {
        m: (PAPER_KEY[m], _CODE_TO_PAPER[m])
        for m in set(PAPER_KEY) & set(_CODE_TO_PAPER)
        if m not in waivers and PAPER_KEY[m] != _CODE_TO_PAPER[m]
    }
    assert not disagreements, f"paper-key drift between save_paths.PAPER_KEY and _CODE_TO_PAPER: {disagreements}"


def test_deterministic_models_known_to_runtime_resolver() -> None:
    """Every deterministic campaign model must also be registered in the runtime resolver.

    Catches "added a model to the generator parser but forgot ``save_paths.PAPER_KEY``"
    — the asymmetry that let pix2pix3d slip the instance-AP track.
    """
    from generate_grouped_eval_configs import _DETERMINISTIC_MODELS

    from dynacell.evaluation.save_paths import PAPER_KEY

    missing = [m for m in _DETERMINISTIC_MODELS if m not in PAPER_KEY]
    assert not missing, f"deterministic campaign models absent from save_paths.PAPER_KEY: {missing}"


@pytest.mark.parametrize(
    "name, expect",
    [
        # dual nucleus+membrane predicts (own track; ``dual_`` prefix).
        ("dual_nucl_memb_fcmae_vscyto3d_pretrained_cytoland_mock.zarr", True),
        ("dual_nucl_memb_fcmae_vscyto3d_pretrained_infectionft.zarr", True),
        # no-FT ablations (Track A/B infixes).
        ("memb_fcmae_vscyto3d_pretrained_randinit_zikv.zarr", True),
        ("tomm20_fcmae_vscyto3d_pretrained_randinit.zarr", True),
        ("sec61b_fcmae_vscyto3d_pretrained_cytoland_denv.zarr", True),
        ("nucl_fcmae_vscyto3d_pretrained_infectionft_mock.zarr", True),
        # FT-combined ablations covered via substring (cytolandft / infectionft_dynacellft).
        ("memb_vscyto3d_cytolandft_a549trained_mock.zarr", True),
        ("memb_vscyto3d_infectionft_dynacellft_a549trained_mock.zarr", True),
        # in-scope campaign zarrs must NOT be flagged.
        ("memb_fcmae_vscyto3d_pretrained_a549trained_mock.zarr", False),
        ("tomm20_fcmae_vscyto3d_pretrained_a549trained.zarr", False),
        ("sec61b_celldiff_r2_iterative__sec61b_mock.zarr", False),
        ("nucl_fnet3d_paper_jointtrained_denv.zarr", False),
    ],
)
def test_is_ablation_track_zarr(name: str, expect: bool) -> None:
    """Ablation / dual prediction families are recognized; campaign zarrs are not."""
    assert _is_ablation_track_zarr(name) is expect


# ---------------------------------------------------------------------------
# 2. Save_dir + dataset_ref derivation
# ---------------------------------------------------------------------------


def _make(rel: str) -> ParsedZarr:
    return parse_zarr_name(Path("/fake/root") / rel, dynacell_root=Path("/fake/root"))


def test_save_dir_canonical_ipsc_ipsc_trained() -> None:
    """iPSC-trained iPSC-test save_dir → evaluations_with_embeddings/eval_<paper>_<organelle>."""
    parsed = _make("ipsc/predictions/sec61b_fnet3d_paper.zarr")
    sd = save_dir_for(parsed, dynacell_root=Path("/X"))
    assert sd == Path("/X/ipsc/evaluations_with_embeddings/eval_fnet3d_er")


def test_save_dir_canonical_a549_joint() -> None:
    """Joint-trained A549-test save_dir uses _jointtrained_ infix + the A549 dataset root."""
    parsed = _make("a549/joint_predictions/memb_celldiff_r2_denv.zarr")
    sd = save_dir_for(parsed, dynacell_root=Path("/X"))
    assert sd == Path("/X/a549/evaluations_jointtrained_with_embeddings/eval_celldiff_r2_jointtrained_membrane_denv")


def test_dataset_ref_ipsc() -> None:
    """For iPSC, dataset_ref points at aics-hipsc + logical organelle target key."""
    parsed = _make("ipsc/predictions/sec61b_fnet3d_paper.zarr")
    assert benchmark_dataset_ref(parsed) == {"dataset": "aics-hipsc", "target": "sec61b"}


def test_dataset_ref_a549_nucleus_uses_h2b() -> None:
    """A549 nucleus dataset_ref uses the gene-marker target key (h2b), not the logical name."""
    parsed = _make("a549/predictions/nucl_fnet3d_paper_jointtrained_mock.zarr")
    assert benchmark_dataset_ref(parsed) == {
        "dataset": "a549-mantis-h2b-mock",
        "target": "h2b",
    }


def test_dataset_ref_a549_membrane_uses_caax() -> None:
    """A549 membrane dataset_ref uses the gene-marker target key (caax), not the logical name."""
    parsed = _make("a549/predictions/memb_fcmae_vscyto3d_scratch_a549trained_zikv.zarr")
    assert benchmark_dataset_ref(parsed) == {
        "dataset": "a549-mantis-caax-zikv",
        "target": "caax",
    }


def test_pred_cache_dir_a549() -> None:
    """A549 pred_cache_dir condition segment is ``<gene>_<cond>`` (e.g. sec61b_denv)."""
    parsed = _make("a549/joint_predictions/sec61b_celldiff_r2_denv.zarr")
    pc = pred_cache_dir_for(parsed, dynacell_root=Path("/X"))
    assert pc == Path("/X/a549/eval_cache_pred/joint/celldiff_r2/sec61b_denv")


def test_pred_cache_dir_ipsc() -> None:
    """For iPSC, the pred_cache_dir condition segment is ``<organelle>_ipsc``.

    iPSC has no plate condition, so the segment is namespaced by the logical
    organelle (a bare ``ipsc`` would collapse all four organelles onto one dir
    and race the manifest's ``pred.plate_path``).
    """
    parsed = _make("ipsc/predictions/tomm20_fnet3d_paper.zarr")
    pc = pred_cache_dir_for(parsed, dynacell_root=Path("/X"))
    assert pc == Path("/X/ipsc/eval_cache_pred/ipsc_trained/fnet3d_paper/mitochondria_ipsc")


# ---------------------------------------------------------------------------
# 3. Live data checks
# ---------------------------------------------------------------------------


@pytest.mark.requires_data
@pytest.mark.skipif(
    not _DYNACELL_ROOT.exists(),
    reason=f"dynacell training root absent: {_DYNACELL_ROOT}",
)
def test_walk_predictions_yields_known_buckets() -> None:
    """Smoke check that all 4 organelles × 3 train_sets buckets are populated."""
    pool = walk_predictions(_DYNACELL_ROOT)
    assert len(pool) > 100
    buckets = {(p.organelle, p.train_set) for p in pool}
    expected = {
        (org, ts)
        for org in ("er", "mitochondria", "nucleus", "membrane")
        for ts in ("ipsc_trained", "joint", "a549_trained")
    }
    assert buckets == expected


@pytest.mark.requires_data
@pytest.mark.skipif(
    not _DYNACELL_ROOT.exists(),
    reason=f"dynacell training root absent: {_DYNACELL_ROOT}",
)
def test_all_pred_paths_exist_after_dedupe() -> None:
    """Every emitted pred_path must be a directory on disk."""
    pool = walk_predictions(_DYNACELL_ROOT)
    missing = [str(p.pred_path) for p in pool if not p.pred_path.is_dir()]
    assert not missing, f"missing pred_paths: {missing[:5]}"


@pytest.mark.requires_data
@pytest.mark.skipif(
    not _DYNACELL_ROOT.exists(),
    reason=f"dynacell training root absent: {_DYNACELL_ROOT}",
)
def test_walk_predictions_excludes_ablation_track() -> None:
    """walk_predictions must skip dual/ablation zarrs instead of crashing on them.

    These families coexist on disk with campaign zarrs; the walk used to raise
    ``unknown prediction zarr grammar`` on the first ``dual_`` entry.
    """
    pool = walk_predictions(_DYNACELL_ROOT)
    leaked = [str(p.pred_path) for p in pool if _is_ablation_track_zarr(p.pred_path.name)]
    assert not leaked, f"ablation-track zarrs leaked into the pool: {leaked[:5]}"


# ---------------------------------------------------------------------------
# 4. Real composition + resolver check per generated leaf
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_eval_grouped_config():
    """Compose the eval_grouped primary config the same way Hydra does."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    repo_root = Path(__file__).resolve().parents[3]
    base_dir = repo_root / "applications/dynacell/src/dynacell/evaluation/_configs"
    feature_extractor_overlay_dir = (
        repo_root / "applications/dynacell/configs/benchmarks/virtual_staining/_internal/shared/eval"
    )
    # Defensive clear: this file lives under tools/, outside the tests/conftest.py
    # that provides the `clear_global_hydra` fixture, so a Hydra-initializing test
    # elsewhere in the run can leave the global singleton dirty. initialize_config_dir
    # raises "GlobalHydra is already initialized" on a dirty singleton — an
    # intermittent failure here. Clear before init; the with-block clears again on
    # exit, so this fixture never leaks the singleton onward.
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.2", config_dir=str(base_dir)):
        # Bring in the external feature_extractor groups via searchpath override.
        cfg = compose(
            config_name="eval_grouped",
            overrides=[
                f"hydra.searchpath=[file://{feature_extractor_overlay_dir}]",
            ],
        )
    return cfg


@pytest.mark.requires_data
@pytest.mark.skipif(
    not (_LEAF_OUT_ROOT.exists() and any(_LEAF_OUT_ROOT.glob("*/eval_grouped.yaml"))),
    reason=f"grouped leaves not yet generated under {_LEAF_OUT_ROOT}",
)
def test_each_leaf_composes_and_resolves(base_eval_grouped_config) -> None:
    """Compose each leaf onto the base + run apply_dataset_ref per condition.

    Verifies every ``benchmark.dataset_ref`` actually resolves (no silent
    None from a partial dict) and every overlay's explicit io fields
    agree with the manifest-spliced fields.
    """
    from omegaconf import OmegaConf

    from dynacell.evaluation._ref_hook import apply_dataset_ref
    from dynacell.evaluation.pipeline import (
        _MODEL_LOADING_FIELDS,
        _check_grouped_field_invariants,
        _seg_model_required,
        _snapshot_field,
    )

    leaf_paths = sorted(_LEAF_OUT_ROOT.glob("*/eval_grouped.yaml"))
    assert leaf_paths, f"no leaves under {_LEAF_OUT_ROOT}"
    for leaf_path in leaf_paths:
        leaf = OmegaConf.load(leaf_path)
        merged_base = OmegaConf.merge(base_eval_grouped_config, leaf)
        conditions = OmegaConf.to_container(merged_base.conditions, resolve=False) or []
        assert conditions, f"{leaf_path}: empty conditions"

        # Snapshot the model-loading fields from the conditions-stripped base.
        base_for_snapshot = OmegaConf.create(OmegaConf.to_container(merged_base, resolve=False))
        if "conditions" in base_for_snapshot:
            del base_for_snapshot["conditions"]
        base_snapshot = {field: _snapshot_field(base_for_snapshot, field) for field in _MODEL_LOADING_FIELDS}
        base_seg_required = _seg_model_required(base_for_snapshot)

        for idx, cond in enumerate(conditions):
            # Build a fresh merged config per condition (mirrors driver).
            merged = OmegaConf.create(OmegaConf.to_container(merged_base, resolve=False))
            if "conditions" in merged:
                del merged["conditions"]
            merged = OmegaConf.merge(merged, OmegaConf.create(cond))
            if "name" in merged:
                del merged["name"]
            # Resolve dataset_ref — splices manifest fields.
            apply_dataset_ref(merged)
            # Verify model-loading invariants.
            _check_grouped_field_invariants(
                base_snapshot,
                base_seg_required,
                merged,
                cond.get("name", str(idx)),
            )


# ---------------------------------------------------------------------------
# 5. Instance-AP wiring in the unified grouped pass (nucleus & membrane only)
# ---------------------------------------------------------------------------


def test_nucleus_grouped_leaf_enables_cellpose_instance_ap() -> None:
    """Nucleus bucket computes instance AP in the SAME pass as features.

    backend=cellpose, no nuclei seeds, and compute_feature_metrics stays on — the
    instance masks feed both the AP_*/mAP/instance_dice columns and the semantic
    Dice/IoU rows, not a separate track.
    """
    leaf = build_leaf_yaml("nucleus", "joint", [_make("ipsc/predictions/nucl_fnet3d_paper_jointtrained.zarr")])
    assert leaf["compute_instance_ap"] is True
    assert leaf["compute_feature_metrics"] is True
    assert leaf["segmentation"]["backend"] == "cellpose"
    assert "nuclei_channel_name" not in leaf["segmentation"]
    assert "nuclei_gt_path" not in leaf["conditions"][0]["io"]


def test_membrane_a549_grouped_leaf_wires_cross_store_nuclei() -> None:
    """Membrane × a549 → watershed backend + per-condition H2B nuclei_gt_path."""
    conds = [
        _make("a549/predictions/memb_fnet3d_paper_a549trained_mock.zarr"),
        _make("a549/predictions/memb_fcmae_vscyto3d_scratch_a549trained_zikv.zarr"),
    ]
    leaf = build_leaf_yaml("membrane", "a549_trained", conds)
    assert leaf["compute_instance_ap"] is True
    assert leaf["compute_feature_metrics"] is True
    assert leaf["segmentation"]["backend"] == "cellpose_watershed"
    assert leaf["segmentation"]["nuclei_channel_name"] == "Nuclei"
    for block in leaf["conditions"]:
        nuclei_gt = block["io"]["nuclei_gt_path"]
        assert "H2B" in nuclei_gt and nuclei_gt.endswith(".ozx")


def test_membrane_ipsc_grouped_leaf_has_no_nuclei_gt_path() -> None:
    """Membrane × iPSC reads nuclei from the same cell.zarr → no separate nuclei_gt_path."""
    leaf = build_leaf_yaml("membrane", "ipsc_trained", [_make("ipsc/predictions/memb_fnet3d_paper.zarr")])
    assert leaf["segmentation"]["backend"] == "cellpose_watershed"
    assert leaf["segmentation"]["nuclei_channel_name"] == "Nuclei"
    assert "nuclei_gt_path" not in leaf["conditions"][0]["io"]


def test_er_and_mito_grouped_leaves_have_no_instance_ap() -> None:
    """ER/mito have no cell instances → no instance AP, no segmentation backend override."""
    for rel, organelle, train_set in (
        ("ipsc/predictions/sec61b_fnet3d_paper_jointtrained.zarr", "er", "joint"),
        ("ipsc/predictions/tomm20_fnet3d_paper_jointtrained.zarr", "mitochondria", "joint"),
    ):
        leaf = build_leaf_yaml(organelle, train_set, [_make(rel)])
        assert "compute_instance_ap" not in leaf
        assert "segmentation" not in leaf
        assert leaf["compute_feature_metrics"] is True
