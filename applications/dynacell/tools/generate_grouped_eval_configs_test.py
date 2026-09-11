"""Smoke tests for ``generate_grouped_eval_configs.py``.

Tests fall into three groups:

1. Pure-Python path grammar (no dependencies).
2. Live-data checks (require the dynacell training tree on disk; marked
   ``@pytest.mark.slow``, so the root ``addopts`` ``-m 'not slow'`` deselects
   them by default — they assert on a shared tree that regen campaigns move).
3. Real composition + resolver check per generated leaf (requires the
   composed eval base config + dynacell Hydra search path).

Run everything including the live-data checks::

    uv run pytest applications/dynacell/tools/generate_grouped_eval_configs_test.py \
        -v -m 'slow or not slow'
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
    _CANONICAL_TRAIN_SET_TO_BUCKET,
    _DYNACELL_ROOT,
    _LEAF_OUT_ROOT,
    _TRAIN_SETS,
    ParsedZarr,
    benchmark_dataset_ref,
    build_leaf_yaml,
    leaf_test_set,
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
        # ER (deconv provenance on a549/joint) — iPSC test.
        (
            "er/fnet3d_paper/ipsc/ipsc/prediction.zarr",
            ("er", "fnet3d_paper", None, "ipsc_trained", "ipsc", "ipsc", None),
        ),
        (
            "er/fnet3d_paper/joint__legacy_deconvgt/ipsc/prediction.zarr",
            ("er", "fnet3d_paper", None, "joint", "joint__legacy_deconvgt", "ipsc", None),
        ),
        (
            "er/fnet3d_paper/a549__deconv/ipsc/prediction.zarr",
            ("er", "fnet3d_paper", None, "a549_trained", "a549__deconv", "ipsc", None),
        ),
        # ER (deconv provenance) — A549 test.
        (
            "er/fnet3d_paper/a549__deconv/a549__mock/prediction.zarr",
            ("er", "fnet3d_paper", None, "a549_trained", "a549__deconv", "a549", "mock"),
        ),
        (
            "er/fnet3d_paper/joint__legacy_deconvgt/a549__mock/prediction.zarr",
            ("er", "fnet3d_paper", None, "joint", "joint__legacy_deconvgt", "a549", "mock"),
        ),
        # CellDiff R2 variant dirs split into (model, variant).
        (
            "er/celldiff_r2_iterative/a549__deconv/ipsc/prediction.zarr",
            ("er", "celldiff_r2", "iterative", "a549_trained", "a549__deconv", "ipsc", None),
        ),
        (
            "membrane/celldiff_r2_sliding_window/ipsc/ipsc/prediction.zarr",
            ("membrane", "celldiff_r2", "sliding_window", "ipsc_trained", "ipsc", "ipsc", None),
        ),
        (
            "mito/celldiff_r2_iterative/a549__deconv/a549__mock/prediction.zarr",
            ("mitochondria", "celldiff_r2", "iterative", "a549_trained", "a549__deconv", "a549", "mock"),
        ),
        (
            "er/celldiff_r2/joint__legacy_deconvgt/ipsc/prediction.zarr",
            ("er", "celldiff_r2", None, "joint", "joint__legacy_deconvgt", "ipsc", None),
        ),
        (
            "membrane/celldiff_r2/joint/a549__denv/prediction.zarr",
            ("membrane", "celldiff_r2", None, "joint", "joint", "a549", "denv"),
        ),
        # nucleus / membrane (raw a549 / joint provenance).
        (
            "nucleus/fnet3d_paper/joint/a549__mock/prediction.zarr",
            ("nucleus", "fnet3d_paper", None, "joint", "joint", "a549", "mock"),
        ),
        (
            "membrane/fcmae_vscyto3d_pretrained/joint/ipsc/prediction.zarr",
            ("membrane", "fcmae_vscyto3d_pretrained", None, "joint", "joint", "ipsc", None),
        ),
        (
            "membrane/fcmae_vscyto3d_scratch/a549/a549__zikv/prediction.zarr",
            ("membrane", "fcmae_vscyto3d_scratch", None, "a549_trained", "a549", "a549", "zikv"),
        ),
        # pix2pix3d_unetvit across pools + test sets.
        (
            "nucleus/pix2pix3d_unetvit/ipsc/ipsc/prediction.zarr",
            ("nucleus", "pix2pix3d_unetvit", None, "ipsc_trained", "ipsc", "ipsc", None),
        ),
        (
            "nucleus/pix2pix3d_unetvit/a549/ipsc/prediction.zarr",
            ("nucleus", "pix2pix3d_unetvit", None, "a549_trained", "a549", "ipsc", None),
        ),
        (
            "nucleus/pix2pix3d_unetvit/joint/a549__mock/prediction.zarr",
            ("nucleus", "pix2pix3d_unetvit", None, "joint", "joint", "a549", "mock"),
        ),
        (
            "nucleus/pix2pix3d_unetvit/a549/a549__zikv/prediction.zarr",
            ("nucleus", "pix2pix3d_unetvit", None, "a549_trained", "a549", "a549", "zikv"),
        ),
        (
            "membrane/pix2pix3d_unetvit/ipsc/a549__denv/prediction.zarr",
            ("membrane", "pix2pix3d_unetvit", None, "ipsc_trained", "ipsc", "a549", "denv"),
        ),
    ],
)
def test_parse_zarr_name(rel: str, expect: tuple) -> None:
    """Path-grammar dispatch covers every pool/test/provenance + celldiff variants.

    The expected tuple is
    ``(organelle, model, variant, train_set, train_set_canonical, test_set, condition)``.
    ``train_set`` is the generator bucket label; ``train_set_canonical`` is the full
    canonical token carrying deconv provenance (``a549__deconv`` /
    ``joint__legacy_deconvgt`` for ER/mito).
    """
    fake_root = Path("/fake/root")
    parsed = parse_zarr_name(fake_root / rel, dynacell_root=fake_root)
    assert (
        parsed.organelle,
        parsed.model,
        parsed.variant,
        parsed.train_set,
        parsed.train_set_canonical,
        parsed.test_set,
        parsed.condition,
    ) == expect


def test_parse_zarr_name_malformed_path_raises() -> None:
    """A non-canonical / legacy path (not the 5-part prediction-store grammar) must raise."""
    fake_root = Path("/fake/root")
    with pytest.raises(ValueError, match="canonical prediction store"):
        parse_zarr_name(fake_root / "ipsc/predictions/sec61b_fnet3d_paper.zarr", dynacell_root=fake_root)


def test_parse_zarr_name_unknown_model_raises() -> None:
    """An unknown model directory must raise ValueError."""
    fake_root = Path("/fake/root")
    with pytest.raises(ValueError, match="unknown model"):
        parse_zarr_name(fake_root / "er/madeup_model/ipsc/ipsc/prediction.zarr", dynacell_root=fake_root)


def test_parse_zarr_name_unknown_celldiff_variant_raises() -> None:
    """An unknown CellDiff variant directory must raise ValueError."""
    fake_root = Path("/fake/root")
    with pytest.raises(ValueError, match="unknown CellDiff variant"):
        parse_zarr_name(fake_root / "er/celldiff_r2_fakevariant/ipsc/ipsc/prediction.zarr", dynacell_root=fake_root)


def test_parse_zarr_name_unknown_organelle_raises() -> None:
    """An unknown organelle root directory must raise ValueError."""
    fake_root = Path("/fake/root")
    with pytest.raises(ValueError, match="organelle"):
        parse_zarr_name(fake_root / "bogus/fnet3d_paper/ipsc/ipsc/prediction.zarr", dynacell_root=fake_root)


def test_parse_zarr_name_out_of_scope_train_set_raises() -> None:
    """A paths-valid but out-of-scope train_set raises ValueError, not KeyError.

    Uses ``a549__bf__deconv``, which ``paths.py`` calls a "grammar-ready follow-up"
    with no fits behind it yet. This test previously used ``a549__bf``; that token
    was registered as its own bucket once the 8 brightfield fits completed, so it no
    longer exercises the out-of-scope path.
    """
    fake_root = Path("/fake/root")
    with pytest.raises(ValueError, match="grouped-campaign bucket"):
        parse_zarr_name(
            fake_root / "er/fnet3d_paper/a549__bf__deconv/a549__mock/prediction.zarr",
            dynacell_root=fake_root,
        )


# ---------------------------------------------------------------------------
# Registry drift guard (this module's _CODE_TO_PAPER vs the runtime resolver)
# ---------------------------------------------------------------------------


def test_paper_key_maps_agree_on_overlap() -> None:
    """The campaign map and the runtime resolver must never assign DIFFERENT paper keys.

    ``paths.PAPER_KEY`` (single-condition submitter + paper aggregation) and
    this module's ``_CODE_TO_PAPER`` (grouped campaign) are separate maps with
    intentionally different *membership* — but where they overlap they must agree,
    or eval outputs land in mismatched dirs. Only documented differences are waived.
    """
    from generate_grouped_eval_configs import _CODE_TO_PAPER

    from dynacell.evaluation.paths import PAPER_KEY

    # Documented intentional difference: the grouped campaign keeps `celldiff`
    # literal; the runtime resolver collapses it to `celldiff_iterative`.
    waivers = {"celldiff"}
    disagreements = {
        m: (PAPER_KEY[m], _CODE_TO_PAPER[m])
        for m in set(PAPER_KEY) & set(_CODE_TO_PAPER)
        if m not in waivers and PAPER_KEY[m] != _CODE_TO_PAPER[m]
    }
    assert not disagreements, f"paper-key drift between paths.PAPER_KEY and _CODE_TO_PAPER: {disagreements}"


def test_deterministic_models_known_to_runtime_resolver() -> None:
    """Every deterministic campaign model must also be registered in the runtime resolver.

    Catches "added a model to the generator parser but forgot ``paths.PAPER_KEY``"
    — the asymmetry that let pix2pix3d slip the instance-AP track.
    """
    from generate_grouped_eval_configs import _DETERMINISTIC_MODELS

    from dynacell.evaluation.paths import PAPER_KEY

    missing = [m for m in _DETERMINISTIC_MODELS if m not in PAPER_KEY]
    assert not missing, f"deterministic campaign models absent from paths.PAPER_KEY: {missing}"


# ---------------------------------------------------------------------------
# 2. Save_dir + dataset_ref derivation
# ---------------------------------------------------------------------------


def _make(rel: str) -> ParsedZarr:
    return parse_zarr_name(Path("/fake/root") / rel, dynacell_root=Path("/fake/root"))


def test_save_dir_canonical_ipsc_ipsc_trained() -> None:
    """iPSC-trained iPSC-test save_dir → canonical <organelle>/<model>/<train>/<test> leaf."""
    parsed = _make("er/fnet3d_paper/ipsc/ipsc/prediction.zarr")
    sd = save_dir_for(parsed, dynacell_root=Path("/X"))
    assert sd == Path("/X/er/fnet3d_paper/ipsc/ipsc")


def test_save_dir_canonical_a549_joint() -> None:
    """Joint-trained A549-test save_dir → canonical leaf with the <test>__<cond> segment."""
    parsed = _make("membrane/celldiff_r2/joint/a549__denv/prediction.zarr")
    sd = save_dir_for(parsed, dynacell_root=Path("/X"))
    assert sd == Path("/X/membrane/celldiff_r2/joint/a549__denv")


def test_save_dir_er_deconv_provenance_preserved() -> None:
    """ER a549__deconv save_dir keeps the deconv token (not the lossy a549 bucket)."""
    parsed = _make("er/fnet3d_paper/a549__deconv/a549__mock/prediction.zarr")
    sd = save_dir_for(parsed, dynacell_root=Path("/X"))
    assert sd == Path("/X/er/fnet3d_paper/a549__deconv/a549__mock")


def test_dataset_ref_ipsc() -> None:
    """For iPSC, dataset_ref points at aics-hipsc + logical organelle target key."""
    parsed = _make("er/fnet3d_paper/ipsc/ipsc/prediction.zarr")
    assert benchmark_dataset_ref(parsed) == {"dataset": "aics-hipsc", "target": "sec61b"}


def test_dataset_ref_a549_nucleus_uses_h2b() -> None:
    """A549 nucleus dataset_ref uses the gene-marker target key (h2b), not the logical name."""
    parsed = _make("nucleus/fnet3d_paper/joint/a549__mock/prediction.zarr")
    assert benchmark_dataset_ref(parsed) == {
        "dataset": "a549-mantis-h2b-mock",
        "target": "h2b",
    }


def test_dataset_ref_a549_membrane_uses_caax() -> None:
    """A549 membrane dataset_ref uses the gene-marker target key (caax), not the logical name."""
    parsed = _make("membrane/fcmae_vscyto3d_scratch/a549/a549__zikv/prediction.zarr")
    assert benchmark_dataset_ref(parsed) == {
        "dataset": "a549-mantis-caax-zikv",
        "target": "caax",
    }


def test_pred_cache_dir_a549() -> None:
    """A549 ER joint pred_cache_dir keeps the deconv-provenance token (joint__legacy_deconvgt)."""
    parsed = _make("er/celldiff_r2/joint__legacy_deconvgt/a549__denv/prediction.zarr")
    pc = pred_cache_dir_for(parsed, dynacell_root=Path("/X"))
    assert pc == Path("/X/a549/eval_cache_pred/er/celldiff_r2/joint__legacy_deconvgt/a549__denv")


def test_pred_cache_dir_ipsc() -> None:
    """IPSC canonical pred_cache_dir keeps the organelle in the tuple (mito normalized).

    The canonical grammar namespaces the pred cache on the full tuple
    ``<test>/eval_cache_pred/<organelle>/<model>/<train>/<test>``, so the four
    organelles never collapse onto one dir.
    """
    parsed = _make("mito/fnet3d_paper/ipsc/ipsc/prediction.zarr")
    pc = pred_cache_dir_for(parsed, dynacell_root=Path("/X"))
    assert pc == Path("/X/ipsc/eval_cache_pred/mito/fnet3d_paper/ipsc/ipsc")


# ---------------------------------------------------------------------------
# 2b. Test-set scoping of the shared canonical tree
# ---------------------------------------------------------------------------


def _touch_prediction(root: Path, organelle: str, model: str, train_set: str, leaf: str) -> Path:
    """Create an empty canonical prediction dir on a tmp tree."""
    pred = root / organelle / model / train_set / leaf / "prediction.zarr"
    pred.mkdir(parents=True)
    return pred


def test_walk_predictions_skips_out_of_scope_test_sets(tmp_path: Path) -> None:
    """A hek__<arm> prediction in the shared tree must not enter the campaign pool.

    The canonical tree is shared across branches, so an out-of-scope probe would
    otherwise be bucketed by (organelle, train_set) into the 12 committed campaign
    leaves and then fed to benchmark_dataset_ref, which keys off the A549 condition
    vocabulary.
    """
    _touch_prediction(tmp_path, "membrane", "fnet3d_paper", "a549", "a549__mock")
    _touch_prediction(tmp_path, "membrane", "fnet3d_paper", "a549", "hek__a549xy")

    pool = walk_predictions(tmp_path)
    assert [p.test_set for p in pool] == ["a549"]


def test_walk_predictions_can_opt_in_a_test_set(tmp_path: Path) -> None:
    """Explicitly requesting a test set includes it (and excludes the others)."""
    _touch_prediction(tmp_path, "membrane", "fnet3d_paper", "a549", "a549__mock")
    _touch_prediction(tmp_path, "membrane", "fnet3d_paper", "a549", "hek__a549xy")

    pool = walk_predictions(tmp_path, test_sets=frozenset({"hek"}))
    assert [(p.test_set, p.condition) for p in pool] == [("hek", "a549xy")]


def test_leaf_test_set_prefix_read() -> None:
    """leaf_test_set must not raise on an unknown segment -- it must be filterable."""
    assert leaf_test_set("ipsc") == "ipsc"
    assert leaf_test_set("a549__mock") == "a549"
    assert leaf_test_set("hek__a549xy") == "hek"
    assert leaf_test_set("somethingnew__v2") == "somethingnew"


# ---------------------------------------------------------------------------
# 3. Live data checks
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not _DYNACELL_ROOT.exists(),
    reason=f"dynacell training root absent: {_DYNACELL_ROOT}",
)
def test_walk_predictions_yields_known_buckets() -> None:
    """The 4x3 core grid is complete, and nothing outside the bucket table appears.

    Asserted as "core is present" plus "no bucket outside the registered labels"
    rather than as equality against a hardcoded 12. The brightfield-input ablation
    adds ``{ipsc,a549}_bf_trained`` for nucleus and ER only, and its predictions land
    incrementally, so an equality check would fail for as long as the arms are
    partially predicted -- without that ever indicating a real problem.
    """
    pool = walk_predictions(_DYNACELL_ROOT)
    assert len(pool) > 100
    buckets = {(p.organelle, p.train_set) for p in pool}
    core = {
        (org, ts)
        for org in ("er", "mitochondria", "nucleus", "membrane")
        for ts in ("ipsc_trained", "joint", "a549_trained")
    }
    assert core <= buckets, f"core grid incomplete: missing {sorted(core - buckets)}"
    unregistered = {ts for _, ts in buckets} - set(_CANONICAL_TRAIN_SET_TO_BUCKET.values())
    assert not unregistered, f"bucket labels not in the table: {sorted(unregistered)}"


def test_brightfield_tokens_get_their_own_buckets() -> None:
    """``__bf`` must not fold into the phase buckets -- it is a different input.

    ``a549__deconv`` folding into ``a549_trained`` is a misleading precedent: it
    marks target provenance on the same ``Phase3D`` input. ``__bf`` swaps the input
    channel, so sharing a bucket would collide a brightfield and a phase prediction
    on one ``canonical_identity`` and let one silently shadow the other.
    """
    table = _CANONICAL_TRAIN_SET_TO_BUCKET
    assert table["ipsc__bf"] == "ipsc_bf_trained"
    assert table["a549__bf"] == "a549_bf_trained"
    assert table["ipsc__bf"] != table["ipsc"]
    assert table["a549__bf"] != table["a549"]
    # And the phase buckets stay exactly as they were.
    assert table["a549__deconv"] == table["a549"] == "a549_trained"


def test_every_registered_bucket_label_is_emittable() -> None:
    """Guard that ``_TRAIN_SETS`` stays DERIVED from the bucket table.

    ``main`` and ``emit_readme`` iterate ``_ORGANELLES x _TRAIN_SETS``, so a bucket
    label the table registers but that tuple omits parses fine, lands in
    ``buckets``, and is then never written -- no error, no empty-bucket message,
    just a silently absent leaf. That is exactly how the brightfield buckets stayed
    invisible after 5a25142e registered them.

    ``_TRAIN_SETS`` is now built with ``dict.fromkeys`` over the table's values, so
    this holds by construction. The test is kept to fail loudly if anyone
    re-hardcodes the tuple, which is what reintroduces the whole bug class. Order is
    asserted too: the tuple drives README row order, so the table's literal order is
    load-bearing and a reorder there should be a deliberate choice, not a surprise.
    """
    registered = set(_CANONICAL_TRAIN_SET_TO_BUCKET.values())
    missing = registered - set(_TRAIN_SETS)
    assert not missing, f"registered but never emitted: {sorted(missing)}"
    assert _TRAIN_SETS == tuple(dict.fromkeys(_CANONICAL_TRAIN_SET_TO_BUCKET.values())), (
        f"_TRAIN_SETS is no longer derived from the bucket table: {_TRAIN_SETS}. "
        f"Re-hardcoding it lets a registered bucket go silently unemitted."
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not _DYNACELL_ROOT.exists(),
    reason=f"dynacell training root absent: {_DYNACELL_ROOT}",
)
def test_all_pred_paths_exist_after_dedupe() -> None:
    """Every emitted pred_path must be a directory on disk."""
    pool = walk_predictions(_DYNACELL_ROOT)
    missing = [str(p.pred_path) for p in pool if not p.pred_path.is_dir()]
    assert not missing, f"missing pred_paths: {missing[:5]}"


@pytest.mark.slow
@pytest.mark.skipif(
    not _DYNACELL_ROOT.exists(),
    reason=f"dynacell training root absent: {_DYNACELL_ROOT}",
)
def test_walk_predictions_excludes_celldiff_r1() -> None:
    """walk_predictions must drop R1 CellDiff (bare + variant dirs) via _SKIP_MODELS.

    The canonical tree carries ``celldiff`` / ``celldiff_iterative`` /
    ``celldiff_denoise`` / ``celldiff_sliding_window`` dirs; all resolve to model
    ``celldiff`` and must be skipped, leaving only the R2 family in scope.
    """
    pool = walk_predictions(_DYNACELL_ROOT)
    leaked = [str(p.pred_path) for p in pool if p.model == "celldiff"]
    assert not leaked, f"celldiff R1 leaked into the pool: {leaked[:5]}"


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


@pytest.mark.slow
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


def test_nucleus_grouped_leaf_enables_cpdino_instance_ap() -> None:
    """Nucleus bucket computes instance AP in the SAME pass as features.

    backend=cpdino, no nuclei seeds, and compute_feature_metrics stays on — the
    instance masks feed both the AP_*/mAP/instance_dice columns and the semantic
    Dice/IoU rows, not a separate track.
    """
    leaf = build_leaf_yaml("nucleus", "joint", [_make("nucleus/fnet3d_paper/joint/ipsc/prediction.zarr")])
    assert leaf["compute_instance_ap"] is True
    assert leaf["compute_feature_metrics"] is True
    assert leaf["segmentation"]["backend"] == "cpdino"
    assert "nuclei_channel_name" not in leaf["segmentation"]
    assert "watershed" not in leaf["segmentation"]  # cpdino nucleus path has no watershed stage
    assert "nuclei_gt_path" not in leaf["conditions"][0]["io"]


def test_membrane_a549_grouped_leaf_wires_cross_store_nuclei() -> None:
    """Membrane × a549 → cpdino backend + per-condition dual-store nuclei_gt_path."""
    conds = [
        _make("membrane/fnet3d_paper/a549/a549__mock/prediction.zarr"),
        _make("membrane/fcmae_vscyto3d_scratch/a549/a549__zikv/prediction.zarr"),
    ]
    leaf = build_leaf_yaml("membrane", "a549_trained", conds)
    assert leaf["compute_instance_ap"] is True
    assert leaf["compute_feature_metrics"] is True
    assert leaf["segmentation"]["backend"] == "cpdino"
    assert leaf["segmentation"]["nuclei_channel_name"] == "Nuclei"
    # Carved is canonical (6aedf52f): the leaf must NOT override subtract_nuclei,
    # so both semantic + AP inherit the eval.yaml carved default (subtract_nuclei=true).
    assert "watershed" not in leaf["segmentation"]
    for block in leaf["conditions"]:
        nuclei_gt = block["io"]["nuclei_gt_path"]
        assert "dual_nucl_memb" in nuclei_gt and nuclei_gt.endswith(".zarr")


def test_membrane_ipsc_grouped_leaf_has_no_nuclei_gt_path() -> None:
    """Membrane × iPSC reads nuclei from the same cell.zarr → no separate nuclei_gt_path."""
    leaf = build_leaf_yaml("membrane", "ipsc_trained", [_make("membrane/fnet3d_paper/ipsc/ipsc/prediction.zarr")])
    assert leaf["segmentation"]["backend"] == "cpdino"
    assert leaf["segmentation"]["nuclei_channel_name"] == "Nuclei"
    assert "nuclei_gt_path" not in leaf["conditions"][0]["io"]


def test_er_and_mito_grouped_leaves_have_no_instance_ap() -> None:
    """ER/mito have no cell instances → no instance AP, no segmentation backend override."""
    for rel, organelle, train_set in (
        ("er/fnet3d_paper/joint__legacy_deconvgt/ipsc/prediction.zarr", "er", "joint"),
        ("mito/fnet3d_paper/joint__legacy_deconvgt/ipsc/prediction.zarr", "mitochondria", "joint"),
    ):
        leaf = build_leaf_yaml(organelle, train_set, [_make(rel)])
        assert "compute_instance_ap" not in leaf
        assert "segmentation" not in leaf
        assert leaf["compute_feature_metrics"] is True
