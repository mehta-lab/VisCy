"""Contract tests for the canonical artifact-path grammar (``paths.py``).

Covers:
- round-trip of every canonical tuple through the forward grammar
  (``prediction_store``/``eval_leaf``/``checkpoint_dir``) — well-formed, unique
  paths, and tuple-validity rejection of the invalid combinations;
- ``normalize_legacy`` mapping ONE real on-disk path of each legacy form to the
  expected :class:`CanonicalKey` (or ``None`` for deliberate skips);
- no two distinct canonical tuples collide on one on-disk path.

The legacy-path fixtures below are literal on-disk names verified under
``/hpc/projects/virtual_staining/training/dynacell`` at authoring time. They are
paths, not I/O — the tests never touch the filesystem.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from dynacell.evaluation.paths import (
    MODELS_ROOT,
    ORGANELLE_EVAL_TARGET,
    PAPER_KEY,
    CanonicalKey,
    canonical_model_name,
    checkpoint_dir,
    eval_leaf,
    gt_cache_dir,
    iter_organelle_evals,
    key_from_prediction_store,
    metrics_repo_dir,
    normalize_legacy,
    paper_key,
    pred_cache_dir,
    prediction_store,
    resolve_model,
)

_DATA_ROOT = "/hpc/projects/virtual_staining/training/dynacell"

# ---------------------------------------------------------------------------
# Canonical tuple enumeration
# ---------------------------------------------------------------------------

_SINGLE_ORGS = ("nucleus", "membrane", "er", "mito")
_MODELS = ("fcmae_vscyto3d_scratch", "fnet3d_paper", "celldiff_r2", "pix2pix3d_unetvit")
# ``ipsc__bf`` / ``a549__bf`` are the brightfield-input ablation (model input = raw
# Brightfield stack instead of the Phase3D volume reconstructed from it). Listed here so
# they inherit the uniqueness and inverse-round-trip guards below: their whole point is
# that they must never collide with the plain ``ipsc``/``a549`` phase arms they are
# compared against.
_FORWARD_TRAINS = ("ipsc", "ipsc__bf", "a549", "a549__bf", "joint")
_TEST_CONDS = (("ipsc", None), ("a549", "mock"), ("a549", "denv"), ("a549", "zikv"))


def _valid_canonical_tuples() -> list[tuple[str, str, str, str, str | None]]:
    """Enumerate valid (organelle, model, train_set, test_set, condition) tuples."""
    out: list[tuple[str, str, str, str, str | None]] = []
    for org, model, train, (test, cond) in itertools.product(_SINGLE_ORGS, _MODELS, _FORWARD_TRAINS, _TEST_CONDS):
        out.append((org, model, train, test, cond))
    # ER/mito deconv forward tuples (a549__deconv only).
    for org, model, (test, cond) in itertools.product(("er", "mito"), _MODELS, _TEST_CONDS):
        out.append((org, model, "a549__deconv", test, cond))
    return out


# ---------------------------------------------------------------------------
# Round-trip: forward grammar produces well-formed, unique paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tup", _valid_canonical_tuples())
def test_prediction_and_eval_leaf_forward(tup: tuple[str, str, str, str, str | None]) -> None:
    org, model, train, test, cond = tup
    pred = prediction_store(org, model, train, test, cond, data_root=_DATA_ROOT)
    leaf = eval_leaf(org, model, train, test, cond, data_root=_DATA_ROOT)
    # The eval leaf top is the parent of prediction.zarr (shared leaf dir).
    assert pred.name == "prediction.zarr"
    assert pred.parent == leaf
    # Structure: <root>/<org>/<model>/<train>/<test[__cond]>/...
    rel = leaf.relative_to(_DATA_ROOT)
    assert rel.parts[0] == org
    assert rel.parts[1] == model
    assert rel.parts[2] == train
    expected_leaf = f"a549__{cond}" if test == "a549" else "ipsc"
    assert rel.parts[3] == expected_leaf


def test_forward_paths_are_unique_per_tuple() -> None:
    """No two distinct tuples map to the same eval leaf / prediction store."""
    seen_leaf: dict[Path, tuple] = {}
    seen_pred: dict[Path, tuple] = {}
    for tup in _valid_canonical_tuples():
        org, model, train, test, cond = tup
        leaf = eval_leaf(org, model, train, test, cond, data_root=_DATA_ROOT)
        pred = prediction_store(org, model, train, test, cond, data_root=_DATA_ROOT)
        assert leaf not in seen_leaf, f"leaf collision: {tup} vs {seen_leaf[leaf]} -> {leaf}"
        assert pred not in seen_pred, f"pred collision: {tup} vs {seen_pred[pred]} -> {pred}"
        seen_leaf[leaf] = tup
        seen_pred[pred] = tup


# ---------------------------------------------------------------------------
# Inverse: key_from_prediction_store recovers the identity (incl. deconv provenance)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tup", _valid_canonical_tuples())
def test_key_from_prediction_store_roundtrip(tup: tuple[str, str, str, str, str | None]) -> None:
    org, model, train, test, cond = tup
    pred = prediction_store(org, model, train, test, cond, data_root=_DATA_ROOT)
    key = key_from_prediction_store(pred, data_root=_DATA_ROOT)
    assert key == CanonicalKey(organelle=org, model=model, train_set=train, test_set=test, condition=cond)


def test_key_from_prediction_store_preserves_deconv_provenance() -> None:
    """ER/mito deconv markers must survive the inverse (the walker-fix core property)."""
    for train in ("a549__deconv", "joint__legacy_deconvgt"):
        pred = prediction_store("er", "celldiff_r2", train, "a549", "denv", data_root=_DATA_ROOT)
        key = key_from_prediction_store(pred, data_root=_DATA_ROOT)
        assert key.train_set == train


def test_key_from_prediction_store_rejects_noncanonical_paths() -> None:
    root = Path(_DATA_ROOT)
    # Legacy prediction layout (must NOT parse as canonical).
    with pytest.raises(ValueError):
        key_from_prediction_store(root / "a549/predictions/sec61b_celldiff_r2_denv.zarr", data_root=_DATA_ROOT)
    # Missing the prediction.zarr leaf (a leaf dir, not the zarr).
    with pytest.raises(ValueError):
        key_from_prediction_store(root / "er/celldiff_r2/a549__deconv/a549__denv", data_root=_DATA_ROOT)
    # Path outside data_root.
    with pytest.raises(ValueError):
        key_from_prediction_store("/somewhere/else/prediction.zarr", data_root=_DATA_ROOT)
    # Unknown condition in the leaf segment.
    with pytest.raises(ValueError):
        key_from_prediction_store(
            root / "er/celldiff_r2/a549__deconv/a549__badcond/prediction.zarr", data_root=_DATA_ROOT
        )


def test_track_subdir() -> None:
    base = eval_leaf("nucleus", "fnet3d_paper", "ipsc", "ipsc", data_root=_DATA_ROOT)
    ap = eval_leaf("nucleus", "fnet3d_paper", "ipsc", "ipsc", track="instance_ap", data_root=_DATA_ROOT)
    assert ap == base / "instance_ap"


def test_multi_target_component_subdir() -> None:
    shared_pred = prediction_store(
        "dual_nucleus_membrane", "fcmae_vscyto3d_pretrained", "a549", "a549", "mock", data_root=_DATA_ROOT
    )
    nuc = eval_leaf(
        "dual_nucleus_membrane",
        "fcmae_vscyto3d_pretrained",
        "a549",
        "a549",
        "mock",
        component="nucleus",
        data_root=_DATA_ROOT,
    )
    memb = eval_leaf(
        "dual_nucleus_membrane",
        "fcmae_vscyto3d_pretrained",
        "a549",
        "a549",
        "mock",
        component="membrane",
        data_root=_DATA_ROOT,
    )
    # Both components share the ONE prediction.zarr at the leaf top.
    assert nuc.parent == memb.parent == shared_pred.parent
    assert nuc.name == "nucleus"
    assert memb.name == "membrane"


def test_multi_target_requires_component() -> None:
    with pytest.raises(ValueError):
        eval_leaf("dual_nucleus_membrane", "fcmae_vscyto3d_pretrained", "a549", "a549", "mock", data_root=_DATA_ROOT)
    with pytest.raises(ValueError):
        eval_leaf(
            "dual_nucleus_membrane",
            "fcmae_vscyto3d_pretrained",
            "a549",
            "a549",
            "mock",
            component="er",  # not a component of the dual token
            data_root=_DATA_ROOT,
        )


def test_single_target_rejects_component() -> None:
    with pytest.raises(ValueError):
        eval_leaf("nucleus", "fnet3d_paper", "ipsc", "ipsc", component="nucleus", data_root=_DATA_ROOT)


# ---------------------------------------------------------------------------
# Tuple validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "org, train",
    [
        ("nucleus", "a549__deconv"),  # deconv is er/mito-only
        ("membrane", "a549__deconv"),
    ],
)
def test_deconv_invalid_for_nucleus_membrane(org: str, train: str) -> None:
    with pytest.raises(ValueError):
        prediction_store(org, "fnet3d_paper", train, "a549", "mock", data_root=_DATA_ROOT)


def test_ipsc_and_joint_deconv_are_invalid_forward_tokens() -> None:
    for train in ("ipsc__deconv", "joint__deconv"):
        with pytest.raises(ValueError):
            prediction_store("er", "fnet3d_paper", train, "a549", "mock", data_root=_DATA_ROOT)


def test_ipsc_brightfield_config_spellings_normalize_to_one_token() -> None:
    """Every config-side spelling of the brightfield iPSC train set lands on ``ipsc__bf``.

    The train leaves live in a directory named ``ipsc_confocal_brightfield`` while the
    on-disk grammar uses ``ipsc__bf``; if the alias table missed a spelling, the eval-side
    path builders would raise "unknown train_set token" only once predictions existed.
    ``ipsc__bf`` must also stay distinct from plain ``ipsc`` — collapsing them would merge
    the brightfield-input arm onto the phase baseline it is measured against.
    """
    canonical = checkpoint_dir("er", "fnet3d_paper", "ipsc__bf")
    for spelling in ("ipsc_confocal_bf", "ipsc_confocal_brightfield", "ipsc__bf"):
        assert checkpoint_dir("er", "fnet3d_paper", spelling) == canonical
    assert canonical == MODELS_ROOT / "ipsc__bf" / "er" / "fnet3d_paper" / "checkpoints"
    assert checkpoint_dir("er", "fnet3d_paper", "ipsc_confocal") != canonical


def test_legacy_joint_deconvgt_valid_only_for_er_mito() -> None:
    # Valid for ER/mito (closed legacy marker).
    p = eval_leaf("er", "fnet3d_paper", "joint__legacy_deconvgt", "a549", "mock", data_root=_DATA_ROOT)
    assert "joint__legacy_deconvgt" in p.parts
    # Invalid for nucleus/membrane.
    with pytest.raises(ValueError):
        eval_leaf("nucleus", "fnet3d_paper", "joint__legacy_deconvgt", "a549", "mock", data_root=_DATA_ROOT)


# ---------------------------------------------------------------------------
# checkpoint_dir / gt_cache_dir / pred_cache_dir / metrics_repo_dir
# ---------------------------------------------------------------------------


def test_checkpoint_dir() -> None:
    got = checkpoint_dir("er", "fnet3d_paper", "a549")
    assert got == MODELS_ROOT / "a549" / "er" / "fnet3d_paper" / "checkpoints"


def test_gt_cache_dir_case_inconsistency() -> None:
    # iPSC ER/mito use UPPERCASE gene; nucleus/membrane lowercase logical organelle.
    assert gt_cache_dir("er", "ipsc", data_root=_DATA_ROOT) == Path(_DATA_ROOT) / "ipsc/eval_cache/SEC61B"
    assert gt_cache_dir("mito", "ipsc", data_root=_DATA_ROOT) == Path(_DATA_ROOT) / "ipsc/eval_cache/TOMM20"
    assert gt_cache_dir("nucleus", "ipsc", data_root=_DATA_ROOT) == Path(_DATA_ROOT) / "ipsc/eval_cache/nucleus"
    # A549 lowercase gene + condition.
    root = Path(_DATA_ROOT)
    assert gt_cache_dir("er", "a549", "mock", data_root=_DATA_ROOT) == root / "a549/eval_cache/sec61b_mock"
    assert gt_cache_dir("nucleus", "a549", "zikv", data_root=_DATA_ROOT) == root / "a549/eval_cache/h2b_zikv"


def test_pred_cache_dir_track_separation() -> None:
    d = pred_cache_dir("nucleus", "celldiff_r2", "joint", "a549", "mock", data_root=_DATA_ROOT)
    ap = pred_cache_dir("nucleus", "celldiff_r2", "joint", "a549", "mock", track="instance_ap", data_root=_DATA_ROOT)
    assert "eval_cache_pred" in d.parts and "eval_cache_pred_instance_ap" not in d.parts
    assert "eval_cache_pred_instance_ap" in ap.parts


def test_metrics_repo_dir_mirrors_leaf() -> None:
    got = metrics_repo_dir("er", "fnet3d_paper", "a549", "a549", "mock", repo_root="/repo")
    assert got == Path("/repo/applications/dynacell/results/metrics/er/fnet3d_paper/a549/a549__mock")
    comp = metrics_repo_dir(
        "dual_nucleus_membrane", "fcmae_vscyto3d_pretrained", "ipsc", "ipsc", component="nucleus", repo_root="/repo"
    )
    assert comp.name == "nucleus"


def test_metrics_repo_dir_default_root_not_doubled() -> None:
    """The default repo_root must resolve to the repo root, not double `applications/`."""
    got = metrics_repo_dir("er", "fnet3d_paper", "a549", "a549", "mock")
    assert "applications/applications" not in str(got)
    assert got.parts[-8:] == (
        "applications",
        "dynacell",
        "results",
        "metrics",
        "er",
        "fnet3d_paper",
        "a549",
        "a549__mock",
    )
    # base is <repo>/applications/dynacell/results/metrics -> the repo root exists.
    assert (got.parents[7] / "applications" / "dynacell").is_dir()


def test_ablation_model_rejects_forward_train_set() -> None:
    """An ablation model is valid only with its ablation train_set, never a forward one."""
    with pytest.raises(ValueError):
        eval_leaf("nucleus", "fcmae_vscyto3d_pretrained_randinit", "a549", "ipsc")
    ok = eval_leaf("nucleus", "fcmae_vscyto3d_pretrained_randinit", "randinit", "ipsc", data_root="/d")
    assert ok == Path("/d/nucleus/fcmae_vscyto3d_pretrained_randinit/randinit/ipsc")


# ---------------------------------------------------------------------------
# iter_organelle_evals spans single-target + dual component subdirs
# ---------------------------------------------------------------------------


def test_iter_organelle_evals_spans_arities() -> None:
    dirs = iter_organelle_evals(
        "nucleus",
        models=["fcmae_vscyto3d_pretrained"],
        train_sets=["a549"],
        test_set="a549",
        condition="mock",
        data_root=_DATA_ROOT,
    )
    single = eval_leaf("nucleus", "fcmae_vscyto3d_pretrained", "a549", "a549", "mock", data_root=_DATA_ROOT)
    dual = eval_leaf(
        "dual_nucleus_membrane",
        "fcmae_vscyto3d_pretrained",
        "a549",
        "a549",
        "mock",
        component="nucleus",
        data_root=_DATA_ROOT,
    )
    assert single in dirs
    assert dual in dirs


# ---------------------------------------------------------------------------
# resolve_model — never trust bare model_name: celldiff
# ---------------------------------------------------------------------------


_MODELS_ROOT_STR = "/hpc/projects/comp.micro/virtual_staining/models/dynacell"


def test_resolve_model_celldiff_r2_from_ckpt() -> None:
    ckpt = f"{_MODELS_ROOT_STR}/cell_diff_vs_viscy/a549_mantis/sec61b/celldiff_r2/checkpoints/last.ckpt"
    assert resolve_model({"model_name": "celldiff"}, ckpt) == "celldiff_r2"


def test_resolve_model_joint_celldiff_is_r2() -> None:
    ckpt = f"{_MODELS_ROOT_STR}/cell_diff_vs_viscy/joint_ipsc_confocal_a549_mantis/nucleus/celldiff/checkpoints/l.ckpt"
    assert resolve_model({"model_name": "celldiff", "train_set": "joint"}, ckpt) == "celldiff_r2"


def test_resolve_model_unext2_dir_is_timm_scratch() -> None:
    ckpt = f"{_MODELS_ROOT_STR}/a549_mantis/er/unext2/checkpoints/last.ckpt"
    assert resolve_model({"model_name": "unext2"}, ckpt) == "unext2_timm_scratch"


def test_resolve_model_bare_unext2_name_is_ambiguous() -> None:
    with pytest.raises(ValueError):
        resolve_model({"model_name": "unext2"}, ckpt_path=None)


def test_resolve_model_deterministic_from_name() -> None:
    assert resolve_model({"model_name": "fnet3d_paper"}, None) == "fnet3d_paper"


@pytest.mark.parametrize(
    "variant",
    ["celldiff_r2_iterative", "celldiff_r2_sliding_window", "celldiff_r2_denoise"],
)
def test_resolve_model_celldiff_r2_variant_not_collapsed(variant: str) -> None:
    """R2 variants keep their own model key — a substring match would collapse them."""
    ckpt = f"{_MODELS_ROOT_STR}/cell_diff_vs_viscy/a549_mantis/sec61b/{variant}/checkpoints/last.ckpt"
    assert resolve_model({"model_name": "celldiff"}, ckpt) == variant


@pytest.mark.parametrize(
    ("run_dir_name", "expected"),
    [
        # recipe-suffixed run dirs -> canonical code key (the two live non-canonical forms)
        ("fcmae_vscyto3d_pretrained_ws8500", "fcmae_vscyto3d_pretrained"),
        ("pix2pix3d_unetvit_modernized_lambdaL1_10_lecam_40ep", "pix2pix3d_unetvit"),
        # already-canonical run-dir names pass through unchanged
        ("fcmae_vscyto3d_scratch", "fcmae_vscyto3d_scratch"),
        ("fnet3d_paper", "fnet3d_paper"),
        ("unetvit3d", "unetvit3d"),
        ("celldiff_r2", "celldiff_r2"),
        # longest-match: an exact ablation key wins over its prefix
        ("fcmae_vscyto3d_pretrained_randinit", "fcmae_vscyto3d_pretrained_randinit"),
        # longest-match: celldiff_r2_iterative wins over celldiff_r2 / celldiff
        ("celldiff_r2_iterative", "celldiff_r2_iterative"),
        # in-focus 2D track: canonical run dirs must not fall back to their 3D
        # namesakes (fcmae_vscyto2d_* is not a suffix-recipe of fcmae_vscyto3d_*).
        ("fcmae_vscyto2d_scratch", "fcmae_vscyto2d_scratch"),
        ("fcmae_vscyto2d_pretrained", "fcmae_vscyto2d_pretrained"),
        ("fnet2d", "fnet2d"),
    ],
)
def test_canonical_model_name(run_dir_name: str, expected: str) -> None:
    assert canonical_model_name(run_dir_name) == expected


def test_canonical_model_name_unknown_raises() -> None:
    with pytest.raises(ValueError):
        canonical_model_name("totally_unknown_model")


# ---------------------------------------------------------------------------
# normalize_legacy — one real on-disk path of each legacy form
# ---------------------------------------------------------------------------

_R = _DATA_ROOT

# (legacy_path, expected CanonicalKey or None)
_LEGACY_CASES: list[tuple[str, CanonicalKey | None]] = [
    # --- prediction zarrs -------------------------------------------------
    # iPSC test, iPSC-trained (no infix).
    (
        f"{_R}/ipsc/predictions/nucl_fnet3d_paper.zarr",
        CanonicalKey("nucleus", "fnet3d_paper", "ipsc", "ipsc"),
    ),
    # iPSC test, A549-trained (a549trained infix); nucleus -> stays a549 (raw).
    (
        f"{_R}/ipsc/predictions/memb_fcmae_vscyto3d_pretrained_a549trained.zarr",
        CanonicalKey("membrane", "fcmae_vscyto3d_pretrained", "a549", "ipsc"),
    ),
    # iPSC test, joint (jointtrained infix).
    (
        f"{_R}/ipsc/predictions/memb_fnet3d_paper_jointtrained.zarr",
        CanonicalKey("membrane", "fnet3d_paper", "joint", "ipsc"),
    ),
    # joint_predictions/ dir -> joint train_set (celldiff_r2, no infix).
    (
        f"{_R}/ipsc/joint_predictions/nucl_celldiff_r2.zarr",
        CanonicalKey("nucleus", "celldiff_r2", "joint", "ipsc"),
    ),
    # A549 test, modern form with condition.
    (
        f"{_R}/a549/predictions/memb_fcmae_vscyto3d_scratch_mock.zarr",
        CanonicalKey("membrane", "fcmae_vscyto3d_scratch", "ipsc", "a549", "mock"),
    ),
    # A549 test, a549trained + condition.
    (
        f"{_R}/a549/predictions/memb_fnet3d_paper_a549trained_denv.zarr",
        CanonicalKey("membrane", "fnet3d_paper", "a549", "a549", "denv"),
    ),
    # A549 test, jointtrained + condition.
    (
        f"{_R}/a549/joint_predictions/memb_fnet3d_paper_jointtrained_zikv.zarr",
        CanonicalKey("membrane", "fnet3d_paper", "joint", "a549", "zikv"),
    ),
    # Legacy ER double-underscore form, iPSC-trained -> ER a549 test scored raw;
    # bare (no infix left of `__`) = iPSC-trained; deconv-provenance is a GT-eval
    # concern for evals, not the prediction train_set (model IS iPSC-trained).
    (
        f"{_R}/a549/predictions/sec61b_fnet3d_paper__sec61b_mock.zarr",
        CanonicalKey("er", "fnet3d_paper", "ipsc", "a549", "mock"),
    ),
    # ER A549-trained prediction -> deconv-provenance train_set relabel.
    (
        f"{_R}/a549/predictions/sec61b_fnet3d_paper_a549trained_zikv.zarr",
        CanonicalKey("er", "fnet3d_paper", "a549__deconv", "a549", "zikv"),
    ),
    # Mito joint prediction -> legacy-joint-deconvgt marker.
    (
        f"{_R}/a549/predictions/tomm20_fcmae_vscyto3d_pretrained_jointtrained_mock.zarr",
        CanonicalKey("mito", "fcmae_vscyto3d_pretrained", "joint__legacy_deconvgt", "a549", "mock"),
    ),
    # dual_nucl_memb multi-target prediction (ablation infix cytoland -> SKIP).
    (
        f"{_R}/a549/predictions/dual_nucl_memb_fcmae_vscyto3d_pretrained_cytoland_mock.zarr",
        None,
    ),
    # Deliberate skips.
    (f"{_R}/ipsc/predictions/sec61b_fnet3d.zarr", None),  # alias-dup
    (f"{_R}/ipsc/predictions/sec61b_unext2.zarr", None),  # alias-dup
    # celldiff R1 is RECOGNIZED (distinct from R2) — recognition != migration.
    (
        f"{_R}/ipsc/predictions/memb_celldiff_iterative.zarr",
        CanonicalKey("membrane", "celldiff_iterative", "ipsc", "ipsc"),
    ),
    # Nonstandard `nucleus_` prefix (unetvit3d A549-test nucleus predicts) -> maps
    # like `nucl_` via longest-prefix match (iPSC-trained on A549 test).
    (
        f"{_R}/a549/predictions/nucleus_unetvit3d_denv.zarr",
        CanonicalKey("nucleus", "unetvit3d", "ipsc", "a549", "denv"),
    ),
    # --- eval dirs --------------------------------------------------------
    # iPSC-trained eval (with_embeddings), unext2 paper key -> code model.
    (
        f"{_R}/ipsc/evaluations_with_embeddings/eval_unext2_mitochondria",
        CanonicalKey("mito", "fcmae_vscyto3d_scratch", "ipsc", "ipsc"),
    ),
    # A549-trained eval (a549trained_with_embeddings) -> ER deconv-provenance.
    (
        f"{_R}/ipsc/evaluations_a549trained_with_embeddings/eval_unext2_a549trained_er",
        CanonicalKey("er", "fcmae_vscyto3d_scratch", "a549__deconv", "ipsc"),
    ),
    # Joint eval -> nucleus stays joint (raw).
    (
        f"{_R}/ipsc/evaluations_jointtrained_with_embeddings/eval_vscyto3d_jointtrained_membrane",
        CanonicalKey("membrane", "fcmae_vscyto3d_pretrained", "joint", "ipsc"),
    ),
    # A549 test iPSC-trained ER eval == legacy DECONV-GT track: DROPPED, NOT migrated -> None.
    (f"{_R}/a549/evaluations_with_embeddings/eval_fnet3d_er_mock", None),
    # A549 test iPSC-trained nucleus eval (raw).
    (
        f"{_R}/a549/evaluations_with_embeddings/eval_fnet3d_nucleus_denv",
        CanonicalKey("nucleus", "fnet3d_paper", "ipsc", "a549", "denv"),
    ),
    # Ablation eval parent -> ablation model + legacy train_set.
    (
        f"{_R}/ipsc/evaluations_randinit/eval_vscyto3d_randinit_nucleus",
        CanonicalKey("nucleus", "fcmae_vscyto3d_pretrained_randinit", "randinit", "ipsc"),
    ),
    # A549 ablation eval dir carries a trailing condition — must map (not UNMAPPED)
    # AND keep the condition (the branch strips the cond before the organelle).
    (
        f"{_R}/a549/evaluations_randinit/eval_vscyto3d_randinit_er_denv",
        CanonicalKey("er", "fcmae_vscyto3d_pretrained_randinit", "randinit", "a549", "denv"),
    ),
    # dynacell-FT ablation model eval parents (no _with_embeddings suffix) -> map to
    # the registered model + the parent's train_set (iPSC-FT base / a549-trained).
    (
        f"{_R}/a549/evaluations_cytolandft/eval_vscyto3d_cytolandft_nucleus_mock",
        CanonicalKey("nucleus", "vscyto3d_cytolandft", "ipsc", "a549", "mock"),
    ),
    (
        f"{_R}/a549/evaluations_infectionft_dynacellft_a549trained/eval_vscyto3d_infectionft_dynacellft_membrane_zikv",
        CanonicalKey("membrane", "vscyto3d_infectionft_dynacellft", "a549", "a549", "zikv"),
    ),
    # instance_ap eval parent -> track=instance_ap; a549trained infix.
    (
        f"{_R}/ipsc/evaluations_instance_ap/eval_celldiff_r2_a549trained_nucleus",
        CanonicalKey("nucleus", "celldiff_r2", "a549", "ipsc", track="instance_ap"),
    ),
    # Stale pre-2D eval parents -> None.
    (f"{_R}/ipsc/evaluations/eval_fnet3d_er", None),
    (f"{_R}/ipsc/joint_evaluations/eval_fnet3d_jointtrained_er", None),
    # Outside data_root -> None.
    ("/some/other/place/thing.zarr", None),
]


@pytest.mark.parametrize("legacy_path, expected", _LEGACY_CASES)
def test_normalize_legacy(legacy_path: str, expected: CanonicalKey | None) -> None:
    got = normalize_legacy(legacy_path, data_root=_DATA_ROOT)
    assert got == expected, f"{legacy_path} -> {got} (expected {expected})"


def test_normalize_legacy_no_two_paths_collide_across_forms() -> None:
    """Distinct legacy paths that DO map must not share a CanonicalKey unless intended.

    (evaluations_with_embeddings and evaluations_instance_ap of the same tuple are
    intentionally distinguished by ``track``, so they do NOT collide.)
    """
    mapped: dict[CanonicalKey, str] = {}
    for legacy_path, expected in _LEGACY_CASES:
        if expected is None:
            continue
        key = normalize_legacy(legacy_path, data_root=_DATA_ROOT)
        assert key is not None
        if key in mapped:
            # Two distinct on-disk paths -> same tuple: allowed only if the same
            # path (dedupe). Here every fixture path is distinct, so this is a
            # genuine collision and should fail.
            assert mapped[key] == legacy_path, (
                f"two distinct legacy paths collide on {key}: {mapped[key]} and {legacy_path}"
            )
        mapped[key] = legacy_path


# ---------------------------------------------------------------------------
# Retained display registry (both directions)
# ---------------------------------------------------------------------------


def test_paper_key_retained_entries() -> None:
    # pix2pix3d + 5 ablation keys must not regress.
    assert paper_key("pix2pix3d_unetvit") == "pix2pix3d"
    for m in (
        "fcmae_vscyto3d_pretrained_randinit",
        "fcmae_vscyto3d_pretrained_cytoland",
        "fcmae_vscyto3d_pretrained_infectionft",
        "vscyto3d_cytolandft",
        "vscyto3d_infectionft_dynacellft",
    ):
        assert m in PAPER_KEY
    # Newly-added live path tokens.
    assert paper_key("celldiff_r2") == "celldiff_r2"
    assert paper_key("unext2_timm_scratch") == "unext2_timm_scratch"


def test_paper_key_2d_track() -> None:
    """The 2D track's display names must stay distinct from their 3D namesakes.

    Pins the three in-focus 2D model keys: collapsing ``fcmae_vscyto2d_pretrained``
    onto ``vscyto3d`` (or the scratch pair onto ``unext2``) would silently merge the
    2D and 3D rows of the 2D-vs-3D comparison into one eval dir.
    """
    assert paper_key("fcmae_vscyto2d_scratch") == "unext2_2d"
    assert paper_key("fcmae_vscyto2d_pretrained") == "vscyto2d"
    assert paper_key("fnet2d") == "fnet2d"
    assert paper_key("fcmae_vscyto3d_scratch") == "unext2"
    assert paper_key("fcmae_vscyto3d_pretrained") == "vscyto3d"
    assert paper_key("fnet3d_paper") == "fnet3d"


def test_organelle_eval_target() -> None:
    assert ORGANELLE_EVAL_TARGET["er"] == "er_sec61b"
    assert ORGANELLE_EVAL_TARGET["mito"] == "mito_tomm20"
    assert ORGANELLE_EVAL_TARGET["nucleus"] == "nucleus"
    assert ORGANELLE_EVAL_TARGET["membrane"] == "membrane"


# ---------------------------------------------------------------------------
# HEK third-cell-type probe (NeurIPS response item O)
# ---------------------------------------------------------------------------

_HEK_ORGS = ("membrane", "mito")
_HEK_MODELS = ("fnet3d_paper", "fcmae_vscyto3d_pretrained", "celldiff_r2", "celldiff_r2_iterative")


@pytest.mark.parametrize("org", _HEK_ORGS)
@pytest.mark.parametrize("model", _HEK_MODELS)
@pytest.mark.parametrize("train", ("ipsc", "a549", "joint"))
def test_hek_roundtrip(org: str, model: str, train: str) -> None:
    """HEK tuples survive prediction_store -> key_from_prediction_store."""
    pred = prediction_store(org, model, train, "hek", "a549xy", data_root=_DATA_ROOT)
    leaf = eval_leaf(org, model, train, "hek", "a549xy", data_root=_DATA_ROOT)
    assert pred.parent == leaf
    assert leaf.relative_to(_DATA_ROOT).parts == (org, model, train, "hek__a549xy")
    key = key_from_prediction_store(pred, data_root=_DATA_ROOT)
    assert key == CanonicalKey(organelle=org, model=model, train_set=train, test_set="hek", condition="a549xy")


def test_hek_arm_is_required_and_validated() -> None:
    """The HEK condition slot carries the geometry arm, and only a known arm."""
    with pytest.raises(ValueError, match="geometry arm"):
        prediction_store("membrane", "fnet3d_paper", "a549", "hek", None, data_root=_DATA_ROOT)
    # An A549 treatment token is not a HEK arm.
    with pytest.raises(ValueError, match="unknown HEK arm"):
        prediction_store("membrane", "fnet3d_paper", "a549", "hek", "mock", data_root=_DATA_ROOT)
    with pytest.raises(ValueError, match="unknown HEK arm"):
        key_from_prediction_store(
            Path(_DATA_ROOT) / "membrane/fnet3d_paper/a549/hek__native/prediction.zarr",
            data_root=_DATA_ROOT,
        )


def test_hek_pred_cache_dir_lands_under_its_own_test_root() -> None:
    """HEK pred caches must not share a root with the a549/ipsc caches."""
    d = pred_cache_dir("mito", "fnet3d_paper", "joint", "hek", "a549xy", data_root=_DATA_ROOT)
    assert d == Path(_DATA_ROOT) / "hek/eval_cache_pred/mito/fnet3d_paper/joint/hek__a549xy"


def test_hek_gt_cache_dir_is_marker_and_arm_keyed() -> None:
    root = Path(_DATA_ROOT)
    assert gt_cache_dir("membrane", "hek", "a549xy", data_root=_DATA_ROOT) == root / "hek/eval_cache/kras_a549xy"
    assert gt_cache_dir("mito", "hek", "a549xy", data_root=_DATA_ROOT) == root / "hek/eval_cache/tomm70a_a549xy"


@pytest.mark.parametrize("organelle", ("nucleus", "er"))
def test_hek_gt_cache_dir_rejects_qc_failed_organelles(organelle: str) -> None:
    """Nucleus (HIST2H2BE) and ER (SEC61B) failed QC, so they have no HEK GT.

    Must be a ValueError like every other invalid tuple in this module, not the
    bare KeyError a raw dict lookup would raise.
    """
    with pytest.raises(ValueError, match="no HEK GT for organelle"):
        gt_cache_dir(organelle, "hek", "a549xy", data_root=_DATA_ROOT)


def test_hek_gt_cache_dir_requires_an_arm() -> None:
    with pytest.raises(ValueError, match="requires a geometry arm"):
        gt_cache_dir("membrane", "hek", None, data_root=_DATA_ROOT)


def test_hek_leaves_do_not_collide_with_a549_or_ipsc() -> None:
    """A HEK leaf is distinct from the same model's a549/ipsc leaves."""
    args = ("membrane", "fcmae_vscyto3d_pretrained", "a549")
    hek = eval_leaf(*args, "hek", "a549xy", data_root=_DATA_ROOT)
    ipsc = eval_leaf(*args, "ipsc", None, data_root=_DATA_ROOT)
    a549 = eval_leaf(*args, "a549", "mock", data_root=_DATA_ROOT)
    assert len({hek, ipsc, a549}) == 3


def test_normalize_legacy_ignores_hek() -> None:
    """There are no legacy HEK artifacts; the legacy mapper must not invent one."""
    assert normalize_legacy(Path(_DATA_ROOT) / "hek/eval_cache/kras_a549xy") is None
