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
    checkpoint_dir,
    eval_leaf,
    gt_cache_dir,
    iter_organelle_evals,
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
_FORWARD_TRAINS = ("ipsc", "a549", "joint")
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


def test_track_and_gt_repr_subdirs() -> None:
    base = eval_leaf("nucleus", "fnet3d_paper", "ipsc", "ipsc", data_root=_DATA_ROOT)
    ap = eval_leaf("nucleus", "fnet3d_paper", "ipsc", "ipsc", track="instance_ap", data_root=_DATA_ROOT)
    assert ap == base / "instance_ap"
    deconv = eval_leaf("er", "fnet3d_paper", "a549", "a549", "mock", gt_repr="deconv", data_root=_DATA_ROOT)
    assert deconv == eval_leaf("er", "fnet3d_paper", "a549", "a549", "mock", data_root=_DATA_ROOT) / "deconv_gt"


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


def test_gt_cache_dir_deconv_variant() -> None:
    got = gt_cache_dir("er", "a549", "denv", gt_repr="deconv", data_root=_DATA_ROOT)
    assert got == Path(_DATA_ROOT) / "a549/eval_cache/sec61b_denv_deconv"
    with pytest.raises(ValueError):
        gt_cache_dir("nucleus", "a549", "mock", gt_repr="deconv", data_root=_DATA_ROOT)


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
    # A549 test iPSC-trained ER eval -> deconv-GT track (gt_repr=deconv), train_set stays ipsc.
    (
        f"{_R}/a549/evaluations_with_embeddings/eval_fnet3d_er_mock",
        CanonicalKey("er", "fnet3d_paper", "ipsc", "a549", "mock", gt_repr="deconv"),
    ),
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


def test_organelle_eval_target_gt_repr_aware() -> None:
    assert ORGANELLE_EVAL_TARGET["er"] == "er_sec61b"
    assert ORGANELLE_EVAL_TARGET.for_repr("er", "deconv") == "er_sec61b_deconvolved"
    assert ORGANELLE_EVAL_TARGET.for_repr("mito", "deconv") == "mito_tomm20_deconvolved"
    with pytest.raises(ValueError):
        ORGANELLE_EVAL_TARGET.for_repr("nucleus", "deconv")
