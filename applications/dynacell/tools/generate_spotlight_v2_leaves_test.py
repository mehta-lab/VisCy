"""Tests for ``generate_spotlight_v2_leaves.py``."""

from __future__ import annotations

from collections import Counter

import pytest
import yaml
from generate_spotlight_v2_leaves import (
    _FIT_RENAMES,
    ARMS,
    BASELINES,
    BENCHMARKS,
    BG_TARGET_ARGS,
    ORGANELLES,
    POOL,
    SEED_SOURCES,
    SEG_AUX_WEIGHTS,
    SEGAUXSELF_BASELINES,
    WEIGHT_SCALES,
    Arm,
    _unseeded,
    allowed_diff,
    build_fit,
    build_leaves,
    changed_keys,
)
from jsonargparse import ArgumentParser
from lightning.pytorch import LightningModule

from dynacell._compose_hook import _dynacell_ref_resolver
from dynacell.evaluation.paths import PAPER_KEY, canonical_model_name
from viscy_utils.compose import load_composed_config
from viscy_utils.losses import BackgroundLowPass


def _suffix(leaf_dir_name: str) -> str:
    return next(arm.suffix for arm in ARMS if arm.model == leaf_dir_name)


@pytest.fixture(scope="module")
def leaves() -> dict:
    """One generation for the whole module (~6-8 s each); tests only read it."""
    return build_leaves()


def test_committed_leaves_match_the_generator(leaves: dict) -> None:
    """Every generated leaf on disk is byte-identical to a fresh generation."""
    stale = [str(p.relative_to(BENCHMARKS)) for p, t in leaves.items() if not p.is_file() or p.read_text() != t]
    assert not stale, f"regenerate with generate_spotlight_v2_leaves.py: {stale}"


def test_leaf_counts_per_arm(leaves: dict) -> None:
    """Count 85 fits and 342 predicts per arm (fit+predict).

    segaux 19+76, segauxself 8+32, seed1 9+36, v2 7+28, probes 3+6 (l1 adds A549), cjoint/ccond 2+8 each, last 0+8,
    l1segaux 1+4, l1seed1 1+4, segaux_seed1 10+40, v2_seed1 7+28,
    segaux_halfw 2+8, segaux_doublew 2+8, segaux_sauna 5+20, segaux_cldice 3+12, bglp 2+8, bgflat 2+8.

    The jointsteps and safecrop probes are iPSC-only; every other arm also predicts the 3 A549 legs.
    """
    fits = Counter(_suffix(p.parent.parent.name) for p in leaves if p.name == "train.yml")
    predicts = Counter(_suffix(p.parent.parent.name) for p in leaves if p.name != "train.yml")
    assert fits == {
        "segaux": 19,
        "segauxself": 8,
        "seed1": 9,
        "v2": 7,
        "jointsteps": 1,
        "l1": 1,
        "safecrop": 1,
        "cjoint": 2,
        "ccond": 2,
        "l1segaux": 1,
        "l1seed1": 1,
        "segaux_seed1": 10,
        "v2_seed1": 7,
        "segaux_halfw": 2,
        "segaux_doublew": 2,
        "segaux_sauna": 5,
        "segaux_cldice": 3,
        "bglp": 2,
        "bgflat": 2,
    }
    assert predicts == {
        "segaux": 76,
        "segauxself": 32,
        "last": 8,
        "seed1": 36,
        "v2": 28,
        "jointsteps": 1,
        "l1": 4,
        "safecrop": 1,
        "cjoint": 8,
        "ccond": 8,
        "l1segaux": 4,
        "l1seed1": 4,
        "segaux_seed1": 40,
        "v2_seed1": 28,
        "segaux_halfw": 8,
        "segaux_doublew": 8,
        "segaux_sauna": 20,
        "segaux_cldice": 12,
        "bglp": 8,
        "bgflat": 8,
    }


def test_every_arm_token_is_registered_and_does_not_collapse() -> None:
    """Each arm's model and store-dir token is its own PAPER_KEY entry, never a baseline's."""
    for arm in ARMS:
        for token in {arm.model, arm.store_dir}:
            assert token in PAPER_KEY, token
            assert canonical_model_name(token) == token


@pytest.mark.parametrize(
    ("baseline", "suffix", "expected"),
    [
        (
            "fnet2d",
            "segaux",
            {"data.init_args.fg_mask_key", "model.init_args.seg_aux", "model.init_args.seg_aux_weight"},
        ),
        (
            "celldiff",
            "segaux",
            {
                "data.init_args.fg_mask_key",
                "model.init_args.seg_aux",
                "model.init_args.seg_aux_weight",
                "model.init_args.seg_aux_t0",
            },
        ),
        ("fnet2d", "seed1", {"seed_everything"}),
        ("fcmae_vscyto3d_scratch", "v2", {"base.5"}),
        ("pix2pix3d_unetvit", "v2", set()),
    ],
)
def test_recipe_delta_is_only_the_arm_change(baseline: str, suffix: str, expected: set[str]) -> None:
    """Beyond the renames, a fit leaf changes exactly the arm's own keys."""
    src = BENCHMARKS / "nucleus" / baseline / POOL / BASELINES[baseline].fit_leaf
    base_cfg = yaml.safe_load(src.read_text())
    arm = Arm(baseline, suffix, ("nucleus",), a549=False)
    renames, _ = allowed_diff(arm, "fit")
    changed = changed_keys(base_cfg, build_fit(arm, "nucleus", base_cfg)) - renames
    collapsed = {"model.init_args.seg_aux" if k.startswith("model.init_args.seg_aux.") else k for k in changed}
    assert collapsed == expected


@pytest.mark.parametrize("baseline", ["fnet2d", "pix2pix3d_unetvit", "celldiff_2d"])
def test_segaux_composes_with_the_engine_args_and_keeps_the_baseline_recipe(baseline: str) -> None:
    """The composed segaux fit adds only the SegAuxDice args; loss and normalization stay the baseline's."""
    arm_leaf = BENCHMARKS / "membrane" / f"{baseline}_segaux" / POOL / "train.yml"
    base_leaf = BENCHMARKS / "membrane" / baseline / POOL / BASELINES[baseline].fit_leaf
    arm_cfg = load_composed_config(arm_leaf, resolver=_dynacell_ref_resolver)
    base_cfg = load_composed_config(base_leaf, resolver=_dynacell_ref_resolver)
    args = arm_cfg["model"]["init_args"]
    assert args["seg_aux"] == {"class_path": "viscy_utils.losses.SegAuxDice", "init_args": {"c": 0.1}}
    assert args["seg_aux_weight"] == SEG_AUX_WEIGHTS[("membrane", f"{baseline}_segaux")]
    assert arm_cfg["data"]["init_args"]["fg_mask_key"] == "fg_mask"
    assert "min_nonzero_fraction" not in arm_cfg["data"]["init_args"]
    for key in ("loss_function", "recon_loss", "lambda_l1", "lecam_gamma", "ema_kimg"):
        assert args.get(key) == base_cfg["model"]["init_args"].get(key)
    assert arm_cfg["data"]["init_args"]["normalizations"] == base_cfg["data"]["init_args"]["normalizations"]
    for key in ("max_epochs", "max_steps", "precision"):
        assert arm_cfg["trainer"].get(key) == base_cfg["trainer"].get(key)


def test_predict_leaves_point_at_the_arm_checkpoint_and_store(leaves: dict) -> None:
    """Predict leaves name the arm's own store dir, never the baseline's, and the arm's own ckpt dir.

    The exception is a predict-only arm (no fit), which reads its baseline's checkpoint dir.
    """
    for path, text in leaves.items():
        if path.name == "train.yml":
            continue
        cfg = yaml.safe_load(text)
        model, organelle = path.parent.parent.name, path.parent.parent.parent.name
        arm = next(a for a in ARMS if a.model == model)
        ckpt_dir = model if arm.fit else BASELINES[arm.baseline].ckpt_dir
        assert cfg["model"]["init_args"]["ckpt_path"].endswith(f"/ipsc/{organelle}/{ckpt_dir}/checkpoints/last.ckpt")
        store = cfg["trainer"]["callbacks"][0]["init_args"]["output_store"]
        assert f"/{organelle}/{arm.store_dir}/ipsc/" in store
        assert store.startswith(cfg["launcher"]["run_root"] + "/")


def test_ccond_predict_reads_the_same_leg_fnet_segaux_mask(leaves: dict) -> None:
    """C-cond predicts condition on the same-dim FNet segaux store of the same test leg, with no fg_mask_key."""
    fnet = {"celldiff_2d_ccond": "fnet2d_segaux", "celldiff_ccond": "fnet3d_paper_segaux"}
    n = 0
    for path, text in leaves.items():
        model = path.parent.parent.name
        if model not in fnet or path.name == "train.yml":
            continue
        cfg = yaml.safe_load(text)
        store = cfg["trainer"]["callbacks"][0]["init_args"]["output_store"]
        source = cfg["model"]["init_args"]["cond_mask_source"]["init_args"]
        leg = store.split("/ipsc/", 1)[1]
        assert (
            source["data_path"] == f"/hpc/projects/virtual_staining/training/dynacell/nucleus/{fnet[model]}/ipsc/{leg}"
        )
        assert source["channel"] == "Nuclei_prediction"
        # Per-window Otsu, which transfers across intensity scales (see CCOND_THRESHOLD).
        assert source["threshold"] == "otsu"
        assert "fg_mask_key" not in cfg.get("data", {}).get("init_args", {})
        n += 1
    assert n == 8


@pytest.mark.parametrize("organelle", ["nucleus", "membrane"])
@pytest.mark.parametrize("baseline", SEGAUXSELF_BASELINES)
def test_segauxself_differs_from_segaux_by_the_dice_label_only(baseline: str, organelle: str) -> None:
    """Composed, the segauxself fit equals the segaux fit except SegAuxDice label and the run identity."""
    load = lambda suffix: load_composed_config(  # noqa: E731
        BENCHMARKS / organelle / f"{baseline}_{suffix}" / POOL / "train.yml", resolver=_dynacell_ref_resolver
    )
    segaux, segauxself = load("segaux"), load("segauxself")
    renames, _ = allowed_diff(Arm(baseline, "segauxself", (organelle,), a549=False), "fit")
    assert changed_keys(segaux, segauxself) - renames == {"model.init_args.seg_aux.init_args.label"}
    assert segauxself["model"]["init_args"]["seg_aux"]["init_args"]["label"] == "target"


def test_last_arm_predicts_the_baselines_own_checkpoint_into_its_own_store(leaves: dict) -> None:
    """The predict-only `last` arm reads the baseline's ckpt dir but never writes the baseline's store."""
    last = [p for p in leaves if p.parent.parent.name == "pix2pix2d_unetvit_last"]
    assert len(last) == 8 and not any(p.name == "train.yml" for p in last)
    for p in last:
        cfg = yaml.safe_load(leaves[p])
        assert cfg["model"]["init_args"]["ckpt_path"].endswith("/pix2pix2d_unetvit/checkpoints/last.ckpt")
        store = cfg["trainer"]["callbacks"][0]["init_args"]["output_store"]
        assert "/pix2pix2d_unetvit_last/" in store and "/pix2pix2d_unetvit/" not in store


def test_l1_arms_differ_from_the_l1_probe_by_their_own_term_only() -> None:
    """Composed, l1segaux = l1 + the seg-aux keys and l1seed1 = l1 + the seed, nothing else."""
    load = lambda suffix: load_composed_config(  # noqa: E731
        BENCHMARKS / "membrane" / f"fcmae_vscyto2d_scratch_{suffix}" / POOL / "train.yml",
        resolver=_dynacell_ref_resolver,
    )
    l1 = load("l1")
    renames, _ = allowed_diff(Arm("fcmae_vscyto2d_scratch", "l1segaux", ("membrane",), a549=False), "fit")
    seg = changed_keys(l1, load("l1segaux")) - renames
    assert {"model.init_args.seg_aux" if k.startswith("model.init_args.seg_aux.") else k for k in seg} == {
        "data.init_args.fg_mask_key",
        "model.init_args.seg_aux",
        "model.init_args.seg_aux_weight",
    }
    assert changed_keys(l1, load("l1seed1")) - renames == {"seed_everything"}


def test_second_draws_differ_from_their_source_arm_by_the_seed_only(leaves: dict) -> None:
    """Second-draw fit leaves equal their source arm's, up to renames and ``seed_everything: 1``."""
    second_draws = [arm for arm in ARMS if arm.suffix in SEED_SOURCES]
    assert {arm.suffix for arm in second_draws} == set(SEED_SOURCES)
    for arm in second_draws:
        for organelle in arm.organelles:
            draw = yaml.safe_load(leaves[BENCHMARKS / organelle / arm.model / POOL / "train.yml"])
            src = yaml.safe_load(leaves[BENCHMARKS / organelle / _unseeded(arm).model / POOL / "train.yml"])
            assert draw.pop("seed_everything") == 1
            assert "seed_everything" not in src
            assert changed_keys(src, draw) == _FIT_RENAMES


def test_weight_sweep_arms_differ_from_the_segaux_arm_by_the_weight_only(leaves: dict) -> None:
    """Sweep fit leaves equal the segaux arm's, up to renames and ``seg_aux_weight`` x its scale."""
    sweep = [arm for arm in ARMS if arm.suffix in WEIGHT_SCALES]
    assert {arm.suffix for arm in sweep} == {"segaux_halfw", "segaux_doublew"}
    for arm in sweep:
        for organelle in arm.organelles:
            arm_leaf = yaml.safe_load(leaves[BENCHMARKS / organelle / arm.model / POOL / "train.yml"])
            src = yaml.safe_load(leaves[BENCHMARKS / organelle / f"{arm.baseline}_segaux" / POOL / "train.yml"])
            w_arm = arm_leaf["model"]["init_args"].pop("seg_aux_weight")
            w_src = src["model"]["init_args"].pop("seg_aux_weight")
            assert w_arm == pytest.approx(w_src * WEIGHT_SCALES[arm.suffix])
            assert changed_keys(src, arm_leaf) == _FIT_RENAMES


@pytest.mark.parametrize("organelle", ORGANELLES)
@pytest.mark.parametrize("suffix", list(BG_TARGET_ARGS))
def test_bg_target_arms_compose_and_instantiate_with_only_the_target_op_added(suffix: str, organelle: str) -> None:
    """Composed, each Track H fit is celldiff_2d + fg_mask_key + target_bg_lowpass; its model builds."""
    load = lambda model: load_composed_config(  # noqa: E731
        BENCHMARKS / organelle / model / POOL / "train.yml", resolver=_dynacell_ref_resolver
    )
    base, arm = load("celldiff_2d"), load(f"celldiff_2d_{suffix}")
    renames, _ = allowed_diff(Arm("celldiff_2d", suffix, (organelle,), a549=False), "fit")
    changed = changed_keys(base, arm) - renames
    op_key = "model.init_args.target_bg_lowpass"
    assert {op_key if k.startswith(op_key + ".") else k for k in changed} == {"data.init_args.fg_mask_key", op_key}
    assert arm["data"]["init_args"]["fg_mask_key"] == "fg_mask"
    # LightningModule as the base, as LightningCLI does: test_lazy_init evicts dynacell.* from
    # sys.modules, so a DynacellFlowMatching bound at import can differ from the one jsonargparse imports.
    parser = ArgumentParser()
    parser.add_subclass_arguments(LightningModule, "model")
    module = parser.instantiate(parser.parse_object({"model": arm["model"]})).model
    assert f"{type(module).__module__}.{type(module).__qualname__}" == "dynacell.engine.DynacellFlowMatching"
    op = module.target_bg_lowpass
    assert isinstance(op, BackgroundLowPass)
    assert {k: getattr(op, k) for k in BG_TARGET_ARGS[suffix]} == BG_TARGET_ARGS[suffix]
    assert (op.sigma_lp_z, op.sigma_feather_z, op.dilate_radius_z) == (0.0, 0.0, 0)
