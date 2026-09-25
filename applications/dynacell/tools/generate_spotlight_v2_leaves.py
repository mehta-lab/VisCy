#!/usr/bin/env python3
"""Generate the Spotlight-v2 first-wave fit and predict leaves from their baselines.

Plan: ``~/.claude/plans/spotlight-v2.md`` (Stage 0 0b/0c/0g, Stage 1 item 5). Every
arm is a copy of its BASELINE leaf (never of a v1 ``*_spotlight`` leaf) that differs
only in the arm's own change plus the renames that give it its own run dir, wandb
name and job name. :func:`allowed_diff` names both sets per arm, and
:func:`build_leaves` refuses to emit a leaf whose YAML differs from its baseline
anywhere else.

Arms (``<baseline>_<suffix>``):

- ``segaux`` -- 8 models x nucleus/membrane. Data gains ``fg_mask_key: fg_mask``
  (NOT v1's ``min_nonzero_fraction``, which changes patch sampling); the model gains
  ``seg_aux: SegAuxDice(c=0.1)`` and ``seg_aux_weight`` (flow models also
  ``seg_aux_t0: 0.7``). ``seg_aux_weight`` comes from SEG_AUX_WEIGHTS, the
  per-cell gradient-norm calibration. The base loss, normalization,
  sampling, precision, budget and monitored metric are the baseline's, and
  ``loss/validate`` stays base-only in all three engines, so ``--ckpt best`` selects
  on the baseline's criterion. The CellDiff-3D arm's checkpoints go to
  ``celldiff_segaux`` and its stores to ``celldiff_segaux_iterative``, mirroring
  the baseline's ``celldiff_r2`` / ``celldiff_r2_iterative`` split.
- ``seed1`` -- noise-floor replicates: ``seed_everything: 1``, nothing else.
  The global seed also redraws the FOV split, so the spread is init + split.
- ``v2`` -- fresh baselines retrained on today's code under their own run roots,
  for the two 3D models whose existing baseline cannot serve as the control:
  ``fcmae_vscyto3d_scratch`` (the April run stopped at ep 111/147 of 200 under an
  older MS-SSIM) and ``pix2pix3d_unetvit`` (Run D; the checkpoint-write halt froze
  its selectable checkpoints at ep <= 25 of 40, while its S arm will have all 40).
  Each model's ``_segaux`` arm is generated from the same baseline leaf in the same
  pass, so the arm and its v2 control share one recipe by construction.
- UNeXt2-3D wall: ``fcmae_vscyto3d_scratch_{v2,segaux}`` compose
  ``hardware_4gpu_long.yml`` (7 d) instead of ``hardware_4gpu.yml`` (4 d). Measured
  from consecutive April checkpoint mtimes of the same 4-GPU recipe (one
  allocation each): nucleus e96 00:39:27 -> e98 01:38:24 (2.04 ep/h), e98 ->
  e111 08:09:59 (1.99 ep/h); membrane e134 09:21:15 -> e136 10:21:36 (1.99 ep/h),
  e146 15:52:49 -> e147 16:22:49 (2.00 ep/h), all 2026-04-30. 200 epochs from
  scratch at ~2.0 ep/h is ~100 h > 96 h, and 1.68x inside 168 h.
- ``jointsteps`` / ``l1`` / ``safecrop`` -- UNeXt2-2D membrane recipe probes for the
  failed baseline fit (train loss plateau at 0.50). ``jointsteps``: ``max_epochs:
  320`` so the step budget equals the joint membrane run's. Measured from the
  runs' final checkpoints: iPSC ``latest-epoch=199-step=100000`` (500 steps/epoch,
  100000 steps) vs joint ``latest-epoch=199-step=160000`` (800 steps/epoch,
  160000 steps), i.e. 1.6x, not 2x; 320 x 500 = 160000. Per-step samples are
  equal (16) in both. ``l1``: ``MixedLoss(l1_alpha=1, l2_alpha=0, ms_dssim_alpha=0)``. ``safecrop``:
  the data overlay's ``gpu_augmentations`` with ``safe_crop_size: [1, 384, 384]``
  and ``safe_crop_coverage: 0.9`` on the affine, as ``pix2pix2d_unetvit`` does.
- ``cjoint`` / ``ccond`` -- Stage 1b CellDiff (2D + 3D, nucleus only for now).
  ``cjoint`` generates ``[image, mask]`` as one flow (``net_config.in_channels: 2``,
  ``mask_mode: joint``, gated mask Dice with a placeholder ``mask_dice_weight``);
  its predict leaves carry the same two model args. ``ccond`` adds the mask as a
  second conditioning channel (``net_config.cond_channels: 2``, ``mask_mode: cond``)
  seen through ``MaskCorruption`` in training; its predict leaves read the mask
  from the same-dim FNet ``_segaux`` store of the same test leg via
  ``CondMaskSource`` (``Nuclei_prediction``, threshold a string PLACEHOLDER so an
  untuned submit fails at parse) and set no ``fg_mask_key``.

Leaves are emitted with ``yaml.safe_dump``, so they carry no inline comments: the
recipe rationale stays in the baseline leaf each header names, and the reasons for
each arm's delta live here. Predict leaves name ``last.ckpt`` as a PLACEHOLDER
``ckpt_path`` (``--ckpt best`` resolves the checkpoint directory from it): submit
them with ``--ckpt best`` and re-bake afterwards.

Usage::

    python applications/dynacell/tools/generate_spotlight_v2_leaves.py --check
    python applications/dynacell/tools/generate_spotlight_v2_leaves.py
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

BENCHMARKS = Path(__file__).resolve().parents[1] / "configs" / "benchmarks" / "virtual_staining"
MODELS_ROOT = "/hpc/projects/comp.micro/virtual_staining/models/dynacell"
ORGANELLES: tuple[str, ...] = ("nucleus", "membrane")
POOL = "ipsc_confocal"
A549_PREDICTS: tuple[str, ...] = (
    "predict__a549_mantis_mock.yml",
    "predict__a549_mantis_denv.yml",
    "predict__a549_mantis_zikv.yml",
)
SEG_AUX_C = 0.1
SEG_AUX_T0 = 0.7
# seg_aux_weight per (organelle, arm): the median over 20 seeded val batches of
# ||grad base|| / ||grad Dice|| at each BASELINE checkpoint (contrast knee, c=0.1),
# rounded to 2 significant figures, so the Dice term starts with the same gradient
# norm as the usual loss. It varies 0.36-16 even within a family, hence per cell.
# Source: experiments/2026-09-24_spotlight-v2/calibration/summary.csv (w_star_contrast).
SEG_AUX_WEIGHTS: dict[tuple[str, str], float] = {
    ("nucleus", "fnet2d_segaux"): 2.9,
    ("nucleus", "fnet3d_paper_segaux"): 1.9,
    ("nucleus", "fcmae_vscyto2d_scratch_segaux"): 0.73,
    ("nucleus", "fcmae_vscyto3d_scratch_segaux"): 0.82,
    ("nucleus", "pix2pix2d_unetvit_segaux"): 16.0,
    ("nucleus", "pix2pix3d_unetvit_segaux"): 2.5,
    ("nucleus", "celldiff_2d_segaux"): 2.5,
    ("nucleus", "celldiff_segaux"): 1.4,
    ("membrane", "fnet2d_segaux"): 0.63,
    ("membrane", "fnet3d_paper_segaux"): 2.2,
    ("membrane", "fcmae_vscyto2d_scratch_segaux"): 0.36,
    ("membrane", "fcmae_vscyto3d_scratch_segaux"): 0.86,
    ("membrane", "pix2pix2d_unetvit_segaux"): 1.2,
    ("membrane", "pix2pix3d_unetvit_segaux"): 3.1,
    ("membrane", "celldiff_2d_segaux"): 0.58,
    ("membrane", "celldiff_segaux"): 1.0,
}
# C-joint's mask Dice has no baseline to calibrate against (the baseline has no
# mask channel); it borrows the same-dim CellDiff seg-aux weight as a starting point.
MASK_DICE_WEIGHTS: dict[tuple[str, str], float] = {
    ("nucleus", "celldiff_2d_cjoint"): 2.5,
    ("nucleus", "celldiff_cjoint"): 1.4,
}
JOINTSTEPS_MAX_EPOCHS = 320
LONG_WALL_MODELS: frozenset[str] = frozenset({"fcmae_vscyto3d_scratch_v2", "fcmae_vscyto3d_scratch_segaux"})
_WALL_4GPU = "launcher_profiles/hardware_4gpu.yml"
_WALL_4GPU_LONG = "launcher_profiles/hardware_4gpu_long.yml"
SAFE_CROP_SIZE = [1, 384, 384]
MASK_VELOCITY_WEIGHT = 1.0
# C-cond predict conditions on the same-dimensionality FNet `_segaux` prediction
# for the same test leg, thresholded at a val-tuned t. The store path is the
# planned one (it exists once that predict completes); the threshold is a STRING
# placeholder on purpose, so a submit before it is tuned fails at config parse
# instead of silently conditioning on an all-background (or all-foreground) mask.
CCOND_MASK_MODEL: dict[str, str] = {"celldiff_2d": "fnet2d_segaux", "celldiff": "fnet3d_paper_segaux"}
CCOND_MASK_CHANNEL = "Nuclei_prediction"
CCOND_THRESHOLD_PLACEHOLDER = "PLACEHOLDER_val_tuned_threshold"
SAFE_CROP_COVERAGE = 0.9
_UNEXT2_2D_DATA_OVERLAY = BENCHMARKS / "_internal/shared/model/data_overlays/fcmae_vscyto2d_fit.yml"


@dataclass(frozen=True)
class Baseline:
    """Where a baseline model's leaves, checkpoints and prediction stores live."""

    fit_leaf: str
    ckpt_dir: str
    store_dir: str
    ipsc_predict: str
    engine: str  # "unet" | "gan" | "flow"


BASELINES: dict[str, Baseline] = {
    "fnet2d": Baseline("train.yml", "fnet2d", "fnet2d", "predict__ipsc_confocal.yml", "unet"),
    "fnet3d_paper": Baseline("train.yml", "fnet3d_paper", "fnet3d_paper", "predict__ipsc_confocal.yml", "unet"),
    "fcmae_vscyto2d_scratch": Baseline(
        "train.yml", "fcmae_vscyto2d_scratch", "fcmae_vscyto2d_scratch", "predict__ipsc_confocal.yml", "unet"
    ),
    "fcmae_vscyto3d_scratch": Baseline(
        "train.yml", "fcmae_vscyto3d_scratch", "fcmae_vscyto3d_scratch", "predict__ipsc_confocal.yml", "unet"
    ),
    "pix2pix2d_unetvit": Baseline(
        "train.yml", "pix2pix2d_unetvit", "pix2pix2d_unetvit", "predict__ipsc_confocal.yml", "gan"
    ),
    # Run D is train_4gpu_modernized.yml; the sibling train.yml is the stale LSGAN
    # overlay (checkpoint hparams: nonsat, lambda_l1=10, lecam_gamma=0.3, ema_kimg=10).
    "pix2pix3d_unetvit": Baseline(
        "train_4gpu_modernized.yml", "pix2pix3d_unetvit", "pix2pix3d_unetvit", "predict__ipsc_confocal.yml", "gan"
    ),
    "celldiff_2d": Baseline("train.yml", "celldiff_2d", "celldiff_2d", "predict__ipsc_confocal.yml", "flow"),
    "celldiff": Baseline(
        "train.yml", "celldiff_r2", "celldiff_r2_iterative", "predict__ipsc_confocal__iterative.yml", "flow"
    ),
}


@dataclass(frozen=True)
class Arm:
    """One (baseline, suffix) arm, emitted for each of ``organelles``."""

    baseline: str
    suffix: str
    organelles: tuple[str, ...]
    a549: bool

    @property
    def model(self) -> str:
        """The arm's model code key, run-dir token and checkpoint-dir token."""
        return f"{self.baseline}_{self.suffix}"

    @property
    def store_dir(self) -> str:
        """The arm's prediction-store dir token (``celldiff_r2_iterative`` -> ``celldiff_segaux_iterative``)."""
        base = BASELINES[self.baseline]
        return base.store_dir.replace(base.ckpt_dir, self.model, 1)


ARMS: tuple[Arm, ...] = (
    *(Arm(m, "segaux", ORGANELLES, a549=True) for m in BASELINES),
    *(
        Arm(m, "seed1", ORGANELLES, a549=False)
        for m in ("fnet2d", "fcmae_vscyto2d_scratch", "pix2pix2d_unetvit", "celldiff_2d")
    ),
    Arm("fnet3d_paper", "seed1", ("nucleus",), a549=False),
    Arm("fcmae_vscyto3d_scratch", "v2", ORGANELLES, a549=True),
    Arm("pix2pix3d_unetvit", "v2", ORGANELLES, a549=True),
    *(Arm("fcmae_vscyto2d_scratch", s, ("membrane",), a549=False) for s in ("jointsteps", "l1", "safecrop")),
    # Stage 1b, nucleus first: segmentation inside the generative process.
    *(Arm(m, s, ("nucleus",), a549=True) for s in ("cjoint", "ccond") for m in ("celldiff_2d", "celldiff")),
)

_DESCRIPTION: dict[str, str] = {
    "segaux": "data fg_mask_key: fg_mask; model seg_aux (SegAuxDice c=0.1) + seg_aux_weight (+ seg_aux_t0 for flow)",
    "seed1": "seed_everything: 1",
    "v2": "nothing (fresh retrain of the baseline recipe under its own run root)",
    "jointsteps": f"trainer max_epochs: {JOINTSTEPS_MAX_EPOCHS}. Step budget matched to the joint membrane run, "
    "measured from final checkpoints: baseline latest-epoch=199-step=100000 (500 steps/ep), joint "
    "latest-epoch=199-step=160000 (800 steps/ep); 320 x 500 = 160000",
    "l1": "MixedLoss l1_alpha 1.0 / l2_alpha 0.0 / ms_dssim_alpha 0.0 (L1 only)",
    "safecrop": f"affine safe_crop_size {SAFE_CROP_SIZE} + safe_crop_coverage {SAFE_CROP_COVERAGE}",
    "cjoint": "data fg_mask_key: fg_mask; model net_config.in_channels 2, mask_mode joint, mask_dice_weight, "
    f"mask_velocity_weight {MASK_VELOCITY_WEIGHT}, seg_aux_t0 {SEG_AUX_T0}",
    "ccond": "data fg_mask_key: fg_mask; model net_config.cond_channels 2, mask_mode cond, mask_corruption "
    "(MaskCorruption defaults)",
}

# Keys every fit / predict arm renames so it owns its run dir, wandb run and job.
_FIT_RENAMES: frozenset[str] = frozenset(
    {
        "benchmark.model_name",
        "benchmark.experiment_id",
        "trainer.logger.init_args.name",
        "trainer.logger.init_args.save_dir",
        "trainer.callbacks.1.init_args.dirpath",
        "launcher.job_name",
        "launcher.run_root",
    }
)
_PREDICT_RENAMES: frozenset[str] = frozenset(
    {
        "benchmark.model_name",
        "benchmark.experiment_id",
        "model.init_args.ckpt_path",
        "trainer.callbacks.0.init_args.output_store",
        "launcher.job_name",
        "launcher.run_root",
    }
)


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts/lists to ``{"a.b.0.c": leaf}``."""
    if isinstance(node, dict):
        items = node.items()
    elif isinstance(node, list):
        items = enumerate(node)
    else:
        return {prefix: node}
    out: dict[str, Any] = {}
    for key, value in items:
        out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    if not out:  # empty container
        out[prefix] = node
    return out


def changed_keys(baseline: dict, arm: dict) -> set[str]:
    """Return the flattened key paths whose value differs between two leaf dicts."""
    a, b = _flatten(baseline), _flatten(arm)
    return {k for k in a.keys() | b.keys() if a.get(k, object()) != b.get(k, object())}


def allowed_diff(arm: Arm, kind: str) -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(renames, recipe_keys)`` an arm's ``kind`` ("fit"/"predict") leaf may change.

    ``recipe_keys`` entries are prefixes: every flattened key below one is allowed.
    """
    if kind == "predict":
        if arm.suffix == "cjoint":
            return _PREDICT_RENAMES, frozenset({"model.init_args.net_config.in_channels", "model.init_args.mask_mode"})
        if arm.suffix == "ccond":
            return _PREDICT_RENAMES, frozenset(
                {
                    "model.init_args.net_config.cond_channels",
                    "model.init_args.mask_mode",
                    "model.init_args.cond_mask_source",
                }
            )
        return _PREDICT_RENAMES, frozenset()
    recipe: set[str] = set()
    if arm.suffix == "segaux":
        recipe |= {"data.init_args.fg_mask_key", "model.init_args.seg_aux", "model.init_args.seg_aux_weight"}
        if BASELINES[arm.baseline].engine == "flow":
            recipe.add("model.init_args.seg_aux_t0")
    elif arm.suffix == "seed1":
        recipe.add("seed_everything")
    elif arm.suffix == "jointsteps":
        recipe.add("trainer.max_epochs")
    elif arm.suffix == "l1":
        recipe.add("model.init_args.loss_function")
    elif arm.suffix == "safecrop":
        recipe.add("data.init_args.gpu_augmentations")
    elif arm.suffix == "cjoint":
        recipe |= {
            "data.init_args.fg_mask_key",
            "model.init_args.net_config.in_channels",
            "model.init_args.mask_mode",
            "model.init_args.mask_dice_weight",
            "model.init_args.mask_velocity_weight",
            "model.init_args.seg_aux_t0",
        }
    elif arm.suffix == "ccond":
        recipe |= {
            "data.init_args.fg_mask_key",
            "model.init_args.net_config.cond_channels",
            "model.init_args.mask_mode",
            "model.init_args.mask_corruption",
        }
    if arm.model in LONG_WALL_MODELS:
        recipe.add("base")
    return _FIT_RENAMES, frozenset(recipe)


def _unexpected(keys: set[str], renames: frozenset[str], recipe: frozenset[str]) -> set[str]:
    return {k for k in keys if k not in renames and not any(k == r or k.startswith(r + ".") for r in recipe)}


def _replace_segment(value: str, old: str, new: str) -> str:
    """Replace the path segment ``old`` with ``new`` exactly once in ``value``."""
    parts = value.split("/")
    if parts.count(old) != 1:
        raise ValueError(f"expected exactly one {old!r} path segment in {value!r}")
    return "/".join(new if p == old else p for p in parts)


def _safecrop_gpu_augmentations() -> list[dict]:
    """Return the UNeXt2-2D data overlay's ``gpu_augmentations`` with a safe-crop affine."""
    augs = copy.deepcopy(yaml.safe_load(_UNEXT2_2D_DATA_OVERLAY.read_text())["data"]["init_args"]["gpu_augmentations"])
    affines = [a for a in augs if a["class_path"].endswith("BatchedRandAffined")]
    if len(affines) != 1:
        raise ValueError(f"expected one BatchedRandAffined in {_UNEXT2_2D_DATA_OVERLAY}, got {len(affines)}")
    affines[0]["init_args"]["safe_crop_size"] = list(SAFE_CROP_SIZE)
    affines[0]["init_args"]["safe_crop_coverage"] = SAFE_CROP_COVERAGE
    return augs


def _apply_recipe(arm: Arm, organelle: str, cfg: dict) -> None:
    """Apply the arm's recipe delta to a fit-leaf dict in place."""
    data_args = cfg.setdefault("data", {}).setdefault("init_args", {})
    model_args = cfg.setdefault("model", {}).setdefault("init_args", {})
    if arm.suffix == "segaux":
        data_args["fg_mask_key"] = "fg_mask"
        model_args["seg_aux"] = {"class_path": "viscy_utils.losses.SegAuxDice", "init_args": {"c": SEG_AUX_C}}
        model_args["seg_aux_weight"] = SEG_AUX_WEIGHTS[(organelle, arm.model)]
        if BASELINES[arm.baseline].engine == "flow":
            model_args["seg_aux_t0"] = SEG_AUX_T0
    elif arm.suffix == "seed1":
        cfg["seed_everything"] = 1
    elif arm.suffix == "jointsteps":
        cfg.setdefault("trainer", {})["max_epochs"] = JOINTSTEPS_MAX_EPOCHS
    elif arm.suffix == "l1":
        model_args["loss_function"] = {
            "class_path": "viscy_utils.losses.MixedLoss",
            "init_args": {"l1_alpha": 1.0, "l2_alpha": 0.0, "ms_dssim_alpha": 0.0},
        }
    elif arm.suffix == "safecrop":
        data_args["gpu_augmentations"] = _safecrop_gpu_augmentations()
    elif arm.suffix == "cjoint":
        data_args["fg_mask_key"] = "fg_mask"
        model_args.setdefault("net_config", {})["in_channels"] = 2
        model_args["mask_mode"] = "joint"
        model_args["mask_dice_weight"] = MASK_DICE_WEIGHTS[(organelle, arm.model)]
        model_args["mask_velocity_weight"] = MASK_VELOCITY_WEIGHT
        model_args["seg_aux_t0"] = SEG_AUX_T0
    elif arm.suffix == "ccond":
        data_args["fg_mask_key"] = "fg_mask"
        model_args.setdefault("net_config", {})["cond_channels"] = 2
        model_args["mask_mode"] = "cond"
        model_args["mask_corruption"] = {"class_path": "dynacell.mask_conditioning.MaskCorruption"}
    elif arm.suffix != "v2":
        raise ValueError(f"unknown arm suffix {arm.suffix!r}")
    # Drop containers the arm did not need (v2/seed1 add no data/model keys).
    for key in ("data", "model"):
        if cfg[key] == {"init_args": {}}:
            del cfg[key]


def build_fit(arm: Arm, organelle: str, baseline_cfg: dict) -> dict:
    """Return the arm's fit-leaf dict, derived from the baseline fit-leaf dict."""
    base = BASELINES[arm.baseline]
    cfg = copy.deepcopy(baseline_cfg)
    cfg["benchmark"]["model_name"] = arm.model
    cfg["benchmark"]["experiment_id"] = f"{organelle}__{POOL}__{arm.model}"
    logger = cfg["trainer"]["logger"]["init_args"]
    logger["name"] = f"{logger['name']}_{arm.suffix}"
    logger["save_dir"] = _replace_segment(logger["save_dir"], base.ckpt_dir, arm.model)
    ckpt_cbs = [cb for cb in cfg["trainer"]["callbacks"] if cb["class_path"].endswith("ModelCheckpoint")]
    if len(ckpt_cbs) != 1:
        raise ValueError(f"{arm.model}/{organelle}: expected one ModelCheckpoint, got {len(ckpt_cbs)}")
    ckpt_cbs[0]["init_args"]["dirpath"] = _replace_segment(
        ckpt_cbs[0]["init_args"]["dirpath"], base.ckpt_dir, arm.model
    )
    cfg["launcher"]["job_name"] = f"{cfg['launcher']['job_name']}_{arm.suffix}"
    cfg["launcher"]["run_root"] = _replace_segment(cfg["launcher"]["run_root"], base.ckpt_dir, arm.model)
    _apply_recipe(arm, organelle, cfg)
    if arm.model in LONG_WALL_MODELS:
        walls = [i for i, entry in enumerate(cfg["base"]) if entry.endswith(_WALL_4GPU)]
        if len(walls) != 1:
            raise ValueError(f"{arm.model}/{organelle}: expected one {_WALL_4GPU} in base, got {len(walls)}")
        cfg["base"][walls[0]] = cfg["base"][walls[0]].replace(_WALL_4GPU, _WALL_4GPU_LONG)
    return cfg


def build_predict(arm: Arm, organelle: str, baseline_cfg: dict) -> dict:
    """Return the arm's predict-leaf dict, derived from the baseline predict-leaf dict."""
    base = BASELINES[arm.baseline]
    cfg = copy.deepcopy(baseline_cfg)
    bench = cfg["benchmark"]
    old = f"__{bench['model_name']}__"
    if bench["experiment_id"].count(old) != 1:
        raise ValueError(f"expected one {old!r} in experiment_id {bench['experiment_id']!r}")
    bench["experiment_id"] = bench["experiment_id"].replace(old, f"__{arm.model}__")
    bench["model_name"] = arm.model
    cfg["model"]["init_args"]["ckpt_path"] = f"{MODELS_ROOT}/ipsc/{organelle}/{arm.model}/checkpoints/last.ckpt"
    writers = [cb for cb in cfg["trainer"]["callbacks"] if cb["class_path"].endswith("HCSPredictionWriter")]
    if len(writers) != 1:
        raise ValueError(f"{arm.model}/{organelle}: expected one HCSPredictionWriter, got {len(writers)}")
    writer = writers[0]["init_args"]
    writer["output_store"] = _replace_segment(writer["output_store"], base.store_dir, arm.store_dir)
    cfg["launcher"]["job_name"] = f"{cfg['launcher']['job_name']}_{arm.suffix}"
    cfg["launcher"]["run_root"] = _replace_segment(cfg["launcher"]["run_root"], base.store_dir, arm.store_dir)
    model_args = cfg["model"]["init_args"]
    if arm.suffix == "cjoint":
        model_args.setdefault("net_config", {})["in_channels"] = 2
        model_args["mask_mode"] = "joint"
    elif arm.suffix == "ccond":
        model_args.setdefault("net_config", {})["cond_channels"] = 2
        model_args["mask_mode"] = "cond"
        mask_store = _replace_segment(writer["output_store"], arm.store_dir, CCOND_MASK_MODEL[arm.baseline])
        model_args["cond_mask_source"] = {
            "class_path": "dynacell.mask_conditioning.CondMaskSource",
            "init_args": {
                "data_path": mask_store,
                "channel": CCOND_MASK_CHANNEL,
                "threshold": CCOND_THRESHOLD_PLACEHOLDER,
            },
        }
    return cfg


def _header(arm: Arm, organelle: str, baseline_leaf: Path, kind: str) -> str:
    rel = baseline_leaf.relative_to(BENCHMARKS)
    lines = [
        "# GENERATED by applications/dynacell/tools/generate_spotlight_v2_leaves.py -- edit the",
        "# generator, not this file (plan: ~/.claude/plans/spotlight-v2.md).",
        f"# Spotlight-v2 `{arm.suffix}` arm ({kind}) of the baseline leaf {rel}.",
    ]
    if kind == "fit":
        lines.append(f"# Recipe delta vs that leaf: {_DESCRIPTION[arm.suffix]}.")
        if arm.model in LONG_WALL_MODELS:
            lines.append(
                "# Wall: hardware_4gpu_long (7 d); 200 epochs at the measured ~2.0 ep/h is ~100 h > 4 d"
                " (checkpoint-mtime pairs in the generator docstring)."
            )
    else:
        lines.append("# Inference is identical to the baseline's; only the checkpoint, store and names differ.")
        lines.append("# ckpt_path is a PLACEHOLDER (last.ckpt): submit with `--ckpt best`, re-bake afterwards.")
    lines.append("# Everything else, and the rationale for it, is the baseline leaf's.")
    return "\n".join(lines) + "\n"


def _render(cfg: dict, header: str) -> str:
    text = yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False)
    threshold_line = f"        threshold: {CCOND_THRESHOLD_PLACEHOLDER}\n"
    text = text.replace(
        threshold_line,
        "        # PLACEHOLDER: set to the threshold tuned on the val FOVs (plan Stage 1b C-cond);\n"
        "        # data_path is the planned FNet _segaux store, which must be COMPLETE first.\n" + threshold_line,
    )
    return header + text


def build_leaves(benchmarks: Path = BENCHMARKS) -> dict[Path, str]:
    """Return ``{leaf_path: text}`` for every arm leaf, each checked against its baseline.

    Raises
    ------
    ValueError
        If any emitted leaf differs from its baseline outside :func:`allowed_diff`.
    """
    out: dict[Path, str] = {}
    for arm in ARMS:
        base = BASELINES[arm.baseline]
        for organelle in arm.organelles:
            src_dir = benchmarks / organelle / arm.baseline / POOL
            dst_dir = benchmarks / organelle / arm.model / POOL
            jobs = [("fit", base.fit_leaf, "train.yml", build_fit)]
            predicts = [base.ipsc_predict, *(A549_PREDICTS if arm.a549 else ())]
            jobs += [("predict", name, name, build_predict) for name in predicts]
            for kind, src_name, dst_name, build in jobs:
                src = src_dir / src_name
                baseline_cfg = yaml.safe_load(src.read_text())
                cfg = build(arm, organelle, baseline_cfg)
                renames, recipe = allowed_diff(arm, kind)
                changed = changed_keys(baseline_cfg, cfg)
                if bad := _unexpected(changed, renames, recipe):
                    raise ValueError(f"{dst_dir / dst_name}: unexpected diff vs {src}: {sorted(bad)}")
                if missing := renames - changed:
                    raise ValueError(f"{dst_dir / dst_name}: rename(s) did not change anything: {sorted(missing)}")
                out[dst_dir / dst_name] = _render(cfg, _header(arm, organelle, src, kind))
    return out


def main(argv: list[str] | None = None) -> int:
    """Write every arm leaf, or with ``--check`` report leaves that differ from disk."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="write nothing; exit 1 if any leaf on disk differs")
    args = ap.parse_args(argv)
    leaves = build_leaves()
    stale = [p for p, text in leaves.items() if not p.is_file() or p.read_text() != text]
    for path in stale:
        print(f"{'[stale]' if args.check else '[write]'} {path.relative_to(BENCHMARKS)}")
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(leaves[path])
    print(f"{len(leaves)} leaves, {len(stale)} {'stale' if args.check else 'written'}")
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
