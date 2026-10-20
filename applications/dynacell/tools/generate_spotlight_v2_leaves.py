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
  pass, so the arm and its v2 control share one recipe by construction. Also
  ``fnet3d_vscyto3daug`` (Phase 15 Arm B; nucleus, then ER and mito for the topology
  arms below): vanilla FNet fails out of domain, so the FNet verdict on A549 rests on
  this recipe, retrained on today's code.
- UNeXt2-3D wall: ``fcmae_vscyto3d_scratch_{v2,v2_seed1,segaux,segaux_seed1,segaux_sauna,segaux_cpout}`` compose
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
  ``CondMaskSource`` (``Nuclei_prediction``, thresholded per window at Otsu) and
  set no ``fg_mask_key``.
- ``segaux_sauna`` / ``segaux_cldice`` -- Stage 2 #5 on ``fnet3d_vscyto3daug`` nucleus, ER
  and mito (``segaux_sauna`` also on the UNeXt2-3D and pix2pix3d membrane arms): the
  ``segaux`` recipe with one change to the Dice term (``TOPOLOGY_ARGS``): Dice
  sums weighted by the patch mask's SAUNA map, or mixed with soft-clDice. Each carries its
  own calibrated ``seg_aux_weight``, since the change rescales the term's gradient. The ER
  and mito arms (and their ``segaux`` arm) train on masks from the eval's classical
  binarizer, written by ``write_classical_fg_masks.py``.
- ``segaux_cpout`` -- B3b on the UNeXt2-3D membrane arm: the ``segaux`` recipe with a
  thin-wall Dice target (``fg_mask_key: fg_mask_cpout``). The mask lives in a derived
  training store (:data:`CPOUT_TRAIN_STORE`) that mirrors the iPSC ``cell.zarr`` (same
  positions, image arrays symlinked) plus the new array, so the fit leaf disables the
  manifest resolver (``dataset_ref: null``) and declares ``data_path`` and both channels
  inline, as the 2D focus-slab leaves do. Its predict leaves are the baseline's test sets.
  It carries its own calibrated ``seg_aux_weight``.
- ``bglp`` / ``bgflat`` -- Track H on ``celldiff_2d`` nucleus/membrane: data gains
  ``fg_mask_key: fg_mask`` and the model ``target_bg_lowpass: BackgroundLowPass`` with the
  arm's ``BG_TARGET_ARGS``, a training-only target transform that keeps the target inside
  the dilated, feathered foreground and replaces the background by an estimate from
  background pixels only: a small-sigma normalized convolution (``bglp``, background high
  frequencies removed) or each plane's background mean (``bgflat``, autofluorescence and
  illumination removed too). The background stays supervised, unlike a masked loss.
  Validation stays raw, so ``--ckpt best`` selects on the baseline's criterion; predict
  leaves equal the baseline's. Normalization is the baseline's.
- ``segcons`` / ``segcons_ctrl`` -- P1 of the conditional-generation review
  (``experiments/2026-10-15_conditional-generation-review/PROPOSALS.md``), nucleus only, on
  FNet-3D vscyto3daug, UNeXt2-3D and pix2pix2d. Both are fine-tunes ("forks") of the
  baseline recipe's converged final checkpoint (:data:`FORK_CKPTS`, loaded weights-only
  through ``model.init_args.ckpt_path``, so optimizer and schedule start fresh) on a
  shortened budget at a lower LR (:data:`FORK_SCHEDULES`). ``segcons`` adds
  ``SegConsistencyLoss`` with Cellpose-SAM (``cpsam``) as the frozen segmenter;
  ``segcons_ctrl`` is the same fork, steps, LR and seed without it, so the pair is
  stage-matched. ``seg_consistency_weight`` is calibrated per arm at the fork checkpoint
  (:data:`SEG_CONSISTENCY_WEIGHTS`). Both
  predict from ``last.ckpt`` (``--ckpt last``): ``loss/validate`` stays base-only, so a
  best-by-val pick would favour the term arm's earliest epoch.

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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

BENCHMARKS = Path(__file__).resolve().parents[1] / "configs" / "benchmarks" / "virtual_staining"
MODELS_ROOT = "/hpc/projects/comp.micro/virtual_staining/models/dynacell"
ORGANELLES: tuple[str, ...] = ("nucleus", "membrane")
THIN_ORGANELLES: tuple[str, ...] = ("er", "mito")
POOL = "ipsc_confocal"
A549_PREDICTS: tuple[str, ...] = (
    "predict__a549_mantis_mock.yml",
    "predict__a549_mantis_denv.yml",
    "predict__a549_mantis_zikv.yml",
)
SEG_AUX_C = 0.1
SEG_AUX_T0 = 0.7
# Stage 2 #5 (2026-10-01): the segaux arm with one topology-aware change to its Dice term.
# segaux_sauna weights the Dice sums by |SAUNA "h" map| of the patch mask, which needs the
# physical voxel size (read from the train set's manifest); segaux_cldice mixes in soft-clDice.
# Each is calibrated on its own, like any other arm (SEG_AUX_WEIGHTS).
_IPSC_MANIFEST = Path(__file__).resolve().parents[1] / "src/dynacell/_manifests/aics-hipsc/manifest.yaml"
_IPSC_MANIFEST_DATA = yaml.safe_load(_IPSC_MANIFEST.read_text())
_IPSC_SPACING = [_IPSC_MANIFEST_DATA["spacing"][k] for k in "zyx"]
TOPOLOGY_ARGS: dict[str, dict[str, Any]] = {
    "segaux_sauna": {"weighting": "sauna", "spacing": _IPSC_SPACING},
    # 5 skeleton iterations (the official default is 10): ER/mito tubules are ~1-3 voxels in radius,
    # nucleus bodies (~40 px) never skeletonize at either count, and each iteration costs ~0.13 s
    # per (8, 1, 32, 384, 384) step on an A40 (10 iterations: 1.37 s vs 0.07 s without clDice).
    "segaux_cldice": {"topology": "cldice", "cldice_alpha": 0.5, "cldice_iters": 5},
}
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
    # UNeXt2-2D membrane on the L1 recipe (the MixedLoss baseline is a failed fit; the l1
    # probe trains, Dice 0.889). Calibrated on the l1 probe's best ckpt: 0.18 [0.15-0.27].
    ("membrane", "fcmae_vscyto2d_scratch_l1segaux"): 0.18,
    # Calibrated on the July Phase 15 Arm B best ckpt (epoch=28-step=75400-loss=0.776): 4.5 [2.6-6.7],
    # results/nucleus__fnet3d_vscyto3daug__batches.csv (val replay 0.782 vs recorded 0.776).
    ("nucleus", "fnet3d_vscyto3daug_segaux"): 4.5,
    # Stage 2 #5 arms on the same ckpt and val batches (calibrate_variant.py, micro-batch 2), whose
    # plain-Dice row reproduces the 4.5 above (4.51 [2.58-6.72]): SAUNA 5.28 [2.75-8.00],
    # clDice at 5 iterations 0.878 [0.636-1.70]. results/nucleus__fnet3d_vscyto3daug__variants.csv.
    ("nucleus", "fnet3d_vscyto3daug_segaux_sauna"): 5.3,
    ("nucleus", "fnet3d_vscyto3daug_segaux_cldice"): 0.88,
    # ER/mito arms, same protocol on the v2 baseline's best ckpt while it was still training (ER
    # epoch=28-step=63713, mito epoch=18-step=49400; batch fg_frac .07-.11 from the classical masks).
    # results/{er,mito}__fnet3d_vscyto3daug_v2__variants.csv. ER: Dice 0.220 [0.185-0.345], SAUNA 0.211
    # [0.177-0.333], clDice 0.236 [0.196-0.361]. Mito: Dice 0.774 [0.617-0.936], SAUNA 0.661
    # [0.517-0.784], clDice 0.712 [0.547-0.824].
    ("er", "fnet3d_vscyto3daug_segaux"): 0.22,
    ("er", "fnet3d_vscyto3daug_segaux_sauna"): 0.21,
    ("er", "fnet3d_vscyto3daug_segaux_cldice"): 0.24,
    ("mito", "fnet3d_vscyto3daug_segaux"): 0.77,
    ("mito", "fnet3d_vscyto3daug_segaux_sauna"): 0.66,
    ("mito", "fnet3d_vscyto3daug_segaux_cldice"): 0.71,
    # Membrane SAUNA arms (2026-10-10), on the same April baseline ckpts and val batches as their
    # segaux weights (calibrate_variant.py; UNeXt2-3D epoch=136-step=42744 micro 2, pix2pix3d
    # epoch=25-step=83200 generator_ema micro 1), whose plain-Dice rows reproduce 0.86 and 3.1
    # (0.8609 [0.468-2.214], 3.1044 [2.304-6.118]): SAUNA 0.7450 [0.424-1.813], 2.946 [2.193-5.870].
    # results/membrane__{fcmae_vscyto3d_scratch,pix2pix3d_unetvit}__variants.csv.
    ("membrane", "fcmae_vscyto3d_scratch_segaux_sauna"): 0.75,
    ("membrane", "pix2pix3d_unetvit_segaux_sauna"): 2.9,
    # segaux_cpout (B3b, 2026-10-14): the thin mask changes the Dice term's gradient scale, so it is
    # calibrated on its own, on the same ckpt from the derived store (b3b_cpout/calib/), where the
    # canonical fg_mask reproduces 0.8609 [0.468-2.214]: fg_mask_cpout 0.5533 [0.397-0.886].
    ("membrane", "fcmae_vscyto3d_scratch_segaux_cpout"): 0.55,
}
# Data mask key per arm suffix; every other fg-mask arm reads ``fg_mask``.
FG_MASK_KEYS: dict[str, str] = {"segaux_cpout": "fg_mask_cpout"}
# segaux_cpout's derived training store: the iPSC dataset_v4 train cell.zarr's positions, image
# arrays symlinked, plus fg_mask_cpout (experiments/2026-09-24_spotlight-v2/b3b_cpout/).
CPOUT_TRAIN_STORE = "/hpc/projects/virtual_staining/training/dynacell/_spotlight_v2_b3b/ipsc/train/cell_cpout.zarr"
# Segaux-family arms that carry their own calibrated seg_aux_weight rather than the segaux arm's.
_OWN_WEIGHT_SUFFIXES: tuple[str, ...] = (*TOPOLOGY_ARGS, "segaux_cpout")
_SEGAUX_FAMILY: tuple[str, ...] = ("segaux", "segauxself", *_OWN_WEIGHT_SUFFIXES)
# C-joint's mask Dice has no baseline to calibrate against (the baseline has no
# mask channel); it borrows the same-dim CellDiff seg-aux weight as a starting point.
MASK_DICE_WEIGHTS: dict[tuple[str, str], float] = {
    ("nucleus", "celldiff_2d_cjoint"): 2.5,
    ("nucleus", "celldiff_cjoint"): 1.4,
}
# Track H (2026-10-03) BackgroundLowPass args per arm, in pixels, measured on the GT by the H0
# step (experiments/2026-09-24_spotlight-v2/trackh/: target_op_check2.csv, mask_recall.csv).
# bglp smooths the background (sigma_lp 2 removes 97% of its 1-px power and keeps 99% of its
# low frequencies); r2/f1 leaves 3% of that 1-px power in the ring, r4/f2 12-15%. A nucleus the
# mask misses is only blurred, so the tight ring is safe. bgflat flattens the background to
# each plane's mean, which erases any nucleus the mask misses: r4/f2 for recall (cpdino GT
# nuclei <50% covered, iPSC 1.2% at r4 vs 2.9% at r2).
BG_TARGET_ARGS: dict[str, dict[str, float | int | None]] = {
    "bglp": {"sigma_lp": 2.0, "sigma_feather": 1.0, "dilate_radius": 2},
    "bgflat": {"sigma_lp": None, "sigma_feather": 2.0, "dilate_radius": 4},
}
JOINTSTEPS_MAX_EPOCHS = 320
# Second draws (see SEED_SOURCES) inherit their source arm's wall.
# The SAUNA arm keeps its segaux source's wall: the membrane segaux fit ran 2.44 ep/h (epoch=125
# 2026-09-27 13:44:58 -> epoch=154 2026-09-28 01:38:59), 82 h for 200 epochs, and SAUNA cost the
# FNet-3D nucleus fit +7.6% (2-12:01:34 vs 2-07:46:14), ~88 h against the 96 h default.
# The cpout arm is the same segaux recipe on another mask, so the same 2.44 ep/h (82 h) applies.
LONG_WALL_MODELS: frozenset[str] = frozenset(
    {
        "fcmae_vscyto3d_scratch_v2",
        "fcmae_vscyto3d_scratch_segaux",
        "fcmae_vscyto3d_scratch_segaux_sauna",
        "fcmae_vscyto3d_scratch_segaux_cpout",
    }
)
_WALL_4GPU = "launcher_profiles/hardware_4gpu.yml"
_WALL_4GPU_LONG = "launcher_profiles/hardware_4gpu_long.yml"
# P1 segmenter-consistency fine-tunes (2026-10-15): three converged nucleus baselines, each
# forked twice, with and without the term. The fork source is the baseline recipe's final
# checkpoint under its immutable latest-* name (all three runs completed their budget); the
# FNet-3D and UNeXt2-3D forks start from the v2 retrains, the Spotlight-v2 controls on today's code.
SEGCONS_BASELINES: tuple[str, ...] = ("fnet3d_vscyto3daug", "fcmae_vscyto3d_scratch", "pix2pix2d_unetvit")
FORK_CKPTS: dict[tuple[str, str], str] = {
    ("nucleus", "fnet3d_vscyto3daug"): f"{MODELS_ROOT}/ipsc/nucleus/fnet3d_vscyto3daug_v2/checkpoints/"
    "latest-epoch=76-step=200000.ckpt",
    ("nucleus", "fcmae_vscyto3d_scratch"): f"{MODELS_ROOT}/ipsc/nucleus/fcmae_vscyto3d_scratch_v2/checkpoints/"
    "latest-epoch=199-step=62400.ckpt",
    ("nucleus", "pix2pix2d_unetvit"): f"{MODELS_ROOT}/ipsc/nucleus/pix2pix2d_unetvit/checkpoints/"
    "latest-epoch=39-step=20000.ckpt",
}
# Fine-tune budget: 10% of the baseline's (200000 steps, 200 epochs, 40 epochs, read off the fork
# checkpoints' names) at 0.1x its peak LR (FNet-3D keeps its Constant schedule). The WarmupCosine
# families warm up over 5% of the fine-tune's scheduler steps: UNeXt2-3D 20 x 312 = 6240 (62400
# steps / 200 epochs), pix2pix2d 4 x 250 = 1000 (20000 global steps / 40 epochs, two optimizer
# steps per GAN batch). The scheduler is rebuilt for the fine-tune, so a cosine decays to 0 again.
# pix2pix2d predicts from generator_ema, which starts as the fork's own shadow. At the baseline's
# ema_kimg 10 the fine-tune (1000 G steps x 16 samples: 4 ranks x batch 4) would leave
# 0.5 ** (16000 / 10000) = 33% of the fork-start shadow in it, diluting both arms; ema_kimg 2
# leaves 0.5 ** (16000 / 2000) = 0.5 ** 8 = 0.4% (decay = 0.5 ** (global batch / (ema_kimg * 1000))).
FORK_SCHEDULES: dict[str, dict[str, dict[str, Any]]] = {
    "fnet3d_vscyto3daug": {"trainer": {"max_steps": 20000}, "model": {"lr": 1.0e-4}},
    "fcmae_vscyto3d_scratch": {"trainer": {"max_epochs": 20}, "model": {"lr": 4.0e-5, "warmup_steps": 312}},
    "pix2pix2d_unetvit": {
        "trainer": {"max_epochs": 4},
        "model": {"lr_g": 2.0e-5, "lr_d": 2.0e-5, "warmup_steps": 50, "ema_kimg": 2.0},
    },
}
# S_train from gate G4: cpsam (Kendall tau vs cpdino nucleus Dice/mAP .65/.80, not in the eval
# stack). One slab-MIP pass per sample around the target crop's nuclear-area plane, center
# 240-px crop, which cellpose pads to exactly one 256-px tile, the geometry the G4 bench timed
# (a 256-px crop pads to 272 and runs 2 x 2 tiles, 4x the segmenter compute, measured with
# get_pad_yx); see viscy_utils.losses.seg_consistency.
# Every sample of the batch is scored. The with-gradient pass is checkpointed per tile: without
# it each sample holds ~5.6 GB of activations (G4 H200 bench: 42.9 GB at 8 samples, against the
# FNet-3D fit's ~41 GB), with it ~0.07 GB per sample at ~1.8x the term's time (A40, 2026-10-15).
SEG_CONSISTENCY_ARGS: dict[str, Any] = {
    "segmenter": {
        "class_path": "dynacell.cellpose_dense.CellposeDense",
        "init_args": {"model_name": "cpsam", "checkpoint_tiles": 1},
    },
    "slab_halfwidth": 1,
    "crop_size": 240,
}
# seg_consistency_weight per arm: the median over 20 seeded val batches of ||grad base|| /
# ||grad seg_consistency|| at the fork checkpoint, at the leaves' 240-px crop (calibrate_variant.py
# --variants dice segcons --ckpt <fork> --tag segcons_crop240, micro-batch 1, A40 2026-10-16),
# 2 significant figures [IQR]: FNet-3D 0.3670 [0.138-0.654], UNeXt2-3D 0.03238 [0.0243-0.0657],
# pix2pix2d 0.1912 [0.0673-0.342]. Source: experiments/2026-09-24_spotlight-v2/calibration/results/
# nucleus__{fnet3d_vscyto3daug_v2,fcmae_vscyto3d_scratch_v2,pix2pix2d_unetvit}__segcons_crop240{.csv,__meta.json}.
# (The first calibration, at the 4-tile 256-px crop, gave 0.48 / 0.050 / 0.57: results/*__segcons.*.)
# Known limitation, kept for comparability with the segaux weights: the convention is the segaux
# one, so for pix2pix2d the gradients are on generator_ema with base = lambda_l1 * L1 (the
# adversarial term is ignored), and micro-batch gradients are summed to the val batch's mean, not
# averaged over the training batch's composition (per-rank batch, DDP world size).
SEG_CONSISTENCY_WEIGHTS: dict[tuple[str, str], float] = {
    ("nucleus", "fnet3d_vscyto3daug_segcons"): 0.37,
    ("nucleus", "fcmae_vscyto3d_scratch_segcons"): 0.032,
    ("nucleus", "pix2pix2d_unetvit_segcons"): 0.19,
}
_FORK_SUFFIXES: tuple[str, ...] = ("segcons", "segcons_ctrl")
# P1 pix2pix2d follow-ups (user-approved 2026-10-19; experiments/2026-10-15_conditional-generation-review/p1/
# PREREG_followups.md): suffix -> (the fork arm it re-runs, fork-source run dir or None for the original, seed or None
# for the original, seg_consistency_weight factor). `_s1` is a replication pair forked from the converged second
# baseline draw (pix2pix2d_unetvit_seed1, latest-epoch=39-step=20000 = 40/40 epochs) with fine-tune seed 1 (the
# original pair used the leaf's 42); `_x10` is the positive control of the hacking check: the original segcons arm at
# 10x lambda (0.19 -> 1.9), same fork and seed, compared against the existing segcons_ctrl.
P1_FOLLOWUPS: dict[str, tuple[str, str | None, int | None, float]] = {
    "segcons_s1": ("segcons", "pix2pix2d_unetvit_seed1", 1, 1.0),
    "segcons_ctrl_s1": ("segcons_ctrl", "pix2pix2d_unetvit_seed1", 1, 1.0),
    "segcons_x10": ("segcons", None, None, 10.0),
}
FOLLOWUP_FORK_CKPT_NAME = "latest-epoch=39-step=20000.ckpt"
SAFE_CROP_SIZE = [1, 384, 384]
MASK_VELOCITY_WEIGHT = 1.0
# C-cond predict conditions on the same-dimensionality FNet `_segaux` prediction for the
# same test leg (that predict must be COMPLETE first), thresholded per window.
CCOND_MASK_MODEL: dict[str, str] = {"celldiff_2d": "fnet2d_segaux", "celldiff": "fnet3d_paper_segaux"}
CCOND_MASK_CHANNEL = "Nuclei_prediction"
# Per-window Otsu (2026-09-28), label- and scale-free. The val-tuned fixed thresholds it
# replaced (2D 0.30, Dice 0.774; 3D 0.40) marked 96-98% of A549 pixels foreground, where
# FNet predictions sit on another intensity scale. Otsu: iPSC val Dice 0.769; A549 fg
# fraction 7-9% (2D) / 26-35% (3D) vs 12.5% [4.8-18.6%] for the A549 training masks
# (experiments/2026-09-24_spotlight-v2/ccond_threshold*/).
CCOND_THRESHOLD = "otsu"
SAFE_CROP_COVERAGE = 0.9
_UNEXT2_2D_DATA_OVERLAY = BENCHMARKS / "_internal/shared/model/data_overlays/fcmae_vscyto2d_fit.yml"


SEGAUXSELF_BASELINES: tuple[str, ...] = ("fnet2d", "fcmae_vscyto2d_scratch", "pix2pix2d_unetvit", "celldiff_2d")


@dataclass(frozen=True)
class Baseline:
    """Where a baseline model's leaves, checkpoints and prediction stores live."""

    fit_leaf: str
    ckpt_dir: str
    store_dir: str
    ipsc_predict: str
    engine: str  # "unet" | "gan" | "flow"
    # (organelle, leaf names) where an organelle's A549 predict leaves are not A549_PREDICTS.
    a549_predict_overrides: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def a549_predicts(self, organelle: str) -> tuple[str, ...]:
        """Return the baseline's A549 predict leaf names for ``organelle``."""
        return dict(self.a549_predict_overrides).get(organelle, A549_PREDICTS)


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
    # Phase 15 Arm B: FNet-3D on a 384^2 patch with the VSCyto3D augmentation stack, the FNet
    # that works out of domain (A549 nucleus PCC 0.68 vs 0.04 for fnet3d_paper). The ER and mito
    # baseline leaves are never-run templates of the same recipe (Stage 2 #5).
    "fnet3d_vscyto3daug": Baseline(
        "train.yml",
        "fnet3d_vscyto3daug",
        "fnet3d_vscyto3daug",
        "predict__ipsc_confocal.yml",
        "unet",
        a549_predict_overrides=(
            ("nucleus", tuple(f"predict__a549_mantis_h2b_{c}.yml" for c in ("mock", "denv", "zikv"))),
        ),
    ),
}


@dataclass(frozen=True)
class Arm:
    """One (baseline, suffix) arm, emitted for each of ``organelles``."""

    baseline: str
    suffix: str
    organelles: tuple[str, ...]
    a549: bool
    fit: bool = True

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
    *(Arm(m, "segaux", ORGANELLES, a549=True) for m in BASELINES if m != "fnet3d_vscyto3daug"),
    # Seed replicates predict the A549 legs too: most arm effects are A549-transfer effects,
    # and without an A549 seed spread they have no noise floor (readout 2026-09-28).
    *(
        Arm(m, "seed1", ORGANELLES, a549=True)
        for m in ("fnet2d", "fcmae_vscyto2d_scratch", "pix2pix2d_unetvit", "celldiff_2d")
    ),
    Arm("fnet3d_paper", "seed1", ("nucleus",), a549=True),
    Arm("fcmae_vscyto3d_scratch", "v2", ORGANELLES, a549=True),
    Arm("pix2pix3d_unetvit", "v2", ORGANELLES, a549=True),
    *(Arm("fcmae_vscyto2d_scratch", s, ("membrane",), a549=False) for s in ("jointsteps", "safecrop")),
    # The l1 probe won (Dice 0.889 vs 0.380) and is the baseline for l1segaux, so it also predicts A549.
    Arm("fcmae_vscyto2d_scratch", "l1", ("membrane",), a549=True),
    # The comparison rebuilt on the recipe that trains: the l1 probe is the baseline for these.
    Arm("fcmae_vscyto2d_scratch", "l1segaux", ("membrane",), a549=True),
    Arm("fcmae_vscyto2d_scratch", "l1seed1", ("membrane",), a549=True),
    # Stage 1b, nucleus first: segmentation inside the generative process.
    *(Arm(m, s, ("nucleus",), a549=True) for s in ("cjoint", "ccond") for m in ("celldiff_2d", "celldiff")),
    # Stage 2 #1: the segaux arm with a self-consistent Dice reference (2D families first).
    *(Arm(m, "segauxself", ORGANELLES, a549=True) for m in SEGAUXSELF_BASELINES),
    # Predict-only: the pix2pix2d BASELINE's own last.ckpt into a v2-only store. Its canonical
    # stores were predicted from a best-by-val checkpoint (nucleus epoch 22, membrane epoch 37),
    # while every GAN arm predicts from last.ckpt (recorded EMA val L1 does not track the
    # weights), so arm-vs-baseline needs this.
    Arm("pix2pix2d_unetvit", "last", ORGANELLES, a549=True, fit=False),
    # Second draws where the verdict rests on one run (readout 2026-09-28: on A549 the two FNet
    # baseline draws differ by up to 0.39 Dice): the arm itself for the FNet cells whose effect
    # holds against both baseline draws, and the pix2pix3d v2 baseline, which had no replicate.
    Arm("fnet2d", "segaux_seed1", ORGANELLES, a549=True),
    Arm("fnet3d_paper", "segaux_seed1", ("nucleus",), a549=True),
    Arm("pix2pix3d_unetvit", "v2_seed1", ORGANELLES, a549=True),
    # UNeXt2-3D v2 baseline, drawn per organelle once its segaux verdict came back non-null
    # (nucleus, wave2d 2026-09-28: A549 instance Dice +0.029..+0.047, PCC -0.020..-0.025;
    # membrane, wave2f 2026-09-29: iPSC mAP +0.128, instance Dice +0.099, PCC +0.027; CIs exclude 0).
    Arm("fcmae_vscyto3d_scratch", "v2_seed1", ORGANELLES, a549=True),
    # ... and the UNeXt2-3D membrane segaux arm itself, so the one arm that improves every
    # in-domain metric gets the full 2x2 (arm draw x baseline draw) on A549, where the
    # FNet-2D arm's two draws disagreed (nucleus Dice +0.15..+0.20 vs +0.01..+0.02).
    # Nucleus too (2026-09-30): vanilla FNet fails out-of-domain by design, so the A549 verdict
    # rests on the families that already work there; their nucleus arms get a second draw.
    Arm("fcmae_vscyto3d_scratch", "segaux_seed1", ORGANELLES, a549=True),
    # pix2pix3d nucleus segaux on A549: Dice 0.863 -> 0.890, mAP 0.385 -> 0.483, PCC 0.708 -> 0.678.
    # Membrane added 2026-10-07: its single draw is one of the few seg gains on both iPSC and
    # A549 (final table, interior-only FG), so it gets the replicate the nucleus gate withheld.
    Arm("pix2pix3d_unetvit", "segaux_seed1", ORGANELLES, a549=True),
    # Loss-weight sweep on the one cell whose effect held against both baseline draws
    # (FNet-3D nucleus at w=1.9: A549 Dice +0.21..+0.44, mAP +0.15..+0.22, readout 2026-09-28).
    Arm("fnet3d_paper", "segaux_halfw", ("nucleus",), a549=True),
    Arm("fnet3d_paper", "segaux_doublew", ("nucleus",), a549=True),
    # The FNet-3D verdict on the recipe that works out of domain (vanilla FNet fails there by
    # design, so its A549 gains are not evidence): the full 2x2 from the start. v2 retrains the
    # July baseline under today's code, selection and predict path, like the other 3D v2s.
    *(Arm("fnet3d_vscyto3daug", s, ("nucleus",), a549=True) for s in ("v2", "v2_seed1", "segaux", "segaux_seed1")),
    # Stage 2 #5: SAUNA-weighted Dice vs soft-clDice on the same cell, against the 2x2 above.
    *(Arm("fnet3d_vscyto3daug", s, ("nucleus",), a549=True) for s in TOPOLOGY_ARGS),
    # ... and on the thin organelles the topology terms are meant for. Their training masks come
    # from the eval's classical ER/mito segmenter (CLAHE+Otsu fills the ER cytoplasm). The two
    # baseline draws need no mask, so they train first; the arms are calibrated on their ckpt.
    *(Arm("fnet3d_vscyto3daug", s, THIN_ORGANELLES, a549=True) for s in ("v2", "v2_seed1")),
    *(Arm("fnet3d_vscyto3daug", s, THIN_ORGANELLES, a549=True) for s in ("segaux", *TOPOLOGY_ARGS)),
    # Second segaux draw on the thin organelles (2026-10-07): their iPSC Dice gain rests on one run.
    Arm("fnet3d_vscyto3daug", "segaux_seed1", THIN_ORGANELLES, a549=True),
    # Loss-weight sweep on the out-of-domain seg gain (2026-10-10): A549 Dice beats both baseline
    # draws on 3/3 legs for both segaux draws at w=4.5, while on fnet3d_paper nucleus both w/2
    # and 2w lost the 1x arm's iPSC mAP gain (final table).
    *(Arm("fnet3d_vscyto3daug", s, ("nucleus",), a549=True) for s in ("segaux_halfw", "segaux_doublew")),
    # SAUNA-weighted Dice on the two membrane arms whose seg gain holds on iPSC and A549 (2026-10-10):
    # on ER it beat plain segaux on A549 FG_PCC on 3/3 legs (+.016/+.008/+.028, CIs > 0).
    *(Arm(m, "segaux_sauna", ("membrane",), a549=True) for m in ("fcmae_vscyto3d_scratch", "pix2pix3d_unetvit")),
    # Second SAUNA draw on pix2pix3d membrane (2026-10-13): draw 1 passed B1 with an A549 FG_PCC gain
    # over both baselines and both segaux draws, which rule R3 does not decide on one draw.
    Arm("pix2pix3d_unetvit", "segaux_sauna_seed1", ("membrane",), a549=True),
    # B3b (2026-10-14): UNeXt2-3D membrane segaux on a thin-wall Dice target.
    Arm("fcmae_vscyto3d_scratch", "segaux_cpout", ("membrane",), a549=True),
    # Track H: CellDiff-2D trained on a background-low-passed target (H1; launched as H2).
    *(Arm("celldiff_2d", s, ORGANELLES, a549=True) for s in BG_TARGET_ARGS),
    # P1 (2026-10-15): segmenter-consistency fine-tunes and their no-term controls, nucleus only.
    *(Arm(m, s, ("nucleus",), a549=True) for m in SEGCONS_BASELINES for s in _FORK_SUFFIXES),
    # ... and the pix2pix2d follow-ups (P1_FOLLOWUPS): a replication pair and the 10x-lambda positive control.
    *(Arm("pix2pix2d_unetvit", s, ("nucleus",), a549=True) for s in P1_FOLLOWUPS),
)
# Second draws: suffix -> the arm whose recipe it re-draws with seed_everything: 1.
SEED_SOURCES: dict[str, str] = {
    "segaux_seed1": "segaux",
    "v2_seed1": "v2",
    "l1seed1": "l1",
    "segaux_sauna_seed1": "segaux_sauna",
}
# Loss-weight sweep arms: the segaux arm's recipe with seg_aux_weight scaled by this factor.
WEIGHT_SCALES: dict[str, float] = {"segaux_halfw": 0.5, "segaux_doublew": 2.0}


def _unseeded(arm: Arm) -> Arm:
    """Return the arm a second draw re-draws (:data:`SEED_SOURCES`); any other arm unchanged."""
    return replace(arm, suffix=SEED_SOURCES.get(arm.suffix, arm.suffix))


def _long_wall(arm: Arm) -> bool:
    return _unseeded(arm).model in LONG_WALL_MODELS


_DESCRIPTION: dict[str, str] = {
    "segaux": "data fg_mask_key: fg_mask; model seg_aux (SegAuxDice c=0.1) + seg_aux_weight (+ seg_aux_t0 for flow)",
    "segauxself": "data fg_mask_key: fg_mask; model seg_aux (SegAuxDice c=0.1, label=target) + the segaux "
    "arm's seg_aux_weight (+ seg_aux_t0 for flow). One field from the segaux arm: the Dice reference is the "
    "target through the same sigmoid, so the term is zero at pred == target",
    "seed1": "seed_everything: 1",
    "segaux_seed1": "the segaux arm's recipe + seed_everything: 1 (a second draw of the arm)",
    "v2_seed1": "seed_everything: 1 (a second draw of the v2 baseline)",
    "segaux_sauna": "the segaux arm's recipe + SegAuxDice weighting=sauna (Dice sums weighted by |SAUNA h map| "
    "of the patch mask, iPSC voxel size) at its own calibrated seg_aux_weight",
    "segaux_sauna_seed1": "the segaux_sauna arm's recipe + seed_everything: 1 (a second draw of the arm)",
    "segaux_cpout": "the segaux arm's recipe + data fg_mask_key: fg_mask_cpout, a thin-wall target from cpdino "
    "whole-cell outer outlines on the GT membrane within a wall band, canonical mask outside (recipe in "
    "experiments/2026-09-24_spotlight-v2/b3b_cpout/cpout_target.py), read from the derived training store "
    f"{CPOUT_TRAIN_STORE} (dataset_ref: null, data_path and channels inline) at its own seg_aux_weight",
    "segaux_cldice": "the segaux arm's recipe + SegAuxDice topology=cldice (0.5 Dice + 0.5 soft-clDice, 5 "
    "skeleton iterations) at its own calibrated seg_aux_weight",
    "segaux_halfw": "the segaux arm's recipe with seg_aux_weight x0.5 (loss-weight sweep)",
    "segaux_doublew": "the segaux arm's recipe with seg_aux_weight x2 (loss-weight sweep)",
    "v2": "nothing (fresh retrain of the baseline recipe under its own run root)",
    "jointsteps": f"trainer max_epochs: {JOINTSTEPS_MAX_EPOCHS}. Step budget matched to the joint membrane run, "
    "measured from final checkpoints: baseline latest-epoch=199-step=100000 (500 steps/ep), joint "
    "latest-epoch=199-step=160000 (800 steps/ep); 320 x 500 = 160000",
    "l1": "MixedLoss l1_alpha 1.0 / l2_alpha 0.0 / ms_dssim_alpha 0.0 (L1 only)",
    "l1segaux": "the l1 arm's MixedLoss (L1 only) + data fg_mask_key: fg_mask + model seg_aux (SegAuxDice "
    "c=0.1) + seg_aux_weight. Its baseline is the l1 probe, from which it differs by the seg-aux term only",
    "l1seed1": "the l1 arm's MixedLoss (L1 only) + seed_everything: 1 (noise floor for l1segaux vs l1)",
    "safecrop": f"affine safe_crop_size {SAFE_CROP_SIZE} + safe_crop_coverage {SAFE_CROP_COVERAGE}",
    "cjoint": "data fg_mask_key: fg_mask; model net_config.in_channels 2, mask_mode joint, mask_dice_weight, "
    f"mask_velocity_weight {MASK_VELOCITY_WEIGHT}, seg_aux_t0 {SEG_AUX_T0}",
    "ccond": "data fg_mask_key: fg_mask; model net_config.cond_channels 2, mask_mode cond, mask_corruption "
    "(MaskCorruption defaults)",
    **{
        suffix: "data fg_mask_key: fg_mask; model target_bg_lowpass (BackgroundLowPass "
        + ", ".join(f"{k}={v}" for k, v in args.items())
        + "), a training-only target transform; validation and prediction see the raw target"
        for suffix, args in BG_TARGET_ARGS.items()
    },
    "segcons_s1": "the segcons arm's recipe (same lambda 0.19) forked from the converged pix2pix2d_unetvit_seed1 "
    "checkpoint with seed_everything: 1 (P1 replication draw)",
    "segcons_ctrl_s1": "the segcons_ctrl fork forked from the converged pix2pix2d_unetvit_seed1 checkpoint with "
    "seed_everything: 1 (no-term control of segcons_s1)",
    "segcons_x10": "the segcons arm's recipe with seg_consistency_weight x10 (0.19 -> 1.9): the hacking check's "
    "positive control, same fork and seed as segcons",
    "segcons_ctrl": "fork: model ckpt_path (the converged baseline checkpoint, weights only) + the fine-tune "
    "budget and LR of FORK_SCHEDULES; no other change (the no-term control of segcons)",
    "segcons": "the segcons_ctrl fork + model seg_consistency (SegConsistencyLoss, Cellpose-SAM cpsam, slab "
    "halfwidth 1, 240-px crop (one cellpose tile), per-tile activation checkpointing) + seg_consistency_weight "
    "(gradient-norm calibrated at the fork checkpoint)",
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
    if arm.suffix in SEED_SOURCES:
        renames, recipe_keys = allowed_diff(_unseeded(arm), kind)
        return renames, recipe_keys | {"seed_everything"}
    if arm.suffix in WEIGHT_SCALES:
        return allowed_diff(replace(arm, suffix="segaux"), kind)
    if arm.suffix in P1_FOLLOWUPS:
        renames, recipe_keys = allowed_diff(replace(arm, suffix=P1_FOLLOWUPS[arm.suffix][0]), kind)
        return renames, recipe_keys | ({"seed_everything"} if P1_FOLLOWUPS[arm.suffix][2] is not None else set())
    if arm.suffix == "l1segaux":
        renames, l1_keys = allowed_diff(replace(arm, suffix="l1"), kind)
        return renames, l1_keys | allowed_diff(replace(arm, suffix="segaux"), kind)[1]
    recipe: set[str] = set()
    if arm.suffix in _SEGAUX_FAMILY:
        recipe |= {"data.init_args.fg_mask_key", "model.init_args.seg_aux", "model.init_args.seg_aux_weight"}
        if BASELINES[arm.baseline].engine == "flow":
            recipe.add("model.init_args.seg_aux_t0")
        if arm.suffix == "segaux_cpout":
            recipe |= {
                "benchmark.dataset_ref",
                "data.init_args.data_path",
                "data.init_args.source_channel",
                "data.init_args.target_channel",
            }
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
    elif arm.suffix in BG_TARGET_ARGS:
        recipe |= {"data.init_args.fg_mask_key", "model.init_args.target_bg_lowpass"}
    elif arm.suffix in _FORK_SUFFIXES:
        schedule = FORK_SCHEDULES[arm.baseline]
        recipe.add("model.init_args.ckpt_path")
        recipe |= {f"trainer.{k}" for k in schedule["trainer"]} | {f"model.init_args.{k}" for k in schedule["model"]}
        if arm.suffix == "segcons":
            recipe |= {"model.init_args.seg_consistency", "model.init_args.seg_consistency_weight"}
    if _long_wall(arm):
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
    if arm.suffix in SEED_SOURCES:
        _apply_recipe(_unseeded(arm), organelle, cfg)
        cfg["seed_everything"] = 1
        return
    if arm.suffix in WEIGHT_SCALES:
        _apply_recipe(replace(arm, suffix="segaux"), organelle, cfg)
        cfg["model"]["init_args"]["seg_aux_weight"] *= WEIGHT_SCALES[arm.suffix]
        return
    if arm.suffix in P1_FOLLOWUPS:
        source_suffix, fork_run, seed, weight_factor = P1_FOLLOWUPS[arm.suffix]
        _apply_recipe(replace(arm, suffix=source_suffix), organelle, cfg)
        model_args = cfg["model"]["init_args"]
        if fork_run is not None:
            model_args["ckpt_path"] = f"{MODELS_ROOT}/ipsc/{organelle}/{fork_run}/checkpoints/{FOLLOWUP_FORK_CKPT_NAME}"
        if seed is not None:
            cfg["seed_everything"] = seed
        if weight_factor != 1.0:
            model_args["seg_consistency_weight"] = round(model_args["seg_consistency_weight"] * weight_factor, 6)
        return
    if arm.suffix == "l1segaux":
        # The l1 recipe plus the segaux terms, at the weight calibrated on the l1 probe.
        _apply_recipe(replace(arm, suffix="l1"), organelle, cfg)
        _apply_recipe(replace(arm, suffix="segaux"), organelle, cfg)
        cfg["model"]["init_args"]["seg_aux_weight"] = SEG_AUX_WEIGHTS[(organelle, arm.model)]
        return
    data_args = cfg.setdefault("data", {}).setdefault("init_args", {})
    model_args = cfg.setdefault("model", {}).setdefault("init_args", {})
    if arm.suffix in _SEGAUX_FAMILY:
        data_args["fg_mask_key"] = FG_MASK_KEYS.get(arm.suffix, "fg_mask")
        seg_args: dict[str, Any] = {"c": SEG_AUX_C, **TOPOLOGY_ARGS.get(arm.suffix, {})}
        if arm.suffix == "segauxself":
            seg_args["label"] = "target"
        model_args["seg_aux"] = {"class_path": "viscy_utils.losses.SegAuxDice", "init_args": seg_args}
        # segauxself reuses the segaux arm's calibrated weight so the two differ by the label only;
        # the topology and cpout arms change the term's gradient scale, so they carry their own.
        weight_arm = arm.model if arm.suffix in _OWN_WEIGHT_SUFFIXES else f"{arm.baseline}_segaux"
        model_args["seg_aux_weight"] = SEG_AUX_WEIGHTS[(organelle, weight_arm)]
        if BASELINES[arm.baseline].engine == "flow":
            model_args["seg_aux_t0"] = SEG_AUX_T0
        if arm.suffix == "segaux_cpout":
            # The resolver raises on data fields co-declared with a full dataset_ref, so disable it
            # and declare the three fields it would splice, channels from the same manifest.
            cfg["benchmark"]["dataset_ref"] = None
            data_args["data_path"] = CPOUT_TRAIN_STORE
            data_args["source_channel"] = _IPSC_MANIFEST_DATA["channels"]["source"]
            data_args["target_channel"] = _IPSC_MANIFEST_DATA["targets"][organelle]["target_channel"]
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
    elif arm.suffix in BG_TARGET_ARGS:
        data_args["fg_mask_key"] = "fg_mask"
        model_args["target_bg_lowpass"] = {
            "class_path": "viscy_utils.losses.BackgroundLowPass",
            "init_args": dict(BG_TARGET_ARGS[arm.suffix]),
        }
    elif arm.suffix in _FORK_SUFFIXES:
        schedule = FORK_SCHEDULES[arm.baseline]
        model_args["ckpt_path"] = FORK_CKPTS[(organelle, arm.baseline)]
        model_args.update(schedule["model"])
        cfg.setdefault("trainer", {}).update(schedule["trainer"])
        if arm.suffix == "segcons":
            model_args["seg_consistency"] = {
                "class_path": "viscy_utils.losses.SegConsistencyLoss",
                "init_args": copy.deepcopy(SEG_CONSISTENCY_ARGS),
            }
            model_args["seg_consistency_weight"] = SEG_CONSISTENCY_WEIGHTS[(organelle, arm.model)]
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
    if _long_wall(arm):
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
    ckpt_dir = base.ckpt_dir if not arm.fit else arm.model
    cfg["model"]["init_args"]["ckpt_path"] = f"{MODELS_ROOT}/ipsc/{organelle}/{ckpt_dir}/checkpoints/last.ckpt"
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
                "threshold": CCOND_THRESHOLD,
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
        if _long_wall(arm):
            lines.append(
                "# Wall: hardware_4gpu_long (7 d); 200 epochs at the measured ~2.0 ep/h is ~100 h > 4 d"
                " (checkpoint-mtime pairs in the generator docstring)."
            )
    else:
        lines.append("# Inference is identical to the baseline's; only the checkpoint, store and names differ.")
        if not arm.fit:
            lines.append("# Predict-only arm: the baseline's own last.ckpt, own store; submit with `--ckpt last`.")
        elif arm.suffix in _FORK_SUFFIXES or arm.suffix in P1_FOLLOWUPS:
            lines.append("# Fine-tune arm: predict the end of the fine-tune; submit with `--ckpt last`.")
        else:
            lines.append("# ckpt_path is a PLACEHOLDER (last.ckpt): submit with `--ckpt best`, re-bake afterwards.")
    lines.append("# Everything else, and the rationale for it, is the baseline leaf's.")
    return "\n".join(lines) + "\n"


def _render(cfg: dict, header: str) -> str:
    return header + yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False)


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
            jobs = [("fit", base.fit_leaf, "train.yml", build_fit)] if arm.fit else []
            predicts = [base.ipsc_predict, *(base.a549_predicts(organelle) if arm.a549 else ())]
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
