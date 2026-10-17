# Cell-cycle prediction from phase movies with temporal foundation-model embeddings

Phase movies of tracked NIH3T3 nuclei in, the two PIP-FUCCI reporters out. The model, the data and the design choices
are described in the app README (*Temporal embeddings of foundation models*) and in
`Bill/5-cellcycle/README.md`; this file records what was run and what came out. Trained 2026-10-09, the controls
without a foundation model 2026-10-10.

```
[p_t] --E (frozen, cached)--> [e_t] --T (RoPE transformer)--> [ê_t] --f--> Geminin-mKO, PIP-mVenus median in the nucleus
                                                                   └--g(p_t, ê_t)--> the two reporter images
```

## What was run

27 experiments (`experiments/`), one seed each, all on the same spatial split (`Bill/5-cellcycle`: 2,722 / 647 / 574
train / validation / test tracks) and the same budget: regression 15k steps, staining and multitask 30k steps, best
checkpoint on the validation split, scored on the test split (25,720 frames). T is 4 layers × 384 wide in every run;
E is one of four frozen pretrained encoders or one of two controls without pretraining, each frame embedded as
[global token | mean of the central 4×4 patch tokens].

| | MorphEm | CELL-DINO | DINOv3 | C-RADIOv4-H | Untrained ViT | ViT from scratch |
|---|---|---|---|---|---|---|
| regression f | ✓ + single-frame | ✓ + single-frame | ✓ + single-frame | ✓ + single-frame | ✓ + single-frame | ✓ + single-frame |
| virtual staining g | ✓ | ✓ | ✓ | ✓ + single-frame + ungated | ✓ | ✓ |
| multitask f + g | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

plus `stain_unconditioned`, the same UNet from phase alone. *Single-frame* removes T's layers (every ê_t reads its own
frame only); *ungated* applies the FiLM conditioning to every pixel instead of around the tracked nucleus.

*Untrained ViT* (`vit-untrained`) is the frozen control: MorphEm's ViT-S/16 architecture with its own random
initialization (seed 0), frozen and cached like the others (384 token + 384 centre patch mean), so pretraining is the
only difference from MorphEm. The UNet's encoder is trained from scratch in every run (no ImageNet weights), so
`stain_vit-untrained` has no pretrained weights anywhere. *ViT from scratch* (`vit-scratch`) is the same ViT, same
seed, trained with the model (its runs started on Intel nodes, where the seeded init draws different weights than on
the AMD node that embedded the frozen cache, so the two controls start from two draws of the same initialization):
`phase_frames` feeds every frame's centred crop, E runs in every step and its weights are kept in the checkpoint.

Scores (`dynaformer.cellcycle_metrics`): Pearson r per reporter over all test frames; track r, the per-track Pearson r
averaged over tracks (the trajectory, which frame-to-frame jitter destroys); PIP-FUCCI phase accuracy and macro F1
over the frames whose true phase is clear (G1 / S / G2/M). Always answering G1 scores phase accuracy **0.687** and F1
0.27. The stain is read out as the median of the predicted image inside the true nucleus mask, in the regression
head's units; pixel r is the per-crop image correlation.

## Results — test split

### Regression head f(ê_t)

| encoder | r Geminin | r PIP | R² | track r | phase acc | phase F1 |
|---|---|---|---|---|---|---|
| MorphEm | 0.674 | 0.620 | 0.400 | **0.536** | 0.867 | 0.773 |
| CELL-DINO | 0.735 | **0.668** | 0.442 | 0.490 | 0.868 | 0.763 |
| DINOv3 | 0.716 | 0.596 | 0.421 | 0.511 | 0.857 | 0.760 |
| C-RADIOv4-H | **0.738** | 0.610 | **0.445** | 0.512 | **0.868** | **0.777** |
| Untrained ViT | 0.629 | 0.478 | 0.291 | 0.350 | 0.651 | 0.491 |
| ViT from scratch | 0.651 | 0.515 | 0.324 | 0.385 | 0.791 | 0.638 |
| *single-frame:* MorphEm | 0.626 | 0.481 | 0.289 | 0.263 | 0.701 | 0.535 |
| CELL-DINO | 0.646 | 0.513 | 0.324 | 0.267 | 0.714 | 0.541 |
| DINOv3 | 0.592 | 0.470 | 0.261 | 0.225 | 0.697 | 0.525 |
| C-RADIOv4-H | 0.618 | 0.504 | 0.296 | 0.246 | 0.710 | 0.538 |
| Untrained ViT | 0.410 | 0.324 | 0.104 | 0.121 | 0.633 | 0.385 |
| ViT from scratch | 0.570 | 0.447 | 0.237 | 0.200 | 0.705 | 0.535 |

### Virtual staining g(p_t, ê_t), read out in the nucleus

| model | r Geminin | r PIP | R² | track r | phase acc | phase F1 | pixel r |
|---|---|---|---|---|---|---|---|
| UNet, phase only | 0.669 | 0.551 | 0.362 | 0.369 | 0.755 | 0.616 | 0.611 |
| + C-RADIO, single frame | 0.664 | 0.561 | 0.360 | 0.362 | 0.778 | 0.642 | 0.614 |
| + untrained ViT, temporal | 0.708 | 0.623 | 0.417 | 0.434 | 0.825 | 0.705 | 0.597 |
| + ViT from scratch, temporal | 0.648 | 0.494 | 0.327 | 0.383 | 0.674 | 0.426 | 0.564 |
| + MorphEm, temporal | 0.688 | 0.643 | 0.421 | 0.514 | 0.845 | 0.733 | 0.601 |
| + CELL-DINO, temporal | 0.688 | **0.673** | 0.443 | **0.599** | **0.876** | **0.785** | **0.622** |
| + DINOv3, temporal | **0.751** | 0.630 | 0.447 | 0.454 | 0.825 | 0.672 | 0.540 |
| + C-RADIOv4-H, temporal | 0.728 | 0.612 | 0.430 | 0.575 | 0.863 | 0.764 | 0.618 |
| + C-RADIOv4-H, temporal, ungated | 0.750 | 0.644 | **0.476** | 0.525 | 0.740 | 0.518 | 0.616 |

### Multitask f + g (each readout from the checkpoint selected on it)

| encoder | f: r Gem / r PIP | f: track r | f: phase acc | g: r Gem / r PIP | g: track r | g: phase acc | g: pixel r |
|---|---|---|---|---|---|---|---|
| MorphEm | 0.721 / 0.624 | 0.439 | 0.858 | 0.685 / 0.649 | 0.590 | 0.859 | 0.617 |
| CELL-DINO | 0.684 / 0.663 | 0.560 | 0.873 | 0.693 / 0.660 | 0.595 | **0.876** | 0.626 |
| DINOv3 | 0.734 / 0.631 | 0.443 | 0.832 | 0.736 / 0.624 | 0.568 | 0.861 | 0.616 |
| C-RADIOv4-H | 0.728 / 0.592 | 0.508 | **0.874** | 0.719 / 0.604 | 0.552 | 0.865 | 0.618 |
| Untrained ViT | 0.623 / 0.468 | 0.355 | 0.738 | 0.694 / 0.574 | 0.459 | 0.797 | 0.603 |
| ViT from scratch | 0.576 / 0.462 | 0.339 | 0.740 | 0.714 / 0.595 | 0.446 | 0.825 | 0.615 |

### Single-frame linear probe of each embedding (`scripts/probe_embeddings.py`)

Ridge regression from one frame's embedding, test-split r averaged over the two reporters:

| encoder | global token | central patch mean | both |
|---|---|---|---|
| CELL-DINO | 0.506 | 0.554 | **0.557** |
| MorphEm | 0.481 | 0.527 | 0.538 |
| C-RADIOv4-H | 0.471 | 0.523 | 0.530 |
| DINOv3 | 0.429 | 0.497 | 0.511 |
| Untrained ViT | 0.268 | 0.330 | 0.348 |

## What the numbers say

1. **Temporal integration is the largest effect, for every pretrained encoder.** The RoPE transformer roughly
   doubles track r (0.23-0.27 → 0.49-0.54) and lifts phase accuracy from 0.70-0.71, barely above always-G1, to
   0.86-0.87. The single-frame models cannot see S phase: PIP-mVenus drops within minutes at S onset with no visible
   change in one phase frame. `figures/trajectories.png` shows it: per-frame predictions jitter, temporal ones follow
   the G1/S transitions.
2. **The pretrained encoders end up close.** With temporal integration the four are within about 0.07 r of each
   other, and most single differences are of the size a second seed could move (one seed each). CELL-DINO and
   C-RADIOv4-H are at the top of most columns; DINOv3 is the weakest temporal model in phase accuracy. Per frame, the
   two microscopy models carry the most reporter information (linear probe), and MorphEm, a ViT-S, keeps up with the
   ViT-L/H models throughout.
3. **The regression head needs a pretrained E.** The untrained ViT, MorphEm's network with random weights, reaches
   phase accuracy 0.651, below always-G1, against 0.857-0.868 for every pretrained encoder (0.867 for MorphEm itself),
   and track r 0.35 against 0.49-0.54 (validation: 0.645 against 0.836-0.868). One frame's random features carry less
   of the reporters (linear probe r 0.35 against 0.51-0.56). T still integrates something from them (single-frame →
   temporal: track r 0.12 → 0.35, R² 0.10 → 0.29), but frame by frame the result is only as good as one pretrained
   frame (R² 0.29 against 0.26-0.32), and its phase calls stay below always-G1. Training the same ViT from scratch
   with f recovers part of the gap, not all of it: phase accuracy 0.791 (validation 0.774), track r 0.385, R² 0.32.
   The scored checkpoint, best on validation r, is at 5.5k of 15k steps, and validation phase accuracy stays at
   0.78-0.79 from about 8k steps on, so more steps would not close the rest. Single-frame, it matches the pretrained
   single-frame models in phase accuracy (0.705 against 0.697-0.714) but not in R² (0.24 against 0.26-0.32) or track
   r (0.20 against 0.23-0.27).
4. **Staining depends less on pretraining**, as the UNet learns its own features from the phase image: conditioning on
   the untrained ViT still lifts the phase-only UNet from 0.755 to 0.825 in phase accuracy (validation 0.726 →
   0.792), PIP r from 0.55 to 0.62 and track r from 0.37 to 0.43, about as far as DINOv3 (0.825, track r 0.45) and
   short of MorphEm (0.845, 0.51) and CELL-DINO (0.876, 0.60). Training that ViT from scratch through the stain loss
   alone does worse than leaving it random. Validation r, the checkpoint monitor, peaks at 4k steps (0.553) and then
   stays at 0.49-0.52, while the phase-only UNet climbs to 0.537 (at 19k) and the frozen control to 0.646 (at 20k);
   its training loss matches theirs (≈0.031 at 20-25k steps), so the learned conditioning overfits. The scored 4k
   checkpoint, whose UNet has also trained less, falls below the phase-only UNet in phase accuracy (0.674, below
   always-G1), PIP r (0.49) and pixel r (0.564 against 0.611), at about equal track r (0.38 against 0.37). With
   random features, training f together with g lifts f's phase accuracy (0.651 → 0.738; validation 0.645 → 0.737),
   though not its r or track r, and lowers g's (0.825 → 0.797). Trained from scratch with f and g together, the ViT
   does not overfit the stain readout this way: its validation stain r peaks at 22k of 30k steps (0.624) instead of
   at 4k, at about the same training stain loss (≈0.030 at 20-25k steps against 0.031). Its stain readout reaches
   phase accuracy 0.825, track r 0.446 and pixel r 0.615 (stain loss alone: 0.674, 0.383, 0.564; validation phase
   accuracy 0.795 against 0.670), level with the frozen untrained ViT's multitask readout (0.797, 0.459, 0.603). Its
   regression head does worse than when the ViT trains with f alone: track r 0.339 and phase accuracy 0.740 against
   0.385 and 0.791, about the frozen control's multitask head (0.355, 0.738). f does not even fit its training
   targets here: its training loss is still 0.14 at the end (25-30k steps), where the frozen control's multitask head
   and the ViT trained with f alone end at 0.04.
5. **The nucleus-centred patch tokens matter.** For every encoder the mean of the central patch tokens predicts the
   reporters better than the global token, and the concatenation used here is best.
6. **Conditioning the UNet on ê_t helps where the phase image cannot: PIP and the cycle's timing.** Against the
   phase-only UNet, the pretrained encoders take PIP r from 0.55 to 0.61-0.67, track r from 0.37 to 0.45-0.60 and
   phase accuracy from 0.755 to 0.83-0.88; Geminin, which the UNet partly reads from the phase image, gains less
   consistently (0.67 to 0.69-0.75). The per-frame embedding alone adds almost nothing (single-frame row: track r
   unchanged, phase accuracy 0.778), so the gain is temporal. Pixel r stays at 0.60-0.62 (0.54 for DINOv3): the
   conditioning acts on the tracked nucleus, not the whole crop. `figures/stain_track*.png` shows the phase-only UNet
   keeping PIP on through S onset where the conditioned models switch it off.
7. **The centre gate is needed for phase calls.** Conditioning every pixel gives similar correlations but phase
   accuracy 0.740 instead of 0.863 (S-phase F1 0.17): without the gate, PIP is never driven fully off at the tracked
   nucleus, and the phase thresholds are absolute.
8. **Multitask is not a consistent win.** It raises the stain readout's track r for MorphEm (0.51 → 0.59) and DINOv3
   (0.45 → 0.57) and the regression head's for CELL-DINO (0.49 → 0.56), but leaves the best numbers where they were.
   The best models overall are the CELL-DINO stain and multitask models (phase accuracy 0.876, F1 0.785, track r
   0.60) and, for the regression head alone, C-RADIOv4-H and CELL-DINO (0.868).

## Caveats

- **One seed per configuration**, and one random draw per ViT control: seed 0 for both, but drawn on different CPU
  types (above), so not the same weights. Differences of a few hundredths between encoders are within what seeds can
  move.
- **Frozen pretrained encoders.** No pretrained E was fine-tuned; the comparison is of the representations as
  published.
- **The from-scratch control is not tuned.** It uses the frozen runs' hyperparameters (lr 3e-4, weight decay 0.05, 500
  warmup steps), not ones chosen for training a ViT. It is also far slower: E runs in every step (0.5-1 s per step on
  an H100), where the cached runs never run it.
- **Bidirectional T uses future frames**, which suits offline analysis of a recorded movie. Transitions in the last
  ~25 frames of the movie, with little future context, are the commonest misses (`figures/trajectories.png`, track
  69), and a track that continues through a division into a daughter gets smoothed over the division
  (`figures/stain_track1712.png`, t = 0).
- **One experiment.** All tracks come from one 14.5 h mosaic; the split is spatial (blocks, lineages kept together,
  a guard band so no pixel appears in crops of two splits), not across experiments.
- **Earlier runs are superseded.** The first round trained with a sampler that silently skipped 70% of the training
  tracks (all tracks shorter than the 64-frame window); those runs were deleted, and every number above is from the
  corrected rerun. Their W&B runs (started before 2026-10-09 17:20 PDT) remain in the project.

## Where things are

- Runs: `/hpc/projects/comp.micro/virtual_staining/models/dynaformer/runs/cellcycle/<experiment>/` (checkpoints,
  `eval/best_<readout>/{predictions,metrics}_{val,test}.{csv,json}`); tables `summary_{val,test}.csv`; figures in
  `figures/`.
- W&B: project `dynaformer-cellcycle` (runs started after 2026-10-09 17:20 PDT; a run that was requeued appears as one
  W&B run per allocation).
- Reproduce: `run_embed.slurm` per encoder (`vit-untrained` too; `vit-scratch` runs E inside training and needs no
  cache), then `./submit.sh --time=0:30:00 [--constraint="h200|h100|a100"] -- <leaves>`;
  `python ../../scripts/summarize_cellcycle.py --split test`, `plot_cellcycle.py`, `render_stain_examples.py`.
  To retrain a leaf from scratch, move its run directory away first: `run_fit.slurm` resumes from what it finds there.
