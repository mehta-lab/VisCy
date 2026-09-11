# dynacell — Claude Code reference

## Model name conventions (code ↔ paper)

Config keys, prediction-zarr filenames, eval keys, and W&B run names use **code
names**; figures/tables/manuscripts use **paper names**. Translate at any
code/paper boundary. This table is the source of truth (referenced by
`src/dynacell/evaluation/paths.py:PAPER_KEY`).

| Code name | Paper name |
| --- | --- |
| `fcmae_vscyto3d_scratch` | **UNeXt2** |
| `fcmae_vscyto3d_pretrained` | **VSCyto3D** (FCMAE-pretrained UNeXt2) |
| `unetvit3d` | **UNetViT3D** (deterministic; iPSC-trained for nucleus/membrane; ER/mito also have a549/joint checkpoints from the A549 raw-regen campaign gap-fill) |
| `pix2pix3d_unetvit` | **pix2pix3d** (GAN; same UNetViT3D generator, `DynacellGAN` engine) |
| `fnet3d_paper` | **FNet3D** |
| `celldiff` / `celldiff_r2` | **CELL-Diff** (variants: `iterative`, `sliding_window`, `denoise`/Mean Predictor) |
| `fcmae_vscyto2d_scratch` | **UNeXt2-2D** (in-focus 2D track) |
| `fcmae_vscyto2d_pretrained` | **VSCyto2D** (encoder init from the public 2D FCMAE ckpt) |
| `fnet2d` | **FNet2D** |
| `celldiff_2d` | **CellDiff-2D** (CELLDiffNet run Z-preserving at Z=1) |
| `pix2pix2d_unetvit` | **pix2pix2d** (same GAN, UNetViT generator + PatchGAN both run Z-preserving at Z=1) |

The five 2D keys are the in-focus 2D-vs-3D benchmark track. Their paper names are
deliberately distinct from the 3D namesakes (`unext2_2d` vs `unext2`, `vscyto2d` vs
`vscyto3d`, `celldiff_2d` vs `celldiff_r2`, `pix2pix2d` vs `pix2pix3d`) — collapsing them
would merge the 2D and 3D rows into one eval dir. `celldiff_2d` is in `_CELLDIFF_MODELS`
(not `_DETERMINISTIC_MODELS`) because it is flow-matching like its 3D counterpart; the
tuple is ordered longest-first so prefix matching cannot truncate it to bare `celldiff`.
`pix2pix2d_unetvit` goes in `_DETERMINISTIC_MODELS` for the same reason
`pix2pix3d_unetvit` does — a GAN generator is a single deterministic forward at
inference, and the sampling variants that tuple guards against are CellDiff's.

⚠ **The published pix2pix3d weights were NOT trained by the overlay its `train.yml`
leaves compose.** `train.yml` binds the LSGAN baseline
(`pix2pix3d_unetvit_fit.yml`); the weights came from the 12
`train_4gpu_modernized.yml` leaves, i.e. Run D — checkpoint `hyper_parameters` read
`loss_type='nonsat'`, `lr_g=lr_d=2e-4`, `r1_gamma=10`, `ema_kimg=10`,
`lecam_gamma=0.3`, `lambda_l1=10`, plus 201 `generator_ema.*` tensors and
`max_epochs: 40`. `pix2pix2d_unetvit_fit.yml` mirrors Run D so the 2D-vs-3D comparison
isolates geometry, and bakes the three Run-D deltas instead of repeating them per leaf.
Read a checkpoint's own hparams before assuming a leaf's overlay is what ran.

VSCyto3D ablations (in `vscyto3d-ablations`): `*_randinit` (untrained),
`*_cytoland` (public ckpt, no FT), `*_infectionft` (cytoland→A549-infection-FT,
no FT), `vscyto3d_cytolandft` / `vscyto3d_infectionft_dynacellft` (+ dynacell FT,
dual nucleus+membrane). The same suffixes appear as zarr-filename infixes;
`_cytolandft` / `_infectionft_dynacellft` combine with `_a549trained`.

## Training data: A549 condition pooling

A549 training data is **condition-pooled**: both joint (iPSC+A549) and A549-only fits
read a single `<TARGET>_all.zarr` per target combining **mock + ZIKV + DENV** (not
mock-only), at
`/hpc/projects/virtual_staining/training/dynacell/a549/mantis_v1/train/{H2B,CAAX,SEC61B,TOMM20}_all.zarr`.
Built by `assemble.py` (paper repo `dynacell_paper/preprocess/a549_mantis/`) with
`condition=None`, so its per-condition position filter drops nothing — it walks every
train-split position across all three conditions.

- The pooled store is a single flattened well (`0/0`) with sequential FOV names
  (`fov0000…`); the **condition is dropped from the FOV name**, so you cannot read the
  conditions off the on-disk layout — only from the build script / config comment.
- There is **no `*joint*` train-set fragment** — joint train leaves compose
  `_internal/shared/model/train_sets/{a549_mantis,ipsc_confocal}.yml` and author the two
  `HCSDataModule` children of `BatchedConcatDataModule` inline (A549 child →
  `<TARGET>_all.zarr`, iPSC child → `ipsc/dataset_v4/train/cell.zarr`).
- Deliberate asymmetry: **train pools conditions; predict/eval stays per-condition**
  (canonical per-treatment `.ozx` in the manifest registry) — hence test sets and the
  paper tables break out mock/denv/zikv.

## Prediction zarr naming

Set by `trainer.callbacks[…HCSPredictionWriter].init_args.output_store` in each
predict leaf. The infix between model name and the optional plate condition flags
the **training set** of the source model:

| Trained on | Test set | Filename |
| --- | --- | --- |
| iPSC | iPSC | `<org>_<model>.zarr` |
| iPSC | A549 plate | `<org>_<model>_<cond>.zarr` |
| A549 | iPSC | `<org>_<model>_a549trained.zarr` |
| A549 | A549 plate | `<org>_<model>_a549trained_<cond>.zarr` |
| Joint | iPSC | `<org>_<model>_jointtrained.zarr` |
| Joint | A549 plate | `<org>_<model>_jointtrained_<cond>.zarr` |

`<org>` ∈ `nucl`/`memb`/`sec61b`/`tomm20`; `<model>` is the **code name**;
`<cond>` ∈ `mock`/`denv`/`zikv`. The no-infix iPSC-trained form is historical;
don't add a `_ipsctrained` infix. iPSC-test predictions go under
`ipsc/predictions/`, A549-plate predictions under `a549/predictions/`.

Two exceptions to watch:
- **Legacy ER/Mito iPSC zarrs** use `<gene>_<model>__<gene>_<cond>.zarr`
  (double-underscore + redundant gene prefix). Don't propagate to new leaves.
- **CellDiff-R2 joint predicts** live in `{ipsc,a549}/joint_predictions/` (not
  `predictions/`) and carry **no** `_jointtrained` infix — bare
  `<org>_celldiff_r2[_<cond>].zarr`. Sweeps that only walk `predictions/` miss
  them. FCMAE/fnet3d joint zarrs in the same dir do follow `_jointtrained_<cond>`.

## Re-predict completeness: gate on chunk mtime, never chunk count

`--overwrite` makes `HCSPredictionWriter` rewrite the prediction channel **in
place**. It never deletes the previous run's chunks, so every count-based
completeness gate reads FULL from the first second of the run.

Measured 2026-09-11 on the 36 CellDiff-2D A549 re-predicts: counting chunks said
**432/432 FOVs complete across all 36 stores** while all 36 jobs were still
`RUNNING` — which looks exactly like the post-completion hang in the root
CLAUDE.md. It was not. An mtime split put `er/celldiff_2d/a549/a549__denv` at
fov0010 213/480 with fov0011 untouched and **748 chunks still carrying the
previous run's mtimes**; the true mean across the 36 was ~55%, not 100%.

The gate that works — count only chunks this run wrote:

```sh
find $STORE/0/0 -path '*/0/c/*' -type f -newermt '<run submit time>' | wc -l
```

against `n_fov * T * Z`. Three things that make a naive version wrong:

- **Read `T` from each FOV's own `zarr.json` `shape[0]`.** It varies *within* one
  plate — 7 and 10 both occur in the same A549 prediction store — so one `T` for
  the whole store over- or under-counts.
- **The chunk path under `c/` is `<t>/<c>/<z>/<y>/<x>`.** `c/2/0/47/0/0` is
  **timepoint 2**, z-plane 47 — not channel 2. Reading that index as a channel
  makes a 7-timepoint store look like a complete 48-plane one.
- **Comparing declared `T` to the number of timepoint directories present is
  circular** on a re-predict: the previous run already created every directory,
  so they always match.

Also restrict the glob to `*/0/c/*` (resolution level 0). `*/c/*` picks up the
multiscale pyramid levels too and reports >100%.

Separately, a re-predict does **not** invalidate the eval caches — see
`force_recompute` under *Grouped multi-condition eval*. Overwriting a store in
place leaves `final_metrics` happy to rescore masks and embeddings derived from
the old predictions, so a re-predicted bucket needs every `pred_*`
`force_recompute` flag, not just `final_metrics`.

## Eval directory naming

`src/dynacell/evaluation/paths.py` is the writer and the cross-repo contract
(`eval_leaf` for the canonical model-centric layout, `normalize_legacy` for the
pre-canonical `*_with_embeddings` forms; the paper repo vendor-copies it and
asserts parity — must match `compute_all_organelle_precision_recall.py:eval_dir_for`;
pinned by `tests/test_paths.py`). Canonical focus-2D outputs under
`/hpc/projects/virtual_staining/training/dynacell/{ipsc,a549}/`:
`evaluations_with_embeddings/` (ipsc-trained), `evaluations_a549trained_with_embeddings/`,
`evaluations_jointtrained_with_embeddings/` (infix `jointtrained`). Dirs use the
paper key. Plain `evaluations/` + `joint_evaluations/` are **stale pre-2D**
(off-scale FID/KID) — do not read.

## Eval runtime / parallelism

`dynacell.evaluation.runtime` provides thread-cap + optional FOV-level
parallelism, opt-in via the `runtime:` block in `eval.yaml`. Defaults preserve
sequential behavior.

```yaml
runtime:
  fov_workers: 1                          # int | "auto"
  threads_per_worker: "auto"              # int | "auto" -> cpu_count // fov_workers
  executor: "serial"                      # "serial" | "process"
```

- `executor=serial` (default): inline FOV loop, identical to pre-runtime behavior.
- `executor=process`: spawn-context `ProcessPoolExecutor` over FOVs; each worker
  lazy-loads models under an fcntl GPU lock (N model copies resident, one GPU op
  at a time). `fov_workers: "auto"` → 1 under serial; clamps to
  `min(cpu_count // threads_per_worker, n_positions)` under process. `fov_workers=1`
  + process auto-demotes to serial (avoids the spawn cold-start).
- `DYNACELL_THREADS_PER_WORKER=N` — export in SLURM scripts **before** invoking
  `dynacell evaluate` (sets `OMP/MKL/OPENBLAS_NUM_THREADS` at C-extension load).
- `DYNACELL_FORCE_PER_T_HYGIENE=1` — runtime escape hatch to force per-T
  `cuda_empty_cache` + `gc_collect` without a YAML change.

## Grouped multi-condition eval

For the same `(model, organelle)` across multiple I/O variants (typically the 3
A549 plates), use `dynacell evaluate-grouped`: it loads SuperModel + DINOv3 +
DynaCLR + CELL-DINO once, then loops conditions, paying the ~30–90 s cold-start
once instead of per-condition. Leaves live under
`configs/benchmarks/virtual_staining/_internal/leaf/grouped/` (discovered via
`_EXTERNAL_SEARCHPATHS`):

```sh
uv run dynacell evaluate-grouped leaf=grouped/<bucket>/eval_grouped
```

Use `leaf=` (a group override), not `-c` (Hydra's `--cfg`, display-only).
Per-condition overlays may override `io.*`, `save.*`, `runtime.*`,
`limit_positions`, `force_recompute.*`, and a `name` label — but **not**
`target_name`, `feature_extractor.*`, `compute_feature_metrics`, or `use_gpu`
(those gate model loading; run such variants separately). Each condition honors
its own cache: with both `force_recompute.all=false` and
`force_recompute.final_metrics=false`, conditions with existing CSV/NPY skip and
load cached outputs — the idiom for folding a new condition into a done bucket.
`executor=serial` maximizes the cross-condition amortization (process mode
re-loads models per condition).

## Focus-aware 2D projection (`evaluation/focus.py`)

The 3D→2D reduction is focus-aware so projection isn't dominated by out-of-focus
caps (A549 Z=48, in-focus band ~5 planes). Three knobs, all default-off:

- `feature_metrics.focus_slab.{enabled,halfwidth,channel_name}` — deep-feature
  crops + per-cell similarity max-project over a `2*halfwidth+1` slab on the
  in-focus plane (CP regionprops stay 3D). Folds `+focusslab_h{h}_{ch}_{sig}`
  into each deep-feature `preprocess_version` so embedding caches auto-invalidate
  (`{sig}` hashes the `focus.{na_det,lambda_ill,pixel_size}` params).
- `segmentation.slice_selection=focus` — 2D instance seg picks the in-focus plane
  (vs the old `frac=0.30`).
- `precompute-gt build.focus=true` — writes `focus_slice` zattrs to a writable GT
  store.

**Plane source (precedence):** (1) precomputed `focus_slice` zattrs (iPSC `.zarr`
fast path); (2) `io.gt_cache_dir/focus_planes/<channel>/<pos>.json`; (3) compute
from the phase channel + persist. Compute-at-eval-time (3) is why focus works on
the **read-only published A549 `.ozx`** — they carry no zattrs and `pack_ozx`
isn't bit-reproducible, so we derive the plane from the `Phase3D` already in each
store and never fork them. Estimator: `waveorder.focus.focus_from_transverse_band`
(not the `qc` app — keeps the dep graph `applications/ → packages/`).

## Predict submission modes

`tools/submit_benchmark_batch.py` (wrapper `tools/predict_batch.sh`) covers three
mutually exclusive shapes — pick by parallelism, not familiarity:

| Mode | Flag | Per-GPU concurrency |
|---|---|---|
| Serial (default) | (none) | 1 |
| Array | `--array [--max-array-concurrency K]` | 1 per task, K allocations |
| Chunked | `--parallel P` | P backgrounded children on one GPU |

- Minimal queue footprint, contiguous time → serial.
- Many leaves, let SLURM throttle allocations → `--array`.
- Few GPU-light leaves → `--parallel P` (2-up confirmed on A40, 2–4 on H200/H100).
- Mixed hardware profiles → `--array --allow-mixed-directives` (one array per
  directive bucket; **not** compatible with `--parallel`).

For local foreground runs, `tools/predict_local.sh --parallel N` backgrounds on
the current host's GPU (2-up confirmed on the A40 interactive node) — a different
path from the sbatch helper's `--parallel`.

## Repository boundary: paper artifacts do not belong here

**This repo owns data packaging, model training and evaluation. Every paper table
and figure is produced in the dynacell repo** (`/hpc/mydata/alex.kalinin/dynacell`),
which reads the evaluation outputs this repo writes.

The split had blurred in two directions, both since corrected:

- `VisCy/.tmp/` accumulated 58 untracked scripts, 34 of which wrote an artifact. One
  of them, `segcompare_nucleus.py`, filled the mask cache behind a **live** paper
  figure, so the manuscript depended on an untracked scratch directory. It has been
  copied to `dynacell/paper/scripts/` with its cache under
  `$DATA_ROOT/segcompare`; the originals remain in `.tmp/`, which is now purely
  exploratory. `.tmp/` is **not gitignored** — never stage it.
- `src/dynacell/reporting/` and `src/dynacell/evaluation/spectral_pcc/plot_*.py`
  stay here and are correct as they are: `reporting/` is a *library* the dynacell
  generators import, and the `spectral_pcc` plots are metric-validation
  diagnostics, not paper figures.

Before adding a script that emits a `.tex`, a `.pdf`, or a table, ask whether it is
producing a paper artifact. If it is, it belongs in dynacell. The generator index
lives at `dynacell/paper/scripts/CLAUDE.md`, `dynacell/paper/tables/CLAUDE.md` and
`dynacell/paper/figures/CLAUDE.md`, with retired tooling and the full `.tmp/`
classification in `dynacell/paper/archive/README.md`.
