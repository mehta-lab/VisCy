# dynacell — Claude Code reference

## Model name conventions (code ↔ paper)

Config keys, prediction-zarr filenames, eval keys, and W&B run names use **code
names**; figures/tables/manuscripts use **paper names**. Translate at any
code/paper boundary. This table is the source of truth (referenced by
`src/dynacell/evaluation/save_paths.py:PAPER_KEY`).

| Code name | Paper name |
| --- | --- |
| `fcmae_vscyto3d_scratch` | **UNeXt2** |
| `fcmae_vscyto3d_pretrained` | **VSCyto3D** (FCMAE-pretrained UNeXt2) |
| `unetvit3d` | **UNetViT3D** (deterministic; iPSC-trained only) |
| `pix2pix3d_unetvit` | **pix2pix3d** (GAN; same UNetViT3D generator, `DynacellGAN` engine) |
| `fnet3d_paper` | **FNet3D** |
| `celldiff` / `celldiff_r2` | **CELL-Diff** (variants: `iterative`, `sliding_window`, `denoise`/Mean Predictor) |

VSCyto3D ablations (in `vscyto3d-ablations`): `*_randinit` (untrained),
`*_cytoland` (public ckpt, no FT), `*_infectionft` (cytoland→A549-infection-FT,
no FT), `vscyto3d_cytolandft` / `vscyto3d_infectionft_dynacellft` (+ dynacell FT,
dual nucleus+membrane). The same suffixes appear as zarr-filename infixes;
`_cytolandft` / `_infectionft_dynacellft` combine with `_a549trained`.

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

## Eval directory naming

`src/dynacell/evaluation/save_paths.py:eval_save_dir` is the writer and the
cross-repo contract (must match the paper's
`compute_all_organelle_precision_recall.py:eval_dir_for`; pinned by
`tests/test_save_paths.py`). Canonical focus-2D outputs under
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
