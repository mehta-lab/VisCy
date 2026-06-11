# dynacell — Claude Code reference

## Model name conventions

Code names (used in YAML config keys, prediction zarr filenames, eval pipeline keys, W&B run names) differ from the paper names. When writing/reading anything that crosses the code/paper boundary (figures, tables, Confluence pages, manuscripts), translate:

| Code name (config / zarr / W&B) | Paper / display name |
| --- | --- |
| `fcmae_vscyto3d_scratch` | **UNeXt2** |
| `fcmae_vscyto3d_pretrained` | **VSCyto3D** (FCMAE-pretrained is the canonical VSCyto3D variant) |
| `unext2` | UNeXt2 (legacy zarr prefix; superseded by `fcmae_vscyto3d_scratch`) |
| `vscyto3d` | VSCyto3D (display key in Dihan's eval pipeline; sources `*_fcmae_vscyto3d_pretrained` predictions) |
| `unetvit3d` | UNetViT3D |
| `fnet3d_paper` | FNet3D |
| `celldiff` | CELL-Diff (variants: `iterative`, `sliding_window`, `denoise`/Mean Predictor) |
| `fcmae_vscyto3d_pretrained_randinit` | **VSCyto3D-RandInit** (untrained ablation; one frozen ckpt per organelle persisted by `save_random_init_vscyto3d_ckpts.py`) |
| `fcmae_vscyto3d_pretrained_cytoland` | **VSCyto3D-Cytoland** (cytoland public ckpt evaluated without dynacell FT) |
| `fcmae_vscyto3d_pretrained_infectionft` | **VSCyto3D-InfectionFT** (cytoland → A549-infection-FT ckpt evaluated without further FT) |
| `vscyto3d_cytolandft` | **VSCyto3D-CytolandFT** (cytoland ckpt + dynacell FT; dual nucleus+membrane, 2-channel) |
| `vscyto3d_infectionft_dynacellft` | **VSCyto3D-InfectionFT-DynacellFT** (cytoland → A549-infection-FT → dynacell FT; dual nucleus+membrane) |

**Training-set infixes for the no-FT ablations** (Track A/B in `vscyto3d-ablations`):

| Infix in zarr filename | Meaning |
| --- | --- |
| `_randinit` | random init, no training (Track A) |
| `_cytoland` | cytoland public ckpt, no FT (Track B1) |
| `_infectionft` | VSCyto3D-A549-infection-finetune ckpt, no FT (Track B2) |
| `_cytolandft` | cytoland init + dynacell FT (Track C1) — combines with `_a549trained` for A549-trained variants |
| `_infectionft_dynacellft` | infection-FT init + dynacell FT (Track C2) — combines with `_a549trained` similarly |

Eval-pipeline directory naming (`/hpc/projects/virtual_staining/training/dynacell/{ipsc,a549}/evaluations/eval_<model>_<organelle>[_<plate>]`) uses the **paper key** (`unext2`, `vscyto3d`, `fnet3d`, `unetvit3d`, `celldiff_*`), not the config key. So `eval_unext2_membrane` maps to the `fcmae_vscyto3d_scratch` predictions, `eval_vscyto3d_membrane` maps to `fcmae_vscyto3d_pretrained`.

## Prediction zarr naming convention

Set by `trainer.callbacks[…HCSPredictionWriter].init_args.output_store` in each leaf of `applications/dynacell/configs/benchmarks/virtual_staining/<organelle>/<model>/<train_set>/predict__*.yml`. The infix between model name and the optional plate condition flags the **training set** of the source model:

| Trained on | Test set | Filename |
| --- | --- | --- |
| iPSC | iPSC | `<org>_<model>.zarr` |
| iPSC | A549 plate | `<org>_<model>_<cond>.zarr` |
| A549 | iPSC | `<org>_<model>_a549trained.zarr` |
| A549 | A549 plate | `<org>_<model>_a549trained_<cond>.zarr` |
| Joint (iPSC + A549) | iPSC | `<org>_<model>_jointtrained.zarr` |
| Joint (iPSC + A549) | A549 plate | `<org>_<model>_jointtrained_<cond>.zarr` |

Where `<org>` is `nucl` / `memb` / `sec61b` / `tomm20`, `<model>` is the **code name** from the table above (e.g. `fcmae_vscyto3d_scratch`, `fnet3d_paper`), and `<cond>` is `mock` / `denv` / `zikv`. The (no-infix) iPSC-trained naming is historical baggage from before joint/A549 training existed; don't add a `_ipsctrained` infix retroactively. Output dirs: iPSC test predictions land under `ipsc/predictions/`, A549 plate predictions under `a549/predictions/`, regardless of training set.

Caveat: Dihan's earlier ER + Mito iPSC-trained zarrs use a legacy `<gene>_<model>__<gene>_<cond>.zarr` shape (e.g. `sec61b_fcmae_vscyto3d_scratch__sec61b_mock.zarr`, double-underscore + redundant gene prefix). New leaves should follow the table above; do not propagate the legacy form.

### CellDiff-R2 joint predictions: separate dir + no `_jointtrained` infix

CellDiff-R2 joint predicts (model `celldiff_r2` trained on iPSC + A549 mantis) deviate from the convention above in **two** ways. Predict configs live at `applications/dynacell/configs/benchmarks/virtual_staining/<organelle>/celldiff/joint_ipsc_confocal_a549_mantis/predict__*.yml` — the directory name says `celldiff/` but the YAMLs hard-code `ckpt_path: .../celldiff_r2/checkpoints/last.ckpt`, so the model variant is selected by the checkpoint, not by the directory.

The submitter that wires these up is `/hpc/projects/comp.micro/virtual_staining/models/cell_diff_vs_viscy/VisCy/plot_related/run_celldiff_r2_pred_joint.slurm` (16-task array; submitted 2026-05-18, completed 2026-05-20 as job `33021852`).

| Trained on | Test set | Output path |
| --- | --- | --- |
| Joint (iPSC + A549) | iPSC | `ipsc/joint_predictions/<org>_celldiff_r2.zarr` |
| Joint (iPSC + A549) | A549 plate | `a549/joint_predictions/<org>_celldiff_r2_<cond>.zarr` |

Note the differences vs. the joint rows in the main table:
- Output dir is `joint_predictions/`, **not** `predictions/`. Sweeps that only walk `predictions/` will miss every joint zarr.
- Filename has **no** `_jointtrained` infix. The model name `celldiff_r2` alone implies joint here. The same naming inconsistency does not apply to FCMAE / fnet3d joint zarrs, which correctly land at `predictions/<org>_<model>_jointtrained[_<cond>].zarr`.

Counts: 4 iPSC zarrs (one per organelle, 100 positions each) + 12 A549 zarrs (4 organelles × 3 plates, 14 positions each). Coexisting joint predicts for FCMAE/fnet3d under the same `joint_predictions/` dir follow the `_jointtrained_<cond>` convention — only CellDiff-R2's are bare. When inspecting "joint training results for CellDiff-R2," check `joint_predictions/` first.

## Eval runtime / parallelism

`dynacell.evaluation.runtime` (added 2026-05) provides three layered thread-cap entry points + optional FOV-level parallelism via spawn-context `ProcessPoolExecutor`. Defaults preserve sequential behavior; opt in via the `runtime:` block in `eval.yaml`.

### Config block (in `_configs/eval.yaml`)

```yaml
runtime:
  fov_workers: 1                          # int | "auto"
  threads_per_worker: "auto"              # int | "auto" -> cpu_count // fov_workers
  executor: "serial"                      # "serial" | "process"
  cuda_empty_cache_every_n_timepoints: 0  # 0 = off
  gc_collect_every_n_fovs: 0              # 0 = off
```

- `executor=serial` (default): inline FOV loop, identical to pre-runtime-module behavior.
- `executor=process`: spawn-context `ProcessPoolExecutor` over FOVs. Each worker independently lazy-loads `seg_model` + extractors under an fcntl GPU lock at `/tmp/dynacell_gpu_<SLURM_JOB_ID>.lock`, so models stay GPU-resident per worker (N × model weights on the GPU) but only one worker runs GPU work at a time. Suitable when GPU memory has headroom for N model copies.
- `fov_workers: "auto"` resolves to 1 under `executor=serial`; under `executor=process` it clamps to `min(cpu_count // threads_per_worker, n_positions)`.
- Literal `fov_workers > 1` with `executor=serial` raises. Literal `fov_workers=1` with `executor=process` auto-demotes to `serial` to avoid the ~5 s spawn cold-start cost for a single-worker pool.
- Two-phase resolve: parent applies BLAS cap at function entry (provisional), then re-resolves after the position list is built with `freeze_threads_per_worker=` so worker initializers see the same value the parent capped to.

### Env-var entry points

- `DYNACELL_THREADS_PER_WORKER=N` — set in SLURM scripts BEFORE invoking `dynacell evaluate`. Exports `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` at C-extension load time (in-process `apply_thread_budget` is a runtime safety net but can come after BLAS load). The `__main__:main_cli` first statement reads this var.
- `DYNACELL_FORCE_PER_T_HYGIENE=1` — operator escape hatch: flips both `cuda_empty_cache_every_n_timepoints` and `gc_collect_every_n_fovs` to ≥1 at runtime regardless of YAML defaults. Useful for post-ship mitigation of per-T memory degradation without a code change.

### Timing instrumentation

Region timers are always on (overhead is ~120 ms on a 9-h eval, ~4e-6 of wall time). Output goes to `<save_dir>/eval_timing.csv` at end of run with columns `pos_name, t, region, seconds`. Regions tag the FOV-level work (`mask_gt`, `mask_pred`, `cp_gt`, `deep_gt_{dinov3,dynaclr,celldino}`, `features_pred_per_t`, `microssim`, `seg_write`) and the per-T work (`pixel_metrics`, `mask_metrics`, `feature_pairwise`).

Under `executor=process`, workers return their slice of the timing log inside `FovResult.timings`; the parent aggregator concatenates.

### Reality check for process mode

- The parent still loads the seg model once via `load_eval_models(config)` at the start of `evaluate_predictions` in `pipeline.py` (used for checkpoint pre-warm side-effect under `process` mode + the seg model itself under `serial` mode).
- `precompute_deep_features` (the upfront batched feature pre-fill) stays in the parent — single-pass over positions with the `DeepFeatureBatcher` cross-FOV amortization. Parallelizing precompute is a separate plan.
- Workers cannot share open iohub plate handles or torch modules across the pickle boundary; each worker re-opens plates on first FOV.
- `ProcessPoolExecutor.shutdown(wait=False, cancel_futures=True)` cancels queued futures only — in-flight workers continue to completion (~minutes if mid-cellpose). Ctrl-C may need `scancel` to fully release GPU memory.
- Tests for the runtime module + FovResult pickle contract live in `applications/dynacell/tests/test_runtime.py` and `tests/test_evaluation_pipeline_parallel.py`. An end-to-end serial-vs-process parity test on a real iohub fixture is a follow-up (see plan `.claude/plans/eval-parallelism.md` §C5).

### Grouped multi-condition eval

When evaluating the same `(model, organelle)` across multiple I/O variants — typically the three A549 treatment plates (`mock`, `denv`, `zikv`), but also any case where the same trained model is scored on multiple datasets — use `dynacell evaluate-grouped`. The driver loads `SuperModel` + `DinoV3` + `DynaCLR` + `CELL-DINO` once, then loops over the conditions calling `evaluate_predictions(merged_cfg, models=...)` so the per-condition cold-start (~30–90 s of weight load + checkpoint warmup) is paid once total instead of N times.

```yaml
# applications/dynacell/configs/.../eval_grouped_a549_mantis_er.yml
defaults:
  - eval_grouped
  - _self_

target_name: er           # MUST be set at the base; conditions cannot override it
compute_feature_metrics: true
feature_extractor:
  dinov3: { ... }         # shared across all conditions
  dynaclr: { checkpoint: /path/to/dynaclr.ckpt, ... }
  celldino: { weights_path: /path/to/celldino.ckpt, ... }

conditions:
  - name: a549_mock
    io: { pred_path: ..., gt_path: ..., gt_cache_dir: ..., pred_cache_dir: ..., cell_segmentation_path: ... }
    save: { save_dir: /path/to/out/a549_mock }
  - name: a549_denv
    io: { ... }
    save: { save_dir: /path/to/out/a549_denv }
  - name: a549_zikv
    io: { ... }
    save: { save_dir: /path/to/out/a549_zikv }
```

Invoke (leaves live under
`applications/dynacell/configs/benchmarks/virtual_staining/_internal/leaf/grouped/`
and are discovered via `_EXTERNAL_SEARCHPATHS` in `__main__.py`):
```sh
uv run dynacell evaluate-grouped leaf=grouped/<bucket>/eval_grouped
```

`-c` is Hydra's `--cfg` flag (accepts `job`, `hydra`, or `all` for config
display only); it cannot select the leaf. Use the `leaf=` group override
instead — that's how single-condition leaves discover their YAML too.

Constraints (enforced at runtime by `_check_grouped_field_invariants`):
- Per-condition overlays may freely override `io.*`, `save.*`, `runtime.*`, `limit_positions`, `force_recompute.*`, and carry a `name` label.
- Overlays MUST NOT change `target_name`, `feature_extractor.*`, `compute_feature_metrics`, or `use_gpu` — those gate model loading and must be identical across conditions. Run such variants separately.
- Each condition independently honors `_final_metrics_cache_valid` — if a condition's CSV/NPY already exist AND both `force_recompute.all=false` and `force_recompute.final_metrics=false`, the driver skips it and loads the cached outputs. Either flag set to `true` bypasses the cache.

Process-mode caveat: under `runtime.executor=process`, the parent's shared `EvalModels` is passed to `evaluate_predictions(..., models=...)` so the parent-side pre-warm is amortized, but each condition still spawns a fresh `ProcessPoolExecutor` whose workers re-load their own model copies. Use `executor=serial` to maximize the amortization benefit. The driver tiers its message based on the cache mode:

- `executor=process` + `require_complete_cache=true` + `n_conditions > 1` → mild **note**: workers re-init per condition but skip `prepare_segmentation_model` (returns None) and don't instantiate extractors, so the cost is just N pool spawns.
- `executor=process` + `require_complete_cache=false` + `n_conditions > 1` → loud **WARNING**: each condition's worker pool independently loads SuperModel + DINOv3 + DynaCLR + CELL-DINO. Total waste is ~30-90 s × `runtime.fov_workers` × `n_conditions`. Fix: switch to `executor=serial` so the parent's pre-loaded models are reused across conditions; reserve `process` for per-FOV parallelism within a *single* condition.

For an A40 / single-GPU interactive node where you'd run serial anyway, this is the default win.

Tests: `applications/dynacell/tests/test_evaluation_grouped.py` validates byte-equal parity against sequential per-condition runs on the same cache-only fixture used by `test_evaluation_pipeline_parallel_cpu.py`, plus rejection cases for empty `conditions` and forbidden model-loading-field overrides.

## Focus-aware 2D projection (`dynacell/evaluation/focus.py`)

The 3D→2D reduction in eval is **focus-aware** so the projection isn't dominated by
out-of-focus caps (A549 Z=48; the in-focus band is ~5 planes). Three knobs, all
default-off (default behavior unchanged):

- `feature_metrics.focus_slab.{enabled,halfwidth,channel_name}` — deep-feature crops
  (`build_crops`) + `per_cell_similarity` max-project over a `2*halfwidth+1` slab
  centered on the in-focus plane (h=2 → ±2 → 5 planes). CP regionprops stay 3D
  (untagged). Enabling it folds `+focusslab_h{h}_{ch}_{sig}` into each deep-feature
  `preprocess_version` so embedding caches auto-invalidate — `{sig}` is a short hash of
  the focus-compute params (`focus.{na_det,lambda_ill,pixel_size}`), so a focus-param
  change also invalidates (the same params are recorded in the `slice_selection=focus`
  instance-mask cache identity).
- `segmentation.slice_selection=focus` — 2D instance seg picks the in-focus plane
  (single slice, no slab) vs the old `frac=0.30`.
- `precompute-gt build.focus=true` — writes `focus_slice` zattrs (DynaCLR/qc schema)
  to a **writable** GT store.

**Plane source (precedence, in `resolve_focus_planes`):** (1) precomputed
`focus_slice` zattrs in the store (iPSC `.zarr` fast path); (2) the
`io.gt_cache_dir/focus_planes/<channel>/<pos>.json` cache; (3) compute from the
`focus.{channel_name,na_det,lambda_ill,pixel_size,device}` phase channel + persist to
that cache. **Compute-at-eval-time (3) is why focus works on the read-only published
A549 `.ozx`** — they carry no zattrs and `pack_ozx` is not bit-reproducible, so we do
NOT fork them; the plane is derived from the `Phase3D` already inside each store and
the published bytes stay identical to AWS Open Data v1. The estimator is
`waveorder.focus.focus_from_transverse_band` (NOT the `qc` app — keeps the dep graph
`applications/ → packages/`); the zattrs writer uses `viscy_utils.meta_utils`.

Heterogeneous T is real in the pooled A549 `.ozx` (positions differ in timepoint
count); the per-position writer/resolver handle it (per-position `shape[0]`).

**Re-eval campaign:** `tools/run_focus_campaign.slurm` + `tools/submit_focus_campaign.sh`
run it two-phase to produce-then-hit the GT cache: phase 1 = the 4 `*_ipsc_trained`
leaves (each touches its organelle's 3 A549 conditions + iPSC, so collectively they
warm every GT cache), phase 2 = the other 16 leaves (`afterany` phase 1, read-only
warm GT, fan out fully parallel, 2 evals/GPU, 12 h). Overrides: `glcm.enabled=true`,
`focus_slab.enabled=true halfwidth=2`, `slice_selection=focus`,
`force_recompute.final_metrics=true`.

## Predict submission modes

`tools/submit_benchmark_batch.py` (and the `tools/predict_batch.sh` wrapper) covers three submission shapes. They are mutually exclusive — pick by parallelism shape, not by familiarity:

| Mode | Flag | Squeue rows | Per-GPU concurrency | Cross-sbatch concurrency |
|---|---|---|---|---|
| Serial (default) | (none) | 1 | 1 | — |
| Array | `--array [--max-array-concurrency K]` | 1 array (N tasks) | 1 per task | K |
| Chunked | `--parallel P` (P > 1) | ceil(N/P) | P (bare-background `&`) | full queue |

Selection guide:

- **One small set, contiguous time, want minimal queue footprint** → serial. One srun per leaf in series; least queue overhead.
- **Many leaves on different GPUs, want SLURM to throttle concurrent allocations** → `--array --max-array-concurrency K`. Each task gets its own allocation.
- **Few leaves but predict is GPU-light** → `--parallel P`. One GPU runs P leaves in parallel (memory-confirmed 2-up on A40, 2–4 on H200/H100). Faster wall time per chunk than serial without using more total GPU-hours.
- **Leaves span mixed hardware profiles (e.g., some H200, some A40)** → `--array --allow-mixed-directives`. Buckets leaves and submits one array per directive bucket. Only mode that handles this. **NOT compatible with `--parallel`.**

Hardening to know about when you read the rendered sbatch:

- `--parallel > 1` scales `cpus_per_task` by the chunk size and pins `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` per backgrounded process so concurrent children don't oversubscribe by all reading `SLURM_CPUS_PER_TASK`. Per-leaf logs land at `{run_root}/slurm/${SLURM_JOB_ID}_<exp_id>.log`; the sbatch's own `%j.out` only sees the driver banner and any chunk-level failure summary.
- PIDs are captured and `wait $pid` is called per child. Bare `wait` (no args) returns only the LAST child's status and would silently mask earlier crashes as `COMPLETED` — the rendered bash propagates non-zero exit codes explicitly.
- Submission loop catches `sbatch` failures per script and reports queued-vs-skipped (matters for `--parallel > 1` and `--array --allow-mixed-directives` since both produce multiple sbatches per invocation). Single-failure no longer hides an opaque traceback.
- Soft warning at `cpus_per_task > 128`. Most cluster nodes top out around there; scaling `--parallel` past that often makes chunks pend forever.

For local foreground execution (no sbatch), `tools/predict_local.sh --parallel N` has its own backgrounding implementation on the current host's GPU. Confirmed safe 2-up on A40 (`gpu-e-2` interactive). Don't confuse it with `--parallel` on the sbatch helper — different invocation paths.
