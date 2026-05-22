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
