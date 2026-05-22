# dynacell.evaluation

End-to-end evaluation pipeline for virtual staining predictions against fluorescence ground truth.

## Components

| Module | Purpose |
|---|---|
| `pipeline.py` | Hydra orchestrator. CLIs: `dynacell evaluate` (single-condition) and `dynacell evaluate-grouped` (one model load, N I/O conditions). |
| `metrics.py` | Pixel, mask, and feature metrics (CP regionprops + DINOv3 + DynaCLR + CELL-DINO), computed symmetrically for GT/predictions and combined pairwise. |
| `segmentation.py` | `aicssegmentation` workflows + SuperModel for `nucleus`/`membrane`. |
| `cache.py`, `pipeline_cache.py` | Artifact cache: on-disk layout, manifest, identity check, per-FOV load-or-compute wrappers, batched `precompute_deep_features`. |
| `model_loader.py` | Shared `load_eval_models(config, flags=...)` returning an `EvalModels` bundle. Used by both `evaluate` and `precompute-gt`. |
| `runtime.py` | BLAS/OMP thread caps, `ProcessPoolExecutor` worker initializer, `gpu_serialization_lock`, region timers. |
| `precompute_cli.py` | `dynacell precompute-gt` — fills the GT cache without running the eval loop. |
| `utils.py` | `DinoV3FeatureExtractor`, `DynaCLRFeatureExtractor`, `CellDinoFeatureExtractor`, plot helpers. |
| `_configs/*.yaml` | Hydra schemas: `eval.yaml`, `precompute.yaml`, `eval_grouped.yaml`. |

Other files (`io.py`, `torch_ssim.py`, `formatting.py`, `spectral_pcc/`) house readers, GPU SSIM, and bead/PSF diagnostics.

## Inputs

- `io.pred_path` — model predictions, HCS OME-Zarr (channel: `io.pred_channel_name`)
- `io.gt_path` — fluorescence ground truth (channel: `io.gt_channel_name`)
- `io.cell_segmentation_path` — *optional* precomputed cell segmentation HCS OME-Zarr. Required when `compute_feature_metrics=true` or when building CP/DINOv3/DynaCLR/CELL-DINO cache entries. Position layout must match GT/pred 1:1.
- `io.gt_cache_dir`, `io.pred_cache_dir` — *optional* artifact cache directories; must be distinct. See [Caches](#caches).

## Quick start

```bash
uv run dynacell evaluate \
  target=er_sec61b \
  predict_set=ipsc_confocal \
  io.pred_path=/hpc/.../fnet3d_sec61b.zarr \
  save.save_dir=/hpc/.../eval_fnet3d_sec61b
```

Add `compute_feature_metrics=true` to enable feature metrics. Smoke test on a subset of FOVs with `limit_positions=N`.

## Configuration

`dynacell evaluate` is a Hydra entrypoint — override any field with `key=value` (CLI overrides win over groups). Settings that travel with a (target, marker, dataset) combination live in named Hydra **config groups**:

| Group | Options | What it sets | Source |
|---|---|---|---|
| `target` | `er_sec61b`, `mito_tomm20`, `membrane`, `nucleus` | `target_name`, `benchmark.dataset_ref.target` | repo-checkout `_internal/shared/eval/target/` |
| `predict_set` | `ipsc_confocal` | `benchmark.dataset_ref.dataset` | in-package |
| `feature_extractor/dinov3` | `lvd1689m` | `feature_extractor.dinov3.pretrained_model_name` | in-package |
| `feature_extractor/dynaclr` | `default` | `feature_extractor.dynaclr.checkpoint` + 8-field encoder dict | repo-checkout `_internal/shared/eval/feature_extractor/dynaclr/` |
| `leaf` | `<org>/<model>/<train_set>/eval__<predict_set>` | Composes all of the above for one canonical run | repo-checkout `_internal/leaf/` (symlink tree) |

Select a group: `<group>=<option>` (no `+` — groups are declared `optional` in `eval.yaml`).

`io.*` and `pixel_metrics.spacing` resolve from the dataset manifest (`dynacell/data/manifests.py`) via a post-compose hook in `_ref_hook.py`. `pred_channel_name` is derived as `{target_channel}_prediction`.

`target_name` ∈ {`nucleus`, `membrane`, `nucleoli`, `lysosomes`, `er`, `mitochondria`} selects the segmentation workflow. The first four map 1:1 with a `target` group; `nucleoli`/`lysosomes` have no ready-made group — set `target_name=…` directly.

**Group sources**: in-package groups ship in the wheel (schema + path-free reference values). Repo-checkout groups under `configs/benchmarks/virtual_staining/_internal/` are discovered via two `hydra.searchpath` roots that `dynacell.__main__` injects on a repo checkout. Hydra only resolves `.yaml` for group lookup, so eval groups + leaves use `.yaml` (Lightning train/predict leaves stay `.yml`).

### Feature metrics

`feature_extractor/dinov3=lvd1689m` and `feature_extractor/dynaclr=default` auto-select on a repo checkout. CELL-DINO is opt-in: set `feature_extractor.celldino.weights_path=/path/to/celldino.ckpt`, or leave `null` to soft-skip.

Override a checkpoint: `feature_extractor.dynaclr.checkpoint=/hpc/.../other.ckpt`. Disable a backbone: `feature_extractor/dinov3=null`. Enable feature metrics: `compute_feature_metrics=true` (also needs `io.cell_segmentation_path` non-null).

### External users (`--config-dir`)

Wheel installs see only in-package groups. To evaluate your own predictions, point Hydra at your group files:

```yaml
# my_configs/target/mine.yaml
# @package _global_         # REQUIRED — writes into root, not under 'target.*'
target_name: er
io:
  gt_path: /data/mine/gt.zarr
  cell_segmentation_path: /data/mine/seg.zarr
  gt_channel_name: MyGroundTruthChannel
  pred_channel_name: MyPredictionChannel
```

```bash
dynacell evaluate --config-dir /abs/path/my_configs \
  target=mine predict_set=ipsc_confocal \
  io.pred_path=/path/to/preds.zarr save.save_dir=/path/to/out
```

Or skip groups and set `io.*` directly on the CLI. For manifest-based path resolution from a wheel-only install, author a dynacell manifest and set `DYNACELL_MANIFEST_ROOTS`.

**Footguns**: (1) missing `# @package _global_` makes contents land under `cfg.target.*` instead of root → `MissingMandatoryValue`. (2) Saving group as `.yml` instead of `.yaml` makes it undiscoverable → `MissingConfigException`.

### Benchmark eval leaves

Canonical leaves at `configs/benchmarks/virtual_staining/<org>/<model>/<train_set>/eval__<predict_set>.yaml` pin every group + path + save dir:

```bash
uv run dynacell evaluate leaf=er/celldiff/ipsc_confocal/eval__ipsc_confocal
```

Coverage: `(er, membrane, mito, nucleus) × (celldiff, unetvit3d)`. CLI overrides apply on top (e.g. `limit_positions=1` for smoke).

## Caches

Set `io.gt_cache_dir` to write/read GT-side artifacts. Set `io.pred_cache_dir` for prediction-side organelle masks + per-cell features. Sharing one root is rejected.

GT caches are reusable across model checkpoints for the same `(gt_path, gt_channel_name, cell_segmentation_path)`. Prediction caches are reusable for repeated evals of the same `(pred_path, pred_channel_name, cell_segmentation_path)`.

### Layout

```
{cache_dir}/
  manifest.yaml                          # built_at, params, positions per artifact
  organelle_masks/{target_name}.zarr     # HCS plate; channel target_seg or prediction_seg
  features/cp.zarr                       # arrays at {row}/{col}/{fov}/t{t}
  features/dinov3/{model_slug}.zarr      # one plate per DINOv3 model name
  features/dynaclr/{ckpt_sha12}.zarr     # one plate per (checkpoint, encoder_config)
  features/celldino/{weights_sha12}.zarr # one plate per CELL-DINO weights
```

Identity: `(cache_schema_version, plate_path, channel_name, cell_segmentation_path)`. Mismatch raises `StaleCacheError`. The DynaCLR `ckpt_sha256_12` is memoized to a `<ckpt>.sha256` sidecar; touch or replace the checkpoint and the hash recomputes.

### Priming with `precompute-gt`

```bash
uv run dynacell precompute-gt target=er_sec61b predict_set=ipsc_confocal
```

Ad-hoc without HPC groups:

```bash
uv run dynacell precompute-gt \
  target_name=er \
  io.gt_path=/hpc/.../SEC61B.zarr \
  io.cell_segmentation_path=/hpc/.../SEC61B_segmented_cleaned.zarr \
  io.gt_cache_dir=/hpc/.../cache/SEC61B \
  pixel_metrics.spacing=[0.29,0.108,0.108] \
  feature_extractor.dynaclr.checkpoint=/path/to/dynaclr.ckpt
```

Skip families: `build.cp=false build.dinov3=false build.dynaclr=false build.celldino=false build.masks=false`.

### Prediction cache

No `precompute-pred` CLI. The first `dynacell evaluate` run with `io.pred_cache_dir` set fills prediction masks/features; later runs hit it.

### Cache-only fast path: `io.require_complete_cache=true`

With both caches warm, this flag puts `dynacell evaluate` in cache-only mode:

- No model loads (~30-90 s cold-start skipped). The upfront `precompute_deep_features` pass is also skipped.
- Cache misses raise `StaleCacheError` immediately — fail-loud instead of opportunistic rebuild.
- Pixel + mask metrics still run on the original image volumes; deep features served from disk.

Use case: parallel sweeps, downstream metric iteration, crash recovery.

```bash
uv run dynacell evaluate ... \
  io.gt_cache_dir=/hpc/.../cache/SEC61B \
  io.pred_cache_dir=/hpc/.../pred_cache/sec61b_fcmae_v1 \
  io.require_complete_cache=true
```

**Edge case**: when `target_name ∈ {nucleus, membrane}` AND `io.pred_cache_dir=null`, SuperModel still loads — the per-T loop falls back to `segment(predict[t], seg_model=...)` for predictions. The skip-load fast path applies cleanly for organelle targets or when `io.pred_cache_dir` is also configured.

### Force recompute

```yaml
force_recompute:
  all: false              # invalidate everything below
  final_metrics: false    # CSV/NPY under save.save_dir → full re-run of the eval loop
  gt_masks / gt_cp / gt_dinov3 / gt_dynaclr / gt_celldino: false
  pred_masks / pred_cp / pred_dinov3 / pred_dynaclr / pred_celldino: false
```

Each per-artifact flag invalidates that family for its side only:
- `*_masks` — organelle masks for `target_name`
- `*_cp` — CP regionprops
- `*_dinov3` — features keyed on the active DINOv3 model name
- `*_dynaclr` — features keyed on `(ckpt_sha12, encoder_cfg_sha12)`
- `*_celldino` — features keyed on the CELL-DINO weights hash

Without `io.gt_cache_dir` / `io.pred_cache_dir`, only `force_recompute.{final_metrics, all}` matter.

### Invalidation

We deliberately do **not** fingerprint zarr contents. If you modify them in place, either bump `cache_schema_version` in `cache.py`, set the right `force_recompute.*`, or delete the affected cache dir.

## Scaling

### FOV-level parallelism: `runtime.*`

Default is sequential per FOV. Knobs (and their YAML defaults):

```yaml
runtime:
  fov_workers: 1                           # int | "auto"
  threads_per_worker: "auto"               # int | "auto" → cpu_count // fov_workers
  executor: "serial"                       # "serial" | "process"
  cuda_empty_cache_every_n_timepoints: 0   # 0 = off
  gc_collect_every_n_fovs: 0               # 0 = off
```

- `executor=serial` is bit-for-bit identical to pre-runtime behavior.
- `executor=process` uses a spawn-context `ProcessPoolExecutor` over FOVs. Each worker independently lazy-loads models under an `fcntl` GPU lock, so only one worker runs GPU work at a time. Best when (a) the per-FOV CPU phase (cache I/O, regionprops, crop construction) is your bottleneck, or (b) you're under `require_complete_cache=true` so workers do no GPU work.
- `fov_workers="auto"` resolves to 1 under `serial`; under `process` it clamps to `min(cpu_count // threads_per_worker, n_positions)`. A literal `fov_workers > 1` with `executor=serial` raises; a resolved `fov_workers=1` with `executor=process` auto-demotes to `serial` (skips ~5 s spawn cost).

**Env-var hooks**:
- `DYNACELL_THREADS_PER_WORKER=N` — set in SLURM script *before* `dynacell evaluate`; the only thread-cap layer that bites at C-extension load time.
- `DYNACELL_FORCE_PER_T_HYGIENE=1` — operator escape hatch; flips both per-T memory hygiene knobs ≥1 regardless of YAML.

Region timers are always on (~120 ms overhead on a 9 h eval). Output: `<save_dir>/eval_timing.csv` with rows `(pos_name, t, region, seconds)`.

### Grouped multi-condition eval: `dynacell evaluate-grouped`

Score one `(model, organelle)` across multiple I/O variants — typically the three A549 plates (`mock`/`denv`/`zikv`) — paying the model cold-start **once** total.

```yaml
# eval_grouped_a549_mantis_er.yaml
defaults: [eval_grouped, _self_]
target_name: er
compute_feature_metrics: true
feature_extractor: { ... }   # shared across all conditions
conditions:
  - { name: a549_mock, io: { ... }, save: { save_dir: /out/a549_mock } }
  - { name: a549_denv, io: { ... }, save: { save_dir: /out/a549_denv } }
  - { name: a549_zikv, io: { ... }, save: { save_dir: /out/a549_zikv } }
```

```bash
uv run dynacell evaluate-grouped -c eval_grouped_a549_mantis_er
```

Per-condition overlays may override `io.*`, `save.*`, `runtime.*`, `limit_positions`, `force_recompute.*`, and carry a `name` label. They MUST NOT touch `target_name`, `feature_extractor.*`, `compute_feature_metrics`, or `use_gpu` — those gate model loading. Each condition honors `force_recompute.{all, final_metrics}` independently — if CSV/NPY already exist and neither flag is set, the condition is skipped and cached outputs are loaded.

**Process-mode caveat**: under `runtime.executor=process` each condition spawns its own worker pool that re-loads models. Combined with `require_complete_cache=false` this multiplies cold-start across conditions; the driver emits a loud warning. For maximum amortization use `executor=serial`, or precompute caches and run with `require_complete_cache=true`.

## Outputs

Under `save.save_dir`:

```
pixel_metrics.csv / .npy        # per-FOV per-timepoint pixel metrics
mask_metrics.csv / .npy
feature_metrics.csv / .npy      # if compute_feature_metrics=true
segmentation_results.zarr       # HCS plate; channels [prediction_seg, target_seg]
pixel_metrics/*.png             # bar/violin plots per metric
mask_metrics/*.png
feature_metrics/*.png
eval_timing.csv                 # region timer log (always on)
```

## Installation

Heavy optional deps (`aicssegmentation`, `segmenter-model-zoo`, `cubic`, `transformers`, `dynaclr`):

```bash
uv pip install -e "applications/dynacell[eval]"
```

## HPC notes

### Shared Hugging Face hub cache

`dynacell evaluate` and `dynacell precompute-gt` default `HF_HUB_CACHE` to a team-shared directory on project storage when they detect a repo checkout, so gated HF models (DINOv3) download once per team. The default path is set in `dynacell/__main__.py` (`_DEFAULT_SHARED_HF_CACHE`); other sites override it via the `DYNACELL_SHARED_HF_CACHE` env var. Pre-set `HF_HUB_CACHE` and the auto-setter backs off.

We use `HF_HUB_CACHE` (not `HF_HOME`) because `HF_HOME` relocates the auth token file too, breaking per-user gated-repo ACLs. `HF_HUB_CACHE` only relocates weights/datasets; tokens stay per-user. First-time setup: one team member with gated-repo access (see [DINOv3 on HF](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)) runs any eval command to trigger the download; everyone else reuses the shared weights afterward — those reads don't hit HF and don't need a token.
