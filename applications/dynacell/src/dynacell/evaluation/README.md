# dynacell.evaluation

Scores virtual-staining predictions against fluorescence ground truth at three levels: pixel fidelity, segmentation, and single-cell phenotype features.

- [Pipeline](#pipeline)
- [Running evaluations](#running-evaluations)
- [Configuration](#configuration)
- [Segmentation](#segmentation)
- [Metrics](#metrics)
- [Caches](#caches)
- [Outputs](#outputs)
- [Modules](#modules)

## Pipeline

`dynacell evaluate` runs these steps for one prediction store:

1. **Check saved metrics.** If `save.save_dir` already holds metrics that are still valid (see [Reusing saved metrics](#reusing-saved-metrics)), load them and stop.
2. **Compose and validate the config.** Fields under `benchmark.dataset_ref` are resolved from the dataset manifest. The installed cubic version and the segmentation backend/target pairing are checked, and so is the CP reference when feature metrics are on.
3. **Load models**: the segmenter, plus the feature extractors when `compute_feature_metrics=true`.
4. **Open stores** (prediction, GT, cell segmentation, optional separate nuclei store) and reconcile positions. Every store must hold the same positions with the same number of timepoints.
5. **Precompute deep features** for both sides in batches, unless the cache already holds them.
6. **For each FOV and timepoint:**
   1. compute pixel metrics on the full volume;
   2. segment GT and prediction, then compute mask metrics;
   3. extract per-cell features and score each GT/prediction pair.
7. **Dataset-level feature metrics**: distribution distances (KID, FID, precision/recall, MIND), median cosine, and a real-vs-predicted linear probe, per feature space.
8. **Save** CSV/NPY tables, plots, single-cell embeddings, the segmentation plate and `metrics_provenance.json`.

`dynacell evaluate-grouped` loads the models once and runs steps 1, 2 and 4–8 for every entry in `conditions`. It then runs a cross-condition probe for each model group whose save directories include `__mock` plus `__denv` and/or `__zikv`. The probe classifies infected against mock cells and writes `cross_condition_probe.csv` into each infected condition's directory.

## Running evaluations

### Environment

From the repository root:

```bash
uv sync --package dynacell --extra eval --extra eval_gpu
uv run --package dynacell --extra eval --extra eval_gpu dynacell evaluate-grouped leaf=...
```

Every eval command and script runs in this uv-built `.venv`. The `eval` extra pins `cubic` (currently `v0.9.0a4`) and requires `cellpose>=4.2`, which provides Cellpose-DINO. `eval_gpu` adds the CUDA 13 CuPy/cuCIM stack, which the GPU segmentation and metric paths require. The cubic version is part of the numeric contract: an unsupported version fails the run. To use another environment, set uv's `UV_PROJECT_ENVIRONMENT`.

**Hugging Face cache.** On a repo checkout, every eval command sets `HF_HUB_CACHE` to a team-shared cache unless it is already set. Other sites override the location with `DYNACELL_SHARED_HF_CACHE`. Only `HF_HUB_CACHE` moves, so access tokens stay per user. The gated DINOv3 weights must be downloaded once by someone with access.

### Benchmark leaves

The benchmark is scored with **grouped leaves**: one model, many test conditions, one model load.

```bash
dynacell evaluate-grouped leaf=grouped/<bucket>/eval_grouped
```

- **Location:** `configs/benchmarks/virtual_staining/_internal/leaf/grouped/<bucket>/eval_grouped.yaml`.
- **Generated, mostly:** most buckets are written by `tools/generate_grouped_eval_configs.py`, `tools/generate_spotlight_v2_eval_configs.py` or `tools/generate_lite_benchmark_configs.py`; regenerate those rather than hand-editing them. The HEK, `celldiff_r2` and `fcmae3d_rescore` buckets are hand-written.
- **Suffixes** mark variants of a bucket, e.g. `__2d`, `__lite`, `__temporal`.
- **`conditions` entries** may override `io.*`, `save.*`, `runtime.*`, `limit_positions`, `force_recompute.*`, `benchmark.dataset_ref` and `name`. Changing any field that affects model loading or segmentation raises an error.
- **Always re-scored:** grouped leaves set `force_recompute.final_metrics: true`. To add a condition to a finished bucket without re-scoring the others, pass `force_recompute.final_metrics=false`.

To score only some conditions of a bucket, pass their names: `only_conditions=[<name>,...]`. An unknown name raises an error. The cross-condition probe still pairs each rescored condition with the unselected mock or infected conditions of its probe group whose cached metrics are current. Under `force_recompute` no cache counts as current, since the check cannot see recipe changes such as `feature_metrics.focus_slab`. A probe CSV that the run cannot rewrite is removed: one in a rescored infected condition that found no current mock, or one in an infected condition whose mock was rescored.

Use `leaf=`, not `-c` (`-c` is Hydra's display-only `--cfg`).

### SLURM

```bash
# from the repository root
sbatch applications/dynacell/tools/run_eval_direct.slurm grouped/<bucket>/eval_grouped [overrides...]
```

- **What it does:** runs `evaluate-grouped` for grouped leaves and `evaluate` otherwise, in the project `.venv`. It requests one GPU with at least 40 GB, 16 CPUs and 256 GB of memory.
- **One condition:** append `'only_conditions=[<name>]'` to the overrides.

### Common overrides

| Override | Effect |
|---|---|
| `limit_positions=N compute_feature_metrics=false` | Smoke test on the first N FOVs. Feature metrics need every GT position. |
| `only_conditions=[<name>,...]` | Grouped leaves: score only the named conditions. |
| `force_recompute.final_metrics=true` | Re-score even if the saved metrics are valid. |
| `io.require_complete_cache=true` | Cache-only run; see [Caches](#caches). |
| `runtime.executor=process runtime.fov_workers=auto` | Parallel FOVs; see [Parallelism](#parallelism). |
| `+io.exclude_fov_names=[...]` | Skip named FOVs. Requires `compute_feature_metrics=false`. |

### DynaCell-lite

A smaller test set for model iteration: A549 keeps 5 of 10 timepoints, and iPSC keeps 50 of 100 FOVs.

- **Paths:** data and outputs live under `paths.LITE_DATA_ROOT`, with the same layout as `paths.DATA_ROOT`.
- **Datasets and leaves:** the `*-lite` manifests, scored with `grouped/<bucket>__lite/eval_grouped`.
- **Metrics:** lite leaves turn off MicroSSIM, FID, precision/recall and MIND. CP (GLCM+) KID is still computed but left out of lite reports.

## Command-line interface

| Command | Purpose |
|---|---|
| `dynacell evaluate` | Evaluate one prediction store (`_configs/eval.yaml`). |
| `dynacell evaluate-grouped` | Evaluate N conditions with one model load (`_configs/eval_grouped.yaml`). |
| `dynacell precompute-gt` | Fill the GT cache without scoring (`_configs/precompute.yaml`). |
| `dynacell backfill-pixel-scalings` | Recompute `SSIM`/`NRMSE`/`PSNR` and their `SI_*` versions in saved pixel metrics; details below. |
| `dynacell report` | Build comparison tables and figures from saved metrics (`dynacell.reporting`). |

All commands are Hydra entry points; override any field with `key=value`.

`backfill-pixel-scalings` skips leaves that already have `SI_*` columns unless `+backfill.force=true`. It writes nothing for a leaf whose recomputed `PCC` no longer matches the saved value, and exits non-zero if any leaf fails that check.

## Configuration

Defaults live in `_configs/eval.yaml`. `eval_grouped.yaml` adds `conditions`, and `precompute.yaml` adds `build.*`.

### Inputs

| Key | Required | Description |
|---|---|---|
| `target_name` | yes | `nucleus`, `membrane`, `er`, `mitochondria`, `nucleoli` or `lysosomes`. Selects the segmentation workflow. |
| `io.pred_path`, `io.pred_channel_name` | yes | Prediction HCS OME-Zarr and channel. |
| `io.gt_path`, `io.gt_channel_name` | yes | Ground-truth HCS OME-Zarr and channel. |
| `pixel_metrics.spacing` | yes | Voxel size `[z, y, x]` in µm. |
| `save.save_dir` | yes | Output directory. |
| `io.cell_segmentation_path` | for feature metrics | Per-cell label plate; positions must match GT 1:1. Not derived from the evaluated masks. The A549 plates are built by `tools/build_cpdino_seg_cleaned.py`. |
| `feature_extractor.dynaclr.checkpoint` | for feature metrics | Set by the `dynaclr=default` group on a repo checkout. |
| `io.nuclei_gt_path` | membrane, when nuclei are stored separately | Store holding the GT nucleus channel. |
| `io.gt_cache_dir`, `io.pred_cache_dir` | no | Artifact caches; must be distinct. |
| `io.pred_is_source` | no | Score the source (label-free) channel as a no-model baseline. |
| `feature_metrics.cp.reference_path` | no | CP reference; `null` resolves to `DATA_ROOT/cp_reference/<target_name>.json`. |

- **Values from the dataset manifest:** setting `benchmark.dataset_ref.{dataset,target}` fills in `io.gt_path`, `io.gt_channel_name`, `io.cell_segmentation_path`, `io.gt_cache_dir` and `pixel_metrics.spacing` (`_ref_hook.py`).
  - `io.pred_channel_name` becomes `<target_channel>_prediction`.
  - Setting one of these explicitly to a value that disagrees with the manifest raises an error.
- **Your own datasets:** register their manifests with `DYNACELL_MANIFEST_ROOTS`.

### Config groups

| Group | Options | Sets | Source |
|---|---|---|---|
| `target` | `nucleus`, `membrane`, `er_sec61b`, `mito_tomm20` | `target_name`, `benchmark.dataset_ref.target` | repo checkout |
| `predict_set` | `ipsc_confocal`, `a549_mantis_<marker>_<condition>` | `benchmark.dataset_ref.dataset` | package |
| `feature_extractor/dinov3` | `lvd1689m` | DINOv3 model name | package |
| `feature_extractor/morphem` | `default` | MorphEm model and revision | package |
| `feature_extractor/dynaclr` | `default` | DynaCLR checkpoint and encoder | repo checkout |
| `feature_extractor/celldino` | `default` | CELL-DINO weights | repo checkout |
| `leaf` | see [Benchmark leaves](#benchmark-leaves) | a complete run | repo checkout |

- **Repo-checkout groups** live under `configs/benchmarks/virtual_staining/_internal/` and are added to the Hydra search path when running from a checkout.
- **Wheel installs** see only package groups. Supply the rest with `--config-dir`; each group file must start with `# @package _global_` and use the `.yaml` extension.
- **Nucleoli and lysosomes** have no `target` group; set `target_name` and `io.*` directly.

### Main switches

| Key | Default | Effect |
|---|---|---|
| `compute_feature_metrics` | `true` | Feature tier. Needs `io.cell_segmentation_path`. |
| `compute_microssim` | `true` | `MicroMS3IM` pixel metric. |
| `compute_instance_ap` | `false` | Instance segmentation and AP. On in the grouped nucleus/membrane leaves. |
| `segmentation.backend` | `supermodel` | Nucleus/membrane segmenter. `cpdino` in the grouped leaves. |
| `compute_cell_similarity` | `false` | Per-cell `PerCell_*` pixel similarity. |
| `pixel_metrics.foreground.enabled` | `false` | Foreground-weighted `FG_*` pixel metrics. |
| `feature_metrics.compute_{fid,prc,mind}` | `true` | FID, precision/recall/F1, MIND. |
| `feature_metrics.focus_slab.{enabled,halfwidth}` | `true`, `2` | Crop deep features from an in-focus slab of `2·halfwidth+1` planes. |
| `use_gpu` | `true` | GPU for metrics, segmentation and extractors. |

The cross-condition probe follows `compute_feature_metrics`. Disable it with `+cross_condition_probe.enabled=false`; the key is not in the schema, hence the `+`.

## Segmentation

GT and prediction are segmented by the same workflow. The table shows the grouped-leaf settings.

| Target | Segmenter | Geometry | Mask metrics |
|---|---|---|---|
| `nucleus` | Cellpose-DINO ViT-L (`cpdino`) | 2-D in-focus slab | semantic + instance |
| `membrane` | `cpdino` whole cells, GT nucleus footprint removed | 2-D in-focus slab | semantic + instance, on the cytoplasm |
| `er`, `mitochondria` | Allen Cell Structure Segmenter SEC61B / TOMM20 workflows, ported to cubic | 3-D | semantic |
| `nucleoli`, `lysosomes` | `aicssegmentation` NPM1 / LAMP1 workflows | 3-D | semantic |

- **Defaults:** `segmentation.backend: supermodel` with `compute_instance_ap: false` produces 3-D binary nucleus/membrane masks from segmenter-model-zoo, with no instance metrics.
- **In-focus slab:** with `segmentation.dimension: 2d` and `slice_selection: focus`, each FOV and timepoint uses the z-plane with the largest GT nuclear foreground area (`focus_anchor: nucleus_area`). That plane is max-projected over ±`focus_slab_halfwidth` (default 1) planes, and the same plane is used for GT, prediction and nucleus seeds. `dimension: 3d` segments the full volume instead.
- **cpdino:**
  - Input is the raw slice; cellpose applies its own percentile normalization (`normalize: true`), with no CLAHE or rescaling.
  - Instances smaller than `min_size: 15` pixels are dropped.
  - Runs on the GPU in bf16, so masks can differ slightly between GPU models.
  - Membrane also needs `segmentation.nuclei_channel_name`.
- **ER and mitochondria** require a `(Z, Y, X)` stack: the filament filter runs on each z-plane, and normalization, smoothing and size filtering run on the whole volume. The workflows run on NumPy or CuPy.
- **Other instance backends:**
  - `cellpose`: Cellpose-SAM, nucleus only.
  - `cellpose_watershed`: Cellpose-SAM nucleus seeds plus a watershed for whole cells, membrane only. Its settings are under `segmentation.watershed`.

## Metrics

### Pixel (`pixel_metrics.csv`)

Computed on each (FOV, timepoint) volume.

| Columns | Notes |
|---|---|
| `PCC` | Pearson correlation; invariant to affine intensity changes. |
| `SSIM`, `NRMSE`, `PSNR` | Each image min–max normalized independently. |
| `SI_SSIM`, `SI_NRMSE`, `SI_PSNR` | Prediction least-squares fitted to the target first (scale-invariant). |
| `Spectral_PCC` | Frequency-weighted PCC; `pixel_metrics.spectral_pcc`. |
| `XY_FSC_Resolution`, `Z_FSC_Resolution` (3-D) / `FRC_Resolution` (2-D) | Fourier shell/ring correlation resolution; `pixel_metrics.fsc`. |
| `MicroMS3IM` | MicroSSIM; `compute_microssim`. |
| `FG_PCC`, `FG_SI_SSIM`, `FG_SI_NRMSE`, `FG_SI_PSNR`, `FG_frac` | Weighted by a soft foreground map derived from the GT; `pixel_metrics.foreground.enabled`. |
| `PerCell_{PCC,SSIM}_{mean,median}` | Per-cell similarity; `compute_cell_similarity`. |

### Mask (`mask_metrics.csv`)

| Columns | When |
|---|---|
| `Dice`, `IoU`, `Precision`, `Recall`, `Accuracy`, `TP`, `FP`, `FN`, `TN` | always; from instance labels > 0 in instance mode |
| `AP_<iou>` (0.50–0.95), `mAP`, `instance_dice`, `instance_{TP,FP,FN}@0.50`, `n_gt`, `n_pred` | `compute_instance_ap=true` |

### Feature (`feature_metrics.csv`)

| Feature space | Input | When |
|---|---|---|
| `CP` (regionprops + GLCM texture) | 3-D cell volume | always |
| `DINOv3` | 2-D cell crop | always |
| `DynaCLR` | 2-D cell crop | always |
| `CellDINO` | 2-D cell crop | `feature_extractor.celldino.weights_path` set |
| `MorphEm` | 2-D cell crop | `feature_extractor.morphem.pretrained_model_name` set |

"Always" means whenever `compute_feature_metrics=true`. On a repo checkout the default groups enable all five.

Columns per feature space `<F>`:

| Columns | Level |
|---|---|
| `<F>_KID`, `<F>_KID_std`, `<F>_Median_Cosine_Similarity`, `<F>_FID` | per (FOV, timepoint) |
| `Dataset_<F>_KID[_std]`, `Dataset_<F>_FID`, `Dataset_<F>_{Precision,Recall,F1}[_std]`, `Dataset_<F>_MIND`, `Dataset_<F>_Median_Cosine_Similarity` | pooled over the test set |
| `Dataset_<F>_RealVsPred_AUROC[_std]`, `Dataset_<F>_Indistinguishability` | linear probe separating real from predicted cells |
| `CP_clip_frac`, `Dataset_CP_clip_frac` | fraction of predicted cells with a CP feature clipped |

- **2-D crops** are max projections of each cell over the in-focus slab (`feature_metrics.focus_slab`), or over the full stack when the slab is disabled.
- **CP reference:** CP features are scored in a fixed space defined by a reference file built from GT cells only. The file holds one feature mask and, for each test set, the GT mean and standard deviation, applied to both GT and prediction. Standardized values are clipped to ±20 before KID, FID and cosine.
  - **When it fails:** the run fails if its GT cells do not match the reference's GT cell count and per-feature mean/std (relative tolerance 1e-6).
  - **Rebuilding:** `tools/build_cp_reference.py --target <target>`; add `--verify` to check current caches against it.

## Caches

`io.gt_cache_dir` caches GT masks and features; it is model-independent and reused across checkpoints. `io.pred_cache_dir` caches the same artifacts for one prediction store. There is no `precompute-pred`: the first eval fills the prediction cache.

```
<cache_dir>/
  manifest.yaml
  organelle_masks/<target_name>[__<backend>].zarr
  instance_masks/<target_name>__<backend>.zarr
  features/cp.zarr
  features/{dinov3,morphem}/<model_slug>.zarr
  features/dynaclr/<checkpoint_sha12>.zarr
  features/celldino/<weights_sha12>.zarr
  focus_planes/<channel>/<position>.json        # GT cache only
```

### Priming the GT cache

```bash
dynacell precompute-gt target=er_sec61b predict_set=ipsc_confocal
```

`build.{masks,cp,dinov3,dynaclr,celldino,morphem}` default to `true`; `build.instances` and `build.focus` default to `false`.

### Invalidation

| Change | Behavior |
|---|---|
| Plate path, channel, segmentation path, or `cache_schema_version` | `StaleCacheError`; clear the cache directory. |
| Tracked parameter (spacing, patch size, extractor `PREPROCESS_VERSION`, CP recipe) | Warning; the affected artifact is recomputed. |
| Prediction store rewritten (re-predict) | Affected positions are recomputed, detected per position from the writer marker and chunk mtime. |
| Library versions (torch, transformers; cubic for cached artifacts) | Not detected; bump the extractor's `PREPROCESS_VERSION` in the same change. |

- **Cache-only runs:** under `io.require_complete_cache=true` or `limit_positions`, a tracked-parameter change raises instead of recomputing.
- **`io.require_complete_cache=true`:**
  - Requires `io.gt_cache_dir`.
  - Skips the precompute pass, and turns any cache miss into a `StaleCacheError`.
  - Feature extractors still load.
  - The segmenter load is skipped for ER, mitochondria, nucleoli and lysosomes. For nucleus/membrane it is skipped only when `io.pred_cache_dir` is set and `compute_instance_ap=false`.
- **`force_recompute`:**
  - `force_recompute.<side>_<artifact>` rebuilds a single family. `<side>` is `gt` or `pred`; `<artifact>` is `masks`, `instances`, `cp`, `dinov3`, `dynaclr`, `celldino` or `morphem`.
  - `force_recompute.final_metrics` re-scores.
  - `force_recompute.all` does everything.

### Reusing saved metrics

Saved metrics in `save.save_dir` are reused only when all of the following hold:

- `metrics_provenance.json` matches the current cubic version, CP space, prediction sources, foreground recipe and FID implementation.
- The saved tables contain every column the current config produces.
- The run is not partial: `limit_positions` or `io.exclude_fov_names` with feature metrics on always re-scores.

## Parallelism

| Key | Default | Description |
|---|---|---|
| `runtime.executor` | `serial` | `process` runs FOVs in a spawn-context process pool. Each worker loads its own models and shares the GPU under a file lock. |
| `runtime.fov_workers` | `1` | Under `process`, `auto` resolves to `min(cpu_count // T, n_positions)`, where `T` is `threads_per_worker` if it is an integer, else 4. A result of 1 falls back to `serial`. |
| `runtime.threads_per_worker` | `auto` | BLAS/OpenMP threads per worker. `auto` = `cpu_count // fov_workers`. |
| `runtime.cuda_empty_cache_every_n_timepoints`, `runtime.gc_collect_every_n_fovs` | `0` | Memory hygiene; `0` disables. |

- **Grouped evals:** use `serial`. A process pool reloads every model for each condition.
- **Parallelizing across buckets:** run independent buckets on separate GPUs.
- **Environment variables:** `DYNACELL_THREADS_PER_WORKER` sets the thread cap before C extensions load. `DYNACELL_FORCE_PER_T_HYGIENE=1` turns on both hygiene options.

## Outputs

Written to `save.save_dir`. Generated grouped leaves set it with `paths.eval_leaf(...)`:

```
DATA_ROOT/<organelle>/<model>/<train_set>/<test_set>[__<condition>]/[<component>/][instance_ap/]
```

Files:

```
pixel_metrics.csv, .npy
mask_metrics.csv, .npy
feature_metrics.csv, .npy            # compute_feature_metrics
{pixel,mask,feature}_metrics/*.png   # per-metric plots
segmentation_results.zarr            # channels prediction_seg, target_seg
embeddings/{gt,pred}_{cp,dinov3,dynaclr,celldino,morphem}_single_cell_embeddings.npz
cp_selected_feature_mask.json        # CP reference mask and clip fractions
metrics_provenance.json              # versions, CP space and prediction-source hashes
eval_timing.csv                      # per-region timings
cross_condition_probe.csv            # grouped runs, infected conditions only
```

## Modules

| Module | Purpose |
|---|---|
| `pipeline.py` | `evaluate` and `evaluate-grouped`: orchestration, per-FOV loop, saving. |
| `precompute_cli.py` | `precompute-gt`. |
| `model_loader.py` | Loads the segmenter and feature extractors. |
| `_ref_hook.py` | Fills `io.*` and `pixel_metrics.spacing` from `benchmark.dataset_ref`. |
| `segmentation.py` | Binary-mask workflows and segmenter selection. |
| `segmentation_cpdino.py` | Cellpose-DINO nucleus and whole-cell instances. |
| `segmentation_cellpose.py`, `segmentation_whole_cell.py` | Cellpose-SAM nuclei; seeded watershed for whole cells. |
| `focus.py` | In-focus plane and slab selection. |
| `metrics.py` | Pixel, mask, CP and per-cell metrics. |
| `instance_metrics.py` | Instance AP and instance Dice. |
| `feature_metrics.py` | KID, FID, precision/recall, MIND, cosine. |
| `linear_probe.py`, `cross_condition_probe.py` | Real-vs-predicted and infected-vs-mock probes. |
| `cp_reference.py`, `feature_select.py` | CP reference space and GT feature selection. |
| `utils.py` | Deep feature extractors and plotting. |
| `cache.py`, `pipeline_cache.py` | Cache layout, manifest, identity checks, load-or-compute. |
| `provenance.py` | cubic version check and `metrics_provenance.json`. |
| `runtime.py` | Thread caps, process pool, GPU lock, timers. |
| `paths.py` | Canonical paths for checkpoints, predictions, eval outputs and caches. |
| `pixel_scaling_backfill.py` | `backfill-pixel-scalings`. |
| `spectral_pcc/` | Spectral PCC diagnostics. |
| `_configs/` | Hydra schemas and package config groups. |

## See also

- [dynacell](../README.md)
- [Benchmark configs](../../../configs/benchmarks/virtual_staining/README.md)
- [CLAUDE.md](CLAUDE.md): cubic and GPU conventions for contributors
