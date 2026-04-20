# dynacell.evaluation

End-to-end evaluation pipeline for virtual staining predictions against fluorescence ground truth.

## Components

| Module | Purpose |
|---|---|
| `pipeline.py` | Hydra-driven orchestrator. Loads prediction/GT OME-Zarr plates, computes per-FOV per-timepoint metrics, saves CSVs + NPYs + plots. CLI entrypoint: `dynacell evaluate`. |
| `metrics.py` | Pixel metrics (PCC, SSIM, NRMSE, PSNR, FSC resolution, spectral PCC, MicroMS3IM), mask metrics (Dice, IoU, precision, recall, accuracy, TP/FP/FN/TN), feature metrics split into `*_target_*` / `*_pred_*` / `*_pairwise` so GT-side work can be cached separately from predictions. |
| `segmentation.py` | Organelle-specific classical-CV segmentation via `aicssegmentation` workflows (`nucleus`, `membrane`, `nucleoli`, `lysosomes`, `er`, `mitochondria`). Used for mask metrics. |
| `cache.py` | GT artifact cache: on-disk layout, manifest I/O, read/write helpers, staleness check. Keyed by `(cache_schema_version, gt_path, gt_channel_name, cell_segmentation_path)`. |
| `pipeline_cache.py` | Per-FOV load-or-compute wrappers (`fov_gt_masks`, `fov_gt_cp_features`, `fov_gt_deep_features`). Honor `force_recompute.*` flags and the `io.require_complete_cache` contract. |
| `precompute_cli.py` | Hydra entrypoint for `dynacell precompute-gt`. Iterates GT positions and fills the cache; no eval loop. |
| `utils.py` | `DinoV3FeatureExtractor`, `DynaCLRFeatureExtractor`, pairwise feature-similarity helpers, `plot_metrics()` bar/violin plots. |
| `io.py` | OME-Zarr / tiff readers and writers, prediction preprocessing transforms. |
| `torch_ssim.py` | GPU-friendly PyTorch SSIM. |
| `formatting.py` | Metric table formatting helpers. |
| `spectral_pcc/` | Bandlimited spectral PCC diagnostics and bead simulations. |
| `_configs/eval.yaml` | Hydra config for `dynacell evaluate`, with `???` MISSING markers for dataset-specific fields. |
| `_configs/precompute.yaml` | Hydra config for `dynacell precompute-gt`; inherits eval, requires `io.gt_cache_dir`. |

## Inputs

- `io.pred_path` — model predictions, HCS OME-Zarr (channel: `io.pred_channel_name`)
- `io.gt_path` — fluorescence ground truth, HCS OME-Zarr (channel: `io.gt_channel_name`)
- `io.cell_segmentation_path` — *optional* precomputed cell segmentation, HCS OME-Zarr. Required only when `compute_feature_metrics=true` or when building CP/DINOv3/DynaCLR cache entries. Position layout must match GT/pred 1:1.
- `io.gt_cache_dir` — *optional* directory for the GT artifact cache. `null` (default) disables caching; set to a writable path to opt in. Required for `dynacell precompute-gt` and for `io.require_complete_cache=true`.

## Running an evaluation

`dynacell evaluate` is a Hydra entrypoint. Override any field on the CLI with `key=value`.

### Minimal example — pixel + mask metrics only

```bash
uv run dynacell evaluate \
  target_name=er \
  io.pred_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/predictions/fnet3d_sec61b.zarr \
  io.gt_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/SEC61B.zarr \
  io.cell_segmentation_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/SEC61B_segmented_cleaned.zarr \
  pixel_metrics.spacing=[0.29,0.108,0.108] \
  save.save_dir=/hpc/projects/virtual_staining/training/dynacell/ipsc/predictions/eval_fnet3d_sec61b
```

`target_name` ∈ {`nucleus`, `membrane`, `nucleoli`, `lysosomes`, `er`, `mitochondria`} — selects the `aicssegmentation` workflow.

### Smoke test on a subset

```bash
uv run dynacell evaluate ... limit_positions=10
```

### Enable feature metrics (DINOv3 + DynaCLR)

Feature metrics require all three `feature_extractor` fields to be set.
`feature_extractor.dynaclr.encoder` is a dict of kwargs for
`viscy_models.contrastive_encoder.ContrastiveEncoder` — inline on the CLI:

```bash
uv run dynacell evaluate ... \
  compute_feature_metrics=true \
  feature_extractor.dinov3.pretrained_model_name=facebook/dinov3-vitl16-pretrain-lvd1689m \
  feature_extractor.dynaclr.checkpoint=/path/to/dynaclr.ckpt \
  'feature_extractor.dynaclr.encoder={backbone: resnet50, in_channels: 1, in_stack_depth: 15, stem_kernel_size: [5,4,4], embedding_dim: 256, projection_dim: 32, drop_path_rate: 0.0}'
```

Omitting any of the three when `compute_feature_metrics=true` raises
`MissingMandatoryValue` at access time.

### Force recompute

The `force_recompute` block has one flag per cacheable artifact plus a shortcut:

| Flag | What it invalidates |
|---|---|
| `force_recompute.final_metrics` | Saved CSV/NPY under `save.save_dir` — forces a full re-run of the eval loop. |
| `force_recompute.gt_masks` | Cached target-side organelle masks for `target_name`. |
| `force_recompute.gt_cp` | Cached target-side CP regionprops features. |
| `force_recompute.gt_dinov3` | Cached target-side DINOv3 features for the current model name. |
| `force_recompute.gt_dynaclr` | Cached target-side DynaCLR features for the current `(ckpt_sha256, encoder_config_sha256)`. |
| `force_recompute.all` | All of the above. |

Examples:

```bash
# Regenerate only DINOv3 features, keep everything else cached:
uv run dynacell evaluate ... io.gt_cache_dir=/path/to/cache force_recompute.gt_dinov3=true

# Nuke everything and rebuild:
uv run dynacell evaluate ... io.gt_cache_dir=/path/to/cache force_recompute.all=true
```

Without `io.gt_cache_dir`, the cache layer is a no-op (same behavior as before the cache landed), and only `force_recompute.final_metrics` / `.all` have any effect — they control whether the saved CSVs are rebuilt.

## GT artifact cache

Set `io.gt_cache_dir` to write and read back GT-side artifacts so subsequent eval runs skip the expensive per-FOV segmentation and per-cell feature extraction. Typical speedup on SEC61B: ~2× on the second eval run, and scaling with the number of evaluations against the same GT.

### Layout

```
{gt_cache_dir}/
  manifest.yaml                          # built_at, params, positions per artifact
  organelle_masks/{target_name}.zarr     # HCS plate; channel target_seg (bool)
  features/cp.zarr                       # zarr group, arrays at {row}/{col}/{fov}/t{t}
  features/dinov3/{model_slug}.zarr      # one plate per model name
  features/dynaclr/{ckpt_sha12}.zarr     # one plate per (checkpoint, encoder_config)
```

Cache identity is the tuple `(cache_schema_version, gt_path, gt_channel_name, cell_segmentation_path)`. A mismatch raises `StaleCacheError` — no silent mis-serving when you change GT channel, swap segmentations, or bump the computation-logic version.

The DynaCLR checkpoint hash (`ckpt_sha256_12`) is memoized to a
`<ckpt>.sha256` sidecar next to the checkpoint and reused across eval
runs as long as the sidecar's mtime is ≥ the checkpoint's. Touch or
replace the checkpoint and the hash recomputes automatically.

### Priming the cache

```bash
uv run dynacell precompute-gt \
  target_name=er \
  io.gt_path=/hpc/.../SEC61B.zarr \
  io.cell_segmentation_path=/hpc/.../SEC61B_segmented_cleaned.zarr \
  io.gt_cache_dir=/hpc/.../cache/SEC61B \
  pixel_metrics.spacing=[0.29,0.108,0.108] \
  feature_extractor.dinov3.pretrained_model_name=facebook/dinov3-vitl16-pretrain-lvd1689m \
  feature_extractor.dynaclr.checkpoint=/path/to/dynaclr.ckpt \
  'feature_extractor.dynaclr.encoder={backbone: resnet50, in_channels: 1, ...}' \
  build.masks=true build.cp=true build.dinov3=true build.dynaclr=true
```

`build.*` toggles control which artifact families get built (all true by default). Skip families you don't need — for example, mask-only:

```bash
uv run dynacell precompute-gt ... build.masks=true build.cp=false build.dinov3=false build.dynaclr=false
```

### Parallel sweeps

After a full precompute, launch many `dynacell evaluate` jobs in parallel against the same cache with `io.require_complete_cache=true`. Missing entries now raise `StaleCacheError` instead of triggering concurrent writes (zarr `mode="a"` is not safe under concurrent write).

```bash
uv run dynacell evaluate ... io.gt_cache_dir=/hpc/.../cache/SEC61B io.require_complete_cache=true
```

### Cache invalidation

We deliberately do **not** fingerprint the GT or cell_segmentation zarr *contents*. If you modify them in place, either bump `cache_schema_version` in `cache.py`, set the appropriate `force_recompute.*` flag, or delete `{gt_cache_dir}/`.

## Outputs

Under `save.save_dir`:

```
pixel_metrics.csv / .npy        # per-FOV per-timepoint pixel metrics
mask_metrics.csv / .npy         # per-FOV per-timepoint mask metrics
feature_metrics.csv / .npy      # per-FOV per-timepoint feature metrics (if enabled)
segmentation_results.zarr       # HCS plate, channels: [prediction_seg, target_seg]
pixel_metrics/*.png             # bar/violin plots per metric
mask_metrics/*.png
feature_metrics/*.png
```

## Installation

Evaluation pulls heavy optional deps (`aicssegmentation`, `segmenter-model-zoo`, `cubic`, `microssim`, `transformers`, `dynaclr`). Install them with the `eval` extra:

```bash
uv pip install -e "applications/dynacell[eval]"
```
