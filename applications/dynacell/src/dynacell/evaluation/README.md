# dynacell.evaluation

End-to-end evaluation pipeline for virtual staining predictions against fluorescence ground truth.

## Components

| Module | Purpose |
|---|---|
| `pipeline.py` | Hydra-driven orchestrator. Loads prediction/GT OME-Zarr plates, computes per-FOV per-timepoint metrics, saves CSVs + NPYs + plots. CLI entrypoint: `dynacell evaluate`. |
| `metrics.py` | Pixel metrics (PCC, SSIM, NRMSE, PSNR, FSC resolution, spectral PCC, MicroMS3IM), mask metrics (Dice, IoU, precision, recall, accuracy, TP/FP/FN/TN), feature metrics (Frechet distance, polynomial MMD on DINOv3 / DynaCLR / CellProfiler embeddings). |
| `segmentation.py` | Organelle-specific classical-CV segmentation via `aicssegmentation` workflows (`nucleus`, `membrane`, `nucleoli`, `lysosomes`, `er`, `mitochondria`). Used for mask metrics. |
| `utils.py` | `DinoV3FeatureExtractor`, `DynaCLRFeatureExtractor`, pairwise feature-similarity helpers, `plot_metrics()` bar/violin plots. |
| `io.py` | OME-Zarr / tiff readers and writers, prediction preprocessing transforms. |
| `torch_ssim.py` | GPU-friendly PyTorch SSIM. |
| `formatting.py` | Metric table formatting helpers. |
| `spectral_pcc/` | Bandlimited spectral PCC diagnostics and bead simulations. |
| `_configs/eval.yaml` | Hydra config with `???` MISSING markers for dataset-specific fields. |

## Inputs

Three HCS OME-Zarr plates (position layouts must match 1:1):

- `io.pred_path` — model predictions (channel: `io.pred_channel_name`)
- `io.gt_path` — fluorescence ground truth (channel: `io.gt_channel_name`)
- `io.cell_segmentation_path` — precomputed cell segmentation (consumed by feature metrics to crop per-cell patches)

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

Feature metrics require additional config:

```bash
uv run dynacell evaluate ... \
  compute_feature_metrics=true \
  feature_extractor.dinov3.pretrained_model_name=facebook/dinov3-vitl16-pretrain-lvd1689m \
  feature_extractor.dynaclr.checkpoint=/path/to/dynaclr.ckpt \
  +feature_extractor.dynaclr.encoder=@configs/recipes/models/dynaclr_encoder.yml
```

### Force recompute

By default, if `pixel_metrics.npy`, `mask_metrics.npy`, and `feature_metrics.npy` all exist under `save.save_dir`, they are loaded from disk and plots are regenerated. Force a full recompute of the saved CSVs:

```bash
uv run dynacell evaluate ... force_recompute.final_metrics=true
```

Per-artifact flags (`gt_masks`, `gt_cp`, `gt_dinov3`, `gt_dynaclr`) control the GT cache wired up in later commits. `force_recompute.all=true` invalidates everything.

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
