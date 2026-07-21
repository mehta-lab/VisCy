# Inference DAG (Triplet path)

Embedding inference for a trained DynaCLR encoder using `TripletDataModule` —
the **zarr + tracking** path (no parquet). Use this when you want to run a
trained checkpoint directly over an OME-Zarr store and its `ultrack` tracking,
rather than the parquet-first `MultiExperimentDataModule` path
(see [evaluation.md](evaluation.md) for the parquet path).

The triplet path is the one that carries the patch-rescaling
(`reference_pixel_size`) and on-the-fly Z-reduction (`z_reduction`) options, so
a 3D zarr can feed a 2D model without materializing a separate MIP dataset.

## Prerequisites

- A trained checkpoint (`last.ckpt` or a selected epoch) for a
  `dynaclr.engine.ContrastiveModule`.
- The inference dataset as an OME-Zarr store with `normalization` metadata in
  the FOV `zattrs` (so `NormalizeSampled` has per-FOV stats), plus a tracking
  zarr/CSV directory with `track_id, t, y, x` columns.
- The model's training pixel size (µm/px) if the inference dataset was acquired
  at a different magnification — passed as `reference_pixel_size` to rescale
  each patch to the physical area the model was trained on.

## Step-by-step detail

```
dataset.zarr        (preprocessed: normalization in FOV zattrs)
tracking.zarr/CSV   (track_id, t, y, x per cell)
checkpoint.ckpt     (trained ContrastiveModule)
  │
  ├──► predict config (TripletDataModule + ContrastiveModule + EmbeddingWriter)
  ▼
viscy predict --config configs/prediction/predict_triplet.yml
  │  TripletDataModule(fit=False): samples ONE anchor patch per (cell, timepoint)
  │    - extracts z_range window, yx at initial_yx_patch_size
  │    - reference_pixel_size → extract larger patch, BatchedZoomd to final_yx
  │    - z_reduction → BatchedChannelWiseZReductiond collapses Z to 1 (2D model)
  │  ContrastiveModule.predict_step → backbone features (+ projections)
  │  EmbeddingWriter accumulates (features, index) and writes one combined store
  ▼
embeddings.zarr     (AnnData: .X = embedding_key array, mirrored to obsm["X_backbone"]
                     /["X_projections"]; obs = fov_name/track_id/t/...)
  │
  ▼
dynaclr split-embeddings --input embeddings.zarr --output-dir embeddings/
  │  groups rows by obs["experiment"], writes one zarr per experiment
  │  removes the combined store afterwards
  ▼
embeddings/{experiment}.zarr   (one per experiment, informatively named)
  │
  ▼
downstream eval  (reduce-dimensionality, linear classifiers, MMD, pseudotime …)
                 see evaluation.md / pseudotime.md
```

## Pipeline DAG (process dependency)

```
predict config + checkpoint + zarr + tracking
  │
  ▼
viscy predict  (GPU, minutes–hours by cell count)
  │
  ▼
split-embeddings  (CPU, ~1 min, I/O bound)
  │
  ▼
downstream eval  (CPU/GPU, per analysis)
```

## Key commands

| Step             | Command                                                                     | Input                                   | Output                              |
| ---------------- | --------------------------------------------------------------------------- | --------------------------------------- | ----------------------------------- |
| Predict          | `uv run viscy predict --config configs/prediction/predict_triplet.yml`      | predict config + ckpt + zarr + tracking | combined `embeddings.zarr`          |
| Predict (SLURM)  | `sbatch configs/prediction/predict_triplet.sh`                              | same                                    | combined `embeddings.zarr`          |
| Split embeddings | `dynaclr split-embeddings --input embeddings.zarr --output-dir embeddings/` | combined zarr with `obs["experiment"]`  | one `{experiment}.zarr` per dataset |

## What lives where

| Data                              | Location                                          | When written          |
| --------------------------------- | ------------------------------------------------- | --------------------- |
| Pixel data (TCZYX)                | dataset.zarr on VAST                               | data prep             |
| Cell tracks (track_id, t, y, x)   | tracking.zarr / CSV on VAST                        | data prep             |
| Normalization stats (per FOV)     | dataset.zarr FOV `zattrs["normalization"]`         | `viscy preprocess`    |
| Backbone embeddings               | `embeddings.zarr` → `.X` (+ `obsm["X_backbone"]`)  | `viscy predict`       |
| Cell index (fov_name/track_id/t)  | `embeddings.zarr` → `obs`                          | `viscy predict`       |
| Per-experiment embeddings         | `embeddings/{experiment}.zarr`                     | `split-embeddings`    |

## Predict config structure

A ready-to-edit sample lives at
[`configs/prediction/predict_triplet_2d_from_3d.yml`](../../configs/prediction/predict_triplet_2d_from_3d.yml)
(the 2D-from-3D case, with `z_reduction` + `reference_pixel_size`). The skeleton
below annotates the load-bearing fields:

```yaml
seed_everything: 42

trainer:
  accelerator: gpu
  devices: 1
  precision: 32-true
  inference_mode: true
  logger: false
  callbacks:
    - class_path: viscy_utils.callbacks.embedding_writer.EmbeddingWriter
      init_args:
        output_path: /path/to/embeddings/embeddings.zarr
        embedding_key: features        # "projections" for frozen-backbone MLP heads
        overwrite: true

model:
  class_path: dynaclr.engine.ContrastiveModule
  init_args:
    encoder:
      class_path: viscy_models.contrastive.ContrastiveEncoder
      init_args:
        backbone: convnext_tiny
        in_channels: 1
        in_stack_depth: 1              # 2D model — pair with z_reduction below
        # … must match the trained checkpoint's encoder args …

data:
  class_path: viscy_data.TripletDataModule
  init_args:
    data_path: /path/to/dataset.zarr
    tracks_path: /path/to/tracking.zarr
    source_channel: [Phase3D]
    z_range: [0, 16]                   # window collapsed by z_reduction
    final_yx_patch_size: [160, 160]
    reference_pixel_size: 0.1494       # rescale to the model's training pixel size (optional)
    z_reduction: mip                   # collapse z_range to 1 slice for a 2D model (optional)
    batch_size: 400
    num_workers: 0                     # REQUIRED for predict (see Notes)
    predict_cells: false               # true + include_fov_names/include_track_ids to subset
    normalizations:
      - class_path: viscy_transforms.NormalizeSampled
        init_args:
          keys: [Phase3D]
          subtrahend: mean
          divisor: std
    augmentations: []                  # MUST be empty for deterministic predict

ckpt_path: /path/to/checkpoint/last.ckpt
return_predictions: false              # writer persists to zarr; don't hold in memory
```

## Notes

- **`num_workers: 0` is required for the predict path.** `HCSDataModule`/
  `TripletDataModule` predict does not use `mmap_preload`, and >0 workers risks a
  zarr-fork deadlock. This matches the dynacell predict overlay.
- **`augmentations: []`** — predict must be deterministic. The datamodule still
  applies `normalizations` (and the `reference_pixel_size` rescale + `z_reduction`
  collapse) at predict time via `_no_augmentation_transform`; only random
  augmentations are dropped.
- **2D from 3D without a MIP dataset.** Set `z_reduction: mip` (or `center`) to
  collapse the extracted `z_range` window to a single slice. Label-free channels
  (resolved by name via `parse_channel_name`) take the center slice; all other
  channels are max-projected. Pair with `in_stack_depth: 1` on the encoder.
  Center the `z_range` on the focus plane to control which planes are collapsed.
- **Pixel-size rescaling.** When the inference dataset's pixel size differs from
  the model's training pixel size, set `reference_pixel_size` (µm/px) so a larger
  patch covering the same physical area is extracted and bilinearly resized to
  `final_yx_patch_size`. Leave unset for same-resolution datasets.
- **`embedding_key`.** Use `features` for the backbone output (most models) and
  `projections` for frozen-backbone MLP-head models, which writes
  `obsm["X_projections"]` instead.
- **`split-embeddings` requires `obs["experiment"]`** on the combined store. For a
  single-experiment predict run the split step is optional — the combined
  `embeddings.zarr` is already per-experiment.
- Downstream analyses (dimensionality reduction, linear classifiers, MMD,
  pseudotime) consume the per-experiment zarrs and are documented in
  [evaluation.md](evaluation.md) and [pseudotime.md](pseudotime.md).

## Triplet vs parquet (MultiExperimentDataModule)

| Aspect              | Triplet path (this doc)                          | Parquet path (evaluation.md)                       |
| ------------------- | ------------------------------------------------ | -------------------------------------------------- |
| Aspect              | Triplet path (this doc)                                           | Parquet path (evaluation.md)                       |
| ------------------- | ------------------------------------------------------------------ | -------------------------------------------------- |
| Data entry point    | `data_path` zarr + `tracks_path`                                   | `cell_index.parquet` (built + preprocessed)        |
| Setup cost          | reads tracking + zarr shape at init                                | reads parquet only at init                         |
| Focus / z window    | explicit `z_range` or per-FOV `z_extraction_window` from `focus_slice`; `z_reduction` collapses | per-FOV `z_extraction_window` from `focus_slice`   |
| Pixel rescaling     | `reference_pixel_size`                           | `reference_pixel_size_xy_um`                       |
| Best for            | ad-hoc predict over a single zarr + tracking     | large multi-experiment runs, reproducible recipes  |
