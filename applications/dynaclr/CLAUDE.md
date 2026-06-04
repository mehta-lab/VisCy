# DynaCLR — Design Principles for Claude Code Sessions

## Data Pipeline Architecture

### Why `__getitems__` + `collate_fn=lambda x:x` + `on_after_batch_transfer`

This three-part pattern is intentional for performance:

1. **`__getitems__`** — dataset returns an already-batched dict by reading multiple patches in one tensorstore I/O call (`ts.stack(...).read().result()`). Much faster than per-sample `__getitem__` + default collation.
2. **`collate_fn=lambda x:x`** — skips PyTorch's default collation since the dataset already returns `(B, C, Z, Y, X)` tensors.
3. **`on_after_batch_transfer`** — runs normalization and augmentation on GPU after CPU→GPU transfer, keeping CPU workers free for I/O.

Never move transforms back to the CPU workers or use per-sample iteration in `on_after_batch_transfer` — this defeats the purpose.

### Batched Transforms — Always Use `Batched*` Variants

All augmentations must use the GPU-native `Batched*` transforms from `viscy_transforms`, not the standard MONAI wrappers. The standard MONAI dict transforms (e.g., `RandAffined`) are designed for single-sample `(C, Z, Y, X)` input and break on batched `(B, C, Z, Y, X)` tensors.

Instead use our defined `Batched*` versions in `viscy-transforms`.

### Flat Parquet Schema

The cell index parquet has **one row per (cell, timepoint, channel)**. Each row carries:
- `channel_name` — zarr channel name (e.g., `"Phase3D"`, `"raw GFP EX488 EM525-45"`)
- `marker` — protein marker (e.g., `"Phase3D"`, `"TOMM20"`, `"SEC61B"`)
- `perturbation` — perturbation label (e.g., `"uninfected"`, `"ZIKV"`)

The dataset resolves zarr channel indices directly from `channel_name` via `exp.channel_names.index(name)`.

### Channel Naming in Transforms

- **Bag-of-channels mode** (`channels_per_sample: 1`): one channel per sample, key is `"channel_0"`
- **All-channels mode** (`channels_per_sample: null`): keys are the marker labels from the collection
- **Fixed mode** (`channels_per_sample: ["Phase3D", "GFP"]`): keys are the specified zarr channel names

In multi-channel mode, use `allow_missing_keys: true` if a transform should only apply to a subset of channels.

### Normalization Metadata (`norm_meta`)

- `norm_meta` is read per-FOV from zarr zattrs and looked up by zarr channel name directly in `_slice_patch`
- `timepoint_statistics` is pre-resolved to the sample's timepoint `t` in the dataset — `NormalizeSampled` does not need to look up timepoints at transform time
- `_collate_norm_meta` stacks per-sample scalar stats into `(B,)` tensors so normalization is correct when a batch mixes samples from different FOVs

### Multi-Experiment Sampling vs. Old `ConcatDataModule`

The old approach combined multiple `TripletDataModule` instances with `ConcatDataModule`, which gave no control over cross-experiment sampling balance. `MultiExperimentDataModule` uses `FlexibleBatchSampler` with explicit axes:

- `batch_group_by` — groups batches by column(s) (e.g., `["marker"]` for bag-of-channels, `["experiment"]`)
- `stratify_by` — balances by perturbation, organelle, or other metadata columns
- `temporal_enrichment` — oversamples cells near biological events

All experiments share one `MultiExperimentTripletDataset` instance and one tensorstore context — no concat overhead.

### Collection YAML

The YAML is the complete reproducible recipe for building a flat parquet. Channels are defined per-experiment with `name` + `marker`:

```yaml
experiments:
  - name: "2025_07_22_SEC61"
    channels:
      - name: "Phase3D"
        marker: "Phase3D"
      - name: "raw GFP EX488 EM525-45"
        marker: "SEC61B"
    perturbation_wells:
      uninfected: ["C/1"]
      ZIKV: ["C/2"]
```

No `SourceChannel` indirection. The builder reads `exp.channels` directly and explodes each cell observation into one row per channel.
