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


### Channel Naming in Transforms

Transforms reference channels by their **source label** from the collection YAML (`source_channels[].label`), not by zarr channel names or generic `ch_N` indices.

- **Bag-of-channels mode** (`bag_of_channels: true`): one channel per sample, key is always `"channel"`
- **Multi-channel mode**: keys are the source labels, e.g. `"labelfree"`, `"reporter"`

In multi-channel mode, use `allow_missing_keys: true` if a transform should only apply to a subset of channels.

### Normalization Metadata (`norm_meta`)

- `norm_meta` is read per-FOV from zarr zattrs and remapped from zarr channel names → source labels in `_slice_patch`
- `timepoint_statistics` is pre-resolved to the sample's timepoint `t` in the dataset — `NormalizeSampled` does not need to look up timepoints at transform time
- `_collate_norm_meta` stacks per-sample scalar stats into `(B,)` tensors so normalization is correct when a batch mixes samples from different FOVs

### Multi-Experiment Sampling vs. Old `ConcatDataModule`

The old approach combined multiple `TripletDataModule` instances with `ConcatDataModule`, which gave no control over cross-experiment sampling balance. `MultiExperimentDataModule` uses `FlexibleBatchSampler` with explicit axes:

- `experiment_aware` — ensures each batch has representation from multiple experiments
- `stratify_by` — balances by condition, organelle, or other metadata columns
- `temporal_enrichment` — oversamples cells near biological events

All experiments share one `MultiExperimentTripletDataset` instance and one tensorstore context — no concat overhead.

### Collection YAML

- `source_channels[].label` defines the canonical channel names used throughout the pipeline
- `source_channels[].per_experiment` maps labels to actual zarr channel names per experiment (different experiments can have different zarr names for the same biological channel)
- The `ExperimentRegistry` computes `channel_maps` and `norm_meta_key_maps` once at setup time for O(1) lookup during data loading
