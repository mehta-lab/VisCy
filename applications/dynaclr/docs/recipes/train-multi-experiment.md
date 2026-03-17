# Recipe: Train DynaCLR Across Multiple Experiments

## Goal

Train a single contrastive model across multiple time-lapse microscopy
experiments with different fluorescence reporters, imaging intervals, and
conditions. `MultiExperimentDataModule` handles cross-experiment channel
alignment, per-experiment tau conversion, condition balancing, and
bag-of-channels training.

## Prerequisites

- HCS OME-Zarr stores (one per experiment, registered)
- Per-FOV tracking CSVs (from ultrack or similar)

---

## Step 1: Write the collection YAML

The collection YAML defines which experiments to train on and how channel names
map across experiments. See `configs/collections/` for examples.

```yaml
# my_collection.yml
name: my_training_collection
description: "Multi-experiment bag-of-channels training"

provenance:
  created_at: "2026-01-01"
  created_by: your.name

source_channels:
  - label: phase          # canonical label used by transforms
    per_experiment:
      exp_alpha: Phase3D  # zarr channel name for this experiment
      exp_beta: Phase3D
      exp_gamma: Phase3D
  - label: gfp
    per_experiment:
      exp_alpha: raw GFP EX488 EM525-45
      exp_beta: GFP EX488 EM525-45
      # exp_gamma omitted — phase-only experiment, no fluorescence channel

experiments:
  - name: exp_alpha
    data_path: /hpc/projects/.../exp_alpha.zarr
    tracks_path: /hpc/projects/.../exp_alpha/tracking.zarr
    channel_names:
      - Phase3D
      - raw GFP EX488 EM525-45
    condition_wells:
      uninfected: [A/1, A/2]
      infected: [B/1, B/2]
    interval_minutes: 30.0
    start_hpi: 4.0
    marker: SEC61B
    organelle: endoplasmic_reticulum
    date: "2025-01-01"
    moi: 5.0
    exclude_fovs: []
  - name: exp_gamma
    data_path: /hpc/projects/.../exp_gamma.zarr
    tracks_path: /hpc/projects/.../exp_gamma/tracking.zarr
    channel_names:
      - Phase3D        # phase only — no fluorescence channel
    condition_wells:
      uninfected: [A/1]
      infected: [B/1]
    interval_minutes: 20.0
    start_hpi: 0.0
```

**Rules enforced at startup:**
- Each `per_experiment` entry must name a channel that exists in that experiment's `channel_names`
- `data_path` must exist and zarr channel names must match `channel_names`
- Experiments may be omitted from a source channel's `per_experiment` — not every experiment needs every channel (e.g. a phase-only experiment can be mixed with GFP experiments in bag-of-channels mode)

---

## Step 2: Build the cell index parquet

Building the index once saves minutes on every training restart. It opens every
zarr store, reads every tracking CSV, and stores the result as a parquet.

```bash
dynaclr build-cell-index my_collection.yml cell_index.parquet
```

Check it loaded correctly:

```python
import pandas as pd
df = pd.read_parquet("cell_index.parquet")
print(df["experiment"].value_counts())
print(df.shape)
```

**Rebuild whenever:** you add experiments, re-track, or change condition wells.

---

## Step 3: Write the training config

Copy `configs/training/multi_experiment_fit.yml` as your starting point.
Key things to get right:

### Bag-of-channels mode (`in_channels: 1`)

Each sample randomly picks one source channel. The encoder sees one channel at a
time, learning representations that generalize across modalities.

```yaml
model:
  init_args:
    encoder:
      init_args:
        in_channels: 1          # bag-of-channels: one channel per sample
        in_stack_depth: 30      # must match z_window
```

```yaml
data:
  init_args:
    bag_of_channels: true
    z_window: 30
    yx_patch_size: [288, 288]       # extraction size (bigger than final)
    final_yx_patch_size: [192, 192] # final size after crop
    cell_index_path: /path/to/cell_index.parquet  # built in Step 2
    collection_path: /path/to/my_collection.yml
    val_experiments: null           # null = FOV-level split via split_ratio
    split_ratio: 0.8
    # num_workers_index: 4          # parallel index build; omit when cell_index_path is set
```

### Multi-channel mode (`in_channels: 2`)

All source channels are loaded together. Use `channel_dropout_prob` to randomly
drop the fluorescence channel and encourage label-free learning.

```yaml
model:
  init_args:
    encoder:
      init_args:
        in_channels: 2
```

```yaml
data:
  init_args:
    bag_of_channels: false
    channel_dropout_channels: [1]   # index of fluorescence channel
    channel_dropout_prob: 0.5
```

### Transforms — always use `Batched*` variants

Transforms run on GPU in `on_after_batch_transfer` on `(B, C, Z, Y, X)` tensors.
Always use the `Batched*` transforms — standard MONAI dict transforms are
single-sample only and will fail on batched input.

Transform keys use the **source channel labels** from the collection YAML
(`phase`, `gfp`, etc.), not zarr channel names or `ch_N` indices. In
bag-of-channels mode the key is always `channel`.

```yaml
    normalizations:
      - class_path: viscy_transforms.NormalizeSampled
        init_args:
          keys: [channel]           # bag-of-channels
          level: fov_statistics     # or timepoint_statistics
          subtrahend: mean
          divisor: std

    augmentations:
      # Affine: rotate in Z (full 360°), no Y/X rotation, mild XY shear
      - class_path: viscy_transforms.BatchedRandAffined
        init_args:
          keys: [channel]
          prob: 0.8
          rotate_range: [3.14, 0.0, 0.0]
          scale_range: [[0.8, 1.2], [0.8, 1.2], [0.8, 1.2]]
          shear_range: [0.05, 0.05, 0.0, 0.05, 0.0, 0.05]  # XY only, no Z shear

      # Random spatial crop: adds invariance to volume stabilization
      - class_path: viscy_transforms.BatchedRandSpatialCropd
        init_args:
          keys: [channel]
          roi_size: [35, 240, 240]  # slightly larger than final, then center-cropped

      # XY flips only (not Z — cell polarity is meaningful)
      - class_path: viscy_transforms.BatchedRandFlipd
        init_args:
          keys: [channel]
          prob: 0.5
          spatial_axes: [1, 2]

      - class_path: viscy_transforms.BatchedRandAdjustContrastd
        init_args:
          keys: [channel]
          prob: 0.5
          gamma: [0.6, 1.6]
      - class_path: viscy_transforms.BatchedRandScaleIntensityd
        init_args:
          keys: [channel]
          prob: 0.5
          factors: 0.5
      - class_path: viscy_transforms.BatchedRandGaussianSmoothd
        init_args:
          keys: [channel]
          prob: 0.5
          sigma_x: [0.25, 0.50]
          sigma_y: [0.25, 0.50]
          sigma_z: [0.0, 0.0]     # no Z blur
      - class_path: viscy_transforms.BatchedRandGaussianNoised
        init_args:
          keys: [channel]
          prob: 0.5
          mean: 0.0
          std: 0.1
```

**Augmentation design notes:**
- `BatchedRandAffined` uses Kornia's `RandomAffine3D` — applies independent random transforms per sample in the batch
- `shear_range` takes 6 values (Kornia's XY plane pairs): `[xy, xz, yx, yz, zx, zy]` — set Z-coupled shears to 0 for microscopy
- `rotate_range` is in radians, ZYX order — full rotation in Z (`3.14`), none in Y/X
- The random crop + center crop sequence (in `augmentations` + `final_yx_patch_size`) makes the model invariant to small XYZ translations from volume stabilization

---

## Step 4: Sanity check with fast_dev_run

Always validate the pipeline before launching a full run:

```bash
viscy fit -c my_training.yml --trainer.fast_dev_run=true
```

This runs 1 train + 1 val batch and catches: config errors, missing paths,
shape mismatches, transform failures.

---

## Step 5: Launch training

**Local (single GPU):**
```bash
viscy fit -c my_training.yml
```

**SLURM (multi-GPU):**
```bash
sbatch fit_slurm.sh
```

See `slurm-training.md` for the job script template. Make sure to set
`export PYTHONNOUSERSITE=1` in the SLURM script to prevent `~/.local/`
packages from overriding the conda/uv environment.

---

## Key parameters

| Parameter | What it does |
|-----------|-------------|
| `bag_of_channels` | Randomly select one source channel per sample — model learns all channels |
| `experiment_aware` | Each batch comes from one experiment — prevents mixing channel semantics |
| `stratify_by` | Columns to balance within batches, e.g. `[condition, organelle]` |
| `temporal_enrichment` | Over-sample cells near a focal HPI window |
| `channel_dropout_prob` | Probability of zeroing fluorescence — forces label-free learning |
| `tau_range` | Hours window for temporal positive sampling |
| `val_experiments` | Experiment names held out for validation; `null` uses FOV-level split |
| `cell_index_path` | Pre-built parquet for fast startup — skips zarr/CSV traversal |
| `split_ratio` | Fraction of FOVs for training when `val_experiments` is null |
| `num_workers_index` | Parallel processes for building the cell index at startup (default `1`). Set to number of experiments for maximum speedup. Ignored when `cell_index_path` is provided. |
