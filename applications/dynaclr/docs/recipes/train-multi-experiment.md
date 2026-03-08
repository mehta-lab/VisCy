# Recipe: Train DynaCLR Across Multiple Experiments

## Goal

Train a single contrastive model across multiple time-lapse microscopy
experiments with different fluorescence reporters, imaging intervals, and
conditions. `MultiExperimentDataModule` handles positional channel alignment,
per-experiment tau conversion, condition balancing, and channel dropout.

## Prerequisites

- HCS OME-Zarr stores (one per experiment, registered)
- Per-FOV tracking CSVs (from ultrack or similar)
- Optionally: a pre-built cell index parquet (see `build-cell-index.md`)

## Step 1: Define your experiments

Create `experiments.yml` listing each experiment. Source channels are aligned
by **position**, not by name — position 0 is always phase, position 1 is
always fluorescence, regardless of the specific channel.

```yaml
experiments:
  - name: "2025_07_22_SEC61"
    data_path: "/hpc/projects/.../SEC61/registered.zarr"
    tracks_path: "/hpc/projects/.../SEC61/tracks"
    channel_names: ["Phase3D", "GFP", "Mito"]
    source_channel: ["Phase3D", "GFP"]       # position 0=phase, 1=fluor
    condition_wells:
      uninfected: ["A/1", "A/2", "A/3"]
      infected: ["B/1", "B/2", "B/3"]
    interval_minutes: 30.0
    start_hpi: 3.0

  - name: "2025_08_15_TOMM20"
    data_path: "/hpc/projects/.../TOMM20/registered.zarr"
    tracks_path: "/hpc/projects/.../TOMM20/tracks"
    channel_names: ["Phase3D", "RFP"]
    source_channel: ["Phase3D", "RFP"]       # same count, different name
    condition_wells:
      uninfected: ["A/1", "A/2"]
      infected: ["B/1", "B/2"]
      mock: ["C/1"]
    interval_minutes: 15.0
    start_hpi: 2.0
```

See `configs/training/experiments.yml` for an annotated example.

**Validation rules** (enforced by `ExperimentRegistry`):
- All experiments must have the **same number** of `source_channel` entries
- Each `source_channel` entry must exist in that experiment's `channel_names`
- `data_path` must exist and zarr channels must match `channel_names`
- `condition_wells` must be non-empty, `interval_minutes` must be positive

## Step 2: Write the training config

Create `my_training.yml`. See `configs/training/multi_experiment_fit.yml`
for a complete template.

```yaml
seed_everything: 42

trainer:
  accelerator: gpu
  strategy: ddp
  devices: 4
  num_nodes: 1
  precision: 32-true
  max_epochs: 100
  use_distributed_sampler: false  # FlexibleBatchSampler handles DDP

  logger:
    class_path: lightning.pytorch.loggers.TensorBoardLogger
    init_args:
      save_dir: /hpc/projects/.../logs
      version: my_run_v1

  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: loss/val
        save_top_k: 4
        save_last: true

model:
  class_path: dynaclr.engine.ContrastiveModule
  init_args:
    encoder:
      class_path: viscy_models.contrastive.ContrastiveEncoder
      init_args:
        backbone: convnext_tiny
        in_channels: 2          # must match source_channel count
        in_stack_depth: 30      # z_range[1] - z_range[0]
        stem_kernel_size: [5, 4, 4]
        stem_stride: [5, 4, 4]
        embedding_dim: 768
        projection_dim: 32
    loss_function:
      class_path: dynaclr.loss.NTXentHCL
      init_args:
        temperature: 0.07
        beta: 0.5
    lr: 0.00002

data:
  class_path: dynaclr.data.datamodule.MultiExperimentDataModule
  init_args:
    experiments_yaml: /hpc/projects/.../experiments.yml
    cell_index_path: null       # or /path/to/cell_index.parquet

    z_range: [15, 45]
    yx_patch_size: [384, 384]
    final_yx_patch_size: [160, 160]

    val_experiments: ["2025_08_15_TOMM20"]
    tau_range: [0.5, 2.0]
    tau_decay_rate: 2.0

    batch_size: 64
    num_workers: 12

    # Sampling strategy
    experiment_aware: true
    condition_balanced: true
    temporal_enrichment: true
    temporal_window_hours: 2.0
    temporal_global_fraction: 0.3
    channel_dropout_channels: [1]   # drop fluorescence
    channel_dropout_prob: 0.5

    # Transforms use generic ch_0/ch_1 keys (positional)
    normalizations:
      - class_path: viscy_transforms.NormalizeSampled
        init_args:
          keys: [ch_0]
          level: fov_statistics
          subtrahend: mean
          divisor: std
      - class_path: viscy_transforms.ScaleIntensityRangePercentilesd
        init_args:
          keys: [ch_1]
          lower: 50
          upper: 99
          b_min: 0.0
          b_max: 1.0
    augmentations:
      - class_path: viscy_transforms.RandAffined
        init_args:
          keys: [ch_0, ch_1]
          prob: 0.8
          scale_range: [0, 0.2, 0.2]
          rotate_range: [3.14, 0.0, 0.0]
          shear_range: [0.0, 0.01, 0.01]
          padding_mode: zeros
```

## Step 3: Sanity check with fast_dev_run

Before committing to a full training run, validate the pipeline:

```bash
viscy fit -c my_training.yml --trainer.fast_dev_run=true
```

This runs 1 train + 1 val batch and catches config errors, missing paths,
and shape mismatches.

## Step 4: Launch training

```bash
viscy fit -c my_training.yml
```

Or via SLURM (see `slurm-training.md`):

```bash
sbatch fit_slurm.sh
```

## Key parameters explained

| Parameter | What it does |
|-----------|-------------|
| `experiment_aware` | Each batch comes from a single experiment (prevents mixing channel semantics) |
| `condition_balanced` | Balances infected/uninfected/mock within each batch |
| `temporal_enrichment` | Over-samples cells near a focal HPI window |
| `channel_dropout_prob` | Probability of zeroing the fluorescence channel, encouraging label-free learning |
| `tau_range` | Hours window for temporal positive sampling (converted to frames per experiment) |
| `tau_decay_rate` | Exponential decay — favors shorter temporal offsets |
| `val_experiments` | Names of experiments held out for validation |
| `cell_index_path` | Pre-built parquet for fast startup (see `build-cell-index.md`) |

## Tips

- **Start with `fast_dev_run`** to validate the full pipeline before long runs.
- **Channel dropout** is critical for cross-modal distillation — it forces the
  model to learn from phase contrast alone.
- **`val_experiments`** holds out entire experiments, not random cells.
  This tests generalization to unseen reporters/conditions.
- **Transforms use `ch_0`/`ch_1`** (not channel names) because different
  experiments have different channel names but the same positional semantics.
