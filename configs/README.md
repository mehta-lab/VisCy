# VisCy Configuration Guide

## Overview

This directory contains Hydra configuration files following the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) pattern for production-ready ML workflows.

## Configuration Structure

```
configs/
├── train.yaml              # Main configuration file
├── model/                  # Model architectures
│   ├── vsunet.yaml        # VSUNet for image translation
│   └── contrastive.yaml   # Contrastive learning (DynaCLR)
├── data/                   # Data modules
│   ├── hcs.yaml           # HCS (High Content Screening) data
│   └── triplet-classical.yaml       # Triplet sampling for contrastive learning
├── trainer/                # PyTorch Lightning Trainer configs
├── callbacks/              # Training callbacks
├── logger/                 # Experiment loggers
├── augmentation/           # Data augmentation transforms
├── normalization/          # Normalization methods
├── paths/                  # Project path configuration
│   └── default.yaml
├── extras/                 # Debug and development flags
│   └── default.yaml
├── debug/                  # Quick testing configuration
│   └── default.yaml
└── hydra/                  # Hydra output configuration
    └── default.yaml
```

## Quick Start

### Image Translation (VSUNet)

```bash
python -m viscy.train \
  model=vsunet \
  data=hcs \
  augmentation=none \
  normalization=none \
  data.data_path=/path/to/data.zarr \
  data.source_channel=Phase3D \
  data.target_channel=[Nuclei,Membrane]
```

### Contrastive Learning (DynaCLR)

```bash
python -m viscy.train \
  model=contrastive \
  data=triplet \
  augmentation=none \
  normalization=none \
  data.data_path=/path/to/data.zarr \
  data.tracks_path=/path/to/tracks \
  data.source_channel=[Phase3D] \
  data.z_range=[10,25]
```

### Debug Mode

Fast testing with minimal epochs and CPU:

```bash
python -m viscy.train \
  debug=default \
  model=vsunet \
  data=hcs \
  augmentation=none \
  normalization=none \
  data.data_path=/path/to/data.zarr
```

## Configuration Groups

### Required Groups

These must be specified for every run:
- `model`: Model architecture (vsunet, contrastive)
- `data`: Data module (hcs, triplet)
- `augmentation`: Data augmentation (none, or custom)
- `normalization`: Normalization method (none, or custom)

### Optional Groups

- `callbacks`: Training callbacks (default provided)
- `logger`: Experiment logger (tensorboard default)
- `trainer`: Trainer configuration (default provided)
- `paths`: Project paths (default provided)
- `extras`: Development utilities (default provided)
- `debug`: Debug configuration (opt-in)
- `experiment`: Pre-configured experiments (opt-in)

## Key Features

### 1. Task Names

Output is organized by task name:
```yaml
task_name: "train"  # Creates logs/train/runs/{timestamp}/
```

### 2. Tags

Tag your experiments for filtering:
```yaml
tags: ["dev", "experiment-1", "phase3d"]
```

### 3. Optimized Metric

For hyperparameter sweeps:
```yaml
optimized_metric: "val/loss"
```

### 4. Control Flags

```yaml
train: True    # Run training
test: True     # Run testing after training
predict: False # Run prediction
```

### 5. Seed

Reproducibility:
```yaml
seed: 42  # Fixed random seed
```

## Model Configurations

### VSUNet (Image Translation)

```yaml
_target_: viscy.translation.engine.VSUNet

model_config:
  in_channels: 1
  out_channels: 2
  residual: true
  dropout: 0.1

loss_function:
  _target_: viscy.translation.engine.MixedLoss
  l1_alpha: 0.5
  ms_dssim_alpha: 0.5

lr: 0.001
schedule: Constant
```

### Contrastive (DynaCLR)

```yaml
_target_: viscy.representation.engine.ContrastiveModule

encoder:
  _target_: viscy.representation.contrastive.ContrastiveEncoder
  backbone: convnext_tiny  # or convnextv2_tiny, resnet50
  in_channels: 1
  in_stack_depth: 15  # Must match z_range
  projection_dim: 128

loss_function:
  _target_: pytorch_metric_learning.losses.NTXentLoss
  temperature: 0.5

lr: 0.0001
schedule: Constant
```

## Data Configurations

### HCS DataModule

```yaml
_target_: viscy.data.hcs.HCSDataModule

data_path: ???  # Required
source_channel: ???  # Required
target_channel: []

z_window_size: 5
yx_patch_size: [256, 256]
batch_size: 32
num_workers: 16
```

### Triplet DataModule

```yaml
_target_: viscy.data.triplet.TripletDataModule

data_path: ???  # Required
tracks_path: ???  # Required
source_channel: ???  # Required
z_range: ???  # Required, e.g., [10, 25]

initial_yx_patch_size: [512, 512]
final_yx_patch_size: [224, 224]
batch_size: 16
num_workers: 1

time_interval: any
return_negative: false  # For NT-Xent loss
```

## Command-Line Overrides

Override any config value from CLI:

```bash
# Override learning rate
python -m viscy.train model=vsunet ... model.lr=0.01

# Override batch size
python -m viscy.train data=hcs ... data.batch_size=64

# Override multiple values
python -m viscy.train model=vsunet ... \
  model.lr=0.01 \
  data.batch_size=64 \
  trainer.max_epochs=100
```

## Hyperparameter Sweeps

Use multirun mode (`-m`) for sweeps:

```bash
python -m viscy.train -m \
  model=vsunet \
  data=hcs \
  augmentation=none \
  normalization=none \
  model.lr=0.0001,0.001,0.01 \
  data.batch_size=16,32,64
```

## Output Structure

Training outputs are organized as:

```
logs/
└── {task_name}/          # From task_name config
    └── runs/
        └── {timestamp}/   # Auto-generated timestamp
            ├── .hydra/    # Hydra configs
            ├── config_tree.log
            ├── checkpoints/
            └── tensorboard/
```

## View Configuration

Print the full configuration without running:

```bash
python -m viscy.train \
  model=vsunet \
  data=hcs \
  augmentation=none \
  normalization=none \
  --cfg job
```

## Tips

1. **Always use absolute paths** for data paths
2. **Use debug=default** for quick testing
3. **Set tags** for experiment tracking
4. **Use multirun** for sweeps
5. **Check config** with `--cfg job` before long runs

## References

- [Hydra Documentation](https://hydra.cc/)
- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/)
