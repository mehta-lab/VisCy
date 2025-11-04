# AGENTS.md

This file provides detailed guidance to Claude Code agents when working with code in this repository.

## Project Overview

VisCy is a PyTorch Lightning-based deep learning pipeline for image-based phenotyping at single-cell resolution. It supports three main tasks:

1. **Image Translation (Virtual Staining)** - Robust virtual staining using Cytoland models (VSCyto2D, VSCyto3D, VSNeuromast)
2. **Representation Learning** - Self-supervised learning via DynaCLR for cell dynamics
3. **Semantic Segmentation** - Supervised learning of cell states

## Development Commands

### Environment Setup

```sh
# Development installation with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run pre-commit manually
pre-commit run  # on staged files only
```

### Testing and Linting

```sh
# Run all tests
pytest -v

# Run specific test file
pytest tests/path/to/test_file.py -v

# Lint and format code
ruff check viscy
ruff format viscy tests
```

### CLI Usage

The main entry point is the `viscy` CLI command (defined in `viscy/cli.py`):

```sh
# View help
viscy --help

# Training (fit)
viscy fit --config path/to/config.yaml

# Prediction (predict)
viscy predict --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt

# Validation (validate)
viscy validate --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt

# Testing (test)
viscy test --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt

# Preprocessing (compute dataset statistics for normalization)
viscy preprocess --data_path path/to/data.zarr --channel_names channel1 channel2

# Export model to ONNX
viscy export --model.class_path VisCy.Model --ckpt_path checkpoint.ckpt --export_path model.onnx

# Precompute normalized arrays
viscy precompute --data_path input.zarr --output_path output.zarr --channel_names ch1 ch2

# Convert embeddings to AnnData
viscy convert_to_anndata --embeddings_path embeddings.zarr --output_anndata_path output.h5ad
```

## Architecture

### High-Level Structure

VisCy follows PyTorch Lightning patterns with a modular architecture:

- **CLI Layer** (`viscy/cli.py`): Lightning CLI wrapper that extends `LightningCLI` with custom subcommands
- **Trainer** (`viscy/trainer.py`): Custom `VisCyTrainer` extending `lightning.pytorch.Trainer` with preprocessing, export, and conversion methods
- **Task Modules**: Task-specific `LightningModule` implementations:
  - `viscy/translation/engine.py`: `VSUNet` for image translation/virtual staining
  - `viscy/representation/engine.py`: `ContrastiveModule` for self-supervised contrastive learning
  - `viscy/representation/vae.py`: Beta-VAE models for representation learning
- **Data Modules** (`viscy/data/`): Task-specific `LightningDataModule` implementations
- **Networks** (`viscy/unet/networks/`): Network architectures (UNeXt2, Unet2D, Unet25D, FCMAE)
- **Transforms** (`viscy/transforms/`): Custom data augmentations built on MONAI

### Data Flow

1. **Input Format**: OME-Zarr datasets stored in HCS (High-Content Screening) layout via `iohub`
2. **Preprocessing**: Compute channel-wise statistics (mean/std) stored as metadata in OME-Zarr attributes
3. **Training**: Lightning reads configs (YAML), loads data via DataModules, trains models, logs to TensorBoard
4. **Inference**: Load checkpoint, predict on new data, write results to OME-Zarr
5. **Evaluation**: Compare predictions to ground truth using metrics in `viscy/translation/evaluation_metrics.py`
6. **Export**: Models can be exported to ONNX format for deployment

### Key Data Modules

- `viscy/data/hcs.py`: HCS OME-Zarr dataset for high-content screening data
- `viscy/data/triplet.py`: Triplet sampling for contrastive learning (anchor, positive, negative)
- `viscy/data/cell_division_triplet.py`: Cell division-specific triplet sampling
- `viscy/data/gpu_aug.py`: GPU-accelerated augmentation wrapper
- `viscy/data/combined.py`: Multi-task combined data module

### Task-Specific Engines

**VSUNet** (Virtual Staining):
- Loss: `MixedLoss` combining L1, L2, and MS-DSSIM
- Architectures: UNeXt2 (default), Unet2D, Unet25D, FCMAE
- Key metrics: SSIM, Pearson correlation, MSE, MAE

**ContrastiveModule** (DynaCLR):
- Loss: Triplet margin loss, cosine embedding loss, or NTXentLoss
- Encoder + projection head architecture
- Tracks embeddings and similarity metrics

**BetaVae** (Variational Autoencoder):
- Implementations: `BetaVae25D`, `BetaVaeMonai`
- Specialized logging in `vae_logging.py`

### Config System

Uses `jsonargparse` for Lightning CLI configuration. Configs are YAML files specifying:
- `model`: Model class and hyperparameters
- `data`: DataModule class and data paths
- `trainer`: Lightning Trainer settings (GPUs, precision, etc.)

Example structure:
```yaml
model:
  class_path: viscy.translation.engine.VSUNet
  init_args:
    architecture: UNeXt2
    loss_function:
      l1_alpha: 0.5
      ms_dssim_alpha: 0.5

data:
  class_path: viscy.data.hcs.HCSDataModule
  init_args:
    data_path: /path/to/data.zarr
    source_channel: ["Phase"]
    target_channel: ["Nuclei", "Membrane"]

trainer:
  max_epochs: 50
  devices: [0]
```

#### Common Config Issues

**Missing `class_path` error**: If you see an error like:
```
error: Parser key "model":
  Does not validate against any of the Union subtypes
  - Problem with given class_path 'lightning.LightningModule':
      Validation failed: Key 'model_config.in_channels' is not expected
```

This means the config file is missing the `class_path` key. Old configs may have parameters directly under `model:` without specifying the class. Fix by restructuring:

```yaml
# OLD (incorrect):
model:
  model_config:
    in_channels: 1
    out_channels: 2
  lr: 0.0002

# NEW (correct):
model:
  class_path: viscy.translation.engine.VSUNet  # or appropriate model class
  init_args:
    architecture: UNeXt2
    model_config:
      in_channels: 1
      out_channels: 2
    lr: 0.0002
```

**Determining the correct `class_path`**:
- For virtual staining: `viscy.translation.engine.VSUNet`
- For contrastive learning: `viscy.representation.engine.ContrastiveModule`
- For VAE: `viscy.representation.vae.BetaVae25D` or `viscy.representation.vae.BetaVaeMonai`
- For classification: `viscy.representation.classification.ContrastiveClassifier`

Look at the parameters to identify the model type:
- `architecture`, `loss_function` → VSUNet
- `encoder`, `loss_function` (with triplet/contrastive) → ContrastiveModule
- `model_config.encoder_blocks`, `model_config.dims` → UNeXt2-based models

### Transforms and Augmentations

Custom transforms in `viscy/transforms/` extend MONAI's dictionary transforms:
- Support for 2.5D (volumetric time-lapse) data
- GPU-accelerated transforms when used with `GPUTransformDataModule`
- Batched random transforms: elastic deformation, histogram shift, local pixel shuffling, z-stack shift

## File Organization

Data hierarchy follows the structure documented in `viscy/data_organization.md`:

```
project_root/
├── datasets/
│   ├── train/
│   │   └── *.zarr  # Training OME-Zarr datasets
│   └── test/
│       └── *.zarr  # Test datasets
└── models/
    └── experiment_name/
        ├── config.yaml  # Training config
        └── lightning_logs/
            └── version_*/
                ├── checkpoints/*.ckpt
                └── config.yaml  # Auto-saved full config
```

## Testing

Tests mirror the source structure in `tests/`:
- Unit tests for data modules, transforms, models
- Fixtures defined in `tests/conftest.py`
- Run specific test modules: `pytest tests/data/test_hcs.py -v`

## Dependencies

Core dependencies:
- `torch>=2.4.1`: PyTorch framework
- `lightning>=2.3`: PyTorch Lightning
- `iohub[tensorstore]>=0.3a2`: OME-Zarr I/O with HCS layout
- `monai>=1.4`: Medical image transforms and networks
- `kornia`: Differentiable computer vision
- `pytorch-metric-learning>2`: Contrastive learning losses
- `timm>=0.9.5`: Vision model architectures

Optional extras:
- `[metrics]`: Cellpose, torchmetrics, UMAP for evaluation
- `[visual]`: Visualization tools (torchview, plotly, dash)
- `[examples]`: Jupyter, napari for interactive examples
- `[dev]`: All dependencies + pytest, ruff, hypothesis

## CI/Pre-commit

Pre-commit hooks (`.pre-commit-config.yaml`):
- `ruff-check` and `ruff-format` for linting/formatting
- `pyproject-fmt` for pyproject.toml formatting
- Standard hooks: detect-private-key, check-ast, trailing-whitespace, etc.

Formatting configuration in `pyproject.toml`:
- Line length: 88
- Import sorting via ruff with `viscy` as first-party

## Notes

- GPU required for training (tested on NVIDIA Ampere/Hopper with CUDA 12.6)
- Set `VISCY_LOG_LEVEL` environment variable to control logging
- Default random seed is 42 for reproducibility
- TF32 precision enabled by default for performance
- Mixed precision training available via Lightning Trainer flags
