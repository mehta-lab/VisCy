# viscy-models

Neural network architectures for virtual staining microscopy.

Part of the [VisCy](https://github.com/mehta-lab/VisCy) project.

## Installation

### From PyPI (when published)

```bash
pip install viscy-models
```

### For development (from monorepo root)

```bash
# Using uv (recommended)
uv pip install -e packages/viscy-models

# Or via workspace sync
uv sync --package viscy-models
```

## Usage

```python
from viscy_models import UNeXt2, FullyConvolutionalMAE

# All models are pure nn.Modules — no Lightning or Hydra coupling
model = UNeXt2(
    in_channels=1,
    out_channels=2,
    in_stack_depth=5,
    backbone="convnextv2_tiny",
)

# Contrastive encoders
from viscy_models import ContrastiveEncoder, ResNet3dEncoder

encoder = ContrastiveEncoder(
    backbone="convnext_tiny",
    in_channels=2,
    in_stack_depth=15,
    embedding_dim=768,
    projection_dim=128,
)

# Variational autoencoders
from viscy_models import BetaVae25D, BetaVaeMonai
```

## Models

### UNet Family (`viscy_models.unet`)

| Model | Description |
|-------|-------------|
| `UNeXt2` | ConvNeXtV2-based encoder-decoder with 3D-to-2D stem and pixel shuffle head |
| `FullyConvolutionalMAE` | Fully convolutional masked autoencoder for self-supervised pretraining |
| `Unet2d` | Classic 2D UNet with configurable depth and residual connections |
| `Unet25d` | 2.5D UNet with learned 3D-to-2D compression via skip interruption |
| `Unet3d` | 3D U-Net (F-Net, Ounkomol et al. 2018) with recursive encoder-decoder. All spatial dims must be divisible by `2^depth`. |

### Contrastive Encoders (`viscy_models.contrastive`)

| Model | Description |
|-------|-------------|
| `ContrastiveEncoder` | timm backbone (ConvNeXt/ResNet) with 3D-to-2D stem and projection MLP |
| `ResNet3dEncoder` | MONAI ResNetFeatures for native 3D contrastive learning |

### Variational Autoencoders (`viscy_models.vae`)

| Model | Description |
|-------|-------------|
| `BetaVae25D` | 2.5D beta-VAE with timm encoder and custom decoder |
| `BetaVaeMonai` | Beta-VAE wrapping MONAI's VarAutoEncoder |

## Heads (`viscy_models.components.heads`)

Pluggable task heads for multi-task learning. Attach to `ContrastiveModule` via `auxiliary_heads`.

| Class | Description |
|-------|-------------|
| `BaseHead` | Abstract base — subclass to add custom heads. Defines `forward`, `compute_loss`, `log_metrics` |
| `ClassificationHead` | Classification on backbone features. Uses `MLP` + optional `CosineClassifier`. Logs top-1 and top-k accuracy |
| `MLP` | Configurable projection/classification MLP with BN/LN and dropout |
| `CosineClassifier` | L2-normalised linear head with learnable temperature — recommended for large class counts |

### Label routing via `SampleMeta`

Auxiliary heads consume labels from `anchor_meta["labels"]` in the batch — a `dict[str, int]` populated by the dataset. The `batch_key` on each head selects which label to use:

```python
# Dataset populates anchor_meta with integer labels
anchor_meta = [{"labels": {"condition": 0, "gene_ko": 42}}]

# Head config — batch_key must match a key in anchor_meta["labels"]
ClassificationHead(
    head_name="gene_ko",      # used for logging
    batch_key="gene_ko",      # key in anchor_meta["labels"]
    in_dims=768,
    hidden_dims=512,
    num_classes=1001,
    loss_weight=0.5,
)
```

## Features

- Pure `nn.Module` architectures — no Lightning or Hydra dependencies
- Shared components in `components/` (stems, heads, decoder blocks, ConvBlocks)
- Pluggable auxiliary heads via `BaseHead` — extend for custom losses and metrics
- State dict key compatibility with original VisCy checkpoints
- Immutable defaults for all model constructors

## Dependencies

- `torch>=2.10`
- `timm>=1.0.15`
- `monai>=1.5.2`
- `numpy>=2.4.1`

## Documentation

In the works!

## License

BSD-3-Clause - see [LICENSE](../../LICENSE) in repository root.
