# viscy-transforms

Image transforms for virtual staining microscopy.

Part of the [VisCy](https://github.com/mehta-lab/VisCy) project.

## Installation

### From PyPI (when published)

```bash
pip install viscy-transforms
```

### For development (from monorepo root)

```bash
# Using uv (recommended)
uv pip install -e packages/viscy-transforms

# Or via workspace sync
uv sync --package viscy-transforms
```

## Usage

```python
from viscy_transforms import NormalizeSampled, RandAffineTransformSampled

# Transforms follow MONAI dictionary transform pattern
# See documentation for full API reference
```

## Features

- PyTorch-based image transforms optimized for microscopy data
- MONAI Dictionary transform compatibility for DataLoader pipelines
- Kornia-accelerated augmentations (affine, noise, blur)
- Specialized transforms for virtual staining workflows

## Dependencies

- `torch>=2.4.1`
- `kornia`
- `monai>=1.4`
- `numpy`

## Documentation

Full documentation available at [mehta-lab.github.io/VisCy](https://mehta-lab.github.io/VisCy/).

## License

BSD-3-Clause - see [LICENSE](../../LICENSE) in repository root.
