# Dynacell

Benchmark virtual staining application using UNetViT3D and FNet3D architectures.

## Usage

```bash
cd applications/dynacell/examples/configs
uv run dynacell fit -c unetvit3d/fit.yml
uv run dynacell fit -c fnet3d/fit.yml
```

## Architectures

- **UNetViT3D**: 3D U-Net with Vision Transformer bottleneck
- **FNet3D**: Recursive encoder-decoder baseline (Ounkomol et al. 2018)

## Limitations (Stage 2)

- Only `fit` and `validate` subcommands are supported
- `predict` raises `NotImplementedError` (requires DivisiblePad + tiled inference, Stage 3)
- `test` raises `MisconfigurationException` (no `test_step` override)
