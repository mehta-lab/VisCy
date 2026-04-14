# Dynacell

Benchmark virtual staining application for deterministic and generative architectures.

## Usage

Set `data_path` in the config file or pass it on the command line:

```bash
cd applications/dynacell/configs/examples

# Deterministic models
uv run dynacell fit -c fnet3d/fit.yml --data.init_args.data_path=/path/to/data.zarr
uv run dynacell fit -c unext2/fit.yml --data.init_args.data_path=/path/to/data.zarr
uv run dynacell fit -c unetvit3d/fit.yml --data.init_args.data_path=/path/to/data.zarr

# Flow-matching CellDiff
uv run dynacell fit -c celldiff/fit.yml --data.init_args.data_path=/path/to/data.zarr
```

## Architectures

### Deterministic (DynacellUNet)

- **UNetViT3D**: 3D U-Net with Vision Transformer bottleneck
- **UNeXt2**: timm encoder with custom stem, decoder, and head (VSCyto3D backbone)
- **FNet3D**: Recursive encoder-decoder baseline (Ounkomol et al. 2018)

### Generative (DynacellFlowMatching)

- **CellDiff**: Flow-matching virtual staining with CELLDiffNet backbone.
  Uses ODE sampling for inference. No external loss function needed —
  the flow-matching loss is computed internally.

## Config Structure

- `configs/recipes/` — Reusable fragments (model, trainer, data, modes)
- `configs/examples/` — Generic fit/predict pair per model family

Benchmark-specific configs (SEC61B, nuclei-mix) live in the `dynacell-paper` repo.

## Supported subcommands

- `fit` and `validate`: fully supported for all architectures
- `predict`: supported; uses `HCSPredictionWriter` to write predictions to OME-Zarr.
  For UNetViT3D and CellDiff, `yx_patch_size` and `z_window_size` in the data config must match the model's `input_spatial_size`.
- `test`: raises `MisconfigurationException` (no `test_step` override)
