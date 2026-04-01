# Dynacell

Benchmark virtual staining application using UNetViT3D and FNet3D architectures.

## Usage

Set `data_path` in the config file or pass it on the command line:

```bash
cd applications/dynacell/examples/configs
uv run dynacell fit -c unetvit3d/fit.yml --data.init_args.data_path=/path/to/data.zarr
uv run dynacell fit -c fnet3d/fit.yml --data.init_args.data_path=/path/to/data.zarr
uv run dynacell predict -c unetvit3d/predict.yml --data.init_args.data_path=/path/to/data.zarr --ckpt_path=/path/to/checkpoint.ckpt
uv run dynacell predict -c fnet3d/predict.yml --data.init_args.data_path=/path/to/data.zarr --ckpt_path=/path/to/checkpoint.ckpt
```

## Architectures

- **UNetViT3D**: 3D U-Net with Vision Transformer bottleneck
- **FNet3D**: Recursive encoder-decoder baseline (Ounkomol et al. 2018)

## Supported subcommands

- `fit` and `validate`: fully supported for both architectures
- `predict`: supported (Stage 2.5); uses `HCSPredictionWriter` to write predictions to OME-Zarr.
  For UNetViT3D, `yx_patch_size` and `z_window_size` in the data config must match the model's `input_spatial_size`.
- `test`: raises `MisconfigurationException` (no `test_step` override)
