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
- **UNeXt2**: timm encoder with custom stem, decoder, and head (VSCyto3D backbone)
- **FNet3D**: Recursive encoder-decoder baseline (Ounkomol et al. 2018)

## SEC61B Benchmark

Launch SEC61B training from Dynacell (canonical location):

```bash
# FNet3D benchmark config
uv run python -m dynacell fit --config applications/dynacell/examples/configs/sec61b/fit_fnet3d.yml

# FNet3D paper-native baseline config
uv run python -m dynacell fit --config applications/dynacell/examples/configs/sec61b/fit_fnet3d_paper.yml

# UNeXt2 (VSCyto3D)
uv run python -m dynacell fit --config applications/dynacell/examples/configs/sec61b/fit_unext2.yml

# SLURM (H200)
sbatch applications/dynacell/examples/configs/sec61b/run_fnet3d.slurm
sbatch applications/dynacell/examples/configs/sec61b/run_fnet3d_paper.slurm
sbatch applications/dynacell/examples/configs/sec61b/run_unext2.slurm
```

## Supported subcommands

- `fit` and `validate`: fully supported for both architectures
- `predict`: supported (Stage 2.5); uses `HCSPredictionWriter` to write predictions to OME-Zarr.
  For UNetViT3D, `yx_patch_size` and `z_window_size` in the data config must match the model's `input_spatial_size`.
- `test`: raises `MisconfigurationException` (no `test_step` override)
