# Dynacell

Benchmark virtual staining application for deterministic and generative architectures.

## Usage

Set `data_path` in the config file or pass it on the command line:

```bash
cd applications/dynacell/examples/configs

# Deterministic models
uv run dynacell fit -c unetvit3d/fit.yml --data.init_args.data_path=/path/to/data.zarr
uv run dynacell fit -c fnet3d/fit.yml --data.init_args.data_path=/path/to/data.zarr
uv run dynacell predict -c unetvit3d/predict.yml --data.init_args.data_path=/path/to/data.zarr --ckpt_path=/path/to/checkpoint.ckpt
uv run dynacell predict -c fnet3d/predict.yml --data.init_args.data_path=/path/to/data.zarr --ckpt_path=/path/to/checkpoint.ckpt

# Flow-matching CellDiff
uv run dynacell fit -c celldiff/fit.yml --data.init_args.data_path=/path/to/data.zarr
uv run dynacell predict -c celldiff/predict.yml --data.init_args.data_path=/path/to/data.zarr --ckpt_path=/path/to/checkpoint.ckpt
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

- `fit` and `validate`: fully supported for all architectures
- `predict`: supported; uses `HCSPredictionWriter` to write predictions to OME-Zarr.
  For UNetViT3D and CellDiff, `yx_patch_size` and `z_window_size` in the data config must match the model's `input_spatial_size`.
- `test`: raises `MisconfigurationException` (no `test_step` override)
