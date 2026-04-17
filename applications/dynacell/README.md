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

- `configs/recipes/` — reusable fragments (model, trainer, data, modes)
- `configs/examples/` — generic fit/predict pair per model family (stubs with
  `#TODO` placeholders)
- `configs/benchmarks/virtual_staining/` — runnable benchmark leaves composed
  from shared axes. One file per (organelle, train_set, model) for fit and
  one per (organelle, train_set, model, predict_set) for predict. See
  `configs/benchmarks/virtual_staining/README.md` for the layout and
  composition order.
- `tools/submit_benchmark_job.py` — drives one benchmark leaf end-to-end
  (compose → strip launcher metadata → render sbatch → submit). Use
  `--print-script` for a safe preview on any leaf, or `--dry-run` to
  stage artifacts to `launcher.run_root` without submitting (requires
  write permission on that path).
- `tools/LEGACY/` — archived pre-schema CellDiff configs kept as the
  equivalence reference. Not for direct launch; see its README.

### Benchmark submit

```bash
LEAF=applications/dynacell/configs/benchmarks/virtual_staining/train/er/ipsc_confocal/celldiff.yml

# Preview the rendered sbatch to stdout — safe on any leaf, no disk writes:
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --print-script

# Preview the resolved LightningCLI config (launcher+benchmark stripped):
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --print-resolved-config

# Stage artifacts to launcher.run_root without submitting (requires write perms):
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --dry-run

# Submit:
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF

# Dotlist overrides deep-merge after compose (repeatable, no ${...} interpolation):
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF \
    --override trainer.max_epochs=50 \
    --override data.init_args.batch_size=2
```

Flag semantics:

- `--print-script` / `--print-resolved-config` — pure preview: stdout
  only, no disk writes, no submission. Safe against run_roots the caller
  can't write to.
- `--dry-run` alone — write resolved YAML + rendered sbatch under
  `launcher.run_root`, but skip `sbatch`. Requires write permission on
  that path.
- `--dry-run` combined with any `--print-*` — preview wins (no writes).
- Bare invocation — write artifacts **and** submit.

Benchmark leaves carry two reserved top-level YAML keys (`launcher:` and
`benchmark:`) that are stripped automatically before the config reaches
LightningCLI, so `uv run dynacell fit -c <benchmark-leaf.yml>` also works
without the submit tool.

See `configs/benchmarks/virtual_staining/README.md` for the shared-axis
layout, composition order, and reserved-key contract.

## Supported subcommands

- `fit` and `validate`: fully supported for all architectures
- `predict`: supported; uses `HCSPredictionWriter` to write predictions to OME-Zarr.
  For UNetViT3D and CellDiff, `yx_patch_size` and `z_window_size` in the data config must match the model's `input_spatial_size`.
- `test`: raises `MisconfigurationException` (no `test_step` override)
