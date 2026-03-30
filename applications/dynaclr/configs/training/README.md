# DynaCLR Training Configs

Composable training configuration using LightningCLI `--config` stacking.
Each layer is a YAML fragment; later configs deep-merge into earlier ones
(dicts merge, lists replace).

## Structure

```
configs/training/
  _base.yml          Trainer + model defaults (callbacks, optimizer, encoder)
  arch/              Encoder geometry (stem, z_depth, patch size)
    2d_z1.yml        stem=[1,4,4], z_window=1
    3d_z16.yml       stem=[4,4,4], z_window=16, random Z crop
    3d_z30.yml       stem=[5,4,4], z_window=30, 192px patch
  data/              Data pipeline: sampling + normalization + augmentations
    boc_{dim}_{positive_pair}_{batch_composition}.yml
  demo/              Self-contained configs for smoke tests (single --config)
  slurm/             SLURM experiment scripts (sbatch entry points)
    train.sh         Shared launcher (sourced, not sbatch'd directly)
  _legacy/           Old monolithic configs (reference only)
```

## Data config naming convention

```
{channel_mode}_{dim}_{positive_pair_strategy}_{batch_composition}.yml
```

| Segment | Values | Meaning |
|---------|--------|---------|
| channel_mode | `boc` | bag-of-channels (1 random channel per sample) |
| dim | `2d`, `3d` | spatial dimensionality |
| positive_pair | `temporal` | same cell lineage at t+tau |
|  | `gene-reporter` | same gene + same reporter (OPS) |
|  | `self` | SimCLR-style (same crop, different augmentation) |
| batch_composition | `stratify-perturbation` | balance infected/uninfected |
|  | `stratify-perturbation-marker` | balance perturbation and organelle marker |
|  | `stratify-marker` | balance by reporter/marker only |

## Composition

Stack three configs: `_base.yml` + `arch/*.yml` + `data/*.yml`, then
pass experiment-specific values as CLI overrides in the SLURM script.

```bash
viscy fit \
  --config _base.yml \
  --config arch/3d_z16.yml \
  --config data/boc_3d_temporal_stratify-perturbation.yml \
  --trainer.devices 4 \
  --data.init_args.batch_size 512 \
  --data.init_args.collection_path path/to/collection.yml
```

## SLURM scripts

Each experiment is a thin `.sh` that sets `PROJECT`, `RUN_NAME`, `CONFIGS`,
experiment-specific `EXTRA_ARGS`, and sources `train.sh`:

```bash
# Submit
sbatch slurm/DynaCLR-3D-BagOfChannels-v2.sh

# Override run name
RUN_NAME=phase2-hcl sbatch slurm/DynaCLR-3D-BagOfChannels-v2.sh

# Parameter sweep
for TEMP in 0.1 0.2 0.5; do
  RUN_NAME="sweep-temp${TEMP}" \
  EXTRA_ARGS="--model.init_args.loss_function.init_args.temperature ${TEMP}" \
  sbatch slurm/DynaCLR-3D-BagOfChannels-v2.sh
done
```

`train.sh` handles:
- `PYTHONNOUSERSITE=1` (prevents `~/.local/` shadowing conda)
- Creates `${MODEL_ROOT}/${PROJECT}/${RUN_NAME}/` output directory
- Copies config files into the run directory for reproducibility
- Sets WandB logger project/name/save_dir via CLI overrides
- Sets checkpoint dirpath via CLI override

## Adding a new experiment

1. Check if an existing `data/*.yml` matches your sampling strategy.
   If not, create a new one following the naming convention.
2. Create a new `slurm/<experiment>.sh` with SBATCH directives and overrides.
3. Submit with `sbatch slurm/<experiment>.sh`.

## Demo configs

Self-contained single-file configs for quick testing:

```bash
viscy fit --config demo/demo_3d_fit.yml --trainer.fast_dev_run true
```
