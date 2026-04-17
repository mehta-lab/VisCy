# Virtual Staining Benchmark Configs

Composable leaf-per-experiment configs for dynacell virtual-staining benchmarks.

## Reserved top-level keys

Two top-level YAML keys are **reserved for dynacell** and are stripped
from the composed config before it reaches LightningCLI:

- `launcher:` — sbatch directives, runtime env, job metadata. Consumed by
  `applications/dynacell/tools/submit_benchmark_job.py`.
- `benchmark:` — informational experiment metadata (target, train_set,
  experiment_id). Readable by downstream reporting; not consumed by
  Lightning.

The strip happens inside `viscy_utils.cli._maybe_compose_config`. This
means `uv run dynacell fit -c <leaf.yml>` works for any benchmark leaf
without the dedicated submit tool.

## Layout

```
virtual_staining/
  shared/
    train_sets/<name>.yml         # imaging modality + source_channel defaults
    targets/<target>.yml          # target_channel, train data_path, norms, CPU augs
    model_overlays/
      celldiff_fit.yml            # model + fit trainer + train data hparams
      celldiff_predict.yml        # model + predict trainer + predict data hparams
    launcher_profiles/
      mode_<fit|predict>.yml      # launcher.mode
      hardware_<hw>.yml           # sbatch directives + trainer.devices
      runtime_<rt>.yml            # launcher.runtime + launcher.env
    predict_sets/<name>.yml       # predict_set metadata + source_channel
  train/<org>/<train_set>/<model>.yml
  predict/<org>/<train_set>/<model>/<predict_set>.yml
```

## Composition order

Last wins via deep-merge. Lists replace wholesale — layers that own list
fields (`callbacks`, `augmentations`, etc.) own the **full** list.

**Train leaf** (at `train/<org>/<train_set>/<model>.yml`):

```yaml
base:
  - ../../../shared/train_sets/<train_set>.yml
  - ../../../shared/targets/<target>.yml
  - ../../../shared/model_overlays/<model>_fit.yml
  - ../../../shared/launcher_profiles/mode_fit.yml
  - ../../../shared/launcher_profiles/hardware_<hw>.yml
  - ../../../shared/launcher_profiles/runtime_<rt>.yml
```

**Predict leaf** (at `predict/<org>/<train_set>/<model>/<predict_set>.yml`):

```yaml
base:
  - ../../../../shared/predict_sets/<predict_set>.yml
  - ../../../../shared/targets/<target>.yml
  - ../../../../shared/model_overlays/<model>_predict.yml
  - ../../../../shared/launcher_profiles/mode_predict.yml
  - ../../../../shared/launcher_profiles/hardware_<hw>.yml
  - ../../../../shared/launcher_profiles/runtime_<rt>.yml
```

## Running

Direct LightningCLI (no sbatch):

- `uv run dynacell fit -c configs/benchmarks/virtual_staining/train/<org>/<train_set>/<model>.yml`
- `uv run dynacell predict -c configs/benchmarks/virtual_staining/predict/<org>/<train_set>/<model>/<predict_set>.yml`

Via sbatch with `submit_benchmark_job.py`:

```bash
LEAF=configs/benchmarks/virtual_staining/train/er/ipsc_confocal/celldiff.yml

# Pure preview (no disk writes, safe on any run_root):
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --print-script
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --print-resolved-config

# Stage artifacts to launcher.run_root but skip submission (requires write perms):
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --dry-run

# Submit:
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF

# Dotlist overrides deep-merge after compose (repeatable; ${...} interpolation is rejected):
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF \
    --override trainer.max_epochs=50 --override data.init_args.batch_size=2
```

`--dry-run` combined with `--print-*` drops the disk writes (preview
wins). `trainer.devices` and `launcher.sbatch.gpus` must match or
submission fails fast.

## Source channel contract

`data.init_args.source_channel` lives in `train_sets/` and `predict_sets/`
(duplicated — must be kept in sync) because it's a property of the
imaging modality, not the target. Predict leaves don't compose train_sets,
so the predict_set file has to own its own `source_channel`.
