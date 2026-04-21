# Virtual Staining Benchmark Configs

Composable leaf-per-experiment configs for dynacell virtual-staining
benchmarks. Train, predict, and eval leaves for one training run live
side-by-side under `<org>/<model>/<train_set>/` — one subdir per training
experiment so a trained model, its predictions, and its evaluations
form one coherent unit.

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

The reserved top-level YAML key `benchmark:` (above) is unrelated to the
Hydra `leaf=<path>` selector used for eval. The Hydra selector was
previously named `benchmark=`; both names referring to "benchmark" were
a source of confusion and the eval selector has been renamed.

## Layout

```
virtual_staining/
  README.md
  <org>/<model>/<train_set>/
    train.yml                             # LightningCLI fit leaf
    predict__<predict_set>.yml            # LightningCLI predict leaf
    eval__<predict_set>.yaml              # Hydra eval leaf (canonical location)
  _internal/                              # hidden support tree — not for browsing
    shared/
      model/
        train_sets/<name>.yml             # imaging modality + source_channel defaults
        predict_sets/<name>.yml           # predict_set metadata + source_channel
        targets/<target>.yml              # target_channel, train data_path, norms, CPU augs
        model_overlays/
          <model>_fit.yml                 # model + fit trainer + train data hparams
          <model>_predict.yml             # model + predict trainer + predict data hparams
        launcher_profiles/
          mode_<fit|predict>.yml          # launcher.mode
          hardware_<hw>.yml               # sbatch directives + trainer.devices
          runtime_shared.yml              # launcher.runtime + launcher.env
      eval/
        target/<target>.yaml              # GT paths, segmentation paths, GT/pred channel names
        feature_extractor/dynaclr/        # DynaCLR checkpoint + encoder kwargs
    leaf/                                 # symlink tree aliasing canonical eval leaves
      <org>/<model>/<train_set>/eval__<predict_set>.yaml -> ../../../../../<org>/<model>/<train_set>/eval__<predict_set>.yaml
```

Leaves are grouped by **train set** inside each `<org>/<model>/` cell so
that a training experiment (train + the predict/eval variants fed by its
checkpoint) lives in one directory. Adding a new training run — e.g. the
planned `joint_ipsc_confocal_a549_mantis` mix — means creating one new
subdir; deleting one is `rm -r`. Each train-set dir holds one `train.yml`
plus one `predict__<predict_set>.yml` and `eval__<predict_set>.yaml` per
held-out split the model is evaluated on.

The top level of `virtual_staining/` shows only biology (`er/`, `membrane/`,
`mito/`, `nucleus/`) plus `_internal/` — a hidden support tree whose
leading underscore signals "implementation detail; don't browse here for
science." All Hydra group files, all shared composition building blocks,
and the `leaf/` symlink adapter live under `_internal/`.

Train/predict leaves use LightningCLI (`.yml`). Eval leaves use Hydra and
keep `.yaml` because Hydra's group resolution only discovers `.yaml` files.
The `_internal/leaf/` symlink tree aliases each canonical eval leaf so
Hydra's `leaf=<path>` selector can discover them at
`<searchpath>/leaf/<path>.yaml`.

Eval runtime uses two search paths injected by `dynacell.__main__`:
`virtual_staining/_internal/` (for the `leaf/` tree) and
`virtual_staining/_internal/shared/eval/` (for the `target/` and
`feature_extractor/dynaclr/` groups). Schema-only eval configs ship
inside the dynacell package; wheel installs without the repo don't see
the HPC-bound groups and external users provide their own via
`--config-dir`. See `applications/dynacell/src/dynacell/evaluation/README.md`.

## Composition order

Last wins via deep-merge. Lists replace wholesale — layers that own list
fields (`callbacks`, `augmentations`, etc.) own the **full** list.

**Train leaf** (at `<org>/<model>/<train_set>/train.yml`):

```yaml
base:
  - ../../../_internal/shared/model/train_sets/<train_set>.yml
  - ../../../_internal/shared/model/targets/<target>.yml
  - ../../../_internal/shared/model/model_overlays/<model>_fit.yml
  - ../../../_internal/shared/model/launcher_profiles/mode_fit.yml
  - ../../../_internal/shared/model/launcher_profiles/hardware_<hw>.yml
  - ../../../_internal/shared/model/launcher_profiles/runtime_shared.yml
```

**Predict leaf** (at `<org>/<model>/<train_set>/predict__<predict_set>.yml`):

```yaml
base:
  - ../../../_internal/shared/model/predict_sets/<predict_set>.yml
  - ../../../_internal/shared/model/targets/<target>.yml
  - ../../../_internal/shared/model/model_overlays/<model>_predict.yml
  - ../../../_internal/shared/model/launcher_profiles/mode_predict.yml
  - ../../../_internal/shared/model/launcher_profiles/hardware_<hw>.yml
  - ../../../_internal/shared/model/launcher_profiles/runtime_shared.yml
```

**Eval leaf** (at `<org>/<model>/<train_set>/eval__<predict_set>.yaml`):

```yaml
# @package _global_
defaults:
  - override /target: <target>
  - override /predict_set: <predict_set>
  - override /feature_extractor/dinov3: lvd1689m
  - override /feature_extractor/dynaclr: default

io:
  pred_path: /hpc/.../predictions.zarr
  gt_cache_dir: /hpc/.../cache

compute_feature_metrics: true

save:
  save_dir: /hpc/.../eval_results
```

## Running

Direct LightningCLI (no sbatch):

- `uv run dynacell fit -c configs/benchmarks/virtual_staining/<org>/<model>/<train_set>/train.yml`
- `uv run dynacell predict -c configs/benchmarks/virtual_staining/<org>/<model>/<train_set>/predict__<predict_set>.yml`

Hydra eval:

- `uv run dynacell evaluate leaf=<org>/<model>/<train_set>/eval__<predict_set>`

Via sbatch with `submit_benchmark_job.py`:

```bash
LEAF=configs/benchmarks/virtual_staining/er/celldiff/ipsc_confocal/train.yml

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

`data.init_args.source_channel` lives in
`_internal/shared/model/train_sets/` and
`_internal/shared/model/predict_sets/` (duplicated — must be kept in
sync) because it's a property of the imaging modality, not the target.
Predict leaves don't compose train_sets, so the predict_set file has to
own its own `source_channel`.
