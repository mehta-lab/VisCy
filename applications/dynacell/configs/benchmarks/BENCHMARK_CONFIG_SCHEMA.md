# Benchmark Config Schema For Active VisCy Training

This document captures the proposed active benchmark config layout for
`VisCy/applications/dynacell`, using one-file benchmark configs with embedded
launcher metadata.

The goal is to support:

- small public example configs
- real benchmark training configs
- no drift between training config and SLURM resource settings
- scalable organization across model families, train sets, targets, and
  prediction domains

This document covers the active benchmark-training surface for two experiment
phases:

- Phase 1
  - target: `er`
  - train sets:
    - `ipsc_confocal`
    - `ipsc_confocal_plus_mantis`
  - model families:
    - `fnet3d`
    - `unext2_scratch`
    - `unext2_fcmae`
    - `unetvit3d`
    - `celldiff`
- Phase 2
  - targets:
    - `mito`
    - `nucleus`
    - `membrane`
  - train sets:
    - `ipsc_confocal`
    - `ipsc_confocal_plus_mantis`
  - model families:
    - `selected_deterministic`
    - `celldiff`

Prediction, evaluation, and paper orchestration are downstream stages. They are
related, but they are not part of this Phase 1 / Phase 2 training numbering.

## Ownership

- `VisCy/applications/dynacell`
  - owns active runnable benchmark train/predict configs
  - owns launcher metadata and submission tooling
  - owns generic `examples/` and reusable `recipes/`
- `dynacell-paper`
  - keeps archived historical paper configs
  - keeps broader benchmark DAG orchestration, paper scripts, and docs

## Target Tree In VisCy

```text
applications/dynacell/
  configs/
    recipes/
      data/
      models/
      trainer/

    examples/
      celldiff/
        fit.yml
        predict.yml
      fnet3d/
        fit.yml
        predict.yml
      unext2/
        fit.yml
        predict.yml
      unetvit3d/
        fit.yml
        predict.yml

    benchmarks/
      virtual_staining/
        shared/
          train_sets/
            ipsc_confocal.yml
            ipsc_confocal_plus_mantis.yml
          targets/
            er_sec61b.yml
            mito_tomm20.yml
            nucleus.yml
            membrane.yml
          model_overlays/
            fnet3d.yml
            unext2_scratch.yml
            unext2_fcmae.yml
            unetvit3d.yml
            celldiff.yml
          launcher_profiles/
            mode_fit.yml
            mode_predict.yml
            hardware_a6000_single.yml
            hardware_h100x4.yml
            hardware_h200_single.yml
            runtime_ddp.yml
            runtime_single_gpu.yml
            runtime_resume.yml
          predict_sets/
            ipsc_confocal.yml
            mantis_a549.yml
            mantis_a549_zikv.yml
            mantis_a549_denv.yml

        train/
          er/
            ipsc_confocal/
              fnet3d.yml
              unext2_scratch.yml
              unext2_fcmae.yml
              unetvit3d.yml
              celldiff.yml
            ipsc_confocal_plus_mantis/
              fnet3d.yml
              unext2_scratch.yml
              unext2_fcmae.yml
              unetvit3d.yml
              celldiff.yml

          mito/
            ipsc_confocal/
              selected_deterministic.yml
              celldiff.yml
            ipsc_confocal_plus_mantis/
              selected_deterministic.yml
              celldiff.yml

          nucleus/
            ipsc_confocal/
              selected_deterministic.yml
              celldiff.yml
            ipsc_confocal_plus_mantis/
              selected_deterministic.yml
              celldiff.yml

          membrane/
            ipsc_confocal/
              selected_deterministic.yml
              celldiff.yml
            ipsc_confocal_plus_mantis/
              selected_deterministic.yml
              celldiff.yml

        predict/
          er/
            ipsc_confocal/
              fnet3d/
                ipsc_confocal.yml
                mantis_a549.yml
                mantis_a549_zikv.yml
                mantis_a549_denv.yml
              unext2_scratch/
              unext2_fcmae/
              unetvit3d/
              celldiff/
            ipsc_confocal_plus_mantis/
              ...

  tools/
    submit_benchmark_job.py
```

## Key Rule

- `configs/examples/` stays generic and public
- `configs/benchmarks/virtual_staining/...` becomes the real benchmark layer
- archived SEC61B configs in `dynacell-paper` remain historical reference only

## Experiment Phase Mapping

The directory layout is meant to scale without changing shape between phases.
Only the populated leaves change.

### Phase 1

Phase 1 fills the `train/er/...` subtree for all model families and both train
sets:

- `train/er/ipsc_confocal/fnet3d.yml`
- `train/er/ipsc_confocal/unext2_scratch.yml`
- `train/er/ipsc_confocal/unext2_fcmae.yml`
- `train/er/ipsc_confocal/unetvit3d.yml`
- `train/er/ipsc_confocal/celldiff.yml`
- the same five files under `train/er/ipsc_confocal_plus_mantis/`

This is the broad comparison phase used to narrow model choice.

### Phase 2

Phase 2 reuses the same schema and shared-axis files, but fills only the
`mito`, `nucleus`, and `membrane` subtrees, and only for the two shortlisted
model families:

- `selected_deterministic`
- `celldiff`

That produces these leaf patterns:

- `train/mito/ipsc_confocal/selected_deterministic.yml`
- `train/mito/ipsc_confocal/celldiff.yml`
- `train/mito/ipsc_confocal_plus_mantis/selected_deterministic.yml`
- `train/mito/ipsc_confocal_plus_mantis/celldiff.yml`
- the same four-file pattern for `nucleus/`
- the same four-file pattern for `membrane/`

This is intentionally repetitive. That repetition is a feature of the tree, not
a design bug: it keeps every runnable benchmark job addressable by target,
train set, and model family without introducing a second naming system.

### Scalability Constraint

The shared-axis directories are what keep the repeated leaf structure from
turning into a maintenance problem:

- `shared/train_sets/` owns data-source membership and base data paths
- `shared/targets/` owns organelle-specific target-channel choices
- `shared/model_overlays/` owns model-family defaults
- `shared/launcher_profiles/` owns reusable hardware / mode / runtime policy
- `shared/predict_sets/` owns prediction-domain inputs

New organelles or train sets should usually add one shared-axis file plus a new
leaf subtree, not a new config convention.

## Launcher Profile Schema

Launcher metadata should be composable too, not stored in one flat profile
registry.

The reusable axes are:

- mode
  - `fit`
  - `predict`
- hardware class
  - `a6000_single`
  - `h100x4`
  - `h200_single`
- runtime behavior
  - `ddp`
  - `single_gpu`
  - `resume`

Use separate launcher-profile files under:

`applications/dynacell/configs/benchmarks/virtual_staining/shared/launcher_profiles/`

### Mode Profile

`mode_fit.yml`

```yaml
launcher:
  mode: fit
```

### Hardware Profile

`hardware_h100x4.yml`

```yaml
launcher:
  sbatch:
    partition: gpu
    nodes: 1
    ntasks_per_node: 4
    gpus: 4
    cpus_per_task: 12
    mem_per_cpu: "20G"
    time: "48:00:00"
    constraint: "a100_80|h100|h200"
```

### Runtime Profile

`runtime_ddp.yml`

```yaml
launcher:
  runtime:
    use_srun: true
    cleanup_tmp: true
  env:
    PYTHONUNBUFFERED: "1"
    PYTHONFAULTHANDLER: "1"
    NCCL_DEBUG: "INFO"
```

### Example Single-GPU Profile

`hardware_h200_single.yml`

```yaml
launcher:
  sbatch:
    partition: gpu
    nodes: 1
    ntasks_per_node: 1
    gpus: 1
    cpus_per_task: 8
    mem: "256G"
    time: "48:00:00"
    constraint: "h200|h100|a100_80"
```

### Example Predict Profile

`mode_predict.yml`

```yaml
launcher:
  mode: predict
```

## Shared-Axis Config Examples

### Train Set

`applications/dynacell/configs/benchmarks/virtual_staining/shared/train_sets/ipsc_confocal.yml`

```yaml
benchmark:
  train_set: ipsc_confocal
  dataset_group: [ipsc_confocal]

data:
  class_path: viscy_data.hcs.HCSDataModule
  init_args:
    data_path: /hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/train/SEC61B.zarr
    source_channel: Phase3D
    split_ratio: 0.8
    preload: true
    scratch_dir: /dev/shm
    persistent_workers: true
```

### Target

`applications/dynacell/configs/benchmarks/virtual_staining/shared/targets/er_sec61b.yml`

```yaml
benchmark:
  target: er
  gene: SEC61B
  target_id: er_sec61b

data:
  init_args:
    target_channel: Structure
```

### Model Overlay

`applications/dynacell/configs/benchmarks/virtual_staining/shared/model_overlays/celldiff.yml`

```yaml
base:
  - ../../../recipes/models/celldiff_fm.yml

model:
  init_args:
    net_config:
      input_spatial_size: [8, 512, 512]
    lr: 0.0001
    schedule: WarmupCosine
    num_log_steps: 10
    compute_validation_loss: true

trainer:
  precision: bf16-mixed
  max_epochs: 20

data:
  init_args:
    z_window_size: 13
    batch_size: 2
    num_workers: 4
    yx_patch_size: [512, 512]
    normalizations:
      - class_path: viscy_transforms.NormalizeSampled
        init_args:
          keys: [Phase3D]
          level: fov_statistics
          subtrahend: mean
          divisor: std
      - class_path: viscy_transforms.NormalizeSampled
        init_args:
          keys: [Structure]
          level: fov_statistics
          subtrahend: median
          divisor: iqr
    augmentations:
      - class_path: viscy_transforms.RandWeightedCropd
        init_args:
          keys: [Phase3D, Structure]
          w_key: Structure
          spatial_size: [13, 624, 624]
          num_samples: 2
    gpu_augmentations:
      - class_path: viscy_transforms.BatchedRandAffined
        init_args:
          keys: [source, target]
          prob: 0.8
          rotate_range: [3.14, 0, 0]
          shear_range: [0.0, 0.05, 0.05]
          scale_range: [[0.7, 1.3], [0.5, 1.5], [0.5, 1.5]]
          safe_crop_size: [8, 512, 512]
          safe_crop_coverage: 0.9
      - class_path: viscy_transforms.BatchedCenterSpatialCropd
        init_args:
          keys: [source, target]
          roi_size: [8, 512, 512]
      - class_path: viscy_transforms.BatchedRandAdjustContrastd
        init_args:
          keys: [source]
          prob: 0.5
          gamma: [0.8, 1.2]
      - class_path: viscy_transforms.BatchedRandScaleIntensityd
        init_args:
          keys: [source]
          prob: 0.5
          factors: 0.5
      - class_path: viscy_transforms.BatchedRandGaussianNoised
        init_args:
          keys: [source]
          prob: 0.5
          mean: 0.0
          std: 0.3
      - class_path: viscy_transforms.BatchedRandGaussianSmoothd
        init_args:
          keys: [source]
          prob: 0.5
          sigma_x: [0.25, 0.75]
          sigma_y: [0.25, 0.75]
          sigma_z: [0.25, 0.75]
    val_gpu_augmentations:
      - class_path: viscy_transforms.BatchedCenterSpatialCropd
        init_args:
          keys: [source, target]
          roi_size: [8, 512, 512]
```

Analogous overlays should be defined for:

- `fnet3d.yml`
- `unext2_scratch.yml`
- `unext2_fcmae.yml`
- `unetvit3d.yml`

## Leaf Train Config Schema

Example:

`applications/dynacell/configs/benchmarks/virtual_staining/train/er/ipsc_confocal/celldiff.yml`

```yaml
base:
  - ../../../shared/train_sets/ipsc_confocal.yml
  - ../../../shared/targets/er_sec61b.yml
  - ../../../shared/model_overlays/celldiff.yml
  - ../../../shared/launcher_profiles/mode_fit.yml
  - ../../../shared/launcher_profiles/hardware_h200_single.yml
  - ../../../shared/launcher_profiles/runtime_single_gpu.yml
  - ../../../../recipes/trainer/fit_fm_4gpu.yml

benchmark:
  task: virtual_staining
  phase: phase1
  organelle: er
  train_set: ipsc_confocal
  model_name: celldiff
  experiment_id: er__ipsc_confocal__celldiff

trainer:
  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      project: dynacell
      name: er__ipsc_confocal__celldiff
      save_dir: /hpc/projects/comp.micro/virtual_staining/models/dynacell/er/ipsc_confocal/celldiff
  callbacks:
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: step
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        every_n_epochs: 1
        save_top_k: -1
        save_last: true
        dirpath: /hpc/projects/comp.micro/virtual_staining/models/dynacell/er/ipsc_confocal/celldiff/checkpoints

launcher:
  job_name: er_ipsc_celldiff
  run_root: /hpc/projects/comp.micro/virtual_staining/models/dynacell/er/ipsc_confocal/celldiff
  sbatch:
    time: "48:00:00"
  env:
    VISCY_WANDB_GROUP: phase1_er
```

## Leaf Predict Config Schema

Example:

`applications/dynacell/configs/benchmarks/virtual_staining/predict/er/ipsc_confocal/celldiff/mantis_a549.yml`

```yaml
base:
  - ../../../../shared/predict_sets/mantis_a549.yml
  - ../../../../shared/targets/er_sec61b.yml
  - ../../../../shared/launcher_profiles/mode_predict.yml
  - ../../../../shared/launcher_profiles/hardware_h200_single.yml
  - ../../../../shared/launcher_profiles/runtime_single_gpu.yml

benchmark:
  task: virtual_staining
  organelle: er
  trained_on: ipsc_confocal
  predict_set: mantis_a549
  model_name: celldiff
  experiment_id: er__ipsc_confocal__celldiff__mantis_a549

model:
  ckpt_path: /hpc/projects/comp.micro/virtual_staining/models/dynacell/er/ipsc_confocal/celldiff/checkpoints/last.ckpt

io:
  pred_path: /hpc/projects/virtual_staining/predictions/er/ipsc_confocal/celldiff/mantis_a549/prediction.zarr

launcher:
  job_name: pred_er_a549_celldiff
  run_root: /hpc/projects/virtual_staining/predictions/er/ipsc_confocal/celldiff/mantis_a549
```

## Submit Tool Contract

File:

`applications/dynacell/tools/submit_benchmark_job.py`

Behavior:

1. Accept one config path.
2. Compose it using the same base-resolution logic VisCy already uses.
3. Read the resolved `launcher:` block after composition.
4. Strip `launcher:` from the resolved config.
5. Write:
   - resolved runtime config to:
     - `<run_root>/resolved/<mode>.resolved.yml`
   - rendered SLURM script to:
     - `<run_root>/slurm/<timestamp>_<job_name>.sbatch`
6. Submit with `sbatch`, unless `--dry-run`.

Command shape:

```bash
uv run python applications/dynacell/tools/submit_benchmark_job.py \
  applications/dynacell/configs/benchmarks/virtual_staining/train/er/ipsc_confocal/celldiff.yml
```

Optional flags:

```bash
--dry-run
--print-script
--print-resolved-config
--override trainer.max_epochs=10
```

## Important Rule

`launcher:` is for the submit tool, not for direct `dynacell fit`.

So:

- direct `dynacell fit -c <leaf.yml>` is not the primary entrypoint
- primary entrypoint is:
  - `submit_benchmark_job.py <leaf.yml>`
- the submit tool produces the stripped resolved config and then runs
  `dynacell fit -c <resolved.yml>`

This avoids any risk that Lightning/Hydra rejects unknown top-level keys.

## Implementation Sequence

### Phase 1 Files To Create

Create the shared-axis files first, then create the ER leaves:

```text
applications/dynacell/configs/benchmarks/virtual_staining/shared/train_sets/ipsc_confocal.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/train_sets/ipsc_confocal_plus_mantis.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/targets/er_sec61b.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/targets/mito_tomm20.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/targets/nucleus.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/targets/membrane.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/model_overlays/fnet3d.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/model_overlays/unext2_scratch.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/model_overlays/unext2_fcmae.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/model_overlays/unetvit3d.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/model_overlays/celldiff.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/launcher_profiles/mode_fit.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/launcher_profiles/mode_predict.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/launcher_profiles/hardware_a6000_single.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/launcher_profiles/hardware_h100x4.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/launcher_profiles/hardware_h200_single.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/launcher_profiles/runtime_ddp.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/launcher_profiles/runtime_single_gpu.yml
applications/dynacell/configs/benchmarks/virtual_staining/shared/launcher_profiles/runtime_resume.yml

applications/dynacell/configs/benchmarks/virtual_staining/train/er/ipsc_confocal/fnet3d.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/er/ipsc_confocal/unext2_scratch.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/er/ipsc_confocal/unext2_fcmae.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/er/ipsc_confocal/unetvit3d.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/er/ipsc_confocal/celldiff.yml

applications/dynacell/configs/benchmarks/virtual_staining/train/er/ipsc_confocal_plus_mantis/fnet3d.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/er/ipsc_confocal_plus_mantis/unext2_scratch.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/er/ipsc_confocal_plus_mantis/unext2_fcmae.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/er/ipsc_confocal_plus_mantis/unetvit3d.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/er/ipsc_confocal_plus_mantis/celldiff.yml

applications/dynacell/tools/submit_benchmark_job.py
```

That is enough to cover the current Phase 1 matrix.

### Phase 2 Extension Files

After Phase 1 results select the deterministic shortlist winner, add the Phase
2 leaves by reusing the same shared files and changing only:

- target subtree: `mito/`, `nucleus/`, `membrane/`
- model leaf names: `selected_deterministic.yml`, `celldiff.yml`
- phase metadata in `benchmark.phase`

The required file patterns are:

```text
applications/dynacell/configs/benchmarks/virtual_staining/train/mito/ipsc_confocal/selected_deterministic.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/mito/ipsc_confocal/celldiff.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/mito/ipsc_confocal_plus_mantis/selected_deterministic.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/mito/ipsc_confocal_plus_mantis/celldiff.yml

applications/dynacell/configs/benchmarks/virtual_staining/train/nucleus/ipsc_confocal/selected_deterministic.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/nucleus/ipsc_confocal/celldiff.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/nucleus/ipsc_confocal_plus_mantis/selected_deterministic.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/nucleus/ipsc_confocal_plus_mantis/celldiff.yml

applications/dynacell/configs/benchmarks/virtual_staining/train/membrane/ipsc_confocal/selected_deterministic.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/membrane/ipsc_confocal/celldiff.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/membrane/ipsc_confocal_plus_mantis/selected_deterministic.yml
applications/dynacell/configs/benchmarks/virtual_staining/train/membrane/ipsc_confocal_plus_mantis/celldiff.yml
```

Only one extra naming decision is needed at that point: replace
`selected_deterministic.yml` with the actual winning model family
(`fnet3d.yml`, `unext2_scratch.yml`, `unext2_fcmae.yml`, or `unetvit3d.yml`).
