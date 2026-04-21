# Dynacell Benchmark Config Schema

This document defines the target benchmark-config layout for
`applications/dynacell/configs/benchmarks/virtual_staining/`.

It replaces the older split where:

- train and predict leaves lived under `configs/benchmarks/virtual_staining/`
- eval leaves and eval target groups lived in a separate top-level
  `configs/evaluation/` tree

That split is not worth keeping. The benchmark unit is simple:

- choose an organelle
- train one or more models
- run prediction on a test set
- evaluate the resulting prediction store

The filesystem layout should follow that workflow directly.

## Goals

- keep one canonical internal tree for the full virtual-staining workflow
- make train, predict, and eval leaves sit next to each other
- keep reusable public schema in the installed `dynacell` package
- keep HPC-specific benchmark instances out of `src/`
- make every canonical benchmark run addressable by
  `(organelle, train_set, model, stage, predict_set)`

## Design Rules

1. `configs/recipes/` stays public and reusable.
2. `configs/examples/` stays small and generic.
3. `configs/benchmarks/virtual_staining/` is the canonical internal benchmark tree.
4. Train, predict, and eval leaves for the same benchmark live under the same
   `(organelle, train_set, model)` directory.
5. Eval using Hydra is an implementation detail. It must not create a second
   top-level config universe.
6. Only path-free eval schema ships inside
   `src/dynacell/evaluation/_configs/`.
7. Any config with HPC paths, internal checkpoints, or canonical benchmark
   run bindings lives under `configs/benchmarks/virtual_staining/`.

## Public vs Internal Split

### In-package, path-free eval schema

These files stay under
`applications/dynacell/src/dynacell/evaluation/_configs/`:

- `eval.yaml`
- `precompute.yaml`
- `predict_set/ipsc_confocal.yaml`
- `feature_extractor/dinov3/lvd1689m.yaml`
- `spectral_pcc/*.yaml`

These files define schema, path-free defaults, or diagnostic-tool defaults that
can ship to external users.

### Internal benchmark instances

These files live under
`applications/dynacell/configs/benchmarks/virtual_staining/`:

- train leaves
- predict leaves
- eval leaves
- eval targets with GT paths and channel bindings
- internal DynaCLR checkpoint config
- launcher metadata
- benchmark output roots and save dirs

These files are for repo users on our infrastructure. They are not part of the
public package contract.

## Target Tree

```text
applications/dynacell/
  configs/
    recipes/
    examples/
    benchmarks/
      BENCHMARK_CONFIG_SCHEMA.md
      UNEXT2_VS_FCMAE_CLASSES.md
      virtual_staining/
        README.md
        shared/
          model/
            train_sets/
              ipsc_confocal.yml
            predict_sets/
              ipsc_confocal.yml
            targets/
              er_sec61b.yml
              mito_tomm20.yml
              membrane.yml
              nucleus.yml
            model_overlays/
              celldiff_fit.yml
              celldiff_predict.yml
              fcmae_vscyto3d_fit.yml
              fnet3d_paper_fit.yml
              unetvit3d_fit.yml
              unetvit3d_predict.yml
              unext2_fit.yml
            launcher_profiles/
              mode_fit.yml
              mode_predict.yml
              hardware_4gpu.yml
              hardware_gpu_any_long.yml
              hardware_h200_single.yml
              runtime_shared.yml
          eval/
            target/
              er_sec61b.yml
              mito_tomm20.yml
              membrane.yml
              nucleus.yml
            feature_extractor/
              dynaclr/
                default.yml
        er/
          ipsc_confocal/
            celldiff/
              train.yml
              predict/
                ipsc_confocal.yml
              eval/
                ipsc_confocal.yml
            fcmae_vscyto3d_pretrained/
              train.yml
            fcmae_vscyto3d_scratch/
              train.yml
            fnet3d_paper/
              train.yml
            unetvit3d/
              train.yml
              predict/
                ipsc_confocal.yml
              eval/
                ipsc_confocal.yml
            unext2/
              train.yml
        membrane/
          ipsc_confocal/
            celldiff/
              train.yml
              predict/
                ipsc_confocal.yml
              eval/
                ipsc_confocal.yml
            fnet3d_paper/
              train.yml
            unetvit3d/
              train.yml
              predict/
                ipsc_confocal.yml
              eval/
                ipsc_confocal.yml
        mito/
          ipsc_confocal/
            ...
        nucleus/
          ipsc_confocal/
            ...
```

## Ownership by Subtree

### `shared/model/`

Owns internal benchmark building blocks used by Lightning train and predict
leaves:

- `train_sets/`: imaging modality and training data defaults
- `predict_sets/`: imaging modality and prediction-domain defaults
- `targets/`: target-channel choices, train data paths, normalizations, and
  target-specific augmentations
- `model_overlays/`: model-family defaults
- `launcher_profiles/`: launcher mode, hardware, and runtime policy

### `shared/eval/`

Owns internal benchmark building blocks used only by eval:

- `target/`: GT paths, segmentation paths, GT and prediction channel names
- `feature_extractor/dynaclr/`: internal DynaCLR checkpoint and encoder config

### `<org>/<train_set>/<model>/`

Owns canonical runnable leaves for one benchmark cell:

- `train.yml`
- `predict/<predict_set>.yml`
- `eval/<predict_set>.yml`

That directory is the canonical place to inspect one benchmark configuration.

## Leaf Addressing

### Train

Path:

```text
configs/benchmarks/virtual_staining/<org>/<train_set>/<model>/train.yml
```

Example:

```text
applications/dynacell/configs/benchmarks/virtual_staining/er/ipsc_confocal/celldiff/train.yml
```

Invocation:

```bash
uv run dynacell fit -c applications/dynacell/configs/benchmarks/virtual_staining/er/ipsc_confocal/celldiff/train.yml
```

### Predict

Path:

```text
configs/benchmarks/virtual_staining/<org>/<train_set>/<model>/predict/<predict_set>.yml
```

Example:

```text
applications/dynacell/configs/benchmarks/virtual_staining/er/ipsc_confocal/celldiff/predict/ipsc_confocal.yml
```

Invocation:

```bash
uv run dynacell predict -c applications/dynacell/configs/benchmarks/virtual_staining/er/ipsc_confocal/celldiff/predict/ipsc_confocal.yml
```

### Eval

Path:

```text
configs/benchmarks/virtual_staining/<org>/<train_set>/<model>/eval/<predict_set>.yml
```

Example:

```text
applications/dynacell/configs/benchmarks/virtual_staining/er/ipsc_confocal/celldiff/eval/ipsc_confocal.yml
```

Hydra selector:

```text
leaf=<org>/<train_set>/<model>/eval/<predict_set>
```

Invocation:

```bash
uv run dynacell evaluate leaf=er/ipsc_confocal/celldiff/eval/ipsc_confocal
```

The eval selector is named `leaf`, not `benchmark`. The benchmark already has a
natural filesystem location; the selector should point at that leaf directly.

## Composition Rules

### Train leaf

Train leaves continue to compose through the existing `viscy_utils.compose`
mechanism.

Canonical shape:

```yaml
base:
  - ../../../shared/model/train_sets/<train_set>.yml
  - ../../../shared/model/targets/<target>.yml
  - ../../../shared/model/model_overlays/<model>_fit.yml
  - ../../../shared/model/launcher_profiles/mode_fit.yml
  - ../../../shared/model/launcher_profiles/hardware_<hw>.yml
  - ../../../shared/model/launcher_profiles/runtime_shared.yml
```

### Predict leaf

Canonical shape:

```yaml
base:
  - ../../../../shared/model/predict_sets/<predict_set>.yml
  - ../../../../shared/model/targets/<target>.yml
  - ../../../../shared/model/model_overlays/<model>_predict.yml
  - ../../../../shared/model/launcher_profiles/mode_predict.yml
  - ../../../../shared/model/launcher_profiles/hardware_<hw>.yml
  - ../../../../shared/model/launcher_profiles/runtime_shared.yml
```

### Eval leaf

Eval leaves compose through Hydra, not `viscy_utils.compose`.

Canonical shape:

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

Hydra resolves:

- `target` from `shared/eval/target/`
- `feature_extractor/dynaclr` from `shared/eval/feature_extractor/dynaclr/`
- `predict_set` from the in-package public group
- `leaf` from the benchmark tree itself

## Hydra Search-Path Contract

The eval runtime uses:

- packaged schema under `src/dynacell/evaluation/_configs/`
- repo-local benchmark groups under:
  - `configs/benchmarks/virtual_staining/shared/eval/`
  - `configs/benchmarks/virtual_staining/`

`dynacell.__main__` injects those two roots through
`hydra.searchpath=[file://...]` when running from a repo checkout.

Hydra group resolution requires a physical `leaf/` directory in the
searchpath so that `leaf=<org>/<train_set>/<model>/eval/<predset>`
resolves to `<searchpath>/leaf/<path>.yaml`. The canonical eval leaves
live next to their train/predict siblings at
`<org>/<train_set>/<model>/eval/<predset>.yaml`; a parallel symlink tree
under `virtual_staining/leaf/` mirrors the benchmark structure and
points each symlink back at the canonical file, so the schema's
"one directory per benchmark" goal stays intact.

Wheel installs do not see those internal benchmark groups. External users get:

- `eval.yaml`
- `precompute.yaml`
- path-free groups such as `predict_set/ipsc_confocal`
- the ability to provide their own groups via `--config-dir`

## Exact Migration From Current Tree

### Shared model files

Move:

- `shared/train_sets/*` -> `shared/model/train_sets/*`
- `shared/predict_sets/*` -> `shared/model/predict_sets/*`
- `shared/targets/*` -> `shared/model/targets/*`
- `shared/model_overlays/*` -> `shared/model/model_overlays/*`
- `shared/launcher_profiles/*` -> `shared/model/launcher_profiles/*`

### Eval shared files

Move:

- `configs/evaluation/target/er_sec61b.yaml` -> `shared/eval/target/er_sec61b.yml`
- `configs/evaluation/target/mito_tomm20.yaml` -> `shared/eval/target/mito_tomm20.yml`
- `configs/evaluation/target/membrane.yaml` -> `shared/eval/target/membrane.yml`
- `configs/evaluation/target/nucleus.yaml` -> `shared/eval/target/nucleus.yml`
- `configs/evaluation/feature_extractor/dynaclr/default.yaml` ->
  `shared/eval/feature_extractor/dynaclr/default.yml`

### Train leaves

Move:

- `train/er/ipsc_confocal/celldiff.yml` -> `er/ipsc_confocal/celldiff/train.yml`
- `train/er/ipsc_confocal/fcmae_vscyto3d_pretrained.yml` ->
  `er/ipsc_confocal/fcmae_vscyto3d_pretrained/train.yml`
- `train/er/ipsc_confocal/fcmae_vscyto3d_scratch.yml` ->
  `er/ipsc_confocal/fcmae_vscyto3d_scratch/train.yml`
- `train/er/ipsc_confocal/fnet3d_paper.yml` ->
  `er/ipsc_confocal/fnet3d_paper/train.yml`
- `train/er/ipsc_confocal/unetvit3d.yml` ->
  `er/ipsc_confocal/unetvit3d/train.yml`
- `train/er/ipsc_confocal/unext2.yml` ->
  `er/ipsc_confocal/unext2/train.yml`

Apply the same pattern for `membrane`, `mito`, and `nucleus`.

### Predict leaves

Move:

- `predict/er/ipsc_confocal/celldiff/ipsc_confocal.yml` ->
  `er/ipsc_confocal/celldiff/predict/ipsc_confocal.yml`
- `predict/er/ipsc_confocal/unetvit3d/ipsc_confocal.yml` ->
  `er/ipsc_confocal/unetvit3d/predict/ipsc_confocal.yml`

Apply the same pattern for `membrane`, `mito`, and `nucleus`.

### Eval leaves

Move:

- `configs/evaluation/benchmark/er/ipsc_confocal/celldiff/ipsc_confocal.yaml` ->
  `er/ipsc_confocal/celldiff/eval/ipsc_confocal.yml`
- `configs/evaluation/benchmark/er/ipsc_confocal/unetvit3d/ipsc_confocal.yaml` ->
  `er/ipsc_confocal/unetvit3d/eval/ipsc_confocal.yml`

Apply the same pattern for `membrane`, `mito`, and `nucleus`.

### Cleanup

After the move:

- delete top-level `configs/evaluation/`
- update all relative `base:` paths in moved train/predict leaves
- rename Hydra eval selector from `benchmark` to `leaf`

## Files That Must Change With The Migration

### Code

- `applications/dynacell/src/dynacell/__main__.py`
- `applications/dynacell/src/dynacell/evaluation/_configs/eval.yaml`

### Docs

- `applications/dynacell/README.md`
- `applications/dynacell/configs/benchmarks/virtual_staining/README.md`
- `applications/dynacell/src/dynacell/evaluation/README.md`
- `applications/dynacell/configs/benchmarks/BENCHMARK_CONFIG_SCHEMA.md`

### Tooling

- `applications/dynacell/tools/submit_benchmark_job.py`

### Tests

- `applications/dynacell/tests/test_benchmark_config_composition.py`
- `applications/dynacell/tests/test_cli_routing.py`
- `applications/dynacell/tests/test_submit_benchmark_job.py`

## Verification

### Train compose

```bash
uv run dynacell fit -c applications/dynacell/configs/benchmarks/virtual_staining/er/ipsc_confocal/celldiff/train.yml --print_config
```

### Predict compose

```bash
uv run dynacell predict -c applications/dynacell/configs/benchmarks/virtual_staining/er/ipsc_confocal/celldiff/predict/ipsc_confocal.yml --print_config
```

### Eval compose

```bash
uv run dynacell evaluate -c job leaf=er/ipsc_confocal/celldiff/eval/ipsc_confocal
```

### Eval smoke

```bash
uv run dynacell evaluate \
  leaf=er/ipsc_confocal/celldiff/eval/ipsc_confocal \
  limit_positions=1 compute_feature_metrics=false \
  save.save_dir=/tmp/eval_leaf_smoke_$(date +%s)
```

### Targeted tests

```bash
uv run pytest \
  applications/dynacell/tests/test_benchmark_config_composition.py \
  applications/dynacell/tests/test_cli_routing.py \
  applications/dynacell/tests/test_submit_benchmark_job.py -q
```

### Full dynacell suite

```bash
uv run pytest applications/dynacell -q
```

## Non-Goals

- Do not change public `recipes/` or `examples/` layout here.
- Do not ship internal HPC benchmark groups in the wheel.
- Do not keep `configs/evaluation/` as a second internal root.
- Do not preserve the old `benchmark=` eval selector only for compatibility.

## Rationale

This schema makes one benchmark inspectable in one place.

For any organelle, train set, and model, the user can open one directory and
see:

- how the model is trained
- how it predicts on the test set
- how those predictions are evaluated

That is the correct filesystem model for dynacell benchmarking.
