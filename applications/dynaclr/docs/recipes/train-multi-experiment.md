# Train across multiple experiments

Train one model from a preprocessed cell-index parquet and a composed Lightning
config. The cell index, rather than the training YAML, is the source of dataset
and channel metadata.

## 1. Prepare the cell index

Create a collection YAML, then build and preprocess its parquet:

```sh
uv run dynaclr build-cell-index \
  applications/dynaclr/configs/collections/<collection>.yml \
  /path/to/collections/<collection>.parquet \
  --num-workers 8

uv run dynaclr preprocess-cell-index \
  /path/to/collections/<collection>.parquet \
  --focus-channel Phase3D
```

See [Prepare a custom dataset](prepare-custom-dataset.md) for the input contract
and [Build a cell index](build-cell-index.md) for validation.

## 2. Create a training leaf

Start from the closest maintained config under
[`applications/dynaclr/configs/training/`](../../configs/training/). Reuse the
trainer, topology, and model recipes instead of copying their contents:

```yaml
base:
  - ../recipes/trainer/fit.yml
  - ../recipes/topology/ddp_2gpu.yml
  - ../recipes/model/contrastive_encoder_convnext_tiny.yml

trainer:
  precision: bf16-mixed
  max_epochs: 150

model:
  init_args:
    encoder:
      init_args:
        in_channels: 1

data:
  class_path: dynaclr.data.datamodule.MultiExperimentDataModule
  init_args:
    cell_index_path: /path/to/collections/<collection>.parquet
    channels_per_sample: 1
    batch_group_by: [experiment]
    stratify_by: [perturbation, marker]
    positive_cell_source: lookup
    positive_match_columns: [lineage_id]
    positive_channel_source: same
    tau_range: [0.5, 2.0]
    batch_size: 256
    num_workers: 4
```

The full leaf must also define compatible crop sizes, z handling,
normalization, and augmentations. Copy those settings from a maintained model
with the same dimensionality and input type. Use
[sampling settings](sampling-strategies.md) to adjust only the sampler and
positive-pair fields.

## 3. Smoke test

```sh
uv run dynaclr fit -c /path/to/training.yml \
  --trainer.fast_dev_run=true
```

This should complete one training and validation batch. Resolve missing parquet
columns, invalid channel counts, and tensor-shape errors before submission.

## 4. Train

Interactive or allocated node:

```sh
uv run dynaclr fit -c /path/to/training.yml
```

SLURM:

```sh
sbatch applications/dynaclr/configs/training/<model>.sh
```

Keep the collection YAML, preprocessed parquet, resolved training config, and
checkpoint together as the definition of the learned feature space. The
[training runbook](../DAGs/training.md) contains the Nextflow preprocessing
entry, resume convention, workflow visual, and output checklist.
