# Extract per-marker embeddings

Use `dynaclr predict-triplet` for collection-driven inference. It maps physical
channels to marker names and writes one AnnData zarr per experiment-marker pair.

## Required inputs

- A collection YAML under
  [`applications/dynaclr/configs/collections/`](../../configs/collections/).
- An AI-ready image zarr and tracking store for each experiment.
- A checkpoint compatible with the selected model family.
- Model, run, and checkpoint names used to construct the output path.

The collection must include `data_path`, `tracks_path`, `channels`, and
`perturbation_wells` for every experiment. Use `channels[].wells` when the same
physical channel represents different markers in different wells.

## Run

```sh
uv run dynaclr predict-triplet \
  -c applications/dynaclr/configs/collections/<collection>.yml \
  --checkpoint /path/to/checkpoint.ckpt \
  --model-family <model-family> \
  --run <run-name> \
  --ckpt-name <checkpoint-name> \
  --datasets-root /path/to/datasets-root \
  --z-range 15 45 \
  --z-reduction mip \
  --reference-pixel-size 0.1494 \
  --yx-patch-size 160 160 \
  --batch-size 32 \
  --num-workers 0
```

Use `--markers SEC61B,TOMM20` to select markers or `--no-labelfree` to skip
label-free channels. Keep `--num-workers 0` for zarr-backed prediction.

The model's training config determines the correct spatial size,
normalization, z reduction, and pixel size. Do not guess these values from the
inference dataset.

## Validate

Outputs follow:

```text
<dataset>/2-phenotyping/predictions/<model-family>/<run>/<checkpoint>/<marker>.zarr
```

Inspect each output:

```sh
uv run dynaclr info /path/to/<marker>.zarr
```

Confirm the row count, feature count, marker, checkpoint provenance, and cell
identifiers. Treat different checkpoints as different feature spaces.

See [per-marker inference](../DAGs/inference_triplet.md) for the workflow visual,
batch prediction, and complete option reference. Use
[evaluation](../DAGs/evaluation.md) when prediction should start from a
cell-index parquet and continue directly into evaluation.
