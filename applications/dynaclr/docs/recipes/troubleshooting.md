# Troubleshoot DynaCLR workflows

Start with the smallest command that resolves configuration without launching a
full job.

| Stage | Check |
| --- | --- |
| Cell index | Run `build-cell-index` with `--num-workers 1`, then `preprocess-cell-index`. |
| Training | Run `dynaclr fit ... --trainer.fast_dev_run=true`. |
| Evaluation | Run `dynaclr prepare-eval-configs -c <config>`. |
| Embeddings | Run `dynaclr info <embeddings.zarr>`. |

## Data and index errors

### Missing or multiple tracking CSVs

Every included image FOV must have exactly one CSV at
`{tracks_path}/{row}/{column}/{fov}/*.csv`. The minimum columns are
`track_id`, `t`, `y`, and `x`.

### Channel not found in zarr

`channels[].name` in the collection must exactly match the OME-Zarr channel
metadata. `channels[].marker` is a semantic output label and does not need to
match the physical name.

### Missing focus or normalization fields

Run the [AI-ready dataset workflow](../DAGs/ai_ready_datasets.md), rebuild the
cell index, and rerun `preprocess-cell-index`. Training should consume the
preprocessed parquet, not the raw index.

### Very few valid anchors

Check track length, lineage identifiers, imaging interval, and `tau_range`.
Temporal lookup requires another observation in the same positive group within
the per-experiment frame range.

## Training errors

### Tensor channel or shape mismatch

Make `model.init_args.encoder.init_args.in_channels` agree with
`data.init_args.channels_per_sample`. Copy crop, z-reduction, and stack-depth
settings from a maintained config with the same model dimensionality.

### GPU out of memory

Reduce `batch_size` first, then patch size or z extraction depth. Confirm mixed
precision is enabled when supported. Re-run the fast development check after
each change.

### DDP stalls

Use a maintained topology recipe and SLURM wrapper. Confirm all ranks receive
the same number of batches and that a custom sampler is not being replaced by
Lightning's distributed sampler.

## Prediction and evaluation errors

### Prediction hangs while opening zarr

Use `--num-workers 0` with `dynaclr predict-triplet`.

### Embeddings are not comparable

Verify checkpoint, feature key, normalization, patch size, z handling, and
pixel-size rescaling. Outputs from different checkpoints or representation keys
are different feature spaces.

### Annotation join produces missing labels

Normalize `fov_name` values and provide either `(fov_name, id)` or
`(fov_name, t, track_id)` in both annotations and embedding `obs`. Duplicate
keys also require `y` and `x` for spatial disambiguation. See the
[annotation contract](../linear_classifiers/annotations_and_linear_classifiers.md).
