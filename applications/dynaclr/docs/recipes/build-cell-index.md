# Build a training-ready cell index

The cell index is one parquet row per cell, timepoint, and selected channel.
Build it from a collection, then preprocess it before training.

## Inputs

- A collection YAML with valid `data_path`, `tracks_path`, channels, wells, and
  acquisition timing for every experiment.
- One tracking CSV under each included FOV:
  `{tracks_path}/{row}/{column}/{fov}/*.csv`.
- AI-ready image metadata for the focus and normalization steps; see
  [AI-ready datasets](../DAGs/ai_ready_datasets.md).

## Build and preprocess

```sh
uv run dynaclr build-cell-index \
  applications/dynaclr/configs/collections/<collection>.yml \
  /path/to/collections/<collection>.parquet \
  --num-workers 8

uv run dynaclr preprocess-cell-index \
  /path/to/collections/<collection>.parquet \
  --focus-channel Phase3D
```

`preprocess-cell-index` updates the parquet in place unless `--output` is set.
It attaches focus and normalization fields and removes empty frames.

To build a subset, repeat either filter as needed:

```sh
uv run dynaclr build-cell-index collection.yml cells.parquet \
  --include-wells A/1 \
  --include-wells B/1 \
  --exclude-fovs A/1/000003
```

For the recommended Nextflow wrapper, use the
[training runbook](../DAGs/training.md).

## Validate

```python
import pandas as pd

df = pd.read_parquet("/path/to/collections/<collection>.parquet")
print(df.groupby(["experiment", "marker"]).size())
print(df[["z_focus_mean", "norm_mean", "norm_std"]].notna().all())
```

Before training, confirm that:

- every requested experiment and marker is present;
- `perturbation`, `hours_post_perturbation`, and tracking identifiers are
  populated;
- focus and normalization columns are not missing;
- excluded wells and FOVs are absent.

Rebuild after changing the collection, tracking results, channel-to-marker
mapping, or included wells.
