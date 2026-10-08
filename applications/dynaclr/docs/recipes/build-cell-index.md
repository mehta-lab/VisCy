# Recipe: Build a Cell Index Parquet

## Goal

Pre-build a **cell index parquet** once, then point the training config at it.
The parquet contains one row per cell observation per timepoint with all
metadata already computed (lineage, conditions, HPI). Training startup drops
from minutes (opening every zarr + reading every FOV's tracks) to a single
`read_parquet` call.

## Prerequisites

- DynaCLR installed (`uv pip install -e applications/dynaclr`)
- A collection YAML (see `train-multi-experiment.md` Step 1)
- Per-FOV tracks for every experiment (see "Where tracks are read from" below)

## Where tracks are read from

Each experiment's `tracks_path` is a root holding one directory per FOV,
`<tracks_path>/<row>/<col>/<fov>/`. In that directory DynaCLR reads:

1. `tracks.geff` if present — the GEFF track graph written by `biahub track`
   (needs the `geff` package, installed with DynaCLR via `viscy-data[tracks]`);
2. otherwise exactly one `*.csv` (the ultrack tracks table).

Both give the same table: `track_id, t, [z], y, x, id, parent_track_id, parent_id`.

`tracks_path` is optional and defaults to `data_path`. Datasets tracked with
current `biahub track` store the tracks inside the image plate
(`<plate>/<row>/<col>/<fov>/tracks.geff`), so omit `tracks_path`:

```yaml
experiments:
  - name: 2026_10_01_A549_H2B_CAAX
    data_path: ${datasets_root}/2026_10_01_A549_H2B_CAAX/2026_10_01_A549_H2B_CAAX.zarr
    # tracks_path defaults to data_path (tracks.geff inside each FOV)
```

Older datasets keep their tracks in a separate tracking zarr; keep setting
`tracks_path` for those:

```yaml
    tracks_path: ${datasets_root}/2025_01_28_A549_G3BP1_ZIKV_DENV/tracking.zarr
```

## Step 1: Build the parquet

```bash
dynaclr build-cell-index my_collection.yml cell_index.parquet
```

You'll see per-experiment progress in the logs:

```
INFO: Building cell index for experiment: 2025_01_28_A549_G3BP1_ZIKV_DENV
INFO: Building cell index for experiment: 2025_07_24_SEC61_TOMM20_G3BP1
INFO: Cell index built: 42 FOVs across 2 experiments
```

Optional filters:

```bash
# Only include specific wells
dynaclr build-cell-index my_collection.yml cell_index.parquet \
    --include-wells A/1 --include-wells A/2

# Exclude problematic FOVs
dynaclr build-cell-index my_collection.yml cell_index.parquet \
    --exclude-fovs B/1/0
```

## Step 2: Inspect the parquet

```python
import pandas as pd
df = pd.read_parquet("cell_index.parquet")
print(df["experiment"].value_counts())
print(df["condition"].value_counts())
print(df.shape)
```

## Step 3: Wire into training config

```yaml
data:
  class_path: dynaclr.data.datamodule.MultiExperimentDataModule
  init_args:
    collection_path: /path/to/my_collection.yml
    cell_index_path: /path/to/cell_index.parquet  # <-- add this
    z_window: 30
    # ... rest of config unchanged
```

> **Note:** When `cell_index_path` is provided, `collection_path` is optional.
> The registry can be built directly from the parquet + zarr metadata via
> `ExperimentRegistry.from_cell_index()`. If `collection_path` is also
> provided, it takes precedence.

## How it works

```
Without parquet (slow — minutes):
  collection.yml → open every zarr → read every FOV's tracks (GEFF or CSV)
               → reconstruct lineage → enrich metadata

With parquet (fast — seconds):
  cell_index.parquet → read_parquet → open only the unique zarr/FOV pairs needed
```

## Tips

- **Rebuild when data changes.** If you add experiments, re-track, or change
  condition assignments, rebuild the parquet.
- **One parquet per collection.** Train/val filtering happens at runtime based
  on `val_experiments`, so one parquet covers all splits.
- **Store it with the collection.** Keep the parquet next to the collection YAML
  in `configs/cell_index/` for reproducibility. Collection YAMLs live in `configs/collections/`.
