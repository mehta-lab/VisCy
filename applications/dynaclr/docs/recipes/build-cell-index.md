# Recipe: Build a Cell Index Parquet

## Goal

Pre-build a **cell index parquet** once, then point the training config at it.
The parquet contains one row per cell observation per timepoint with all
metadata already computed (lineage, conditions, HPI). Training startup drops
from minutes (opening every zarr + reading every CSV) to a single
`read_parquet` call.

## Prerequisites

- DynaCLR installed (`uv pip install -e applications/dynaclr`)
- A collection YAML (see `train-multi-experiment.md` Step 1)

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
  collection.yml → open every zarr → read every tracking CSV
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
