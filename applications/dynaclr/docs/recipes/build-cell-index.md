# Recipe: Build a Cell Index Parquet

## Goal

Pre-build a **cell index parquet** once, then point the training config at it.
The parquet contains one row per cell observation per timepoint with all
metadata already computed (lineage, conditions, HPI). Training startup drops
from minutes (opening every zarr + CSV) to a single `read_parquet` call plus
lazy zarr opens for unique FOVs only.

## Prerequisites

- DynaCLR installed (`uv pip install -e applications/dynaclr`)
- HCS OME-Zarr stores with registered data
- Per-FOV tracking CSVs (from ultrack, btrack, etc.)

## Step 1: Write the experiments YAML

Create a YAML file listing your experiments. This is the same format used by
`ExperimentRegistry.from_yaml()`:

```yaml
# experiments.yaml
experiments:
  - name: "2025_07_22_SEC61"
    data_path: "/hpc/projects/.../registered.zarr"
    tracks_path: "/hpc/projects/.../tracks"
    channel_names: ["Phase3D", "GFP", "Mito"]
    source_channel: ["Phase3D", "GFP"]
    condition_wells:
      uninfected: ["A/1", "A/2", "A/3"]
      infected: ["B/1", "B/2", "B/3"]
    interval_minutes: 30.0
    start_hpi: 3.0

  - name: "2025_08_15_TOMM20"
    data_path: "/hpc/projects/.../registered.zarr"
    tracks_path: "/hpc/projects/.../tracks"
    channel_names: ["Phase3D", "RFP"]
    source_channel: ["Phase3D", "RFP"]
    condition_wells:
      uninfected: ["A/1", "A/2"]
      infected: ["B/1", "B/2"]
      mock: ["C/1"]
    interval_minutes: 15.0
    start_hpi: 2.0
```

See `configs/cell_index/example_cell_index.yaml` for a full annotated example.

### Tracking CSV format

Each FOV needs a CSV at `{tracks_path}/{row}/{col}/{fov_idx}/tracks.csv`:

| Column | Required | Description |
|--------|----------|-------------|
| `track_id` | yes | Integer track identifier |
| `t` | yes | Timepoint index |
| `y` | yes | Centroid Y coordinate (pixels) |
| `x` | yes | Centroid X coordinate (pixels) |
| `z` | no | Z-slice index (defaults to 0) |
| `parent_track_id` | no | Parent track for lineage reconstruction |

## Step 2: Build the parquet

```bash
dynaclr build-cell-index experiments.yaml cell_index.parquet
```

Optional filters:

```bash
# Only include specific wells
dynaclr build-cell-index experiments.yaml cell_index.parquet \
    --include-wells A/1 --include-wells A/2

# Exclude problematic FOVs
dynaclr build-cell-index experiments.yaml cell_index.parquet \
    --exclude-fovs B/1/0
```

The output is a single parquet file with the canonical schema:

| Group | Columns |
|-------|---------|
| **Core** | `cell_id`, `experiment`, `store_path`, `tracks_path`, `fov`, `well`, `y`, `x`, `z`, `source_channels` |
| **Grouping** | `condition`, `channel_name` |
| **Timelapse** | `t`, `track_id`, `global_track_id`, `lineage_id`, `parent_track_id`, `hours_post_infection` |

## Step 3: Wire into training config

Add `cell_index_path` to your training YAML:

```yaml
data:
  class_path: dynaclr.data.datamodule.MultiExperimentDataModule
  init_args:
    cell_index_path: /hpc/projects/.../cell_index.parquet  # <-- add this
    experiments_yaml: /hpc/projects/.../experiments.yaml    # still required
    z_range: [15, 45]
    yx_patch_size: [384, 384]
    # ... rest of config unchanged
```

> **Note:** `experiments_yaml` is still required even with a parquet — the
> registry provides `channel_maps` (for cross-experiment channel remapping)
> and `tau_range_frames()` (for per-experiment temporal sampling) which are
> not stored in the parquet.

## Step 4: Train as usual

```bash
dynaclr fit --config multi_experiment_fit.yml
```

The same parquet works for both train and val — `MultiExperimentIndex`
automatically filters to the experiments in each registry based on the
`val_experiments` split.

## How it works

```
Without parquet (slow):
  experiments.yaml → open every zarr → read every CSV → reconstruct lineage → enrich metadata

With parquet (fast):
  cell_index.parquet → read_parquet → filter by registry → open unique zarr/FOV pairs only
```

## Tips

- **Rebuild when data changes.** If you add FOVs, re-track, or change
  condition assignments, rebuild the parquet.
- **One parquet for all splits.** Train/val filtering happens at runtime
  based on the registry, so you only need one parquet per experiment set.
- **Inspect the parquet** with pandas:
  ```python
  from viscy_data.cell_index import read_cell_index
  df = read_cell_index("cell_index.parquet")
  print(df.shape, df["experiment"].value_counts())
  ```
