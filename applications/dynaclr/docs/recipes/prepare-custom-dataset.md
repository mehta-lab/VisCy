# Recipe: Prepare a Custom Dataset for DynaCLR

## Goal

Format time-lapse microscopy data (TIFFs, ND2, etc.) for DynaCLR training
or inference.

## What DynaCLR expects

Two inputs per experiment:

1. **HCS OME-Zarr store** — image data in `TCZYX` axis order, organized as
   `{row}/{col}/{fov}/0` (plate/well/position layout)
2. **Tracking CSVs** — one CSV per FOV with cell centroid coordinates and
   track IDs, at `{tracks_root}/{row}/{col}/{fov}/tracks.csv`

## Step 1: Convert images to HCS OME-Zarr

Use [iohub](https://github.com/czbiohub-sf/iohub) to convert your data:

```python
from iohub.ngff import open_ome_zarr
import numpy as np

channel_names = ["Phase3D", "GFP"]

with open_ome_zarr("my_experiment.zarr", layout="hcs", mode="w",
                    channel_names=channel_names) as plate:
    # Create positions (row, col, fov_index)
    pos = plate.create_position("A", "1", "0")

    # Write image data: shape = (T, C, Z, Y, X)
    pos.create_zeros("0", shape=(100, 2, 30, 2048, 2048), dtype=np.float32)

    # Fill with your data
    pos["0"][:] = your_image_array  # shape must match
```

**Resulting layout:**
```
my_experiment.zarr/
  A/
    1/
      0/          # FOV
        0/        # multiscale level 0 (primary data)
      1/          # another FOV in same well
  B/
    1/
      0/
```

**Key constraints:**
- All positions must have the same channel names and count
- Axis order is always `TCZYX`
- Channel names must match what you put in `experiments.yml`

## Step 2: Generate tracking CSVs

DynaCLR needs per-FOV tracking CSVs with cell centroids. You can generate
these from a cell tracker (ultrack, btrack) or from segmentation masks.

### Required CSV columns

| Column | Type | Description |
|--------|------|-------------|
| `track_id` | int | Unique cell track identifier (per FOV) |
| `t` | int | Timepoint index |
| `y` | float | Centroid Y coordinate in pixels |
| `x` | float | Centroid X coordinate in pixels |

### Optional CSV columns

| Column | Type | Description |
|--------|------|-------------|
| `z` | int | Z-slice index (defaults to 0) |
| `parent_track_id` | int | Parent track ID for cell division lineage |
| `id` | int | Unique observation ID |

### Example CSV

```csv
track_id,t,y,x,parent_track_id
0,0,128.5,256.3,
0,1,130.2,255.8,
0,2,131.0,254.1,
1,5,200.1,100.4,0
1,6,201.3,101.2,0
```

### Pseudo-tracking from segmentation

If you have segmentation masks but no tracker, extract centroids directly:

```python
import numpy as np
import pandas as pd

def extract_centroids(seg_mask, timepoint):
    """Extract cell centroids from a 2D segmentation mask."""
    rows = []
    for label_id in np.unique(seg_mask):
        if label_id == 0:
            continue  # skip background
        ys, xs = np.where(seg_mask == label_id)
        rows.append({
            "track_id": int(label_id),
            "t": timepoint,
            "y": float(ys.mean()),
            "x": float(xs.mean()),
        })
    return pd.DataFrame(rows)
```

See `examples/data_preparation/classical_sampling/` for a full working example.

### File layout

Place CSVs to mirror the zarr FOV structure:

```
tracks/
  A/
    1/
      0/
        tracks.csv    # matches FOV A/1/0
      1/
        tracks.csv    # matches FOV A/1/1
  B/
    1/
      0/
        tracks.csv
```

## Step 3: Write the experiments YAML

```yaml
experiments:
  - name: "my_experiment"
    data_path: "/path/to/my_experiment.zarr"
    tracks_path: "/path/to/tracks"
    channel_names: ["Phase3D", "GFP"]
    source_channel: ["Phase3D", "GFP"]
    perturbation_wells:
      control: ["A/1"]
      treated: ["B/1"]
    interval_minutes: 30.0
    start_hpi: 0.0
```

## Step 4: Validate

Quick sanity check that everything loads:

```python
from dynaclr.data.experiment import ExperimentRegistry

registry = ExperimentRegistry.from_collection("experiments.yml")
print(f"{len(registry.experiments)} experiments validated")
for exp in registry.experiments:
    print(f"  {exp.name}: {[ch.marker for ch in exp.channels]}")
```

`ExperimentRegistry` will raise clear errors if:
- `data_path` doesn't exist
- `perturbation_wells` is empty
- `interval_minutes` is not positive
- `data_path` doesn't exist
- `perturbation_wells` is empty

## Step 5: (Optional) Build cell index parquet

For faster training startup, pre-build the cell index:

```bash
dynaclr build-cell-index experiments.yml cell_index.parquet
```

See `build-cell-index.md` for details.

## Common issues

**"No tracking CSV in ..., skipping"** — CSV file is missing or not in the
expected directory structure. Check that the path is
`{tracks_path}/{row}/{col}/{fov}/something.csv`.

**"channel_names mismatch"** — The `channel_names` in your YAML doesn't
match what's actually in the zarr. Open the zarr and check:
```python
from iohub.ngff import open_ome_zarr
plate = open_ome_zarr("my_experiment.zarr", mode="r")
pos = next(iter(plate.positions()))[1]
print(pos.channel_names)
```

**Cells at image borders** — DynaCLR clamps centroids inward (not excluded)
so border cells still contribute to training. Cells with coordinates
completely outside the image boundary (e.g., `y < 0`) are dropped.
