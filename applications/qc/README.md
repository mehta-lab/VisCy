# QC Metrics Pipeline

Composable quality control metrics for HCS OME-Zarr datasets. Results are written to `.zattrs` at both plate and position levels.

## Installation

```bash
pip install -e ".[qc]"
```

This installs `waveorder` as an optional dependency for focus detection.

## Usage

```bash
viscy qc -c applications/qc/qc_config.yml
```

## Available Metrics

### FocusSliceMetric

Detects the in-focus z-slice per timepoint using midband spatial frequency power (GPU-batched FFT via waveorder).

**Parameters:**

| Parameter | Description |
|---|---|
| `NA_det` | Detection numerical aperture |
| `lambda_ill` | Illumination wavelength (same units as `pixel_size`) |
| `pixel_size` | Object-space pixel size (camera pixel size / magnification) |
| `channel_names` | List of channel names, or `-1` for all channels in the dataset |
| `midband_fractions` | Inner/outer fractions of cutoff frequency (default `[0.125, 0.25]`) |
| `device` | Torch device (`cpu` or `cuda`) |
| `batch_size` | Max timepoints per GPU batch (auto-calculated from GPU memory when omitted) |

## Configuration

```yaml
data_path: /path/to/dataset.zarr
num_workers: 4
metrics:
  - class_path: viscy.preprocessing.focus.FocusSliceMetric
    init_args:
      NA_det: 1.35
      lambda_ill: 0.450
      pixel_size: 0.1494
      channel_names:
        - Phase3D
        - GFP
      device: cuda
```

Use `channel_names: -1` to run on all channels:

```yaml
metrics:
  - class_path: viscy.preprocessing.focus.FocusSliceMetric
    init_args:
      NA_det: 1.35
      lambda_ill: 0.450
      pixel_size: 0.1494
      channel_names: -1
      device: cuda
```

Multiple metrics can be composed in the `metrics` list.

## Output Structure

### Plate-level `.zattrs`

```json
{
  "focus_slice": {
    "Phase3D": {
      "dataset_statistics": {
        "z_focus_mean": 5.3,
        "z_focus_std": 1.2,
        "z_focus_min": 3,
        "z_focus_max": 8
      }
    }
  }
}
```

### Position-level `.zattrs`

```json
{
  "focus_slice": {
    "Phase3D": {
      "dataset_statistics": {"z_focus_mean": 5.3, "z_focus_std": 1.2, "z_focus_min": 3, "z_focus_max": 8},
      "fov_statistics": {"z_focus_mean": 5.5, "z_focus_std": 0.7},
      "per_timepoint": {"0": 5, "1": 6, "2": 5}
    }
  }
}
```

## Inspecting Results

```python
from iohub import open_ome_zarr

ds = open_ome_zarr("/path/to/dataset.zarr", mode="r")
print(ds.zattrs["focus_slice"])

for name, pos in ds.positions():
    print(name, pos.zattrs["focus_slice"])
    break
```

## Adding Custom Metrics

Subclass `QCMetric` and implement `channels()` and `__call__()`:

```python
from viscy.preprocessing.qc_metrics import QCMetric

class MyMetric(QCMetric):
    field_name = "my_metric"

    def __init__(self, channel_names, ...):
        self.channel_names = channel_names

    def channels(self):
        return self.channel_names  # list[str] or -1 for all

    def __call__(self, position, channel_name, channel_index, num_workers=4):
        # compute metric per FOV
        return {
            "fov_statistics": {"key": value},
            "per_timepoint": {"0": value, "1": value},
        }
```

Then add it to the config:

```yaml
metrics:
  - class_path: my_module.MyMetric
    init_args:
      channel_names: -1
```
