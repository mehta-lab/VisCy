# QC Metrics Pipeline

Composable quality control metrics for HCS OME-Zarr datasets. Results are written to `.zattrs` at both plate and position levels.

## Usage

```bash
viscy qc -c applications/qc/qc_config.yml
```

## Available Metrics

### FocusSliceMetric

Detects the in-focus z-slice per timepoint using midband spatial frequency power (GPU-batched FFT via waveorder).

**Requires:** `pip install -e ".[qc]"` (installs waveorder)

**Parameters:**

| Parameter | Description |
|---|---|
| `channel_params` | Per-channel optical parameters (see below) |
| `device` | Torch device (`cpu` or `cuda`) |
| `batch_size` | Max timepoints per GPU batch (`null` = all at once) |

**Per-channel optical parameters:**

| Parameter | Description |
|---|---|
| `NA_det` | Detection numerical aperture |
| `lambda_ill` | Illumination wavelength (same units as `pixel_size`) |
| `pixel_size` | Object-space pixel size (camera pixel size / magnification) |
| `midband_fractions` | Optional inner/outer fractions of cutoff frequency (default `[0.125, 0.25]`) |

## Configuration

```yaml
trainer:
  qc:
    data_path: /path/to/dataset.zarr
    num_workers: 4
    metrics:
      - class_path: viscy.preprocessing.focus.FocusSliceMetric
        init_args:
          device: cuda
          batch_size: 5
          channel_params:
            Phase:
              NA_det: 0.55
              lambda_ill: 0.532
              pixel_size: 0.325
            GFP:
              NA_det: 1.2
              lambda_ill: 0.488
              pixel_size: 0.103
```

Multiple metrics can be composed in the `metrics` list. Each metric specifies its own channels and parameters.

## Output Structure

### Plate-level `.zattrs`

```json
{
  "focus_slice": {
    "Phase": {
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
    "Phase": {
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

    def channels(self) -> list[str]:
        return ["Phase"]

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
    init_args: {}
```
