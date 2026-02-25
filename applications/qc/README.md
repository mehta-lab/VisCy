# QC Metrics

Composable quality-control metrics for HCS OME-Zarr datasets.

## Available Metrics

### Focus Slice Detection

Detects the in-focus z-slice per timepoint using midband spatial frequency power via [waveorder](https://github.com/mehta-lab/waveorder).

Results are written to `.zattrs` at both plate and position levels under the `focus_slice` field.

## Usage

```bash
# Install (from repo root)
uv sync

# Run QC metrics
qc run -c applications/qc/qc_config.yml
```

## Configuration

See `qc_config.yml` for an example. Key fields:

```yaml
data_path: /path/to/dataset.zarr
num_workers: 4

focus_slice:
  channel_names:
    - Phase
  NA_det: 0.55
  lambda_ill: 0.532
  pixel_size: 0.325
  midband_fractions:
    - 0.125
    - 0.25
  device: cpu
```

## Adding New Metrics

1. Subclass `QCMetric` from `qc.qc_metrics`
2. Implement `field_name`, `channels()`, and `__call__()`
3. Add a Pydantic config model in `config.py`
4. Wire it into `cli.py`
