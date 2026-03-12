# QC Metrics

Composable quality-control metrics for HCS OME-Zarr datasets.

## Available Functions

| Function | Config key | Description | Output location |
|----------|------------|-------------|-----------------|
| Focus slice detection | `focus_slice` | Detects the in-focus z-slice per timepoint using midband spatial frequency power via [waveorder](https://github.com/mehta-lab/waveorder) | `.zattrs["focus_slice"]` (plate + position) |
| Metadata annotation | `annotation` | Writes `channel_annotation` and `experiment_metadata` to `.zattrs` from a YAML config. The schema is defined in the [Airtable README](../airtable/README.md#unified-zattrs-schema). | `.zattrs["channel_annotation"]` (plate + position), `.zattrs["experiment_metadata"]` (position) |



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

annotation:
  channel_annotation:
    Phase3D:
      channel_type: labelfree
    raw GFP EX488 EM525-45:
      channel_type: fluorescence
      biological_annotation:
        organelle: endoplasmic_reticulum
        marker: SEC61B
        marker_type: protein_tag
        fluorophore: eGFP
  experiment_metadata:
    A/1:
      perturbations:
        - name: ZIKV
          type: virus
          hours_post: 48.0
          moi: 5.0
      time_sampling_minutes: 30.0
```

## Adding New Metrics

1. Subclass `QCMetric` from `qc.qc_metrics`
2. Implement `field_name`, `channels()`, and `__call__()`
3. Add a Pydantic config model in `config.py`
4. Wire it into `cli.py`
