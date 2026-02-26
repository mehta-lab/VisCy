# Airtable Utils

Interface to the **Computational Imaging Database** on Airtable, with utilities for syncing experiment metadata between Airtable and OME-Zarr datasets.

Part of the [VisCy](https://github.com/mehta-lab/VisCy) monorepo.

## Installation

```bash
# From the VisCy monorepo root
uv pip install -e "applications/airtable"
```

### Environment Variables

Create a `.env` file in the repo root (gitignored):

```bash
# .env
AIRTABLE_API_KEY=patXXXXXXXXXXXXXX   # Personal access token
AIRTABLE_BASE_ID=appXXXXXXXXXXXXXX   # Computational Imaging Database base ID
```

Or export them in your shell / `.bashrc`.

## Usage

### Python API

```python
from airtable_utils import AirtableDatasets, DatasetRecord, parse_channel_name

db = AirtableDatasets()

# List unique dataset names
datasets = db.get_unique_datasets()

# Get all FOV records for a dataset
records = db.get_dataset_records("2024_10_16_A549_SEC61_ZIKV_DENV")

# Get experiment metadata dict (for writing to .zattrs)
for rec in records:
    meta = rec.to_experiment_metadata()

# All records as a DataFrame
df = db.list_records()

# Filter with Airtable formula
df = db.list_records(filter_formula="NOT({data_path} = '')")

# Parse channel names from zarr labels
parse_channel_name("Phase3D")
# {'channel_type': 'labelfree'}

parse_channel_name("raw GFP EX488 EM525-45")
# {'channel_type': 'fluorescence', 'filter_cube': 'GFP', 'excitation_nm': 488, 'emission_nm': 525}

parse_channel_name("nuclei_prediction")
# {'channel_type': 'virtual_stain'}
```

### Updating Records Programmatically

```python
from airtable_utils import AirtableDatasets

db = AirtableDatasets()

# Get records for a dataset
records = db.get_dataset_records("2024_10_16_A549_SEC61_ZIKV_DENV")

# Update a single record
db.batch_update([{
    "id": records[0].record_id,
    "fields": {"perturbation": "ZIKV", "moi": 10}
}])

# Update multiple records (e.g. fix data_path to FOV-level)
updates = []
for rec in records:
    if rec.fov and rec.data_path and rec.fov not in rec.data_path:
        updates.append({
            "id": rec.record_id,
            "fields": {"data_path": f"{rec.data_path}/{rec.well_id}/{rec.fov}"}
        })
db.batch_update(updates)
```

## Airtable Schema

The Datasets table uses snake_case column names. Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `dataset` | text | Dataset name |
| `well_id` | text | Well identifier (e.g. "B/1") |
| `fov` | text | Field of view (e.g. "000000") |
| `cell_type` | select | Cell type (e.g. "A549") |
| `cell_line` | multiselect | Cell line(s) |
| `perturbation` | select | Perturbation applied |
| `hours_post_perturbation` | number | Hours post-perturbation |
| `moi` | number | Multiplicity of infection |
| `channel_N_name` | text | Zarr channel label (populated from zarr) |
| `channel_N_biology` | select | Biological meaning (filled by scientist) |
| `data_path` | text | Path to FOV-level zarr position |
| `t/c/z/y/x_shape` | number | Array dimensions (populated from zarr) |


### Scripts

The script has two subcommands that correspond to different stages of the metadata workflow:

#### Step 1: `register` — zarr → Airtable

Expand well-level platemap records into per-FOV records using zarr position data.

```bash
# Dry run — see what would be created
uv run --package airtable-utils \
    applications/airtable/scripts/write_experiment_metadata.py \
    register /path/to/dataset.zarr --dry-run

# Create per-FOV records in Airtable
uv run --package airtable-utils \
    applications/airtable/scripts/write_experiment_metadata.py \
    register /path/to/dataset.zarr
```

This will:
- Read zarr positions and match them to well-level Airtable records by `well_id`
- Create per-FOV records with platemap metadata, channel names, shapes, and FOV-level `data_path`
- Skip FOVs that already have records
- Print a channel validation table for manual review

#### Step 2: `write` — Airtable → zarr

After reviewing/correcting channel biology in Airtable, write `experiment_metadata` to each FOV's `.zattrs`.

```bash
# Dry run — see what metadata would be written
uv run --package airtable-utils \
    applications/airtable/scripts/write_experiment_metadata.py \
    write /path/to/dataset.zarr --dry-run

# Write metadata to zarr
uv run --package airtable-utils \
    applications/airtable/scripts/write_experiment_metadata.py \
    write /path/to/dataset.zarr
```

This will:
- Read per-FOV records from Airtable (must have `fov` set — run `register` first)
- Write `experiment_metadata` to each position's `.zattrs`
- Update `data_path` to FOV-level if it was plate-level
- Track processed datasets in `experiment_metadata_tracking.csv`

### Verification

```python
from iohub import open_ome_zarr

plate = open_ome_zarr("/path/to/dataset.zarr", mode="r")
for name, pos in plate.positions():
    print(name, pos.zattrs.get("experiment_metadata"))
```
