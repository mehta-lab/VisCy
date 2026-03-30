---
name: airtable-register
description: Register zarr positions into the Computational Imaging Database on Airtable, write channels_metadata/experiment_metadata to zarr .zattrs, or bulk-update Airtable records via MCP. Use when the user asks to "register a dataset", "register zarr positions", "update airtable from zarr", "write metadata to zarr", "run register on", "sync airtable", "populate channel markers", "update airtable records", "backfill fields", or "fill in missing fields". Also use for Marker Registry questions.
version: 3.0.0
author: ai-x-imaging
tags: [Airtable, OME-Zarr, Metadata, Registration, DynaCLR, VisCy]
---

# Airtable Registration & Update Skill

Manages bidirectional metadata sync between OME-Zarr datasets and the Computational Imaging Database on Airtable, and supports bulk field updates via MCP.

## Airtable Configuration

- **Base ID**: `app8vqaoWyOwa0sB5` (Computational Imaging Database)
- **Datasets table ID**: `tblaFzrDMlVZHPZIj`
- **Collections table ID**: `tblu0Rbj9OnLl7vJf`
- **Models table ID**: `tblVZhRA48tDMWj8U`
- **Marker Registry table**: `tblmP8l2GmpCeERyD`
- **Script**: `applications/airtable/scripts/write_experiment_metadata.py`
- **Core logic**: `applications/airtable/src/airtable_utils/registration.py`
- **Schemas**: `applications/airtable/src/airtable_utils/schemas.py`
- **Database interface**: `applications/airtable/src/airtable_utils/database.py`
- **Channel parsing**: `packages/viscy-data/src/viscy_data/channel_utils.py`
- `MAX_CHANNELS = 8` (defined in `schemas.py`)
- `AIRTABLE_API_KEY` and `AIRTABLE_BASE_ID` must be set in environment

## Operations

### 1. Register (zarr -> Airtable)

Reads zarr metadata and writes per-FOV records to Airtable.

**Fields written by register:**
- `data_path` — full path to zarr position
- `channel_{i}_name` — zarr channel names (up to 8)
- `channel_{i}_marker` — protein marker, derived from Marker Registry
- `t_shape`, `c_shape`, `z_shape`, `y_shape`, `x_shape` — array dimensions
- `pixel_size_xy_um`, `pixel_size_z_um` — from zarr coordinate transforms

**Marker derivation rules:**
- **labelfree** channels -> marker = channel name (e.g. `"Phase3D"`, `"BF"`, `"DIC"`)
- **virtual_stain** channels -> marker = channel name (e.g. `"nuclei_prediction"`)
- **fluorescence** channels -> substring-match aliases against channel name -> protein marker from registry (e.g. `"TOMM20"`, `"SEC61B"`)

#### Commands

```bash
# Dry run first (always recommended)
uv run --package airtable-utils \
    applications/airtable/scripts/write_experiment_metadata.py \
    register --dry-run /path/to/dataset.zarr/*/*/*

# Register all positions
uv run --package airtable-utils \
    applications/airtable/scripts/write_experiment_metadata.py \
    register /path/to/dataset.zarr/*/*/*

# Single position
uv run --package airtable-utils \
    applications/airtable/scripts/write_experiment_metadata.py \
    register /path/to/dataset.zarr/A/1/000000

# Override dataset name (when zarr stem doesn't match Airtable)
uv run --package airtable-utils \
    applications/airtable/scripts/write_experiment_metadata.py \
    register --dataset my_dataset /path/to/dataset.zarr/*/*/*
```

#### Parquet Readiness Report

After registration, the CLI prints a **Parquet Readiness** report that flags any fields still needed before a flat parquet cell index can be built. Fields are split by source:

- **zarr** fields (auto-filled by `register`): `data_path`, `channel_N_name`, `channel_N_marker`, `pixel_size_xy_um`, `pixel_size_z_um`
- **platemap** fields (biologist fills in Airtable): `tracks_path`, `perturbation`, `time_interval_min`, `hours_post_perturbation`, `cell_type`

If any platemap fields are missing, the report shows what to fill in and how (Airtable UI or MCP bulk update).

### 2. Write (Airtable -> zarr)

Writes `channels_metadata` and `experiment_metadata` to zarr `.zattrs`.

```bash
uv run --package airtable-utils \
    applications/airtable/scripts/write_experiment_metadata.py \
    write /path/to/dataset.zarr/*/*/*
```

#### channels_metadata schema

```json
{
  "Phase3D": {
    "channel_type": "labelfree",
    "biological_annotation": {"marker": "Phase3D"}
  },
  "raw GFP EX488 EM525-45": {
    "channel_type": "fluorescence",
    "biological_annotation": {
      "marker": "TOMM20",
      "marker_type": "protein_tag",
      "fluorophore": null
    }
  }
}
```

#### experiment_metadata schema

```json
{
  "perturbations": [
    {"name": "ZIKV", "type": "unknown", "hours_post": 48.0, "moi": 5.0}
  ],
  "time_sampling_minutes": 15.0
}
```

### 3. Bulk Update (via Airtable MCP)

For updating fields that don't come from zarr (e.g. `tracks_path`, `organelle`, manually-curated fields).

**Process:**

1. Fetch target records with `mcp__airtable__list_records` using `filterByFormula`
2. Compute new values (python/jq)
3. Batch update with `mcp__airtable__update_records` (max 10 per call, send all batches in parallel)
4. Verify with the same filter query (must return zero remaining records)

**Pagination warning:** `mcp__airtable__list_records` returns ~100 records max per call. If count equals ~100, re-query with tighter filters.

## Datasets Table Fields

| Field | Description | Written by register? |
|---|---|---|
| `dataset` | Experiment name | on create |
| `well_id` | Well path (e.g. `B/2`) | on create |
| `fov` | FOV identifier | on create |
| `data_path` | Path to HCS OME-Zarr position | yes |
| `tracks_path` | Path to tracking zarr | no (manual/MCP) |
| `channel_0_name` .. `channel_7_name` | Zarr channel names | yes |
| `channel_0_marker` .. `channel_7_marker` | Protein marker per channel | yes |
| `t_shape` .. `x_shape` | Array dimensions | yes |
| `pixel_size_xy_um` | Physical XY pixel size (um) | yes |
| `pixel_size_z_um` | Physical Z pixel size (um) | yes |
| `marker` | Well-level primary marker | template copy |
| `organelle` | Target organelle | template copy |
| `perturbation` | Perturbation applied | template copy |
| `cell_type` | Cell type (e.g. `A549`) | template copy |
| `cell_state` | Condition label | template copy |
| `time_interval_min` | Minutes between frames | template copy |
| `hours_post_perturbation` | HPI at imaging start | template copy |
| `moi` | Multiplicity of infection | template copy |
| `fluorescence_modality` | Imaging modality | template copy |

## Marker Registry

Table `tblmP8l2GmpCeERyD` — maps constructs to protein markers.

| Field | Type | Example |
|---|---|---|
| `marker-fluorophore` | text (primary) | `TOMM20-GFP` |
| `channel_name_aliases` | text | `GFP, FITC` |
| `marker` | text | `TOMM20` |

Matching is substring-based: `"GFP" in "raw GFP EX488 EM525-45"` -> match.

## Flat Parquet Alignment

The `register` command writes all fields needed to build a flat parquet cell index:
- `channel_{i}_name` -> parquet `channel_name`
- `channel_{i}_marker` -> parquet `marker`
- `pixel_size_xy_um`, `pixel_size_z_um` -> parquet pixel size columns
- `data_path` -> parquet `store_path`

## Dataset Directory Conventions

For **organelle_dynamics** datasets:
```
data_path:    /hpc/projects/intracellular_dashboard/organelle_dynamics/{EXP}/2-assemble/{EXP}.zarr
tracks_path:  /hpc/projects/intracellular_dashboard/organelle_dynamics/{EXP}/1-preprocess/label-free/3-track/{EXP}_cropped.zarr
```

Other families (organelle_box, viral-sensor) have non-standard structures — check filesystem.

## Example Invocations

- "register this dataset /path/to/dataset.zarr/*/*/*"
- "write metadata to zarr for dataset X"
- "update tracks_path for all organelle_dynamics datasets"
- "fill in pixel_size_xy_um for all records where it's missing"
- "set organelle = 'mitochondria' for all 2024_11_21 records"
