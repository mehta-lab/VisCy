# Airtable Utils

Interface to the **Computational Imaging Database** on Airtable, with utilities for syncing experiment metadata between Airtable and OME-Zarr datasets.

Part of the [VisCy](https://github.com/mehta-lab/VisCy) monorepo.

---

## Setup

### Installation

```bash
# From the VisCy monorepo root
uv sync --all-packages --all-extras
```

### Environment Variables

```bash
export AIRTABLE_API_KEY=patXXXXXXXXXXXXXX   # Personal access token
export AIRTABLE_BASE_ID=app8vqaoWyOwa0sB5    # Computational Imaging Database
```

Add to your `.bashrc` or a `.env` file (gitignored).

---

## Airtable Tables

| Table | ID | Purpose |
|---|---|---|
| **Datasets** | `tblaFzrDMlVZHPZIj` | One record per FOV. Biologists fill well-level metadata; the registration script expands to per-FOV. |
| **Marker Registry** | `tblmP8l2GmpCeERyD` | LUT mapping each construct to its fluorescent channel aliases and biology. Used to auto-derive `channel_*_biology` at registration time. |

### Marker Registry

Each row maps one construct (one fluorophore) to a biology label:

| Field | Example | Notes |
|---|---|---|
| `marker-fluorophore` | `TOMM20-GFP` | `PROTEIN-FLUOROPHORE` or `PLASMID-FLUOROPHORE`, dashes only |
| `channel_name_aliases` | `GFP, FITC` | Comma-separated substrings to substring-match against zarr channel names |
| `biology` | `mitochondria` | snake_case biological annotation |

One row per construct. Compound lines (e.g. `TOMM20-GFP pAL40`) are represented as two linked entries: `TOMM20-GFP` (GFP→mitochondria) + `pAL40-mCherry` (mCherry→viral_sensor).

### Datasets Schema

Key fields (snake_case):

| Field | Type | Source |
|---|---|---|
| `dataset` | text | Must match zarr stem |
| `well_id` | text | e.g. `B/1` |
| `fov` | text | e.g. `000000` — set by `register` |
| `cell_type` | select | e.g. `A549` |
| `cell_line` | linked records | Links to Marker Registry |
| `marker` | select | Primary protein marker |
| `organelle` | select | Target organelle |
| `perturbation` | select | e.g. `ZIKV`, `DENV` |
| `channel_N_name` | text | Zarr channel label — set by `register` (N = 0–7) |
| `channel_N_biology` | text | Biology annotation — derived from Marker Registry (N = 0–7) |
| `data_path` | text | Path to zarr position — set by `register` |
| `t/c/z/y/x_shape` | number | Array dimensions — set by `register` |

---

## Workflow

### Step 1 — Biologist: fill well-level platemap in Airtable

Before the zarr exists, create one Datasets record per well with:
- `dataset`, `well_id`, `cell_type`, `cell_line` (linked to Marker Registry), `marker`, `organelle`, `perturbation`, `moi`, `time_interval_min`, `fluorescence_modality`

Leave all zarr-derived fields empty — they are filled by `register`.

> **New construct?** Add it to the Marker Registry first (with aliases + biology), then link it in the Datasets record.

### Step 2 — Engineer: register zarr positions after QC

```bash
# Dry run — see what would be created/updated
uv run --package airtable-utils \
    applications/airtable/scripts/write_experiment_metadata.py \
    register --dry-run /path/to/dataset.zarr/*/*/*

# Run for real
uv run --package airtable-utils \
    applications/airtable/scripts/write_experiment_metadata.py \
    register /path/to/dataset.zarr/*/*/*
```

The atomic unit is a **position path** (e.g. `dataset.zarr/A/1/000000`). Shell globbing handles batch registration. For a single position:

```bash
uv run --package airtable-utils \
    applications/airtable/scripts/write_experiment_metadata.py \
    register /path/to/dataset.zarr/A/1/000000
```

What `register` does per position:
- Reads channel names (up to 8) and array shape from the zarr
- Fetches Marker Registry once per run
- Resolves `cell_line` linked records → aliases → derives `channel_*_biology` via substring match
- Creates new per-FOV record or updates existing one

### Step 3 — Engineer: write metadata to zarr `.zattrs`

```bash
uv run --package airtable-utils \
    applications/airtable/scripts/write_experiment_metadata.py \
    write /path/to/dataset.zarr/*/*/*
```

Writes `channels_metadata` and `experiment_metadata` to each position's `.zattrs`.

---

## Using with Claude Code (AI-assisted workflows)

This application ships with a **Claude Code skill** that lets you run registration tasks conversationally without flooding the context with large Airtable API responses.

### Setup

1. Install the skill (one-time, already done for `eduardo.hirata`):

```bash
# The skill lives at:
~/.claude/skills/airtable-register/SKILL.md
```

2. Open Claude Code in the VisCy repo:

```bash
cd /hpc/mydata/eduardo.hirata/repos/viscy
claude
```

### Usage

Claude will automatically invoke the skill when you ask things like:

```
register 2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV
```
```
re-register all datasets in Airtable
```
```
add LAMP2-GFP to the Marker Registry with biology lysosome
```
```
check which positions are missing channel biology
```

The skill runs registration as a subagent, keeping Airtable MCP responses out of the main conversation context.

### Installing the skill for a new user

Copy the skill to your Claude config:

```bash
cp -r /hpc/mydata/eduardo.hirata/repos/viscy/applications/airtable/.claude/skills/airtable-register \
      ~/.claude/skills/
```

> Note: The skill is not checked into the repo (it lives in `~/.claude/skills/`) because it contains HPC-specific paths. Copy and adapt as needed.

---

## Python API

```python
from airtable_utils import AirtableDatasets, DatasetRecord, parse_channel_name
from airtable_utils.registration import register_fovs, parse_position_path

db = AirtableDatasets()

# Get all FOV records for a dataset
records = db.get_dataset_records("2024_10_16_A549_SEC61_ZIKV_DENV")

# Get Marker Registry (keyed by record ID)
registry = db.get_marker_registry()

# Register positions programmatically
from pathlib import Path
positions = list(Path("/path/to/dataset.zarr").glob("*/*/*"))
result = register_fovs(positions, db=db)
print(f"created={len(result.created)} updated={len(result.updated)}")
if result.updated:
    db.batch_update(result.updated)
if result.created:
    db.batch_create(result.created)

# Parse channel names from zarr labels
parse_channel_name("Phase3D")
# {'channel_type': 'labelfree'}
parse_channel_name("raw GFP EX488 EM525-45")
# {'channel_type': 'fluorescence', 'filter_cube': 'GFP', 'excitation_nm': 488, 'emission_nm': 525}
```

---

## `.zattrs` Schema

Written to each position by the `write` subcommand:

**`channels_metadata`** — keyed by channel name:
```json
{
    "Phase3D": {"channel_type": "labelfree", "biological_annotation": null},
    "raw GFP EX488 EM525-45": {
        "channel_type": "fluorescence",
        "biological_annotation": {
            "organelle": "mitochondria",
            "marker": "TOMM20",
            "marker_type": "protein_tag",
            "fluorophore": null
        }
    },
    "raw mCherry EX561 EM600-37": {
        "channel_type": "fluorescence",
        "biological_annotation": {
            "organelle": "viral_sensor",
            "marker": "unknown",
            "marker_type": "protein_tag",
            "fluorophore": null
        }
    }
}
```

**`experiment_metadata`**:
```json
{
    "perturbations": [{"name": "ZIKV", "type": "virus", "hours_post": 4.0, "moi": 5.0}],
    "time_sampling_minutes": 30.0
}
```

---

## Testing

```bash
uv run pytest applications/airtable/ -v
```

Tests use mocked Airtable and zarr — no credentials or network required.
