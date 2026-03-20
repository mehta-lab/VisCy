---
name: airtable:update-records
description: Bulk-update records in the Computational Imaging Database on Airtable, with automatic pagination handling and verification
---

# Update Airtable Records

Bulk-update one or more fields across many records in the Computational Imaging Database.

## Airtable Configuration

- **Base ID**: `app8vqaoWyOwa0sB5` (Computational Imaging Database)
- **Datasets table ID**: `tblaFzrDMlVZHPZIj`
- **Collections table ID**: `tblu0Rbj9OnLl7vJf`
- **Models table ID**: `tblVZhRA48tDMWj8U`

### Key Datasets Table Fields

| Field | Description |
|---|---|
| `dataset` | Experiment name |
| `well_id` | Well path (e.g. `B/2`) |
| `fov` | FOV identifier |
| `data_path` | Path to HCS OME-Zarr store |
| `tracks_path` | Path to tracking zarr (`*cropped.zarr`) |
| `organelle` | Target organelle/structure |
| `marker` | Protein marker name |
| `perturbation` | Perturbation applied |
| `cell_type` | Cell type (e.g. `A549`) |
| `cell_state` | Condition label |
| `time_interval_min` | Minutes between frames |
| `hours_post_perturbation` | HPI at imaging start |
| `t_shape`, `c_shape`, `z_shape`, `y_shape`, `x_shape` | Array dimensions |
| `pixel_size_xy_um`, `pixel_size_z_um` | Pixel sizes |
| `channel_0_name` .. `channel_3_name` | Zarr channel names |
| `channel_0_biology` .. `channel_3_biology` | Biological meaning per channel |

### Dataset Directory Conventions

For **organelle_dynamics** datasets, `data_path` and `tracks_path` follow:
```
data_path:    /hpc/projects/intracellular_dashboard/organelle_dynamics/{EXPERIMENT}/2-assemble/{EXPERIMENT}.zarr
tracks_path:  /hpc/projects/intracellular_dashboard/organelle_dynamics/{EXPERIMENT}/1-preprocess/label-free/3-track/{EXPERIMENT}_cropped.zarr
```

Other dataset families (organelle_box, viral-sensor) have non-standard track directory structures — check the filesystem before assuming.

## Process

### Step 1: Understand the task

Ask the user (or infer from context):
- Which table to update (default: Datasets)
- Which field(s) to update
- How to compute the new values (from another field, filesystem, formula, etc.)
- Which records to target (all? a subset filtered by dataset name, organelle, etc.)

### Step 2: Fetch all target records (with pagination)

**Critical**: `mcp__airtable__list_records` returns at most ~100 records per call and does NOT paginate automatically. To get all matching records:

1. Use a `filterByFormula` to narrow scope as much as possible (e.g. filter to records where the target field is empty AND the source field is non-empty).
2. After the call, check the record count. If it equals ~100, there may be more — re-query with a tighter filter or a different approach.
3. Save the full result to a file (the MCP will do this automatically if the result is large).
4. Use `jq` to extract record IDs and field values from the saved file.

Example filter to find records missing `tracks_path` in organelle_dynamics:
```
AND(NOT({data_path} = ""), FIND("organelle_dynamics", {data_path}), {tracks_path} = "")
```

Repeat this verification query after updating to confirm zero records remain.

### Step 3: Compute new field values

Use `jq` + `python3` to:
- Parse the saved JSON
- Derive the new value from existing fields (e.g. transform `data_path` → `tracks_path`)
- Group into batches of 10 (the MCP `update_records` limit)
- Write batches to `/tmp/airtable_batches.jsonl` (one JSON array per line)

Example python snippet for deriving `tracks_path` from `data_path`:
```python
import json, re, sys

with open('/path/to/records.json') as f:
    data = json.load(f)

pairs = []
for rec in data['records']:
    dp = rec['fields'].get('data_path', '')
    m = re.match(r'(.*/organelle_dynamics/([^/]+))/2-assemble/', dp)
    if m:
        base, exp = m.group(1), m.group(2)
        tracks = f'{base}/1-preprocess/label-free/3-track/{exp}_cropped.zarr'
        pairs.append({'id': rec['id'], 'tracks_path': tracks})

batches = [pairs[i:i+10] for i in range(0, len(pairs), 10)]
for b in batches:
    records = [{'id': r['id'], 'fields': {'tracks_path': r['tracks_path']}} for r in b]
    print(json.dumps(records))
```

### Step 4: Update records in parallel batches

Send all batches using `mcp__airtable__update_records` in a **single message with multiple parallel tool calls** (up to 10 calls at once). Each call handles one batch of 10 records.

Do NOT send batches sequentially one at a time — send all available in parallel.

### Step 5: Verify completeness

After all batches complete, re-run the same filter query from Step 2 to confirm zero records remain with empty fields. If any remain (due to pagination gaps in the initial fetch), repeat Steps 2-4 for the remaining records until the verification query returns `{"records": []}`.

This verification step is mandatory — do not skip it.

## Example Invocations

- "update tracks_path for all organelle_dynamics datasets"
- "fill in pixel_size_xy_um for all records where it's missing"
- "set organelle = 'mitochondria' for all 2024_11_21 records"
- "backfill t_shape for records where data_path contains 'viral-sensor'"
