---
name: airtable:build-collection
description: Query Airtable MCP to fetch experiment metadata and generate a DynaCLR collection YAML for training
---

# Build Collection from Airtable

Build a collection YAML for DynaCLR training by querying the Computational Imaging Database on Airtable.

## Airtable Configuration

- **Base ID**: `app8vqaoWyOwa0sB5` (Computational Imaging Database)
- **Table ID**: `tblaFzrDMlVZHPZIj` (Datasets)

Key fields in the Datasets table:

| Field | Description |
|---|---|
| `dataset` | Experiment name (e.g. `2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV`) |
| `well_id` | Well path (e.g. `B/2`) |
| `fov` | FOV identifier |
| `cell_state` | Condition label (e.g. `infected`, `uninfected`) |
| `organelle` | Target organelle |
| `perturbation` | Perturbation applied |
| `hours_post_perturbation` | HPI at imaging start |
| `moi` | Multiplicity of infection |
| `time_interval_min` | Minutes between frames |
| `data_path` | Path to HCS OME-Zarr store |
| `channel_0_name` .. `channel_3_name` | Zarr channel names |
| `channel_0_biology` .. `channel_3_biology` | Biological meaning of each channel |
| `t_shape`, `c_shape`, `z_shape`, `y_shape`, `x_shape` | Array dimensions |

## Usage

The user will describe what they want in natural language, e.g.:

- "fetch the dataset from 2025_07_24 with all the organelles from that experiment"
- "build a collection with the SEC61 and TOMM20 experiments from July 2025"
- "make a collection for all ZIKV infection datasets"

## Process

### Step 1: Query Airtable

Search for matching records using `mcp__airtable__search_records` or `mcp__airtable__list_records` with `filterByFormula`.

Common filter patterns:
- By dataset name: `SEARCH("2025_07_24", {dataset})`
- By organelle: `{organelle} = "SEC61"`
- By perturbation: `{perturbation} = "ZIKV"`
- Combined: `AND(SEARCH("2025_07", {dataset}), {organelle} = "TOMM20")`

Use `mcp__airtable__list_records` with `filterByFormula` for precise filtering.
Use `mcp__airtable__search_records` for fuzzy text matching.

### Step 2: Group and Summarize

Group records by `dataset`. **If a single dataset contains multiple markers/organelles** (different `marker` values across wells), split it into one experiment entry per marker. The experiment name gets a `_{MARKER}` suffix (e.g. `2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV_TOMM20`). All split entries share the same `data_path` and `tracks_path` but have different `perturbation_wells`, `marker`, and `organelle`.

This is handled automatically by `build_collection()` in `packages/viscy-data/src/viscy_data/collection.py` via the `_group_records()` helper, which groups by `(dataset, marker)` when multiple markers are present.

Present a summary table to the user showing:

- Dataset names found (with split entries if multi-organelle)
- Number of FOVs per dataset
- Organelles and markers
- Channel names and biology
- Conditions (from `cell_state` or inferred from `perturbation`)
- Wells per condition
- Whether `tracks_path` is available (check the field, but note it may be missing — the user may need to provide it or derive it from convention)

**Note on cell_state**: In Airtable, `cell_state` may be "Live" for all records. In that case, infer infection status from the `perturbation` field: wells with a perturbation are "infected", wells without are "uninfected".

Ask the user to confirm which datasets to include.

### Step 3: Determine Source Channels

This is the key mapping step. Source channels define semantic labels that map across experiments with different physical channel names.

Rules:
1. **labelfree** channel: Look for channels where `channel_X_biology` is "brightfield", "phase", or similar. The zarr channel name is typically "Phase3D" or similar.
2. **reporter** channels: Look for fluorescence channels. Group by `channel_X_biology` — channels with the same biology across experiments are semantically equivalent even if they have different zarr names.

Present the proposed source channel mapping to the user for confirmation:
```
Source channels:
  - labelfree:
      exp_a → Phase3D
      exp_b → Phase3D
  - reporter (mitochondria):
      exp_a → raw GFP EX488 EM525-45
      exp_b → raw RFP EX561 EM600-50
```

### Step 4: Determine tracks_path

The `tracks_path` field may not be in Airtable. If missing, use convention:
```
{data_path parent}/3-tracking/{dataset_name}_tracks
```
or ask the user.

### Step 5: Naming Convention

Collection filenames follow: `{cell_line}_{perturbation}_{organelle}.yml`

- **Single organelle**: use the organelle name, e.g. `A549_ZIKV_SEC61.yml`
- **Multiple organelles**: use `multiorganelle`, e.g. `A549_ZIKV_multiorganelle.yml`
- **No version suffix** — versioning is handled by git history
- The `name` field inside the YAML should match the filename (without `.yml`)

### Step 6: Generate Collection YAML

Use the Collection schema from `packages/viscy-data/src/viscy_data/collection.py`:

```yaml
name: <filename without .yml>
description: "<what this collection contains>"

provenance:
  airtable_base_id: app8vqaoWyOwa0sB5
  airtable_query: "<the filter formula used>"
  record_ids: []
  created_at: "<ISO 8601 timestamp>"
  created_by: "<user name if known>"

source_channels:
  - label: labelfree
    per_experiment:
      <exp_name>: <zarr_channel_name>
  - label: <reporter_label>
    per_experiment:
      <exp_name>: <zarr_channel_name>

experiments:
  - name: <dataset_name>
    data_path: <from airtable>
    tracks_path: <from airtable or convention>
    channel_names: [<all zarr channel names>]
    perturbation_wells:
      <cell_state>:
        - <well_id>
        - ...
    interval_minutes: <time_interval_min>
    start_hpi: <hours_post_perturbation or 0.0>
    organelle: <organelle>
    moi: <moi or 0.0>
```

Derive `perturbation_wells` by grouping unique `(cell_state, well_id)` pairs per dataset.
Derive `channel_names` from `channel_0_name` through `channel_3_name` (skip None).

### Step 7: Save and Validate

1. Save to `applications/dynaclr/configs/collections/<name>.yml`
2. Validate by loading with `viscy_data.collection.load_collection(path)` using a quick Python check
3. Show the user the final YAML and validation result

## Important Notes

- Each experiment in the collection must have the **same number of source channels** (the Collection validator enforces this)
- Every experiment must appear in every source channel's `per_experiment` mapping
- `interval_minutes` must be > 0
- `perturbation_wells` must not be empty
- The zarr channel names in `source_channels.per_experiment` must exist in that experiment's `channel_names`
