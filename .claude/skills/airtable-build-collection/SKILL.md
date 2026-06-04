---
name: airtable-build-collection
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
| `marker` | Protein marker (e.g. `SEC61B`, `TOMM20`, `pAL10`) |
| `organelle` | Target organelle |
| `perturbation` | Perturbation applied |
| `hours_post_perturbation` | HPI at imaging start |
| `moi` | Multiplicity of infection |
| `time_interval_min` | Minutes between frames |
| `data_path` | Path to HCS OME-Zarr store (FOV-level — extract zarr root by trimming well/fov) |
| `tracks_path` | Path to tracking zarr (may be absent) |
| `channel_0_name` .. `channel_N_name` | Zarr channel names |
| `channel_0_marker` .. `channel_N_marker` | Protein marker for each channel |
| `t_shape`, `c_shape`, `z_shape`, `y_shape`, `x_shape` | Array dimensions |
| `pixel_size_xy_um`, `pixel_size_z_um` | Physical pixel sizes |

## Usage

The user will describe what they want in natural language, e.g.:

- "fetch the dataset from 2025_07_24 with all the organelles from that experiment"
- "build a collection with the SEC61 and TOMM20 experiments from July 2025"
- "make a collection for all ZIKV infection datasets"

## Process

### Step 1: Query Airtable

Search for matching records using `mcp__airtable__list_records` with `filterByFormula`.

Common filter patterns:
- By dataset name: `SEARCH("2025_07_24", {dataset})`
- By organelle: `{organelle} = "SEC61"`
- By perturbation: `{perturbation} = "ZIKV"`
- Combined: `AND(SEARCH("2025_07", {dataset}), {organelle} = "TOMM20")`

Use `mcp__airtable__list_records` with `filterByFormula` for precise filtering.
Use `mcp__airtable__search_records` for fuzzy text matching.

### Step 2: Group and Summarize

Group records by `dataset`. **If a single dataset contains multiple markers/organelles** (different `marker` values across wells), split it into one experiment entry per marker. The experiment name gets a `_{MARKER}` suffix (e.g. `2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV_TOMM20`). All split entries share the same `data_path` and `tracks_path` but have different `perturbation_wells`, `marker`, and `organelle`.

This is handled automatically by `build_collection()` in `packages/viscy-data/src/viscy_data/collection.py` via the `_group_records()` helper.

Present a summary table to the user showing:

- Dataset names found (with split entries if multi-organelle)
- Number of FOVs per dataset
- Organelles and markers
- Channel names and markers
- Conditions (inferred from `perturbation` field — see note below)
- Wells per condition
- Whether `tracks_path` is available

**Note on cell_state**: In Airtable, `cell_state` is typically "Live" for all records. Infer infection status from the `perturbation` field: wells with a perturbation value are "infected", wells without are "uninfected".

Ask the user to confirm which datasets to include.

### Step 3: Determine Channels

Each experiment entry has a `channels` list where each entry maps a zarr channel name to a protein marker:

```yaml
channels:
  - name: "Phase3D"           # zarr channel name
    marker: "Phase3D"         # protein marker / semantic label
  - name: "raw GFP EX488 EM525-45"
    marker: "SEC61B"
```

Rules for mapping:
1. `channel_X_name` from Airtable → `name` field (the zarr channel name)
2. `channel_X_marker` from Airtable → `marker` field (the protein marker)
3. Only include channels relevant to the experiment — typically Phase3D (labelfree) and the fluorescence channel(s) for the marker of interest

Present the proposed channel mapping to the user for confirmation:
```
Channels per experiment:
  2025_07_24_SEC61:
    - Phase3D → Phase3D
    - raw GFP EX488 EM525-45 → SEC61B
  2024_08_14_ZIKV:
    - Phase3D → Phase3D
    - MultiCam_GFP_BF → pAL10
```

### Step 4: Determine tracks_path

Check the `tracks_path` field in Airtable. If missing, ask the user.

### Step 5: Naming Convention

Collection filenames follow: `{cell_line}_{perturbation}_{organelle}.yml`

- **Single organelle**: use the organelle name, e.g. `A549_ZIKV_SEC61.yml`
- **Multiple organelles**: use `multiorganelle`, e.g. `A549_ZIKV_multiorganelle.yml`
- **No version suffix** — versioning is handled by git history
- The `name` field inside the YAML should match the filename (without `.yml`)

### Step 6: Generate Collection YAML

Use the Collection schema from `packages/viscy-data/src/viscy_data/collection.py`.

The current schema uses per-experiment `channels` (list of `{name, marker}` entries), NOT `source_channels`:

```yaml
name: <filename without .yml>
description: "<what this collection contains>"

provenance:
  airtable_base_id: app8vqaoWyOwa0sB5
  airtable_query: "<the filter formula used>"
  record_ids: []
  created_at: "<ISO 8601 timestamp>"
  created_by: "<user name if known>"

experiments:
  - name: <dataset_name or dataset_marker split name>
    data_path: <zarr store root — trim well/fov from airtable data_path>
    tracks_path: <from airtable or user>
    channels:
      - name: <zarr_channel_name>
        marker: <protein_marker>
      - name: <zarr_channel_name>
        marker: <protein_marker>
    perturbation_wells:
      uninfected:
        - <well_id>
      <perturbation_name>:
        - <well_id>
    interval_minutes: <time_interval_min>
    start_hpi: <hours_post_perturbation or 0.0>
    marker: <primary marker>
    organelle: <organelle>
    moi: <moi or 0.0>
    pixel_size_xy_um: <from airtable>
    pixel_size_z_um: <from airtable>
```

Key notes:
- `data_path` should be the zarr store root (up to `.zarr`), NOT the FOV-level path from Airtable
- `perturbation_wells` uses `uninfected` / `<perturbation>` keys inferred from the `perturbation` field
- `channels` lists only the channels needed for training (not all channels in the zarr)
- `marker` at the experiment level is the primary marker for this experiment entry

### Step 7: Save and Validate

1. Save to `applications/dynaclr/configs/collections/<name>.yml`
2. Validate by loading with `viscy_data.collection.load_collection(path)` using a quick Python check
3. Show the user the final YAML and validation result

## Important Notes

- `interval_minutes` must be > 0
- `perturbation_wells` must not be empty
- Zarr channel names in `channels[].name` must match actual zarr channel names
- For multi-marker datasets, split into separate experiment entries per marker
- Reference existing collections in `applications/dynaclr/configs/collections/` for format examples
