---
name: prepare-dataset
description: Prepare datasets for training on VAST storage (NFS -> VAST rechunked zarr v3 pipeline). Use when the user asks to "prepare a dataset", "run prepare", "rechunk dataset", "copy dataset to VAST", "run QC and preprocess", or references the `prepare` CLI.
---

# Prepare Dataset for Training (NFS -> VAST)

## Overview

This skill runs the `prepare` CLI from `applications/airtable/` to create rechunked zarr v3 copies of NFS datasets on VAST storage, with QC (focus slice) and preprocessing (normalization stats).

## Pipeline Steps

1. **Airtable validation** — dataset must be registered
2. **Discover wells/channels** — reads NFS zarr via iohub; auto-detects raw channels (`Phase3D` + `raw *`)
3. **biahub concatenate** — rechunks to zarr v3 with sharding (submits own SLURM jobs via submitit)
4. **Copy tracking zarr** — rsync from NFS
5. **QC + preprocess** — SLURM job running focus slice QC (GPU) and normalization (CPU) in parallel

## Key Files

- **CLI**: `applications/airtable/src/airtable_utils/prepare_cli.py`
- **Core logic**: `applications/airtable/src/airtable_utils/prepare.py`
- **Default config**: `applications/airtable/configs/prepare_config.yml`

## Commands

```bash
# Check status of one or more datasets
uv run --package airtable-utils \
    prepare status <dataset_name> [<dataset_name> ...] \
    -c applications/airtable/configs/prepare_config.yml

# Dry run (generate configs + scripts, don't execute)
uv run --package airtable-utils \
    prepare run <dataset_name> \
    -c applications/airtable/configs/prepare_config.yml --dry-run

# Full run
uv run --package airtable-utils \
    prepare run <dataset_name> \
    -c applications/airtable/configs/prepare_config.yml

# Force overwrite existing VAST zarr
uv run --package airtable-utils \
    prepare run <dataset_name> \
    -c applications/airtable/configs/prepare_config.yml --force
```

## Running Multiple Datasets

Run `prepare run` sequentially for each dataset. The concatenation step blocks until biahub's internal SLURM jobs complete, then submits the QC+preprocess SLURM job. Example:

```bash
for ds in 2025_01_28_A549_G3BP1_ZIKV_DENV 2025_04_15_A549_H2B_CAAX_ZIKV_DENV; do
    uv run --package airtable-utils \
        prepare run "$ds" \
        -c applications/airtable/configs/prepare_config.yml
done
```

## Output Layout

```
/hpc/projects/organelle_phenotyping/datasets/{dataset_name}/
  {dataset_name}.zarr       # zarr v3 rechunked (OME-Zarr 0.5)
  tracking.zarr             # copied from NFS
  crop_concat.yml           # generated biahub config
  qc_config.yml             # generated QC config
  sbatch_overrides.sh       # SLURM overrides for biahub (if configured)
  01_concatenate.sh         # bash: biahub concatenate + tracking copy
  02_qc_preprocess.sh       # SLURM: QC + preprocess
```

## Config Reference

The config at `applications/airtable/configs/prepare_config.yml` has these key settings:

| Section | Field | Default | Notes |
|---|---|---|---|
| `concatenate` | `channel_names` | `null` (auto-detect) | Set explicitly to override; auto picks `Phase3D` + `raw *` |
| `concatenate` | `chunks_czyx` | `[1, 16, 256, 256]` | ~4MB chunks for training |
| `concatenate` | `shards_ratio` | `[1, 1, 8, 8, 8]` | Sharding for zarr v3 |
| `concatenate` | `sbatch_overrides` | `{partition: preempted}` | Overrides biahub's internal SLURM via `-sb` |
| `qc` | `channel_names` | `[Phase3D]` | Channels for focus slice detection |
| `slurm.qc_preprocess` | `partition` | `gpu` | QC needs GPU for torch FFT |
| `slurm.qc_preprocess` | `cpus_per_task` | `16` | |
| `slurm.qc_preprocess` | `time` | `01:00:00` | |

## Extracting Dataset Names from a Collection YAML

To get unique dataset names from a collection:

```bash
uv run python3 -c "
import yaml
from pathlib import Path
with open('path/to/collection.yml') as f:
    col = yaml.safe_load(f)
datasets = sorted(set(
    Path(e['data_path']).parts[
        Path(e['data_path']).parts.index('organelle_dynamics') + 1
    ]
    for e in col['experiments']
    if 'organelle_dynamics' in e['data_path']
))
for d in datasets:
    print(d)
"
```

## Troubleshooting

- **"Dataset not found in Airtable"**: Register it first with the `airtable-register` skill
- **Channel validation fails**: Check `channel_names` in config; set to `null` for auto-detection
- **biahub concatenate fails**: Check conda env exists (`conda run -n biahub which biahub`)
- **QC/preprocess SLURM job pending**: Check `squeue -u $USER` and partition availability
