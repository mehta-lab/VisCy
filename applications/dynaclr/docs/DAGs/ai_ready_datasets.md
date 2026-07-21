# Data Preparation DAG

## Entry point

`prepare run <dataset_name> -c prepare_config.yaml` (from `airtable_utils`) discovers wells and
channels from NFS, generates all configs and SLURM scripts, and submits the pipeline.

```bash
prepare run 2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV \
    -c /path/to/prepare_config.yaml

# Dry-run: generate configs/scripts without submitting
prepare run 2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV \
    -c /path/to/prepare_config.yaml \
    --dry-run
```

## Step-by-step detail

```
NFS assembled zarr  (intracellular_dashboard/organelle_dynamics/{dataset}/2-assemble/)
  │
  ▼
prepare run                              # discovers wells + channels from NFS zarr
  │  airtable_utils.prepare_cli          # validates dataset is in Airtable
  │  airtable_utils.prepare              # generates all configs and scripts
  ▼
{vast_output_dir}/
  ├── crop_concat.yml                    # biahub concatenate config (wells × channels)
  ├── qc_config.yml                      # focus-slice QC config
  ├── sbatch_overrides.sh                # optional SLURM overrides for biahub's internal jobs
  ├── 01_concatenate.sh                  # bash (not SLURM): runs biahub + rsync tracking
  ├── 02_qc.sh                           # SLURM: GPU focus-slice detection
  └── 03_preprocess.sh                   # SLURM: CPU normalization stats
  │
  ▼
bash 01_concatenate.sh                   # NOT a SLURM job — runs interactively
  │  Step 1: conda run biahub concatenate -c crop_concat.yml -o {dataset}.zarr -m
  │           biahub submits its own SLURM jobs internally via submitit; -m blocks until done
  │  Step 2: rsync tracking zarr (NFS → VAST)
  ▼
{dataset}.zarr  (OME-Zarr v0.5 / zarr v3, rechunked)
tracking.zarr   (cell tracking results)
  │
  ├──► sbatch 02_qc.sh                   # GPU (~30 min)
  │       qc run -c qc_config.yml        # focus-slice detection on Phase3D channel
  │       → writes focus_slice metadata into {dataset}.zarr
  │         (or a CSV sidecar under csv_dir, if set — read-only stores)
  │
  └──► sbatch 03_preprocess.sh           # CPU, preempted partition (~4 hrs)
          viscy preprocess               # computes per-channel normalization stats
              --data_path {dataset}.zarr
          → writes normalization metadata into {dataset}.zarr
            (or a CSV sidecar under csv_dir, if set — read-only stores)
```

If `prepare_config.yaml` sets `csv_dir`, both jobs open `{dataset}.zarr`
read-only and write to a per-store CSV sidecar under `csv_dir` instead —
use this when VAST datasets are mounted read+execute only. See
[training.md](training.md) for the corresponding `--csv-dir` flag on
`preprocess-cell-index`.

## Pipeline DAG (process dependency)

```
NFS zarr (assembled)
  │
  ▼
prepare run  ──── generates configs + scripts
  │
  ▼
01_concatenate.sh  (interactive bash, blocks until biahub SLURM jobs finish)
  │
  ▼
{dataset}.zarr + tracking.zarr
  │
  ├──► 02_qc.sh (SLURM, GPU)       → focus_slice metadata in zarr
  └──► 03_preprocess.sh (SLURM, CPU) → normalization metadata in zarr
```

02_qc and 03_preprocess run in parallel (no dependency between them).
Both write metadata back to the same zarr; their outputs are checked by
`check_preprocessed()` before downstream training or evaluation.

## Key commands


| Step              | Command                                           | Input              | Output                                                          |
| ----------------- | ------------------------------------------------- | ------------------ | --------------------------------------------------------------- |
| Generate + submit | `prepare run <dataset> -c prepare_config.yaml`    | NFS assembled zarr | scripts + configs, submits jobs                                 |
| Status check      | `prepare status <dataset> -c prepare_config.yaml` | -                  | markdown table (NFS/VAST existence, zarr version, preprocessed) |
| Concatenate       | `bash 01_concatenate.sh`                          | crop_concat.yml    | {dataset}.zarr + tracking.zarr                                  |
| QC                | `sbatch 02_qc.sh`                                 | qc_config.yml      | focus_slice metadata in zarr                                    |
| Preprocess        | `sbatch 03_preprocess.sh`                         | {dataset}.zarr     | normalization metadata in zarr                                  |


## prepare_config.yaml format

```yaml
nfs_root: /hpc/projects/intracellular_dashboard/organelle_dynamics
vast_root: /hpc/projects/organelle_phenotyping/datasets
workspace_dir: /hpc/mydata/eduardo.hirata/repos/viscy
csv_dir: null                  # optional: set to a path to preprocess read-only
                                # datasets via CSV sidecars instead of zarr zattrs

concatenate:
  channel_names: null          # null = auto-detect raw channels (Phase3D + "raw " prefix)
  chunks_czyx: [1, 16, 256, 256]
  shards_ratio: [1, 1, 8, 8, 8]
  output_ome_zarr_version: "0.5"
  conda_env: biahub
  sbatch_overrides:            # optional: overrides for biahub's internal SLURM jobs
    partition: preempted
    mem-per-cpu: 16G

qc:
  channel_names: [Phase3D]
  NA_det: 1.35
  lambda_ill: 0.450
  pixel_size: 0.1494
  midband_fractions: [0.125, 0.25]
  device: cuda
  num_workers: 16

preprocess:
  channel_names: -1            # -1 = all channels
  num_workers: 32
  block_size: 32

slurm:
  qc:
    partition: gpu
    gres: gpu:1
    cpus_per_task: 16
    mem_per_cpu: 4G
    time: "00:30:00"
  preprocess:
    partition: preempted
    cpus_per_task: 32
    mem_per_cpu: 4G
    time: "04:00:00"
```

## Notes

- `prepare run` validates the dataset exists in Airtable before generating anything.
Use `--force` to overwrite an existing VAST zarr (e.g. to upgrade from zarr v2 to v0.5).
- `01_concatenate.sh` is an interactive bash script, not a SLURM job. Run it from a login
node or an interactive session; it blocks until biahub's internal SLURM jobs finish (`-m` flag).
- `02_qc.sh` and `03_preprocess.sh` are independent — submit both immediately after
`01_concatenate.sh` completes; no need to wait for QC before running preprocess.
- Channel auto-detection (`channel_names: null`) keeps channels with prefix `Phase3D` or `raw` .
Virtual stains (`nuclei_prediction`, `membrane_prediction`) and deconvolved channels are excluded.
- `check_preprocessed()` checks for `normalization` key in zarr metadata (or, when `csv_dir` is set,
for a `normalization` row in the per-store CSV sidecar); used by `prepare status` and as a gate
before evaluation.
- `csv_dir` is a single shared root for all datasets under one `prepare_config.yaml` — per-store
CSVs are named from a slug of the store path, so multiple datasets can safely share one root.
- Raw channel names written to `crop_concat.yml` are repeated once per well entry — this is a
biahub concatenate requirement.

## Path convention

All AI-ready data lives under `/hpc/projects/organelle_phenotyping/`:


| Directory                  | Contents                                      |
| -------------------------- | --------------------------------------------- |
| `datasets/<dataset_name>/` | Zarr v3 store + `tracking.zarr`               |
| `datasets/annotations/`    | Per-experiment annotation CSVs                |
| `models/collections/`      | Cell index parquets (one per collection YAML) |
| `models/<ModelName>/`      | Training runs (checkpoints, WandB configs)    |


Collection YAMLs use `datasets_root: /hpc/projects/organelle_phenotyping` and
`${datasets_root}/datasets/...` placeholders — resolved at load time by `load_collection()`.

## Batch preprocessing on Reef (many datasets, incremental)

`prepare run` assumes Bruno's NFS-assemble → `biahub concatenate` → VAST
pipeline. On Reef, `{name}.zarr` + `tracking.zarr` stores are usually
Globus-copied from Bruno already assembled — there's nothing to
concatenate, and per-file ownership varies by whoever ran the transfer
(everyone else in the group has read-only access to those files). For this
case, use `prepare batch-preprocess <data_root> -c prepare_config.yaml`
instead of `prepare run` per dataset:

```bash
prepare batch-preprocess /bio/projects/compimaging/data/infectomics/imaging/datasets \
    -c reef_prepare_config.yaml

# Dry-run: generate configs/scripts without submitting
prepare batch-preprocess /bio/projects/compimaging/data/infectomics/imaging/datasets \
    -c reef_prepare_config.yaml \
    --dry-run
```

It discovers every `{data_root}/{name}/{name}.zarr` store (skipping sibling
`tracking.zarr` dirs), skips any already marked preprocessed in the CSV
sidecar (via `check_preprocessed(..., csv_dir=...)`), and generates +
submits `02_qc.sh`/`03_preprocess.sh`-equivalent SLURM jobs for the rest —
**always** via `csv_dir` (required for this command), never `.zattrs`
directly, so it works uniformly regardless of who owns a given dataset.
Re-running it as new datasets land only processes the new ones; nothing
already in the sidecar gets redone.

Since the batch may not have write access into each dataset's own
directory, generated `qc_config.yml`/`02_qc.sh`/`03_preprocess.sh` land
under `{jobs_dir or csv_dir}/_jobs/{dataset_name}/` instead — a writable
scratch location shared by everyone using the same `prepare_config.yaml`.

Reef also needs an explicit `--qos` on every `sbatch` (see
[docs/clusters/reef.md](../../../../docs/clusters/reef.md)) — set
`slurm.qc.qos`/`slurm.preprocess.qos` in the config; unset on Bruno-style
configs, which don't need it. Example Reef config:

```yaml
csv_dir: /bio/projects/compimaging/data/infectomics/metadata_csv
jobs_dir: /bio/projects/compimaging/data/infectomics/metadata_csv   # optional; defaults to csv_dir
workspace_dir: /mnt/main0/home/eduardo.hirata/repos/VisCy

qc:
  channel_names: [Phase3D]
  NA_det: 1.35
  lambda_ill: 0.450
  pixel_size: 0.1494
  device: cuda

preprocess:
  channel_names: -1
  num_workers: 32
  block_size: 32

slurm:
  qc:
    partition: h100_reserved
    gres: gpu:1
    qos: mid
    time: "00:30:00"
  preprocess:
    partition: cpu
    qos: mid
    time: "04:00:00"
```

`nfs_root`/`vast_root`/`concatenate` can be left at their defaults — this
command never reads them.
