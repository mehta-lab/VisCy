# Training DAG

## Prerequisites

Datasets must be AI-ready before building a collection. See [ai_ready_datasets.md](ai_ready_datasets.md)
for the full data preparation pipeline (`prepare run` → concatenate → QC → preprocess).

A dataset is ready when `prepare status` shows `preprocessed: yes` — meaning both
`normalization` and `focus_slice` metadata exist in the zarr zattrs.

## Step-by-step detail

```
zarr stores (preprocessed: normalization + focus_slice in zattrs)
tracking.zarr (per-dataset, synced from NFS)
  │
  ├──► collection.yml              # defines experiments, channels, perturbation_wells
  │                                # versioned in git under configs/collections/
  ▼
dynaclr build-cell-index \
    configs/collections/<collection>.yml \
    /hpc/projects/organelle_phenotyping/models/collections/<collection>.parquet \
    --num-workers 8
  │  reads tracking CSVs + zarr shape metadata
  │  one row per (cell, timepoint, channel)
  │  sets z=0 placeholder (overwritten in next step)
  ▼
<collection>.parquet  (raw: shape columns, z=0, no norm stats)
  │
  ▼
dynaclr preprocess-cell-index \
    /hpc/.../collections/<collection>.parquet \
    --focus-channel Phase3D --focus-level fov
  │  opens each unique FOV once from zarr zattrs:
  │    norm_mean/std/median/iqr/max/min  — per (cell, timepoint, channel)
  │    z_focus                           — focus plane the Z window centers on,
  │                                         from focus_slice at --focus-level
  │                                         (fov = per-FOV mean; per_timepoint =
  │                                         per-timepoint index). `z` (tracked
  │                                         position) is left unchanged.
  │  drops empty frames (max == 0)
  ▼
<collection>.parquet  (ready: self-contained, no zarr reads at training time)
  │
  ▼
viscy fit --config configs/training/<model>.yml
  │  OR: sbatch configs/training/<model>.sh   (SLURM, recommended)
  │  MultiExperimentDataModule reads parquet only at init
  │  tensorstore opens zarr lazily on first batch
  │  Z window centers on the parquet `z_focus` column (per-sample). If z_focus
  │  is null, falls back to the per-experiment z_range from zattrs focus_slice,
  │  else mid-stack. z_focus_offset sets the below/above split.
  ▼
checkpoints/  +  wandb logs
```

## Pipeline DAG (process dependency)

```
collection.yml
  │
  ▼
build-cell-index  (CPU, ~1 min)
  │
  ▼
preprocess-cell-index  (CPU, ~5 min, I/O bound)
  │
  ▼
viscy fit  (GPU, hours–days)
```

## Key commands


| Step                  | Command                                                                   | Input                                  | Output                                                    |
| --------------------- | ------------------------------------------------------------------------- | -------------------------------------- | --------------------------------------------------------- |
| Build cell index      | `dynaclr build-cell-index <collection.yml> <out.parquet> --num-workers 8` | collection YAML + zarr + tracking CSVs | parquet with TCZYX shape columns                          |
| Preprocess cell index | `dynaclr preprocess-cell-index <parquet> --focus-channel Phase3D --focus-level fov` | parquet + zarr zattrs        | parquet with norm stats + z_focus, empties removed        |
| Train (interactive)   | `uv run viscy fit --config configs/training/<model>.yml`                  | training config + parquet              | checkpoints + logs                                        |
| Train (SLURM)         | `sbatch configs/training/<model>.sh`                                      | training config + parquet              | checkpoints + logs                                        |
| Resume (SLURM)        | `CKPT_PATH=.../last.ckpt sbatch configs/training/<model>.sh`              | checkpoint path env var                | resumed checkpoints                                       |


## What lives where


| Data                                    | Location                                                  | When written                                 |
| --------------------------------------- | --------------------------------------------------------- | -------------------------------------------- |
| Pixel data (TCZYX arrays)               | zarr store on VAST                                        | `prepare run` → concatenate                  |
| Cell tracking (y, x, t, track_id)       | tracking.zarr on VAST                                     | `prepare run` → concatenate                  |
| Normalization stats (per FOV/timepoint) | zarr zattrs → parquet `norm_*` columns                    | `viscy preprocess` → `preprocess-cell-index` |
| Tracked cell z (per cell)               | tracking → parquet `z` column (unchanged by preprocess)   | `build-cell-index`                           |
| Focus plane (Z window center)           | zarr zattrs focus_slice → parquet `z_focus`               | `viscy preprocess` → `preprocess-cell-index --focus-level` |
| TCZYX shape per FOV                     | parquet columns                                           | `build-cell-index`                           |
| Collection definition                   | `configs/collections/<name>.yml` in git                   | manually authored                            |
| Parquet                                 | `/hpc/projects/organelle_phenotyping/models/collections/` | `build-cell-index`                           |


## collection.yml format

```yaml
name: <collection-name>
description: "..."

experiments:
  - name: <experiment_name>                     # {date}_{cell}_{marker}_{perturbation}
    data_path: /hpc/projects/.../dataset.zarr
    tracks_path: /hpc/projects/.../tracking.zarr
    channels:
      - name: "raw GFP EX488 EM525-45"         # zarr channel name (exact match)
        marker: G3BP1                           # protein label used in parquet
    perturbation_wells:
      uninfected: [C/1]
      infected: [C/2]
    interval_minutes: 30.0
    start_hpi: 3.5
    marker: G3BP1
    organelle: stress_granules
    moi: 5.0
    pixel_size_xy_um: 0.1494
    pixel_size_z_um: 0.174
```

Experiment name convention: `{date}_{cell_line}_{marker}_{perturbation}` —
perturbation suffix is always included (e.g., `_ZIKV`, `_DENV`, `_ZIKV_DENV`).

## Training config structure

Training configs use Lightning CLI `base:` inheritance:

```yaml
base:
  - ../recipes/trainer/fit.yml                              # seed, logger, callbacks
  - ../recipes/topology/ddp_2gpu.yml                        # accelerator/strategy/devices
  - ../recipes/model/contrastive_encoder_convnext_tiny.yml  # or dinov3_frozen_mlp.yml, cell_dino_frozen_mlp.yml

trainer:
  precision: bf16-mixed
  max_epochs: 150

data:
  cell_index_path: /hpc/.../collections/<collection>.parquet
  ...
```

Recipes are split into orthogonal axes (`trainer/`, `topology/`,
`data/`, `augmentations/`, `model/`) so leaves only re-declare what
varies per experiment.

SLURM `.sh` scripts export `PYTHONNOUSERSITE=1` and launch via `srun` for DDP.

## Reproducibility

Version `collection.yml` in git. The parquet is derived deterministically from:

1. The collection YAML (experiment definitions, channels, wells)
2. Tracking zarrs (cell positions)
3. Zarr zattrs (normalization + focus stats from `viscy preprocess` + `qc run`)

To reproduce: `build-cell-index` → `preprocess-cell-index` from the same collection YAML.

## Notes

- `preprocess-cell-index` overwrites the parquet in-place by default. Pass `--output` to write elsewhere.
- `--focus-channel Phase3D` selects which channel's `focus_slice` metadata feeds the `z_focus` column. Use the channel with sharpest axial contrast (label-free Phase3D for most experiments). `--focus-level {fov,per_timepoint}` picks per-FOV mean vs per-timepoint index.
- **Z window centering is parquet-first.** At training time the datamodule centers the Z window on the per-sample `z_focus` column. If `z_focus` is null it falls back to the per-experiment `z_range` (which `ExperimentRegistry` derives from `plate.zattrs["focus_slice"][channel]["dataset_statistics"]["z_focus_mean"]`, else mid-stack). Existing parquets without a `z_focus` column read as all-null → identical to the previous zattrs-only behavior.
- `z_focus` and `z` are distinct: `z_focus` = where the Z window centers; `z` = the tracked cell position (unchanged by preprocess). Datasets without focus_slice (e.g. static bbox-center data) can seed `z_focus = z` in their own prep script.
- The `z` column is carried through to embeddings obs during predict for downstream consumers.
- For performance tuning (num_workers, pin_memory, batch_size, augmentation placement), see [profiling.md](profiling.md) — authored after the first validated profiling sweep.
