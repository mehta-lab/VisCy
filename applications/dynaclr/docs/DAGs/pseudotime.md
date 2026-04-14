# Pseudotime DAG

Pipeline for DTW-based pseudotime alignment of cell trajectories.
Each stage is a standalone Python script; outputs from one stage feed the next.

## Directory layout

```
pseudotime/
├── multi_template.yaml          # shared config for all stages
├── pred_dirs/                   # per-date symlink dirs → evaluation embeddings
│   ├── 2025_07_24/
│   └── 2025_07_22/
├── 0-build_templates/
│   ├── build_templates.py
│   ├── lineage_overview.py      # optional: track counts by division/infection state
│   └── templates/               # output: template_*.zarr
├── 1-align_cells/
│   ├── align_cells.py
│   ├── plotting.py              # optional: diagnostic plots for alignments
│   └── alignments/              # output: alignments_{template_name}.parquet
├── 2-evaluate_dtw/
│   ├── evaluate_dtw.py
│   └── evaluation/              # output: evaluation_summary.parquet, plots
├── 3-organelle_dynamics/
│   ├── organelle_dynamics.py
│   ├── plotting.py              # optional: cell montage plots along pseudotime
│   └── organelle_dynamics/      # output: organelle_distances.parquet, plots
└── 4-export_anndata/
    ├── export_anndata.py
    └── anndata/                 # output: {dataset_id}_dtw.zarr
```

## DAG

```
[cell_index.parquet]  [annotations.csv]
         │                   │
         ▼                   ▼
  [embedding *.zarr]  ──► 0-build_templates/build_templates.py
  (evaluation_lc_v1/        │   per-template: track filter, align,
   embeddings/)             │   DBA averaging (PCA + z-score)
                            ▼
                  templates/template_*.zarr
                  (one zarr per template name:
                   infection_nondividing,
                   infection_dividing_before,
                   infection_dividing_after)
                            │
                            ▼
  [embedding *.zarr]  ──► 1-align_cells/align_cells.py
  [annotations.csv]         │   DTW-align each track to template
                            │   → pseudotime score per cell
                            ▼
                  alignments/alignments_{template_name}.parquet
                  (fov_name, track_id, t, pseudotime,
                   dataset_id, template_name, ...)
                            │
                            ├──► 1-align_cells/plotting.py (optional)
                            │    --alignments alignments/alignments_{name}.parquet
                            │    → plots/pseudotime_curves.png, etc.
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
  2-evaluate_dtw/              3-organelle_dynamics/
  evaluate_dtw.py              organelle_dynamics.py
  [annotations.csv]            [embedding *.zarr per organelle]
       │                              │
       │  AUC vs infection_state,     │  distance from baseline
       │  onset concordance           │  along pseudotime axis
       ▼                              ▼
  evaluation/                  organelle_dynamics/
  evaluation_summary.parquet   organelle_distances.parquet
  per_timepoint_auc.parquet    aggregated_curves.parquet
  failed_alignments.csv        onset_summary.parquet
  plots/                       plots/
       │
       │ (optional)
       ▼
  4-export_anndata/export_anndata.py
  [embedding *.zarr]
       │
       ▼
  anndata/{dataset_id}_dtw.zarr
  (embeddings + pseudotime + annotations merged)
```

## MIP model note

For the MIP model, embedding zarrs are per-(date, channel) in a flat directory rather than split
by sensor/organelle/phase. The `pred_dirs/` symlink directories solve this: each contains only
the zarrs for one date, so glob patterns like `*_viral_sensor_*.zarr` match exactly one file.
The `data_zarr` field in `multi_template.yaml` points to the source image zarr used for cell
crop montages in `3-organelle_dynamics/plotting.py` — no `--data-zarr` flag needed.

## How to run

Run from each stage's subdirectory — scripts resolve sibling paths relative to their own location.

### Stage 0 — Build templates

```bash
cd 0-build_templates
python build_templates.py --config ../multi_template.yaml
```

Outputs one `templates/template_{name}.zarr` per template in `config["templates"]`.

#### Optional: lineage overview

```bash
python lineage_overview.py --config ../multi_template.yaml
```

Outputs `lineage_overview/{dataset_id}_lineages.csv`, `combined_lineages.csv`, `track_survival_curve.png`.

### Stage 1 — Align cells

```bash
cd 1-align_cells
python align_cells.py --config ../multi_template.yaml
```

Reads `../0-build_templates/templates/template_{template_name}.zarr`.
Outputs `alignments/alignments_{template_name}.parquet`.

#### Optional: diagnostic plots

```bash
python plotting.py \
    --config ../multi_template.yaml \
    --alignments alignments/alignments_infection_nondividing.parquet
```

Outputs `plots/pseudotime_curves.png`, `pseudotime_distribution.png`, `dtw_cost_distribution.png`, `warping_heatmap.png`.

### Stage 2 — Evaluate DTW (optional, needs annotations)

```bash
cd 2-evaluate_dtw
python evaluate_dtw.py --config ../multi_template.yaml
```

Reads all `../1-align_cells/alignments/alignments_*.parquet`.
Outputs `evaluation/evaluation_summary.parquet`, `per_timepoint_auc.parquet`, plots.

### Stage 3 — Organelle dynamics

```bash
cd 3-organelle_dynamics
python organelle_dynamics.py \
    --config ../multi_template.yaml \
    --alignments ../1-align_cells/alignments/alignments_infection_nondividing.parquet
```

Reads the specified alignments parquet.
Outputs `organelle_dynamics/organelle_distances.parquet`, `aggregated_curves.parquet`, plots.

#### Optional: cell montage plots

```bash
python plotting.py \
    --config ../multi_template.yaml \
    --alignments ../1-align_cells/alignments/alignments_infection_nondividing.parquet
```

### Stage 4 — Export AnnData

```bash
cd 4-export_anndata
python export_anndata.py \
    --config ../multi_template.yaml \
    --alignments ../1-align_cells/alignments/alignments_infection_nondividing.parquet
```

Reads the specified alignments parquet.
Outputs `anndata/{dataset_id}_dtw.zarr` with embeddings + pseudotime merged.

## Key config fields (`multi_template.yaml`)

| Field | Used by | Purpose |
|---|---|---|
| `data_zarr` | 3 plotting | source image zarr for cell crop montages |
| `embeddings` | 0, 1, 3 | glob patterns → zarr per channel |
| `datasets` | 0, 1, 3, 4 | pred_dir, annotations, fov_pattern, frame_interval |
| `templates` | 0 | track filters, DBA params, per-template dataset list |
| `alignment` | 1 | which template to align to, min_track_minutes |
| `organelle_dynamics` | 3 | per-organelle embedding key, dataset_ids, baseline range |

## Script arguments added vs upstream

Scripts in this pipeline were patched to accept explicit `--alignments` and related args
so they work with the `alignments_{template_name}.parquet` naming from the multi-template config:

| Script | Added arg | Purpose |
|---|---|---|
| `0-build_templates/lineage_overview.py` | _(none)_ | reads `embeddings.sensor` from config instead of hardcoded pattern |
| `1-align_cells/plotting.py` | `--alignments` | path to alignments parquet (default: `alignments/alignments.parquet`) |
| `3-organelle_dynamics/organelle_dynamics.py` | `--alignments` | path to alignments parquet |
| `3-organelle_dynamics/plotting.py` | `--alignments` | path to alignments parquet |
| `4-export_anndata/export_anndata.py` | `--alignments` | path to alignments parquet |
