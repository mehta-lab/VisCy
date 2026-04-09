# Evaluation DAG

## Running with Nextflow (recommended)

```bash
module load nextflow/24.10.5

nextflow run applications/dynaclr/nextflow/main.nf \
    --eval_config applications/dynaclr/configs/evaluation/DynaCLR-2D-MIP-BagOfChannels.yaml \
    --workspace_dir /hpc/mydata/eduardo.hirata/repos/viscy \
    -resume
```

`-resume` makes Nextflow skip steps whose outputs already exist. Re-run the same command after a failure — Nextflow picks up from where it left off.

### Local test (no SLURM)

```bash
nextflow run applications/dynaclr/nextflow/main.nf \
    --eval_config applications/dynaclr/configs/evaluation/DynaCLR-2D-MIP-BagOfChannels_test.yaml \
    --workspace_dir /hpc/mydata/eduardo.hirata/repos/viscy \
    -profile local \
    -resume
```

## Pipeline entry point

`dynaclr prepare-eval-configs` (also aliased as `dynaclr evaluate`) generates all YAML configs
under `output_dir/configs/` and prints a JSON manifest to stdout. Nextflow reads the manifest
to wire steps together.

```
eval_config.yaml
  │
  ▼
dynaclr prepare-eval-configs -c eval_config.yaml   # writes configs/ + manifest JSON
  │
  ▼
output_dir/configs/
  ├── eval.yaml                    # copy of input config (for re-runs)
  ├── predict.yml                  # GPU step: viscy predict
  ├── reduce.yaml                  # template: dynaclr reduce-dimensionality (per-experiment)
  ├── reduce_combined.yaml         # CPU step: dynaclr combined-dim-reduction (joint)
  ├── smoothness.yaml              # template: dynaclr evaluate-smoothness (per-experiment)
  ├── plot.yaml                    # template: dynaclr plot-embeddings (per-experiment)
  ├── plot_combined.yaml           # CPU step: dynaclr plot-embeddings (all experiments)
  ├── {block_name}.yaml            # template: dynaclr compute-mmd (per-experiment, per-block)
  ├── {block_name}_combined.yaml   # CPU step: dynaclr compute-mmd --combined (per-block)
  └── linear_classifiers.yaml      # CPU step (optional)
```

## Step-by-step detail

```
checkpoint.ckpt + cell_index.parquet
  │
  ▼
viscy predict -c predict.yml              # MultiExperimentDataModule predict mode
  │    EmbeddingWriter callback           # normalizations + z_reduction, no augmentations
  ▼                                       # obs: fov_name, id, t, track_id,
embeddings/embeddings.zarr                #   experiment, marker, perturbation,
  │  (AnnData: .X=features,              #   hours_post_perturbation, organelle, well, microscope
  │   .obs=cell metadata)
  │
  ▼
dynaclr split-embeddings \
    --input embeddings/embeddings.zarr \
    --output-dir embeddings/
  │  Splits by obs["experiment"], deletes combined zarr
  │  Also writes configs/viewer.yaml (datasets: {exp: {hcs_plate, anndata}})
  │  hcs_plate read from obs["store_path"] of each split zarr
  ▼
embeddings/{experiment_A}.zarr
embeddings/{experiment_B}.zarr
  ...
configs/viewer.yaml               # nd-embedding viewer config (also valid input
  ...                             # for combined-dim-reduction via datasets: key)
  │
  ├──► dynaclr reduce-dimensionality      # PCA only (per experiment, parallel SLURM jobs)
  │        -c reduce.yaml                 # __ZARR_PATH__ substituted by Nextflow
  │        → {experiment}.zarr (obsm: X_pca)
  │        NOTE: skip PHATE here to avoid computing it twice
  │
  │  (after reduce-dimensionality finishes for ALL experiments)
  │
  ├──► dynaclr combined-dim-reduction     # joint PCA + PHATE across all experiments
  │        -c reduce_combined.yaml        # fits on concatenated embeddings
  │        → {experiment}.zarr (obsm: X_pca_combined, X_phate_combined)
  │
  │  (after combined-dim-reduction finishes)
  │
  ├──► dynaclr plot-embeddings            # per-experiment PCA scatter (X_pca)
  │        -c plot.yaml                   # parallel SLURM jobs, one per experiment
  │        → plots/{experiment}/*.pdf
  │
  ├──► dynaclr plot-embeddings            # all-experiments combined (X_pca_combined, X_phate_combined)
  │        -c plot_combined.yaml          # concatenates all zarrs into one figure
  │        → plots/combined/*.pdf
  │
  ├──► dynaclr evaluate-smoothness        # temporal smoothness + dynamic range
  │        -c smoothness.yaml             # parallel SLURM jobs, one per experiment
  │        → smoothness/combined_smoothness_stats.csv
  │        → smoothness/*.pdf
  │
  ├──► dynaclr compute-mmd               # one SLURM job per (experiment, block)
  │        -c {block_name}.yaml
  │        # Block: perturbation — biology signal with temporal bins
  │        → perturbation/{experiment}_mmd_results.csv
  │        → perturbation/{experiment}_kinetics.pdf
  │        → perturbation/{experiment}_heatmap.pdf
  │        # Block: batch_qc — microscope comparisons on uninfected cells only
  │        → batch_qc/{experiment}_mmd_results.csv
  │        → batch_qc/{experiment}_heatmap.pdf
  │
  ├──► dynaclr compute-mmd --combined    # cross-experiment MMD with batch centering (optional)
  │        -c {block_name}_combined.yaml # only generated when combined_mode: true
  │        → perturbation_combined/combined_mmd_results.csv
  │        → perturbation_combined/combined_kinetics.pdf
  │        → perturbation_combined/combined_heatmap.pdf
  │
  └──► dynaclr run-linear-classifiers    # logistic regression probe
           -c linear_classifiers.yaml    # reads per-experiment zarrs directory + annotation CSVs
           # joins annotations on (fov_name, t, track_id); trains one LogisticRegression
           # per (task, marker_filter); annotated subset only (~35k cells from 5 experiments)
           → linear_classifiers/metrics_summary.csv
           → linear_classifiers/metrics_summary.pdf  # bar charts + per-task ROC curves
```

## Nextflow DAG (process dependency graph)

```
PREPARE_CONFIGS
  │
  ▼
PREDICT (GPU)
  │
  ▼
SPLIT (CPU light)
  │
  ├─[scatter]─► REDUCE ─[gather]─► REDUCE_COMBINED ─[scatter]─► PLOT
  │                                                 └─[gather]─► PLOT_COMBINED
  │
  ├─[scatter]─► SMOOTHNESS
  ├─[scatter per (exp,block)]─► MMD
  ├─[gather per block]─► MMD_COMBINED
  └─► LINEAR_CLASSIFIERS
```

Key: **scatter** = one SLURM job per experiment (parallel). **gather** = waits for all scatter jobs.

## Key commands

| Step | Command | Input | Output |
|------|---------|-------|--------|
| Config gen | `dynaclr prepare-eval-configs -c eval.yaml` | eval config | configs/ + manifest JSON |
| Predict | `viscy predict -c predict.yml` | checkpoint + parquet | embeddings/embeddings.zarr |
| Split | `dynaclr split-embeddings --input ... --output-dir ...` | combined zarr | per-experiment zarrs + `configs/viewer.yaml` |
| Dim reduction | `dynaclr reduce-dimensionality -c reduce.yaml` | {experiment}.zarr | zarr with X_pca |
| Combined reduction | `dynaclr combined-dim-reduction -c reduce_combined.yaml` | all {experiment}.zarr | zarrs with X_pca_combined/X_phate_combined |
| Plots (per-exp) | `dynaclr plot-embeddings -c plot.yaml` | {experiment}.zarr | plots/{experiment}/*.pdf |
| Plots (combined) | `dynaclr plot-embeddings -c plot_combined.yaml` | all {experiment}.zarr | plots/combined/*.pdf |
| Smoothness | `dynaclr evaluate-smoothness -c smoothness.yaml` | {experiment}.zarr | smoothness_stats.csv |
| MMD (per-exp) | `dynaclr compute-mmd -c mmd.yaml` | {experiment}.zarr | mmd/{experiment}_mmd_results.csv |
| MMD (combined) | `dynaclr compute-mmd --combined -c mmd_combined.yaml` | all {experiment}.zarr | mmd/combined_mmd_results.csv |
| Linear probe | `dynaclr run-linear-classifiers -c clf.yaml` | per-experiment zarrs + annotations | metrics_summary.csv, metrics_summary.pdf |

## Placeholder pattern

Template YAMLs (`reduce.yaml`, `smoothness.yaml`, `mmd.yaml`, `plot.yaml`) contain `__ZARR_PATH__`
as a placeholder for `input_path`. `plot.yaml` also contains `__PLOT_DIR__`. Nextflow process
scripts substitute these inline with Python one-liners before calling the CLI command:

```python
import yaml
with open('reduce.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['input_path'] = '/path/to/experiment.zarr'
with open('reduce_patched.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
```

For `reduce_combined.yaml`, `plot_combined.yaml`, and `mmd_*_combined.yaml`, Nextflow collects
all zarr paths and writes the `input_paths` list directly.

## Notes

- `MultiExperimentDataModule` supports `stage="predict"` since the eval orchestrator was added.
  It uses the full cell index (no train/val split), applies only normalizations + z-reduction (no augmentations).
- `BatchedChannelWiseZReductiond` is architecturally required for 2D models even at inference time
  (converts 3D z-stack → 2D MIP/center-slice). The orchestrator moves it from `augmentations`
  to `normalizations` in the generated predict config.
- Dimensionality reductions (PCA, PHATE) are **not** computed inline during predict.
  They run as separate CPU steps after splitting, keeping predict fast.
- The `combined-dim-reduction` step fits reductions on all experiments jointly and writes
  `X_pca_combined` / `X_phate_combined` back to each per-experiment zarr.
- PHATE is not computed per-experiment by default (`reduce_dimensionality.phate: null`). Run it only jointly via `reduce_combined`.
- `configs/viewer.yaml` is generated after split and can be passed directly to `dynaclr combined-dim-reduction`.
- MMD reads `.X` (raw backbone embeddings) by default. It can also run on `X_pca` or `X_pca_combined` via `embedding_key`.
- Embeddings obs carries `organelle`, `well`, and `microscope` in addition to `experiment`, `marker`, `perturbation`, `hours_post_perturbation`.

## MMD config format

```yaml
# Per-experiment (mmd.yaml template — __ZARR_PATH__ substituted at runtime)
input_path: __ZARR_PATH__
output_dir: /path/to/evaluation/mmd/
group_by: perturbation          # obs column whose values cond_a/cond_b reference
comparisons:
  - cond_a: uninfected          # reference/control group value
    cond_b: ZIKV                # treatment group value
    label: "uninfected vs ZIKV" # used in filenames and plot titles
embedding_key: null             # null = raw .X embeddings; or "X_pca"
mmd:
  n_permutations: 1000
  max_cells: 2000               # subsample per group for tractability
  min_cells: 20                 # skip groups with too few cells
  seed: 42
temporal_bins: [0, 2, 4, 8, 12, 24]  # hours_post_perturbation bin edges (null = aggregate)
save_plots: true
```

## MMD output columns

| Column | Description |
|--------|-------------|
| `experiment` | Experiment name (or "combined" for cross-experiment) |
| `marker` | Organelle marker (e.g., "TOMM20", "SEC61B") |
| `cond_a` | First condition in the comparison (typically reference/control) |
| `cond_b` | Second condition in the comparison (typically treatment) |
| `label` | Human-readable label for this comparison (used in filenames and plot titles) |
| `hours_bin_start` | Start of temporal bin (NaN if no binning) |
| `hours_bin_end` | End of temporal bin (NaN if no binning) |
| `n_a` | Number of cells from `cond_a` used |
| `n_b` | Number of cells from `cond_b` used |
| `mmd2` | Unbiased MMD^2 estimate |
| `p_value` | Permutation test p-value |
| `bandwidth` | Gaussian RBF bandwidth used |
| `effect_size` | mmd2 / bandwidth (normalized, scale-free) |
| `embedding_key` | Which embedding was used ("X" or obsm key) |

## Linear classifiers

### Annotated datasets

The annotated collection covers 5 logical experiments from 2 physical experiments:

| Collection YAML | Parquet |
|---|---|
| `configs/collections/DynaCLR-2D-MIP-BagOfChannels-annotated.yml` | `/hpc/projects/organelle_phenotyping/models/collections/DynaCLR-2D-MIP-BagOfChannels-annotated.parquet` |

Experiments and annotation coverage:

| Experiment | Annotation CSV | Annotated wells | Tasks |
|---|---|---|---|
| `2025_01_28_A549_G3BP1_ZIKV_DENV_G3BP1` | `annotations/2025_01_28_.../...csv` | B/4, C/4 | infection, division, organelle, death |
| `2025_07_24_A549_G3BP1_ZIKV` | `annotations/2025_07_24_.../...csv` | C/1, C/2 | infection, division, organelle, death |
| `2025_07_24_A549_SEC61_ZIKV` | (same) | A/2 (A/1 not annotated) | infection, division, organelle, death |
| `2025_07_24_A549_viral_sensor` | (same) | C/1, C/2, A/2 | infection, division, organelle, death |
| `2025_07_24_A549_Phase3D` | (same) | C/1, C/2, A/2 | infection, division, organelle, death |

TOMM20 (`2025_07_24`) excluded — wells B/1, B/2 not annotated. ALFI excluded for now.

### Annotation join

Embeddings obs does **not** carry the `id` (Ultrack node ID) column. Annotations are joined on the composite key `(fov_name, t, track_id)`, which is unique in both the embeddings and annotation CSVs.

### Config format

```yaml
embeddings_path: /path/to/evaluation/embeddings/  # directory of per-experiment zarrs (post-split)
output_dir: /path/to/evaluation/linear_classifiers/
annotations:
  - experiment: "2025_01_28_A549_G3BP1_ZIKV_DENV_G3BP1"
    path: /hpc/projects/organelle_phenotyping/datasets/annotations/2025_01_28_A549_G3BP1_ZIKV_DENV/2025_01_28_A549_G3BP1_ZIKV_DENV_combined_annotations.csv
  - experiment: "2025_07_24_A549_G3BP1_ZIKV"
    path: /hpc/projects/organelle_phenotyping/datasets/annotations/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV_combined_annotations.csv
  # ... (same CSV repeated for each logical experiment from the same physical experiment)
tasks:
  - task: infection_state      # marker_filters omitted = one classifier across all markers
  - task: cell_division_state
  - task: organelle_state
  - task: cell_death_state
use_scaling: true
split_train_data: 0.8
random_seed: 42
```

### Linear classifiers output columns

| Column | Description |
|--------|-------------|
| `task` | Classification task (e.g., `infection_state`) |
| `marker_filter` | Marker used to filter cells (`null` = all markers) |
| `n_samples` | Total annotated cells used |
| `val_accuracy` | Validation accuracy |
| `val_weighted_f1` | Validation weighted F1 |
| `val_auroc` | Validation AUROC (OvR macro for multiclass) |
| `train_*` | Training set counterparts of the above |
| `val_{class}_f1` | Per-class F1 on validation set |
