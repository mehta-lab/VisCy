# Evaluation DAG

This document describes the **per-run** evaluation pipeline (one model on
one dataset). For the cross-model, cross-dataset matrix layout — including
the central linear-classifier registry that lets Wave-2 datasets fetch LC
pipelines trained on Wave-1 (infectomics-annotated) — see the companion
[`evaluation_matrix.md`](evaluation_matrix.md).

## Running with Nextflow (recommended)

```bash
module load nextflow/24.10.5

nextflow run applications/dynaclr/nextflow/main.nf -entry evaluation \
    --eval_config applications/dynaclr/configs/evaluation/DynaCLR-2D-MIP-BagOfChannels/infectomics-annotated.yaml \
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
  ├── {block_name}_cross_exp.yaml  # CPU step: dynaclr compute-mmd --combined (per-block)
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
  │        → smoothness/{model}_per_marker_smoothness.csv   # one row per marker
  │        → smoothness/{model}_smoothness_stats.csv        # mean ± std across markers
  │        → smoothness/*.pdf                               # per-marker + per-model plots
  │
  ├──► dynaclr compute-mmd               # one SLURM job per (experiment, block)
  │        -c {block_name}.yaml          # __ZARR_PATH__ substituted by Nextflow
  │        → mmd/{block_name}/mmd_results.csv
  │        → mmd/{block_name}/kinetics.pdf
  │        → mmd/{block_name}/activity_heatmap.pdf
  │
  ├──► dynaclr compute-mmd --combined    # pairwise cross-experiment batch effect detection
  │        -c {block_name}_cross_exp.yaml # only generated when combined_mode: true
  │        # For each marker shared by a pair of experiments, runs MMD per
  │        # (condition, time_bin) after per-pair mean centering.
  │        # Conditions are auto-discovered from data intersection.
  │        → mmd/{block_name}_cross_exp/combined_mmd_results.csv
  │        → mmd/{block_name}_cross_exp/kinetics.pdf
  │        → mmd/{block_name}_cross_exp/activity_heatmap.pdf
  │
  ├──► dynaclr run-linear-classifiers    # logistic regression probe
  │        -c linear_classifiers.yaml    # reads per-experiment zarrs directory + annotation CSVs
  │        # joins annotations on (fov_name, t, track_id); trains one LogisticRegression
  │        # per (task, marker); marker_filters omitted → auto-discovers all markers
  │        # writes trained pipelines to linear_classifiers/pipelines/ (in-run staging)
  │        # if publish_dir is set: atomically promotes the bundle to the central
  │        # LC registry as {publish_dir}/vN/ and updates the `latest` symlink.
  │        → linear_classifiers/metrics_summary.csv
  │        → linear_classifiers/{task}_summary.pdf
  │        → linear_classifiers/pipelines/{task}_{marker}.joblib
  │        → linear_classifiers/pipelines/manifest.json
  │        → {publish_dir}/vN/{task}_{marker}.joblib            (when publish_dir set)
  │        → {publish_dir}/vN/manifest.json                     (when publish_dir set)
  │        → {publish_dir}/latest -> vN                         (atomic symlink swap)
  │
  ├──► dynaclr append-annotations        # persist ground truth labels to per-experiment zarrs
  │        -c append_annotations.yaml    # reads annotation CSVs + writes task columns to zarr obs
  │        # only experiments with AnnotationSource entries are processed; others skipped
  │        → {experiment}.zarr (obs: infection_state, organelle_state, ...)
  │
  └──► dynaclr append-predictions        # apply saved classifiers
           -c append_predictions.yaml    # predicts on ALL cells per marker, not just annotated ones
           # pipelines_dir may be either:
           #   (a) in-run: {output_dir}/linear_classifiers/pipelines/ (default), or
           #   (b) external: a `latest` symlink into the central LC registry
           #       (e.g., /hpc/.../linear_classifiers/{model_name}/latest)
           # The symlink is resolved once at startup so the run is consistent
           # even if a new bundle is published mid-run. Logs feature_space (=
           # registry/{model_name}) and version (= vN) for traceability.
           → {experiment}.zarr (obs: predicted_infection_state, ...)
           → {experiment}.zarr (obsm: predicted_infection_state_proba, ...)
           → {experiment}.zarr (uns: predicted_infection_state_classes,
                                     predicted_infection_state_lc_version,
                                     predicted_infection_state_lc_feature_space,
                                     predicted_infection_state_lc_path, ...)

checkpoint.ckpt  (independent of predict/split — runs in parallel)
  │
  ▼
viscy export -c export_onnx.yml          # export backbone to ONNX
  │
  ▼
model.onnx + CTC datasets ({seq}_ERR_SEG/, {seq}/, {seq}_GT/TRA/)
  │
  ▼
dynaclr evaluate-tracking-accuracy \    # ILP tracking on CTC benchmarks
    -c tracking_accuracy.yaml           # loops over (model, dataset, sequence)
  │    builds tracksdata graph from segmentation masks
  │    runs ONNX inference on cell crops → dynaclr_similarity edge cost
  │    solves ILP; compares to GT via evaluate_ctc_metrics
  │    set show_napari: true for interactive inspection
  ▼
tracking_accuracy/results.csv           # one row per (model, dataset, sequence)
tracking_accuracy/                      # grouped mean summary printed to stdout
```

After all enrichment steps complete, per-experiment zarrs contain:

- `.obs`: embeddings metadata + annotations (`infection_state`, etc.) + predictions (`predicted_infection_state`, etc.)
- `.obsm`: `X_pca`, `X_pca_combined`, `X_phate_combined`, `predicted_{task}_proba`
- `.uns`: `predicted_{task}_classes`, `predicted_{task}_lc_version`, `predicted_{task}_lc_feature_space`, `predicted_{task}_lc_path`

This enables plots colored by experiment, perturbation, annotation, and prediction from a single zarr. The `_lc_*` uns fields record exactly which LC bundle produced each predicted column (registry path, version tag, feature_space).

## Central LC registry

Linear-classifier pipelines can be **published** to a central per-model
registry instead of (or in addition to) the per-run `output_dir`. This lets
later evaluations on different datasets reuse the same trained classifiers
without retraining.

### Layout

```
/hpc/projects/organelle_phenotyping/models/linear_classifiers/
├── DynaCLR-2D-MIP-BagOfChannels/
│   ├── latest -> v3                        # symlink (relative target)
│   ├── v1/ {manifest.json, *.joblib}
│   ├── v2/
│   └── v3/
├── DynaCLR-2D-BagOfChannels-v3/  { same }
├── DynaCLR-classical/             { same }
├── DINOv3-temporal-MLP-2D-BagOfChannels-v1/  { same }
└── DINOv3-frozen/                 { same }
```

The directory name (e.g. `DynaCLR-2D-MIP-BagOfChannels`) is the
**feature_space** identifier — pipelines from one model's registry are
*not* applicable to a different model's embeddings (different dim, different
distribution). The model name follows the training-config-stem convention
(see `evaluation_matrix.md` §7).

### Publishing (writer)

A Wave-1 leaf (training run) sets `linear_classifiers.publish_dir`:

```yaml
linear_classifiers:
  publish_dir: /hpc/projects/organelle_phenotyping/models/linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/
  # ... annotations, tasks, ...
```

`run-linear-classifiers` writes pipelines to a temp staging directory,
atomically renames to `vN/` (next available version), then atomically
swaps the `latest` symlink. Crash-safe: a partial bundle never appears as
`vN/`.

### Fetching (reader)

A Wave-2 leaf (evaluation on a different dataset) sets
`append_predictions.pipelines_dir`:

```yaml
append_predictions:
  pipelines_dir: /hpc/projects/organelle_phenotyping/models/linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/latest
```

`append-predictions` resolves the symlink **once** at startup and uses the
resolved `vN/` for the rest of the run, so a publish during the run does
not affect output. The resolved path's parent name (`DynaCLR-2D-MIP-BagOfChannels`)
becomes `feature_space` in the manifest log.

### Manifest format

```json
{
  "trained_at": "2026-04-24T15:33:21+00:00",
  "pipelines": [
    {"task": "infection_state", "marker_filter": "G3BP1", "path": "infection_state_G3BP1.joblib"},
    {"task": "infection_state", "marker_filter": "SEC61B", "path": "infection_state_SEC61B.joblib"}
  ]
}
```

Lineage (model name + version) lives in the directory structure, not the
manifest. Reproducibility comes from pinning a specific `vN` (instead of
`latest`) in paper-rerun scripts.

### Pinning vs. latest

```yaml
# active development — picks up the latest published bundle
pipelines_dir: /hpc/.../linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/latest

# paper rerun — frozen at submission time
pipelines_dir: /hpc/.../linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/v2
```

## Nextflow DAG (process dependency graph)

```
checkpoint.ckpt ──────────────────────────────────────────────────────────────┐
  │                                                                             │
  ▼                                                                             ▼
PREPARE_CONFIGS                                                        EXPORT_ONNX (optional)
  │                                                                             │
  ▼                                                                             ▼
PREDICT (GPU)                                                    model.onnx + CTC datasets
  │                                                                             │
  ▼                                                                             ▼
SPLIT (CPU light)                                              TRACKING_ACCURACY (CPU)
  │                                                              → results.csv
  ├─[scatter]─► REDUCE ─[gather]─► REDUCE_COMBINED ─┐
  │                                                   │
  ├─► APPEND_ANNOTATIONS ───────────────────────────►├─[scatter]─► PLOT
  │                                                   │ [gather]─► PLOT_COMBINED
  ├─► LINEAR_CLASSIFIERS ─► APPEND_PREDICTIONS ─────►┘
  │
  ├─[scatter]─► SMOOTHNESS ─[gather]─► SMOOTHNESS_GATHER
  ├─[scatter per (exp,block)]─► MMD ─[gather]─► MMD_PLOT_HEATMAP
  └─[gather per block]─► MMD_COMBINED
```

Key: **scatter** = one SLURM job per experiment (parallel). **gather** = waits for all scatter jobs.

`TRACKING_ACCURACY` is independent of the embedding pipeline — it reads directly from an ONNX
model and CTC-format data. Run it manually or as a separate Nextflow job alongside the main DAG.

`APPEND_ANNOTATIONS` and `APPEND_PREDICTIONS` emit a `'skip'` signal when not present in
`steps`, so `PLOT` and `PLOT_COMBINED` always proceed once `REDUCE_COMBINED` finishes.

## CTC Tracking Accuracy Benchmark

Standalone benchmark that evaluates whether DynaCLR embeddings improve cell tracking
accuracy on [Cell Tracking Challenge](https://celltrackingchallenge.net/) datasets.
**Not part of the Nextflow embedding pipeline** — run independently after exporting an ONNX model.

### Approach

```
CTC segmentation masks + raw images
  │
  ▼
tracksdata graph (RegionPropsNodes + DistanceEdges)
  │
  ├── baseline: IoU edge weights (no model)
  │
  └── DynaCLR: ONNX inference on cell crops
                → dynaclr_similarity × spatial_dist_weight as ILP edge cost
  │
  ▼
ILPSolver → tracked graph
  │
  ▼
evaluate_ctc_metrics vs. ground truth
  │
  ▼
results.csv (model × dataset × sequence × CTC metrics)
```

### Usage

```bash
dynaclr evaluate-tracking-accuracy -c tracking_accuracy_config.yaml
```

### Config format

```yaml
models:
  - path: /hpc/projects/.../model_ckpt146.onnx
    label: DynaCLR-classical
  - path: /hpc/projects/.../model_ckpt185.onnx
    label: DynaCLR-timeaware
  - path: null          # baseline: IoU + spatial distance only
    label: baseline-iou

datasets:
  - path: /hpc/reference/group.royer/CTC/training/BF-C2DL-HSC
    sequences: ["01", "02"]
  - path: /hpc/reference/group.royer/CTC/training/Fluo-C2DL-Huh7
    sequences: ["01", "02"]

crop_shape: [64, 64]          # must match the model's training resolution
distance_threshold: 325.0     # spatial candidate edge threshold (pixels)
n_neighbors: 10
delta_t: 5                    # max frame gap for candidate edges
batch_size: 128
output_dir: /path/to/tracking_accuracy_results
```

### Output

**`results.csv`** — one row per (model, dataset, sequence):

| Column | Description |
|--------|-------------|
| `model` | Model label |
| `dataset` | CTC dataset name |
| `sequence` | Sequence number (01, 02) |
| `LNK` | CTC Linking metric |
| `TRA` | Tracking metric |
| `DET` | Detection metric |
| `CHOTA` | Cell-specific HOTA |
| `HOTA` | Higher Order Tracking Accuracy |
| `MOTA` | Multiple Object Tracking Accuracy |
| `IDF1` | ID F1 score |
| `BIO(0)` | Biological metric |
| `OP_CLB(0)` | Combined linking+bio score |

Prints a grouped summary (mean over sequences) at the end.

### Prerequisites

1. Export the model to ONNX:
   ```bash
   viscy export -c export_onnx.yml
   ```
2. CTC datasets must have `{seq}_ERR_SEG/`, `{seq}/`, and `{seq}_GT/TRA/` subdirectories.
3. Install eval dependencies: `uv sync --all-packages --extra eval`

## Cross-model comparison

After running evals for multiple models, compare results with:

```bash
python applications/dynaclr/scripts/evaluation/compare_evals.py -c eval_registry.yml
```

Registry format:

```yaml
models:
  - name: DynaCLR-v3
    eval_dir: /path/to/eval_v3
  - name: DINOv3-MLP
    eval_dir: /path/to/eval_dino
output_dir: /path/to/comparison_output
fdr_threshold: 0.05
```

Auto-discovers results from each `eval_dir` and produces overlaid plots and summary CSVs for
smoothness, linear classifiers, and MMD.

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
| Smoothness | `dynaclr evaluate-smoothness -c smoothness.yaml` | {experiment}.zarr | per_marker_smoothness.csv, smoothness_stats.csv |
| MMD (per-exp) | `dynaclr compute-mmd -c {block}.yaml` | {experiment}.zarr | mmd/{block}/mmd_results.csv |
| MMD (combined) | `dynaclr compute-mmd --combined -c {block}_cross_exp.yaml` | all {experiment}.zarr | mmd/{block}_cross_exp/combined_mmd_results.csv |
| MMD (pooled) | `dynaclr compute-mmd --pooled -c pooled.yaml` | all {experiment}.zarr | mmd_results.csv |
| Linear probe | `dynaclr run-linear-classifiers -c clf.yaml` | per-experiment zarrs + annotations | metrics_summary.csv, {task}_summary.pdf, pipelines/ |
| Append annotations | `dynaclr append-annotations -c append_annotations.yaml` | per-experiment zarrs + annotation CSVs | zarrs with obs annotation columns |
| Append predictions | `dynaclr append-predictions -c append_predictions.yaml` | per-experiment zarrs + pipelines/ | zarrs with predicted_{task} in obs/obsm/uns |
| Compare models | `python compare_evals.py -c eval_registry.yml` | multiple eval dirs | comparison CSVs + plots |
| CTC tracking | `dynaclr evaluate-tracking-accuracy -c tracking_accuracy.yaml` | ONNX model + CTC datasets | tracking_accuracy/results.csv |

## Placeholder pattern

Template YAMLs (`reduce.yaml`, `smoothness.yaml`, `{block}.yaml`, `plot.yaml`) contain `__ZARR_PATH__`
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

For `reduce_combined.yaml`, `plot_combined.yaml`, and `{block}_cross_exp.yaml`, Nextflow collects
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

Use `configs/evaluation/recipes/mmd_defaults.yml` as a base to avoid repeating MMD algorithm parameters:

```yaml
# Per-experiment (template — __ZARR_PATH__ substituted at runtime)
base: recipes/mmd_defaults.yml
input_path: __ZARR_PATH__
output_dir: /path/to/evaluation/mmd/perturbation/
group_by: perturbation
comparisons:
  - cond_a: uninfected
    cond_b: ZIKV
    label: "uninfected vs ZIKV"
embedding_key: null             # null = raw .X; or "X_pca", "X_pca_combined"
temporal_bin_size: 4.0          # uniform bin width in hours (null = aggregate)
# temporal_bins: [0, 6, 12, 24] # alternative: explicit bin edges (mutually exclusive)
mmd:
  balance_samples: true         # subsample larger group to match smaller
  share_bandwidth_from: "uninfected vs uninfected"  # reuse bandwidth from baseline comparison
map_settings:
  enabled: true                 # compute mAP via copairs alongside MMD

# Cross-experiment ({block}_cross_exp.yaml — input_paths substituted at runtime)
# No comparisons — conditions auto-discovered from data intersection.
base: recipes/mmd_defaults.yml
input_paths: [__ZARR_PATH__]
output_dir: /path/to/evaluation/mmd/perturbation_cross_exp/
group_by: perturbation
temporal_bin_size: 4.0

# Pooled (standalone CLI only — not generated by orchestrator)
base: recipes/mmd_defaults.yml
input_paths:
  - /path/to/exp_A.zarr
  - /path/to/exp_B.zarr
output_dir: /path/to/evaluation/mmd/pooled/
comparisons:
  - cond_a: uninfected
    cond_b: ZIKV
    label: "uninfected vs ZIKV"
condition_aliases:
  uninfected: [uninfected, uninfected1, uninfected2]  # map variants to canonical name
```

## MMD output columns

### Per-experiment and pooled (`mmd_results.csv`)

| Column | Description |
|--------|-------------|
| `experiment` | Experiment name (absent in pooled output) |
| `marker` | Organelle marker (e.g., "TOMM20", "G3BP1") |
| `cond_a` | Reference/control condition |
| `cond_b` | Treatment condition |
| `label` | Human-readable comparison label |
| `hours_bin_start` | Start of temporal bin (NaN if no binning) |
| `hours_bin_end` | End of temporal bin (NaN if no binning) |
| `n_a` | Cells from `cond_a` used after subsampling |
| `n_b` | Cells from `cond_b` used after subsampling |
| `mmd2` | Unbiased MMD² estimate |
| `p_value` | Permutation test p-value (Phipson & Smyth smoothed) |
| `q_value` | BH-corrected FDR (pooled mode only) |
| `bandwidth` | Gaussian RBF bandwidth |
| `effect_size` | mmd2 / bandwidth (scale-free) |
| `activity_zscore` | (mmd2 − null_mean) / null_std — normalized against permutation null |
| `map_value` | Mean Average Precision (NaN if map_settings.enabled=false) |
| `map_p_value` | mAP permutation p-value (NaN if map_settings.enabled=false) |
| `embedding_key` | Embedding used ("X" or obsm key) |

### Cross-experiment (`combined_mmd_results.csv`)

| Column | Description |
|--------|-------------|
| `marker` | Organelle marker |
| `exp_a` | First experiment in the pair |
| `exp_b` | Second experiment in the pair |
| `condition` | Condition value matched across experiments |
| `hours_bin_start` | Start of temporal bin (NaN if no binning) |
| `hours_bin_end` | End of temporal bin (NaN if no binning) |
| `n_a` | Cells from `exp_a` used |
| `n_b` | Cells from `exp_b` used |
| `mmd2` | Unbiased MMD² estimate |
| `p_value` | Permutation test p-value |
| `bandwidth` | Gaussian RBF bandwidth |
| `effect_size` | mmd2 / bandwidth |
| `activity_zscore` | (mmd2 − null_mean) / null_std |
| `embedding_key` | Embedding used |

## Linear classifiers output columns

| Column | Description |
|--------|-------------|
| `task` | Classification task (e.g., `infection_state`) |
| `marker_filter` | Marker used to filter cells (one row per marker per task) |
| `n_samples` | Total annotated cells used |
| `val_accuracy` | Validation accuracy |
| `val_weighted_f1` | Validation weighted F1 |
| `val_auroc` | Validation AUROC (OvR macro for multiclass) |
| `train_*` | Training set counterparts of the above |
| `val_{class}_f1` | Per-class F1 on validation set |
