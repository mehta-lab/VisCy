# Evaluation DAG

## Orchestrated pipeline (recommended)

```
training_config.yml + checkpoint.ckpt
  │
  ▼
dynaclr evaluate -c eval_config.yaml      # generates all configs + SLURM scripts
  │                                        # reads training config automatically
  │                                        # no manual YAML writing needed
  ▼
output_dir/configs/
  ├── eval.yaml                            # copy of input eval config (for re-runs)
  ├── predict.yml + predict.sh             # GPU step: viscy predict
  ├── split.sh                             # CPU step: dynaclr split-embeddings + viewer.yaml
  ├── reduce.yaml + reduce.sh             # CPU step: dynaclr reduce-dimensionality (per-experiment)
  ├── reduce_combined.yaml + .sh          # CPU step: dynaclr combined-dim-reduction (joint)
  ├── smoothness.yaml + smoothness.sh     # CPU step: dynaclr evaluate-smoothness (per-experiment)
  ├── plot.yaml + plot.sh                 # CPU step: dynaclr plot-embeddings (per-experiment, X_pca)
  ├── plot_combined.yaml + plot_combined.sh  # CPU step: dynaclr plot-embeddings (all experiments, X_pca_combined + X_phate_combined)
  ├── viewer.yaml                          # nd-embedding viewer config (generated after split)
  └── linear_classifiers.yaml + .sh      # CPU step (optional)
  │
  ▼  (submit chained SLURM jobs)
JOB_PREDICT=$(sbatch --parsable predict.sh)
JOB_SPLIT=$(sbatch --parsable --dependency=afterok:$JOB_PREDICT split.sh)
JOB_REDUCE=$(sbatch --parsable --dependency=afterok:$JOB_SPLIT reduce.sh)
JOB_REDUCE_COMBINED=$(sbatch --parsable --dependency=afterok:$JOB_REDUCE reduce_combined.sh)
sbatch --dependency=afterok:$JOB_REDUCE_COMBINED plot.sh
sbatch --dependency=afterok:$JOB_REDUCE_COMBINED plot_combined.sh
sbatch --dependency=afterok:$JOB_SPLIT smoothness.sh
sbatch --dependency=afterok:$JOB_SPLIT linear_classifiers.sh
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
  │  (AnnData: .X=features,              #   hours_post_perturbation
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
  ├──► dynaclr reduce-dimensionality      # PCA only (per experiment)
  │        -c reduce.yaml                 # shell script loops over *.zarr
  │        → {experiment}.zarr (obsm: X_pca)
  │        NOTE: skip PHATE here to avoid computing it twice
  │
  │  (after reduce-dimensionality finishes)
  │
  ├──► dynaclr combined-dim-reduction     # joint PCA + PHATE across all experiments
  │        -c reduce_combined.yaml        # fits on concatenated embeddings
  │        → {experiment}.zarr (obsm: X_pca_combined, X_phate_combined)
  │
  │  (after combined-dim-reduction finishes)
  │
  ├──► dynaclr plot-embeddings            # per-experiment PCA scatter (X_pca)
  │        -c plot.yaml                   # shell script loops over *.zarr
  │        → plots/{experiment}/*.pdf
  │
  ├──► dynaclr plot-embeddings            # all-experiments combined (X_pca_combined, X_phate_combined)
  │        -c plot_combined.yaml          # concatenates all zarrs into one figure
  │        → plots/combined/*.pdf
  │
  ├──► dynaclr evaluate-smoothness        # temporal smoothness + dynamic range
  │        -c smoothness.yaml             # shell script loops over *.zarr
  │        → smoothness/combined_smoothness_stats.csv
  │        → smoothness/*.pdf
  │
  └──► dynaclr run-linear-classifiers    # logistic regression probe (optional)
           -c linear_classifiers.yaml    # reads per-experiment zarrs + annotation CSVs
           → linear_classifiers/metrics_summary.csv
```

## Key commands

| Step | Command | Input | Output |
|------|---------|-------|--------|
| Orchestrate | `dynaclr evaluate -c eval.yaml` | training config + ckpt | configs/ + SLURM scripts |
| Predict | `viscy predict -c predict.yml` | checkpoint + parquet | embeddings/embeddings.zarr |
| Split | `dynaclr split-embeddings --input ... --output-dir ...` | combined zarr | per-experiment zarrs + `configs/viewer.yaml` |
| Dim reduction | `dynaclr reduce-dimensionality -c reduce.yaml` | {experiment}.zarr | zarr with X_pca/X_phate |
| Combined reduction | `dynaclr combined-dim-reduction -c reduce_combined.yaml` | all {experiment}.zarr | zarrs with X_pca_combined/X_phate_combined |
| Plots (per-exp) | `dynaclr plot-embeddings -c plot.yaml` | {experiment}.zarr | plots/{experiment}/*.pdf |
| Plots (combined) | `dynaclr plot-embeddings -c plot_combined.yaml` | all {experiment}.zarr concatenated | plots/combined/*.pdf |
| Smoothness | `dynaclr evaluate-smoothness -c smoothness.yaml` | {experiment}.zarr | smoothness_stats.csv |
| Linear probe | `dynaclr run-linear-classifiers -c clf.yaml` | per-experiment zarrs + annotations | metrics_summary.csv |

## Template YAML pattern

`reduce.yaml`, `smoothness.yaml`, and `plot.yaml` contain `__ZARR_PATH__` as a placeholder
for `input_path`. `plot.yaml` also contains `__PLOT_DIR__` for the per-experiment output dir.
The generated SLURM scripts substitute these at runtime by looping over `embeddings/*.zarr` with `sed`:

```bash
for zarr in "$EMBEDDINGS_DIR"/*.zarr; do
  name=$(basename "$zarr" .zarr)
  sed "s|__ZARR_PATH__|$zarr|g; s|__PLOT_DIR__|$PLOTS_DIR/$name|g" plot.yaml > /tmp/plot_$name.yaml
  uv run ... dynaclr plot-embeddings -c /tmp/plot_$name.yaml
done
```

For `reduce_combined.yaml` and `plot_combined.yaml`, the shell script uses a Python one-liner
to glob all zarrs and write the `input_paths` list dynamically. `plot_combined.yaml` accepts
`input_paths` (list) and concatenates all zarrs into one figure.

**Re-running individual steps:** copy `configs/eval.yaml`, edit the `steps:` list to only the
step(s) you want, and re-run `dynaclr evaluate -c eval_rerun.yaml --mode local`.

## Linear classifiers config format

```yaml
embeddings_path: /path/to/evaluation/embeddings/  # directory of per-experiment zarrs
output_dir: /path/to/evaluation/linear_classifiers/
annotations:
  - experiment: "2025_04_22_A549_ZIKV_TOMM20"
    path: /path/to/annotations.csv
tasks:
  - task: infection_state
    marker_filter: Phase3D   # only use phase-channel embeddings
  - task: organelle_state
    marker_filter: TOMM20
use_scaling: true
split_train_data: 0.8
```

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
- `plot.yaml` plots per-experiment keys (`X_pca`) into `plots/{experiment}/` subdirs — one subdir per experiment.
- `plot_combined.yaml` concatenates all zarrs and plots combined keys (`X_pca_combined`, `X_phate_combined`)
  into `plots/combined/` — one figure across all experiments.
- PHATE is not computed per-experiment by default (`reduce_dimensionality.phate: null`). Run it only jointly via `reduce_combined`.
- `configs/viewer.yaml` is generated after split and can be passed directly to `dynaclr combined-dim-reduction` (uses the `datasets:` key format accepted by `CombinedDimensionalityReductionConfig`).
