# Train and apply linear classifiers

Linear classifiers are logistic-regression probes trained on DynaCLR cell
embeddings. Use the batch workflow for evaluation runs and tracked time series.
Use the standalone workflow only when models must be stored as W&B artifacts.

## Choose a workflow

| Need | Commands | Model storage |
| --- | --- | --- |
| Evaluation metrics, group-aware validation, reusable local bundles | `run-linear-classifiers`, `append-predictions` | Versioned joblib bundle |
| W&B artifact training and inference | `train-linear-classifier`, `apply-linear-classifier` | W&B artifact |
| Dataset contribution analysis | `cross-validate` | CSV and optional PDF report |

All paths require embeddings and annotation CSVs that follow the
[annotation contract](annotations_and_linear_classifiers.md). Train and apply a
classifier only within the same embedding feature space.

## Recommended: evaluation bundle

The full evaluation config places this block under `linear_classifiers`. The
orchestrator generates the direct step config and runs it through Nextflow.

```yaml
linear_classifiers:
  label_source: annotations
  annotations:
    - experiment: experiment-a
      path: /path/to/experiment-a-annotations.csv
    - experiment: experiment-b
      path: /path/to/experiment-b-annotations.csv
  tasks:
    - task: infection_state
      marker_filters: [Phase3D]
  publish_dir: /path/to/linear_classifiers/<feature-space>
  use_scaling: true
  use_pca: false
  split_train_data: 0.8
  split_groups_by: [experiment, fov_name, track_id]
  random_seed: 42
```

Use a group-aware split whenever multiple timepoints from one track are
present. This prevents a track from contributing to both training and
validation.

For a direct run, use the generated YAML or add its two top-level paths:

```yaml
embeddings_path: /path/to/combined-embeddings.zarr
output_dir: /path/to/evaluation/linear_classifiers

label_source: annotations
annotations:
  - experiment: experiment-a
    path: /path/to/experiment-a-annotations.csv
tasks:
  - task: infection_state
    marker_filters: [Phase3D]
split_groups_by: [experiment, fov_name, track_id]
```

```sh
uv run dynaclr run-linear-classifiers -c linear_classifiers.yaml
```

The command trains one classifier per task-marker pair and writes:

```text
<output_dir>/
├── metrics_summary.csv
├── <task>_summary.pdf
└── pipelines/
    ├── manifest.json
    └── <task>_<marker>.joblib
```

When `publish_dir` is set, the finished bundle is promoted to `vN/` and the
`latest` symlink is updated. Pin `vN` for reproducible application.

Apply a saved bundle to a directory of per-experiment zarrs:

```yaml
embeddings_path: /path/to/per-experiment-embeddings
pipelines_dir: /path/to/linear_classifiers/<feature-space>/v2
```

```sh
uv run dynaclr append-predictions -c append_predictions.yaml
```

For the complete Nextflow launch, see the
[evaluation runbook](../DAGs/evaluation.md). Witness-GMM pseudo-labels use this
same training path; see the
[witness-GMM runbook](../DAGs/witness_gmm_classifiers.md).

## Standalone W&B artifacts

This path performs standalone classifier training and uses W&B for storage.
Start from the maintained configs:

- [`example_linear_classifier_train.yaml`](../../configs/linear_classifiers/example_linear_classifier_train.yaml)
- [`example_linear_classifier_inference.yaml`](../../configs/linear_classifiers/example_linear_classifier_inference.yaml)

Required training fields are:

```yaml
task: organelle_state
input_channel: marker
marker: g3bp1
embedding_model_name: DynaCLR-2D-BagOfChannels-timeaware
embedding_model_version: v3
train_datasets:
  - embeddings: /path/to/embeddings_marker.zarr
    annotations: /path/to/annotations.csv
    include_wells: [C/1, C/2]
use_scaling: true
use_pca: false
split_train_data: 0.8
random_seed: 42
wandb_entity: null
wandb_tags: []
```

```sh
wandb login
uv run dynaclr train-linear-classifier \
  -c applications/dynaclr/configs/linear_classifiers/<train>.yaml
```

Apply one or more artifact versions:

```yaml
embedding_model_name: DynaCLR-2D-BagOfChannels-timeaware
embedding_model_version: v3
embeddings_path: /path/to/embeddings.zarr
output_path: /path/to/embeddings-with-predictions.zarr
overwrite: false
models:
  - model_name: linear-classifier-infection_state-phase
    version: v2
  - model_name: linear-classifier-organelle_state-marker-g3bp1
    version: v1
    include_wells: [C/1, C/2]
```

```sh
uv run dynaclr apply-linear-classifier \
  -c applications/dynaclr/configs/linear_classifiers/<inference>.yaml
```

The W&B project is derived as
`linearclassifiers-<embedding_model_name>-<embedding_model_version>`.

## Cross-validation

Use rotating leave-one-dataset-out validation to identify datasets that help or
hurt transfer:

```sh
uv run dynaclr cross-validate \
  -c applications/dynaclr/configs/linear_classifiers/cross_validate_example.yaml \
  --report
```

The output directory contains raw folds, the dataset impact summary,
recommended subsets, and an optional PDF report. This analysis is separate from
the group-aware train/validation split used to fit the final bundle.
