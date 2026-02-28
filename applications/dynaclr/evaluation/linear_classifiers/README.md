# Linear Classifier for Cell Phenotyping

Train and apply logistic regression classifiers on DynaCLR cell embeddings for supervised cell phenotyping tasks.

## Overview

This directory contains:

| File | Description |
|------|-------------|
| `src/utils.py` | Shared functions for discovering predictions, annotations, channel resolution, and path utilities |
| `src/report.py` | PDF report generation for cross-validation and evaluation (optional) |
| `scripts/generate_prediction_scripts.py` | Generates SLURM `.sh`/`.yml` scripts for datasets missing embeddings |
| `scripts/generate_batch_predictions.py` | Batch prediction config & SLURM script generator with auto z-range |
| `scripts/generate_train_config.py` | Generates training YAML configs for all valid task x channel combinations |
| `scripts/train_linear_classifier.py` | CLI for training a classifier from a config |
| `scripts/apply_linear_classifier.py` | CLI for applying a trained classifier to new embeddings |
| `scripts/cross_validation.py` | Leave-one-dataset-out CV with impact scoring (helps/hurts/uncertain) |
| `scripts/evaluate_dataset.py` | Compare embedding models (e.g. 2D vs 3D) on a held-out test set |

## Prerequisites

Install DynaCLR with the eval extras:

```bash
pip install -e "applications/dynaclr[eval]"
```

You also need a [Weights & Biases](https://wandb.ai) account for model storage and tracking. Log in before running:

```bash
wandb login
```

## Workflow

### 1. Discover datasets and generate prediction scripts

If some annotated datasets don't have embeddings yet, generate the SLURM prediction scripts:

```python
# Edit configuration in scripts/generate_prediction_scripts.py, then run cells
# Key parameters:
#   embeddings_dir  - base directory with dataset folders
#   annotations_dir - base directory with annotation CSVs
#   model           - model directory glob pattern
#   version         - model version (e.g. "v3")
#   ckpt_path       - checkpoint to use for ALL datasets
```

This will:
- Discover which annotated datasets are missing predictions
- Use an existing dataset as a template
- Generate `predict_{phase,sensor,organelle}.{sh,yml}` and `run_all.sh` per dataset
- Enforce a single checkpoint across all generated scripts

### 2. Generate training configs

Once datasets have both embeddings and annotations:

```python
# Edit configuration in scripts/generate_train_config.py, then run cells
# Generates one YAML config per (task, channel) combination
```

### 3. Train a classifier

```bash
dynaclr train-linear-classifier -c configs/generated/cell_death_state_phase.yaml
```

### 4. Apply a trained classifier to new data

```bash
dynaclr apply-linear-classifier -c configs/example_linear_classifier_inference.yaml
```

### 5. Cross-validate training datasets

Determine which training datasets help or hurt classifier performance using rotating leave-one-dataset-out CV. Run from the `linear_classifiers/` directory:

```bash
python scripts/cross_validation.py -c configs/cross_validate_example.yaml
python scripts/cross_validation.py -c configs/cross_validate_example.yaml --report  # with PDF
```

Outputs:
- `cv_results.csv` — raw results (one row per fold x seed)
- `cv_summary.csv` — aggregated impact labels per dataset
- `cv_recommended_subsets.csv` — recommended training subsets with harmful datasets excluded
- `cv_report.pdf` — (optional) impact heatmaps, AUROC distributions, temporal curves

Each dataset is labeled as:
- **helps** — removing it hurts performance (keep it)
- **hurts** — removing it improves performance (exclude it)
- **uncertain** — delta within noise
- **unsafe** — fold skipped due to insufficient class samples

### 6. Evaluate models on a held-out test set

Compare embedding models by training classifiers and evaluating on a held-out dataset:

```bash
python scripts/evaluate_dataset.py -c configs/evaluate_dataset_example.yaml
python scripts/evaluate_dataset.py -c configs/evaluate_dataset_example.yaml --report  # with PDF
```

Outputs per model:
- `{model}/{task}_{channel}_pipeline.joblib` — trained classifier
- `{model}/{task}_{channel}_predictions.zarr` — test predictions
- `{model}/metrics_summary.csv` — per-model metrics

Combined outputs:
- `train_metrics_comparison.csv` — validation metrics across models
- `test_metrics_comparison.csv` — test metrics across models

## Training Configuration

Create a YAML config file (see `configs/example_linear_classifier_train.yaml`):

```yaml
task: cell_death_state  # infection_state | organelle_state | cell_division_state | cell_death_state
input_channel: phase    # phase | sensor | organelle
embedding_model: DynaCLR-2D-BagOfChannels-timeaware-v3

train_datasets:
  - embeddings: /path/to/dataset1/embeddings_phase.zarr
    annotations: /path/to/dataset1/annotations.csv
  - embeddings: /path/to/dataset2/embeddings_phase.zarr
    annotations: /path/to/dataset2/annotations.csv
    include_wells: ["A/1", "C/2"]  # optional: filter by well prefix

use_scaling: true
use_pca: false
n_pca_components: null
max_iter: 1000
class_weight: balanced
solver: liblinear
split_train_data: 0.8
random_seed: 42

wandb_project: DynaCLR-2D-linearclassifiers
wandb_entity: null
wandb_tags: []
```

### Well filtering

Each dataset entry can optionally specify `include_wells` — a list of well prefixes (e.g. `["A/1", "B/2"]`) to restrict which FOVs are used. The `fov_name` column in annotations follows the format `{row}/{col}/{position}` (e.g. `B/1/002001`), and filtering matches on the `{row}/{col}/` prefix. If `include_wells` is omitted or null, all wells are used.

This is useful for the `organelle_state` task where different wells contain different organelle markers and remodeling phenotypes differ between them.

### What happens during training

1. Embeddings and annotations are loaded and matched on `(fov_name, id)`
2. If `include_wells` is specified, only matching FOVs are kept
3. Cells with missing or `"unknown"` labels are filtered out
4. Multiple datasets are concatenated
5. Optional preprocessing is applied (StandardScaler, PCA)
6. Data is split into train/validation sets (stratified)
7. A `LogisticRegression` classifier is trained
8. Metrics (accuracy, precision, recall, F1) are logged to W&B
9. The trained model pipeline is saved as a W&B artifact

## Inference Configuration

```yaml
wandb_project: DynaCLR-2D-linearclassifiers
model_name: linear-classifier-cell_death_state-phase
version: latest
wandb_entity: null
embeddings_path: /path/to/embeddings.zarr
output_path: /path/to/output_with_predictions.zarr
overwrite: false
```

### Output format

```python
adata.obs[f"predicted_{task}"]            # Predicted class labels
adata.obsm[f"predicted_{task}_proba"]     # Class probabilities (n_cells x n_classes)
adata.uns[f"predicted_{task}_classes"]    # Ordered list of class names
```

## Supported Tasks and Channels

| Task | Description | Example Labels |
|------|-------------|----------------|
| `infection_state` | Viral infection status | `infected`, `uninfected` |
| `organelle_state` | Organelle morphology | `nonremodel`, `remodeled` |
| `cell_division_state` | Cell cycle phase | `mitosis`, `interphase` |
| `cell_death_state` | Cell viability/death | `alive`, `dead` |

| Channel | Description |
|---------|-------------|
| `phase` | Phase contrast / brightfield |
| `sensor` | Fluorescent reporter |
| `organelle` | Organelle staining |

## Model Naming Convention

```
linear-classifier-{task}-{channel}[-pca{n}]
```

Examples: `linear-classifier-cell_death_state-phase`, `linear-classifier-infection_state-sensor-pca32`

## Further Reference

See `annotations_and_linear_classifiers.md` for the full specification of the annotations schema and naming conventions.
