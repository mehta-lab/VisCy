# Linear Classifier for Cell Phenotyping

Train and apply logistic regression classifiers on DynaCLR cell embeddings for supervised cell phenotyping tasks.

## Prerequisites

Install VisCy with the metrics extras:

```bash
pip install -e ".[metrics]"
```

You also need a [Weights & Biases](https://wandb.ai) account for model storage and tracking. Log in before running:

```bash
wandb login
```

## Quick Start

### Train a classifier

```bash
viscy-dynaclr train-linear-classifier -c configs/example_linear_classifier_train.yaml
```

### Apply a trained classifier to new data

```bash
viscy-dynaclr apply-linear-classifier -c configs/example_linear_classifier_inference.yaml
```

## Training

### Configuration

Create a YAML config file (see `configs/example_linear_classifier_train.yaml`):

```yaml
# Classification task
task: cell_death_state  # infection_state | organelle_state | cell_division_state | cell_death_state

# Input channel used for embeddings
input_channel: phase  # phase | sensor | organelle

# Name of the embedding model
embedding_model: DynaCLR-2D-BagOfChannels-timeaware-v3

# Training datasets (explicit paths, no glob patterns)
train_datasets:
  - embeddings: /path/to/dataset1/embeddings_phase.zarr
    annotations: /path/to/dataset1/annotations.csv
  - embeddings: /path/to/dataset2/embeddings_phase.zarr
    annotations: /path/to/dataset2/annotations.csv

# Preprocessing
use_scaling: true
use_pca: false
n_pca_components: null  # required if use_pca is true

# Classifier hyperparameters
max_iter: 1000
class_weight: balanced  # 'balanced' or null
solver: liblinear

# Train/validation split (set to 1.0 to use all data for training)
split_train_data: 0.8
random_seed: 42

# Weights & Biases
wandb_project: DynaCLR-2D-linearclassifiers
wandb_entity: null
wandb_tags: []
```

### What happens during training

1. Embeddings and annotations are loaded and matched on `(fov_name, id)`
2. Cells with missing or `"unknown"` labels are filtered out
3. Multiple datasets are concatenated
4. Optional preprocessing is applied (StandardScaler, PCA)
5. Data is split into train/validation sets (stratified)
6. A `LogisticRegression` classifier is trained
7. Metrics (accuracy, precision, recall, F1) are logged to W&B
8. The trained model pipeline is saved as a W&B artifact

### Supported tasks

| Task | Description | Example Labels |
|------|-------------|----------------|
| `infection_state` | Viral infection status | `infected`, `uninfected` |
| `organelle_state` | Organelle morphology | `nonremodel`, `remodeled` |
| `cell_division_state` | Cell cycle phase | `mitosis`, `interphase` |
| `cell_death_state` | Cell viability/death | `alive`, `dead` |

### Supported channels

| Channel | Description |
|---------|-------------|
| `phase` | Phase contrast / brightfield |
| `sensor` | Fluorescent reporter |
| `organelle` | Organelle staining |

## Inference

### Configuration

Create a YAML config file (see `configs/example_linear_classifier_inference.yaml`):

```yaml
# W&B project and model artifact
wandb_project: DynaCLR-2D-linearclassifiers
model_name: linear-classifier-cell_death_state-phase
version: latest  # or specific version like 'v0', 'v1'
wandb_entity: null

# Input/output paths
embeddings_path: /path/to/embeddings.zarr
output_path: /path/to/output_with_predictions.zarr
overwrite: false
```

### What happens during inference

1. The trained pipeline (classifier + scaler + PCA) is downloaded from W&B
2. Embeddings are loaded from the zarr file
3. Preprocessing is applied identically to training
4. Predictions and class probabilities are computed
5. Results are saved to a new zarr file

### Output format

The output zarr file contains the original AnnData with added fields:

```python
adata.obs[f"predicted_{task}"]            # Predicted class labels
adata.obsm[f"predicted_{task}_proba"]     # Class probabilities (n_cells x n_classes)
adata.uns[f"predicted_{task}_classes"]    # Ordered list of class names
```

## Annotations Format

Annotations are CSV files with these required columns:

| Column | Type | Description |
|--------|------|-------------|
| `dataset_name` | str | Dataset identifier |
| `fov_name` | str | Field of view (e.g., `/Position_001`) |
| `id` | int | Cell identifier |
| `t` | int | Timepoint |
| `track_id` | int | Tracking ID |
| `parent_id` | int | Parent cell ID |
| `parent_track_id` | int | Parent track ID |
| `x`, `y` | float | Cell coordinates |
| `{task}` | str | Ground truth label column |

Cells without annotations (NaN/empty) and cells labeled `"unknown"` are excluded from training.

Example:

```csv
dataset_name,fov_name,id,t,track_id,parent_id,parent_track_id,x,y,cell_death_state,infection_state
2024_11_07_A549_SEC61_DENV,/Position_001,1,0,1,-1,-1,128.5,256.3,live,uninfected
2024_11_07_A549_SEC61_DENV,/Position_001,2,0,2,-1,-1,450.2,180.5,apoptotic,infected
```

## Model Naming Convention

Trained models follow this naming pattern:

```
linear-classifier-{task}-{channel}[-pca{n}]
```

Examples:
- `linear-classifier-cell_death_state-phase`
- `linear-classifier-infection_state-sensor-pca32`

## W&B Artifact Structure

Each trained model artifact contains:

| File | Description |
|------|-------------|
| `{model_name}.joblib` | Trained classifier |
| `{model_name}_config.json` | Training configuration |
| `{model_name}_scaler.joblib` | StandardScaler (if `use_scaling: true`) |
| `{model_name}_pca.joblib` | PCA transformer (if `use_pca: true`) |

## Further Reference

See `annotations_and_linear_classifiers.md` for the full specification of the annotations schema and naming conventions.
