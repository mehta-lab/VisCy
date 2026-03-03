# Linear Classifier Specification

This document defines the current annotations schema, naming conventions, and specifications for linear classifiers used in DynaCLR cell phenotyping. These standards may evolve as the project develops.

## 1. Model Naming Convention

### 1.1 Naming Pattern

All trained linear classifier models follow this naming pattern:

```
linear-classifier-{task}-{channel}[-pca{n}]
```

**Components:**
- `task` (**REQUIRED**): Classification task identifier
- `channel` (**REQUIRED**): Input channel identifier
- `pca{n}` (OPTIONAL): PCA dimensionality reduction with n components

### 1.2 Valid Tasks

Currently supported tasks:

| Task ID | Description | Example Labels |
|---------|-------------|----------------|
| `infection_state` | Viral infection status | `infected`, `uninfected` |
| `organelle_state` | Organelle morphology | `nonremodel`, `remodeled` |
| `cell_division_state` | Cell cycle phase | `mitosis`, `interphase` |
| `cell_death_state` | Cell viability/death | `alive`, `dead` |

**Conventions:**
- Task identifiers use snake_case
- Task identifiers do not contain hyphens
- New tasks can be added to `VALID_TASKS` in `linear_classifier_config.py`

### 1.3 Valid Channels

Currently supported channels:

| Channel ID | Description | Imaging Modality |
|------------|-------------|------------------|
| `phase` | Phase contrast | Brightfield microscopy |
| `sensor` | Fluorescent reporter | Fluorescence microscopy |
| `organelle` | Organelle staining | Fluorescence microscopy |

**Conventions:**
- Channel identifiers are lowercase
- Channel identifiers do not contain underscores or hyphens
- New channels can be added to `VALID_CHANNELS` in `linear_classifier_config.py`

### 1.4 Examples

| Model Name | Valid |
|------------|-------|
| `linear-classifier-cell_death_state-phase` | ✅ |
| `linear-classifier-infection_state-sensor-pca32` | ✅ |


## 2. Annotations Schema

### 2.1 Required Columns

Annotations are provided as CSV files with:

**Dataset identifier:**
- `dataset_name` (str): Name/identifier of the dataset (e.g., experiment name, date)

**Tracking indices (from Ultrack):**
- `fov_name` (str): Field of view identifier (e.g., `/Position_001`)
- `id` (int): Cell identifier
- `t` (int): Timepoint
- `track_id` (int): Tracking ID
- `parent_id` (int): Parent cell ID
- `parent_track_id` (int): Parent track ID
- `x` (float): X coordinate
- `y` (float): Y coordinate

**Task labels:**
- `{task}` (str): Ground truth label for the classification task

### 2.2 Example CSV

```csv
dataset_name,fov_name,id,t,track_id,parent_id,parent_track_id,x,y,cell_death_state,infection_state
2024_11_07_A549_SEC61_DENV,/Position_001,1,0,1,-1,-1,128.5,256.3,live,uninfected
2024_11_07_A549_SEC61_DENV,/Position_001,1,1,1,-1,-1,129.1,257.0,live,uninfected
2024_11_07_A549_SEC61_DENV,/Position_001,2,0,2,-1,-1,450.2,180.5,apoptotic,infected
2024_11_07_A549_SEC61_DENV,/Position_002,1,0,1,-1,-1,300.0,400.0,,infected
```

### 2.3 Well Filtering

The `fov_name` column follows the format `{row}/{col}/{position}` (e.g. `B/1/002001`), where `{row}/{col}` identifies the well.

Each dataset entry in the training config can optionally specify `include_wells` — a list of well prefixes (e.g. `["A/1", "B/2"]`). When specified, only annotations whose `fov_name` starts with one of the given prefixes are used. If omitted or null, all wells are included.

This is useful when different wells contain different organelle markers and the classification task (e.g. `organelle_state`) should only use specific wells.

### 2.4 Annotation Rules

**Current behavior:**
- Cells without annotations can be left as `NaN` or empty (will be filtered out)
- Label values of `"unknown"` are filtered out during training
- Matching between embeddings and annotations is performed on `(fov_name, id)` tuple
- The intersection of embeddings and annotations is used for training

## 3. CLI Usage

### 3.1 Training

Train a new linear classifier:

```bash
viscy-dynaclr train-linear-classifier -c config.yaml
```

Configuration file must specify:
- `task`: One of the valid tasks
- `input_channel`: One of the valid channels
- `train_datasets`: List of embeddings + annotations paths (with optional `include_wells`)
- `wandb_project`: W&B project name for artifact storage

### 3.2 Inference

Apply a trained classifier to new embeddings:

```bash
viscy-dynaclr apply-linear-classifier -c inference_config.yaml
```

Configuration file must specify:
- `wandb_project`: W&B project where model is stored
- `model_name`: Full model name (e.g., `linear-classifier-cell_death_state-phase`)
- `version`: Artifact version (`latest`, `v0`, `v1`, etc.)
- `embeddings_path`: Path to new embeddings
- `output_path`: Path to save predictions

### 3.3 Model Identification

To identify which model to use for inference:

1. **Check W&B project**: Navigate to `wandb_project` (e.g., `DynaCLR-2D-linearclassifiers`)
2. **Find artifact**: Look for model artifacts following naming convention
3. **Check version**: Use `latest` for most recent, or specific version (`v0`, `v1`) for reproducibility

Example:
- Project: `DynaCLR-2D-linearclassifiers`
- Model: `linear-classifier-infection_state-sensor`
- Version: `latest` or `v2`

## 4. Output Format

### 4.1 Predictions in AnnData

After inference, the output `.zarr` file contains:

```python
adata.obs[f"predicted_{task}"]              # Predicted class labels (n_cells,)
adata.obsm[f"predicted_{task}_proba"]       # Class probabilities (n_cells, n_classes)
adata.uns[f"predicted_{task}_classes"]      # List of class names
```

**Example:**
```python
adata.obs["predicted_cell_death_state"]           # ["live", "live", "apoptotic", ...]
adata.obsm["predicted_cell_death_state_proba"]    # [[0.95, 0.03, 0.02], ...]
adata.uns["predicted_cell_death_state_classes"]   # ["live", "apoptotic", "necrotic"]
```

## 5. Model Storage

### 5.1 W&B Artifact Structure

Trained models are stored in Weights & Biases with:

**Required files:**
- `{model_name}.joblib` - Trained classifier
- `{model_name}_config.json` - Training configuration

**Optional files:**
- `{model_name}_scaler.joblib` - StandardScaler (if used)
- `{model_name}_pca.joblib` - PCA transformer (if used)

**Metadata:**
- Artifact type: `model`
- Training metrics logged to W&B run
- Full configuration logged for reproducibility

---

**Version:** 1.1
**Last Updated:** 2026-02-11
