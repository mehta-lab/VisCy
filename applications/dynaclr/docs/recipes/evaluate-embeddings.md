# Recipe: Evaluate DynaCLR Embeddings

## Goal

Quantify embedding quality — how well they capture infection state, cell
death, temporal smoothness, etc.

## Evaluation tools

DynaCLR provides three evaluation axes:

1. **Linear classifiers** — probe embedding quality for classification tasks
2. **Temporal smoothness** — measure temporal coherence of embeddings
3. **Dimensionality reduction** — visualize embedding structure

## Linear classifiers

### Prepare annotations

Create a CSV with human labels matching your embeddings. Cells are matched
on `(fov_name, id)`:

```csv
dataset_name,fov_name,id,t,track_id,x,y,infection_state,cell_death_state
my_exp,A/1/0,1,0,1,128.5,256.3,uninfected,live
my_exp,A/1/0,1,1,1,129.1,257.0,uninfected,live
my_exp,B/1/0,5,10,5,200.1,100.4,infected,dead
```

See `docs/linear_classifiers/annotations_and_linear_classifiers.md` for the
full annotation schema.

### Train a classifier

Create a training config:

```yaml
# train_classifier.yaml
task: infection_state   # or: cell_death_state, organelle_state, cell_division_state
input_channel: phase

embedding_model_name: DynaCLR-3D
embedding_model_version: v1

train_datasets:
  - embeddings: /path/to/embeddings.zarr
    annotations: /path/to/annotations.csv
  - embeddings: /path/to/dataset2/embeddings.zarr
    annotations: /path/to/dataset2/annotations.csv
    include_wells: ["A/1", "B/1"]   # optional well filter

use_scaling: true
use_pca: false
max_iter: 1000
class_weight: balanced
solver: liblinear
split_train_data: 0.8
random_seed: 42

wandb_entity: null
wandb_tags: []
```

```bash
dynaclr train-linear-classifier -c train_classifier.yaml
```

The trained pipeline is saved as a W&B artifact.
See `configs/linear_classifiers/example_linear_classifier_train.yaml`.

### Apply a trained classifier

```yaml
# apply_classifier.yaml
embedding_model_name: DynaCLR-3D
embedding_model_version: v1
wandb_entity: null

embeddings_path: /path/to/new_embeddings.zarr
output_path: /path/to/predictions.zarr   # optional, defaults to input

models:
  - model_name: linear-classifier-infection_state-phase
    version: latest
  - model_name: linear-classifier-cell_death_state-phase
    version: latest
```

```bash
dynaclr apply-linear-classifier -c apply_classifier.yaml
```

Predictions are written to the zarr:
- `.obs["predicted_{task}"]` — class labels
- `.obsm["predicted_{task}_proba"]` — probability vectors
- `.uns["predicted_{task}_classes"]` — ordered class names

See `configs/linear_classifiers/example_linear_classifier_inference.yaml`.

## Temporal smoothness

Measures whether temporally adjacent cells have similar embeddings
(lower = smoother = better).

```yaml
# smoothness.yaml
models:
  - path: /path/to/embeddings.zarr
    label: DynaCLR-3D-v1

  # Compare multiple models:
  # - path: /path/to/baseline_embeddings.zarr
  #   label: ImageNet-baseline

evaluation:
  distance_metric: cosine
  output_dir: /path/to/smoothness_results
  save_plots: true
  save_distributions: false
  verbose: true
```

```bash
dynaclr evaluate-smoothness -c smoothness.yaml
```

**Metrics produced:**
- `smoothness_score` — mean distance between temporally adjacent observations (lower is better)
- `dynamic_range` — ratio of random vs adjacent distances (higher is better)
- Distance distributions for adjacent and random frame pairs

See `configs/smoothness/example_smoothness.yaml`.

## Dimensionality reduction

Compute PCA, UMAP, and/or PHATE projections for visualization:

```yaml
# reduce.yaml
input_path: /path/to/embeddings.zarr
overwrite_keys: false

pca:
  n_components: 32
  normalize_features: true
umap:
  n_components: 2
  n_neighbors: 15
  normalize: true
phate:
  n_components: 2
  knn: 5
  decay: 40
  scale_embeddings: true
  random_state: 42
```

```bash
dynaclr reduce-dimensionality -c reduce.yaml
```

Results stored in `.obsm` as `X_pca`, `X_umap`, `X_phate`.
See `configs/dimensionality_reduction/example_reduce.yaml`.

## Merging external annotations

Attach columns from a CSV to an existing embeddings zarr:

```bash
dynaclr append-obs \
    -e /path/to/embeddings.zarr \
    --csv /path/to/annotations.csv \
    --prefix annotated_ \
    --merge-key fov_name --merge-key id
```

## Suggested evaluation workflow

1. **Extract embeddings** (`viscy predict`) → `embeddings.zarr`
2. **Reduce dimensions** (`dynaclr reduce-dimensionality`) → adds `X_pca`, `X_umap`, `X_phate`
3. **Merge annotations** (`dynaclr append-obs`) → adds label columns
4. **Train classifiers** (`dynaclr train-linear-classifier`) → saves to W&B
5. **Evaluate smoothness** (`dynaclr evaluate-smoothness`) → temporal coherence metrics
6. **Visualize** in napari or plotly using the `.obsm` projections
