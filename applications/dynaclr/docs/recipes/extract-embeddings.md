# Recipe: Extract Embeddings from a Trained Model

## Goal

Extract per-cell embeddings from a trained DynaCLR checkpoint for downstream
analysis (clustering, classification, visualization). Use `viscy predict`
with an `EmbeddingWriter` callback — the output is an AnnData zarr store
with embeddings in `.X` and optional PCA/PHATE in `.obsm`.

## Step 1: Create the predict config

Create `predict.yml`:

```yaml
seed_everything: 42

trainer:
  accelerator: gpu
  strategy: auto
  devices: auto
  precision: 32-true
  inference_mode: true
  callbacks:
    - class_path: viscy_utils.callbacks.embedding_writer.EmbeddingWriter
      init_args:
        output_path: /path/to/embeddings.zarr
        # Optional: compute PCA and PHATE during prediction
        pca_kwargs:
          n_components: 8
        phate_kwargs:
          knn: 5
          decay: 40
          n_jobs: -1
          random_state: 42
        # Set either to null to skip:
        # pca_kwargs: null
        # phate_kwargs: null

model:
  class_path: dynaclr.engine.ContrastiveModule
  init_args:
    encoder:
      class_path: viscy_models.contrastive.ContrastiveEncoder
      init_args:
        backbone: convnext_tiny
        in_channels: 2
        in_stack_depth: 30
        stem_kernel_size: [5, 4, 4]
        stem_stride: [5, 4, 4]
        embedding_dim: 768
        projection_dim: 32
    example_input_array_shape: [1, 2, 30, 256, 256]

data:
  class_path: viscy_data.triplet.TripletDataModule
  init_args:
    data_path: /path/to/test_data.zarr
    tracks_path: /path/to/test_tracks
    source_channel:
      - Phase3D
      - GFP
    z_range: [15, 45]
    batch_size: 32
    num_workers: 16
    initial_yx_patch_size: [160, 160]
    final_yx_patch_size: [160, 160]
    normalizations:
      - class_path: viscy_transforms.NormalizeSampled
        init_args:
          keys: [Phase3D]
          level: fov_statistics
          subtrahend: mean
          divisor: std
      - class_path: viscy_transforms.ScaleIntensityRangePercentilesd
        init_args:
          keys: [GFP]
          lower: 50
          upper: 99
          b_min: 0.0
          b_max: 1.0

return_predictions: false
ckpt_path: /path/to/checkpoints/best.ckpt
```

See `configs/prediction/predict.yml` for the full template.

**Key differences from training config:**
- `initial_yx_patch_size` = `final_yx_patch_size` (no random crop margin needed)
- No augmentations (deterministic inference)
- `EmbeddingWriter` callback handles output
- Single GPU is usually sufficient

## Step 2: Run prediction

```bash
viscy predict -c predict.yml
```

Or via SLURM:

```bash
#!/bin/bash
#SBATCH --job-name=dynaclr_predict
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=7G
#SBATCH --time=0-01:00:00

WORKSPACE_DIR=/path/to/viscy
uv run --project "$WORKSPACE_DIR" --package dynaclr viscy predict -c predict.yml
```

See `configs/prediction/predict_slurm.sh`.

## Step 3: Inspect the output

The output is an AnnData zarr store:

```python
import anndata as ad

adata = ad.read_zarr("/path/to/embeddings.zarr")
print(adata)
# AnnData object with n_obs x n_vars
#   obs: fov_name, track_id, t, ...
#   obsm: X_pca, X_phate (if configured)
```

- `.X` — embedding vectors (n_cells x embedding_dim)
- `.obs` — cell metadata (FOV, track ID, timepoint, etc.)
- `.obsm["X_pca"]` — PCA projection (if `pca_kwargs` was set)
- `.obsm["X_phate"]` — PHATE projection (if `phate_kwargs` was set)

## Step 4: (Optional) Reduce dimensionality post-hoc

If you skipped PCA/PHATE during prediction, or want to try different
parameters, use the dimensionality reduction CLI:

```yaml
# reduce.yaml
input_path: /path/to/embeddings.zarr
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
```

```bash
dynaclr reduce-dimensionality -c reduce.yaml
```

Results are written to `.obsm` as `X_pca`, `X_umap`, `X_phate`.
See `configs/dimensionality_reduction/example_reduce.yaml`.

## Tips

- **Match normalizations** to training — using different normalization
  at inference will produce degraded embeddings.
- **Patch size at inference** should equal `final_yx_patch_size` from
  training (no augmentation margin needed).
- **Batch size** can be larger at inference since no gradients are stored.
- **Multiple datasets** — run predict separately per dataset, then evaluate
  with linear classifiers that can combine multiple zarr stores.
