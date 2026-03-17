# DynaCLR Evaluation

Evaluation tools for DynaCLR cell embedding models. Each evaluation method lives in its own subdirectory.

## Available Methods

| Method | Directory/Module | Description |
|--------|------------------|-------------|
| Linear classifiers | `linear_classifiers/` | Logistic regression on embeddings for supervised cell phenotyping |
| Temporal smoothness | `benchmarking/smoothness/` | Evaluate how smoothly embeddings change across adjacent time frames |
| Dimensionality reduction | `dimensionality_reduction/` | Compute PCA, UMAP, and/or PHATE on saved AnnData zarr embeddings |
| Pseudotime remodeling | `pseudotime/` | Lineage-aware remodeling timing analysis (annotation, prediction, embedding distance) |
| Append obs | `append_obs.py` | Merge columns from a CSV into an AnnData zarr obs with optional prefix (e.g. `annotated_`, `predicted_`, `feature_`) |
