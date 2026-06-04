# DynaCLR Recipes

Practical, self-contained guides for common DynaCLR workflows.

| Recipe | Description |
|--------|-------------|
| [prepare-custom-dataset](prepare-custom-dataset.md) | Format your data (OME-Zarr + tracking CSVs) for DynaCLR |
| [build-cell-index](build-cell-index.md) | Pre-build a parquet cell index for faster training startup |
| [sampling-strategies](sampling-strategies.md) | When to use each sampling configuration (stratify_by, temporal enrichment, leaky mixing) |
| [train-multi-experiment](train-multi-experiment.md) | End-to-end multi-experiment contrastive training |
| [extract-embeddings](extract-embeddings.md) | Run inference and extract per-cell embeddings |
| [evaluate-embeddings](evaluate-embeddings.md) | Linear classifiers, temporal smoothness, dimensionality reduction |
| [slurm-training](slurm-training.md) | SLURM job scripts for training, prediction, and evaluation |
| [troubleshooting](troubleshooting.md) | Common errors and how to fix them |
