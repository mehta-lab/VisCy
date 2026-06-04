"""Evaluation utilities for learned representations.

Includes:
- Linear classifier accuracy
- Clustering (NMI, ARI)
- Correlation between embeddings and features
- Dimensionality reduction (PCA, UMAP, PHATE)
"""


def __getattr__(name):
    if name == "load_annotation_anndata":
        from viscy_utils.evaluation.annotation import load_annotation_anndata

        return load_annotation_anndata
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["load_annotation_anndata"]
