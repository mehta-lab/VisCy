"""Configuration models for dimensionality reduction."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class PCAConfig(BaseModel):
    """PCA reduction parameters."""

    n_components: Optional[int] = None
    normalize_features: bool = True


class UMAPConfig(BaseModel):
    """UMAP reduction parameters."""

    n_components: int = 2
    n_neighbors: int = 15
    normalize: bool = True


class PHATEConfig(BaseModel):
    """PHATE reduction parameters."""

    n_components: int = 2
    knn: int = 5
    decay: int = 40
    knn_dist: str = "cosine"
    scale_embeddings: bool = False
    random_state: int = 42
    n_pca: Optional[int] = 50
    subsample: Optional[int] = 50_000
    n_jobs: int = 1


class DimensionalityReductionConfig(BaseModel):
    """Configuration for computing dimensionality reductions on saved embeddings.

    Parameters
    ----------
    input_path : str
        Path to AnnData zarr store with features in ``.X``.
    output_path : str, optional
        Path for output zarr. If None, writes back to ``input_path``.
    pca : PCAConfig, optional
        PCA parameters. Set to enable PCA computation.
    umap : UMAPConfig, optional
        UMAP parameters. Set to enable UMAP computation.
    phate : PHATEConfig, optional
        PHATE parameters. Set to enable PHATE computation.
    overwrite_keys : bool
        If True, overwrite existing ``.obsm`` keys. Otherwise raise on conflict.
    """

    input_path: str = Field(...)
    output_path: Optional[str] = None
    pca: Optional[PCAConfig] = None
    umap: Optional[UMAPConfig] = None
    phate: Optional[PHATEConfig] = None
    overwrite_keys: bool = False

    @model_validator(mode="after")
    def validate_config(self):
        if not Path(self.input_path).exists():
            raise ValueError(f"Input path not found: {self.input_path}")
        if self.pca is None and self.umap is None and self.phate is None:
            raise ValueError("At least one reduction method must be specified (pca, umap, or phate)")
        return self


class CombinedDatasetConfig(BaseModel):
    """Input dataset spec for combined reductions.

    Parameters
    ----------
    anndata : str
        Path to AnnData zarr store with features in ``.X``.
    hcs_plate : str, optional
        Path to the raw HCS plate zarr (not used for reductions, but useful for reuse).
    """

    anndata: str = Field(...)
    hcs_plate: Optional[str] = None


class CombinedDimensionalityReductionConfig(BaseModel):
    """Configuration for computing joint dimensionality reductions across multiple AnnData stores.

    Parameters
    ----------
    input_paths : list[str], optional
        Paths to AnnData zarr stores. Embeddings from all stores are concatenated before fitting
        reductions, then per-store slices are written back with a ``_combined`` suffix.
    datasets : dict[str, CombinedDatasetConfig], optional
        Alternative to ``input_paths``. When provided, ``input_paths`` is derived from
        ``datasets[*].anndata``. This matches the multi-dataset YAML used in organelle dynamics.
    pca : PCAConfig, optional
        PCA parameters. Results stored as ``X_pca_combined``.
    umap : UMAPConfig, optional
        UMAP parameters. Results stored as ``X_umap_combined``.
    phate : PHATEConfig, optional
        PHATE parameters. Results stored as ``X_phate_combined``.
    overwrite_keys : bool
        If True, overwrite existing ``.obsm`` keys. Otherwise raise on conflict.
    """

    input_paths: Optional[list[str]] = None
    datasets: Optional[dict[str, CombinedDatasetConfig]] = None
    pca: Optional[PCAConfig] = None
    umap: Optional[UMAPConfig] = None
    phate: Optional[PHATEConfig] = None
    overwrite_keys: bool = False

    @model_validator(mode="after")
    def validate_config(self):
        if self.input_paths is None:
            if not self.datasets:
                raise ValueError("Either input_paths or datasets must be provided")
            self.input_paths = [d.anndata for d in self.datasets.values()]

        if len(self.input_paths) < 1:
            raise ValueError("At least one input path must be provided")

        for p in self.input_paths:
            if not Path(p).exists():
                raise ValueError(f"Input path not found: {p}")
        if self.pca is None and self.umap is None and self.phate is None:
            raise ValueError("At least one reduction method must be specified (pca, umap, or phate)")
        return self
