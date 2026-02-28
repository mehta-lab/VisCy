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
    scale_embeddings: bool = False
    random_state: int = 42


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
