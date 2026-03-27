"""Pydantic configuration model for the compute-mmd CLI."""

from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, model_validator

from viscy.representation.evaluation.lot_correction_config import UninfFilter


class ComputeMMDConfig(BaseModel):
    """Configuration for computing MMD between two groups of embeddings.

    Parameters
    ----------
    zarr_a : str
        Path to AnnData zarr for group A (e.g. uninfected, light-sheet).
    zarr_b : str or None
        Path to AnnData zarr for group B.  When omitted (or same path as
        ``zarr_a``), a single zarr is loaded and both filters are applied to it.
    filter_a : UninfFilter or None
        Obs filter selecting cells for group A.  If omitted, all cells are used.
    filter_b : UninfFilter or None
        Obs filter selecting cells for group B.  If omitted, all cells are used.
    group_by : list[str]
        Obs column names to stratify the comparison by (e.g.
        ``["organelle", "timepoint"]``).  Leave empty for a single overall MMD.
    use_pca : bool
        Fit a shared PCA on the combined filtered embeddings before MMD,
        by default True.
    n_pca : int
        Number of PCA components, by default 50.
    n_perm : int
        Permutation test iterations for p-value estimation, by default 1000.
        Set to 0 to skip the permutation test (p_value will be NaN).
    max_cells : int
        Maximum cells per group passed to the MMD kernel, by default 2000.
    random_seed : int
        Random seed for reproducibility, by default 42.
    output_csv : str
        Path to write the results CSV.
    """

    zarr_a: str = Field(..., min_length=1)
    zarr_b: Optional[str] = Field(default=None)

    filter_a: Optional[UninfFilter] = Field(default=None)
    filter_b: Optional[UninfFilter] = Field(default=None)

    group_by: list[str] = Field(default_factory=list)

    use_pca: bool = Field(default=True)
    n_pca: int = Field(default=50, gt=0)

    n_perm: int = Field(default=1000, ge=0)
    max_cells: int = Field(default=2000, gt=0)
    random_seed: int = Field(default=42)

    output_csv: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_zarr_paths(self):
        if not Path(self.zarr_a).exists():
            raise ValueError(f"zarr_a not found: {self.zarr_a}")
        if self.zarr_b is not None and not Path(self.zarr_b).exists():
            raise ValueError(f"zarr_b not found: {self.zarr_b}")
        return self
