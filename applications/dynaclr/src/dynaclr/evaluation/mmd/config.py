"""Pydantic configuration for MMD² computation."""

from pathlib import Path
from typing import Union

from pydantic import BaseModel, Field, model_validator


class WellSpec(BaseModel):
    """Specification for one well in an AnnData zarr.

    Cells are selected by matching the ``fov_name`` obs column against
    the prefix ``"{well_name}/{well_id}/"``.

    Parameters
    ----------
    zarr_path : str
        Path to the AnnData zarr store containing the embeddings.
    well_name : str
        Well name (e.g. ``"B03"``).
    well_id : str or int
        Well ID (e.g. ``1``).
    """

    zarr_path: str = Field(..., min_length=1)
    well_name: str = Field(..., min_length=1)
    well_id: Union[str, int]

    @model_validator(mode="after")
    def validate_zarr_path(self):
        if not Path(self.zarr_path).exists():
            raise ValueError(f"zarr_path not found: {self.zarr_path}")
        return self


class MMDConfig(BaseModel):
    """Configuration for MMD² computation between two pooled groups of wells.

    Parameters
    ----------
    group_a : list[WellSpec]
        Wells to pool into group A.
    group_b : list[WellSpec]
        Wells to pool into group B.
    n_perm : int, optional
        Number of permutations for the p-value. Set to 0 to skip.
        By default 1000.
    max_cells : int, optional
        Maximum cells per group used in the kernel computation (random
        subsample if larger). By default 2000.
    random_seed : int, optional
        Random seed. By default 42.
    output_csv : str, optional
        Path to write the results CSV. If not set, results are only
        printed to the terminal.

    Example config (YAML)
    ---------------------
    group_a:
      - zarr_path: /path/to/experiment1_embeddings.zarr
        well_name: B03
        well_id: 1
      - zarr_path: /path/to/experiment2_embeddings.zarr
        well_name: B03
        well_id: 1

    group_b:
      - zarr_path: /path/to/experiment3_embeddings.zarr
        well_name: C04
        well_id: 2

    n_perm: 1000
    max_cells: 2000
    random_seed: 42
    output_csv: /path/to/mmd_results.csv
    """

    group_a: list[WellSpec] = Field(..., min_length=1)
    group_b: list[WellSpec] = Field(..., min_length=1)
    n_perm: int = Field(default=1000, ge=0)
    max_cells: int = Field(default=2000, gt=0)
    random_seed: int = Field(default=42)
    output_csv: str | None = Field(default=None)

    def group_a_as_dicts(self) -> list[dict]:
        return [
            {"zarr_path": w.zarr_path, "well_name": w.well_name, "well_id": w.well_id}
            for w in self.group_a
        ]

    def group_b_as_dicts(self) -> list[dict]:
        return [
            {"zarr_path": w.zarr_path, "well_name": w.well_name, "well_id": w.well_id}
            for w in self.group_b
        ]
