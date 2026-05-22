"""Pydantic configuration models for LOT batch correction."""

from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class UninfFilter(BaseModel):
    """Specification for selecting uninfected reference cells from an obs table.

    Exactly one of ``startswith`` or ``equals`` must be provided.

    Parameters
    ----------
    column : str
        Name of the ``.obs`` column to filter on (e.g. ``"fov_name"``).
    startswith : str or list[str], optional
        Keep cells whose column value starts with any of these prefixes.
    equals : str, optional
        Keep cells whose column value equals this string.
    """

    column: str = Field(..., min_length=1)
    startswith: Optional[Union[str, list[str]]] = Field(default=None)
    equals: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def exactly_one_filter(self):
        has_sw = self.startswith is not None
        has_eq = self.equals is not None
        if not has_sw and not has_eq:
            raise ValueError("UninfFilter must specify either 'startswith' or 'equals'.")
        if has_sw and has_eq:
            raise ValueError("UninfFilter must specify only one of 'startswith' or 'equals'.")
        return self

    def to_dict(self) -> dict:
        """Convert to the dict format expected by _apply_filter."""
        d = {"column": self.column}
        if self.startswith is not None:
            d["startswith"] = self.startswith
        else:
            d["equals"] = self.equals
        return d


class LotFitConfig(BaseModel):
    """Configuration for fitting a LOT batch-correction pipeline.

    Parameters
    ----------
    source_zarr : str
        Path to the source AnnData zarr (e.g. light-sheet embeddings).
    target_zarr : str
        Path to the target AnnData zarr (e.g. confocal embeddings).
    source_uninf_filter : UninfFilter
        Filter identifying uninfected cells in the source dataset.
    target_uninf_filter : UninfFilter
        Filter identifying uninfected cells in the target dataset.
    n_pca : int, optional
        Number of PCA components for the shared PCA, by default 50.
    ns_lot : int, optional
        Maximum cells subsampled per dataset for LOT fitting, by default 3000.
    random_seed : int, optional
        Random seed, by default 42.
    output_pipeline : str
        Path to save the fitted pipeline (joblib pickle).
    """

    source_zarr: str = Field(..., min_length=1)
    target_zarr: str = Field(..., min_length=1)
    source_uninf_filter: UninfFilter
    target_uninf_filter: UninfFilter
    n_pca: int = Field(default=50, gt=0)
    ns_lot: int = Field(default=3000, gt=0)
    random_seed: int = Field(default=42)
    output_pipeline: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_paths(self):
        if not Path(self.source_zarr).exists():
            raise ValueError(f"source_zarr not found: {self.source_zarr}")
        if not Path(self.target_zarr).exists():
            raise ValueError(f"target_zarr not found: {self.target_zarr}")
        return self


class LotApplyConfig(BaseModel):
    """Configuration for applying a fitted LOT pipeline to a zarr.

    Parameters
    ----------
    input_zarr : str
        Path to the source AnnData zarr to correct.
    pipeline : str
        Path to the fitted pipeline file (joblib pickle).
    output_zarr : str
        Path to write the corrected AnnData zarr.
    overwrite : bool, optional
        Overwrite output if it exists, by default False.
    """

    input_zarr: str = Field(..., min_length=1)
    pipeline: str = Field(..., min_length=1)
    output_zarr: str = Field(..., min_length=1)
    overwrite: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_paths(self):
        if not Path(self.input_zarr).exists():
            raise ValueError(f"input_zarr not found: {self.input_zarr}")
        if not Path(self.pipeline).exists():
            raise ValueError(f"pipeline file not found: {self.pipeline}")
        output = Path(self.output_zarr)
        if output.exists() and not self.overwrite:
            raise ValueError(
                f"output_zarr already exists: {self.output_zarr}. "
                "Set overwrite: true to overwrite."
            )
        return self
