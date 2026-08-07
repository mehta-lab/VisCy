"""Pydantic configuration models for LOT batch correction."""

from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, model_validator


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


class DatasetSpec(BaseModel):
    """A single embedding zarr plus an optional reference-population filter.

    Parameters
    ----------
    zarr : str
        Path to an AnnData embedding zarr.
    filter : UninfFilter, optional
        Filter selecting the reference population (e.g. uninfected cells).
        When omitted, all cells in the zarr are used.
    """

    zarr: str = Field(..., min_length=1)
    filter: Optional[UninfFilter] = Field(default=None)

    @model_validator(mode="after")
    def validate_path(self):
        if not Path(self.zarr).exists():
            raise ValueError(f"zarr not found: {self.zarr}")
        return self


class LotFitConfig(BaseModel):
    """Configuration for fitting a LOT batch-correction pipeline.

    Source and target are lists of datasets so multiple acquisitions from the
    same platform can be pooled into a single distribution before fitting.

    Parameters
    ----------
    source : list[DatasetSpec]
        Source datasets (e.g. light-sheet embeddings), each with an optional
        reference-population filter. Pooled into one source distribution.
    target : list[DatasetSpec]
        Target datasets (e.g. confocal embeddings), each with an optional
        reference-population filter. Pooled into one target distribution.
    channel : str, optional
        The bag-of-channels channel/marker these embeddings were computed for
        (e.g. ``"Phase3D"``). Recorded in the fitted pipeline for provenance so
        the map is not blindly applied to a different channel. By default
        ``None``.
    n_pca : int or None, optional
        Number of PCA components for the shared PCA. Set to ``null`` to
        disable PCA and fit LOT in the scaled embedding space. By default 50.
    ns_lot : int or None, optional
        Maximum cells subsampled per side for LOT fitting (compute cap on
        covariance estimation). Set to ``null`` to use all pooled cells.
        By default 3000.
    random_seed : int, optional
        Random seed, by default 42.
    output_pipeline : str
        Path to save the fitted pipeline (joblib pickle).
    """

    source: list[DatasetSpec] = Field(..., min_length=1)
    target: list[DatasetSpec] = Field(..., min_length=1)
    channel: Optional[str] = Field(default=None)
    n_pca: Optional[int] = Field(default=50, gt=0)
    ns_lot: Optional[int] = Field(default=3000, gt=0)
    random_seed: int = Field(default=42)
    output_pipeline: str = Field(..., min_length=1)
