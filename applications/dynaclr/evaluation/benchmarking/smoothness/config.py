"""Configuration models for smoothness evaluation."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ModelEntry(BaseModel):
    """A single model to evaluate."""

    path: str
    label: str


class SmoothnessEvalConfig(BaseModel):
    """Configuration for temporal smoothness evaluation.

    Parameters
    ----------
    models : list[ModelEntry]
        List of models to evaluate, each with a zarr path and display label.
    distance_metric : str
        Distance metric for similarity computation.
    time_offsets : list[int]
        Temporal offsets to compute (e.g., [1] for t->t+1).
    output_dir : str
        Directory for results (plots and CSV files).
    save_plots : bool
        Whether to save distribution plots per model.
    save_distributions : bool
        Whether to save full distance distributions as numpy arrays.
    use_optimized : bool
        Whether to use memory-optimized computation.
    verbose : bool
        Print verbose progress messages.
    """

    models: list[ModelEntry] = Field(..., min_length=1)
    distance_metric: Literal["cosine", "euclidean"] = "cosine"
    time_offsets: list[int] = Field(default=[1])
    output_dir: str = Field(...)
    save_plots: bool = True
    save_distributions: bool = False
    use_optimized: bool = True
    verbose: bool = False

    @model_validator(mode="after")
    def validate_paths(self):
        """Check that all model embedding paths exist."""
        for model in self.models:
            if not Path(model.path).exists():
                raise ValueError(f"Embedding not found: {model.path}")
        return self


class ResultFileEntry(BaseModel):
    """A single result CSV file for comparison."""

    path: str
    label: str


class CompareModelsConfig(BaseModel):
    """Configuration for comparing previously saved evaluation results.

    Parameters
    ----------
    result_files : list[ResultFileEntry]
        List of CSV result files to compare.
    metrics : list[str]
        Metric columns to include in the comparison table.
    output_path : Optional[str]
        Path to save combined results.
    output_format : str
        Output format for combined results.
    """

    result_files: list[ResultFileEntry] = Field(..., min_length=1)
    metrics: list[str] = Field(
        default=[
            "smoothness_score",
            "dynamic_range",
            "adjacent_frame_mean",
            "adjacent_frame_peak",
            "random_frame_mean",
            "random_frame_peak",
        ]
    )
    output_path: Optional[str] = None
    output_format: Literal["markdown", "csv", "json"] = "markdown"
