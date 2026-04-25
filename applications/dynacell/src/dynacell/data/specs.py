"""Benchmark spec schemas for reproducible benchmark runs."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from dynacell.data._yaml import load_yaml


class BenchmarkSpec(BaseModel):
    """Executable benchmark recipe tying together pipeline stages."""

    name: str
    version: str
    description: str
    collection_path: Path
    preprocess_configs: list[Path] = Field(default_factory=list)
    train_preset: str | None = None
    predict_preset: str | None = None
    evaluate_config: Path | None = None
    report_config: Path | None = None
    output_root: Path
    checkpoint_path: Path | None = None


def load_benchmark_spec(spec_path: Path) -> BenchmarkSpec:
    """Load and validate a benchmark spec.

    Parameters
    ----------
    spec_path : Path
        Path to a benchmark spec YAML file.

    Returns
    -------
    BenchmarkSpec
        Validated benchmark spec.
    """
    return load_yaml(spec_path, BenchmarkSpec)
