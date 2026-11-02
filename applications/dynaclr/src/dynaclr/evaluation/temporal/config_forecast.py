"""Configuration schema for pre-onset temporal event forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ForecastDataset(BaseModel):
    """One annotated experiment participating in a forecast evaluation."""

    experiment: str
    annotation: str


class ForecastEvent(BaseModel):
    """A binary state transition whose first positive frame defines onset."""

    name: str
    column: str
    negative: str
    positive: str


class TemporalModelSettings(BaseModel):
    """Shared training settings for temporal forecast models."""

    names: list[Literal["tcn", "rope_transformer"]] = Field(default_factory=lambda: ["tcn", "rope_transformer"])
    epochs: int = Field(default=15, ge=1)
    batch_size: int = Field(default=512, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    tcn_hidden: int = Field(default=64, ge=1)
    tcn_dilations: tuple[int, ...] = (1, 2)
    dropout: float = Field(default=0.3, ge=0, lt=1)
    rope_d_model: int = Field(default=128, ge=8)
    rope_heads: int = Field(default=4, ge=1)
    rope_layers: int = Field(default=2, ge=1)
    rope_ff_multiplier: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def validate_models(self) -> "TemporalModelSettings":
        if not self.names or len(self.names) != len(set(self.names)):
            raise ValueError("models.names must be non-empty and unique")
        if not self.tcn_dilations or min(self.tcn_dilations) < 1:
            raise ValueError("models.tcn_dilations must contain positive integers")
        if self.rope_d_model % self.rope_heads:
            raise ValueError("models.rope_d_model must be divisible by models.rope_heads")
        if (self.rope_d_model // self.rope_heads) % 2:
            raise ValueError("RoPE head dimension must be even")
        return self


class TemporalForecastConfig(BaseModel):
    """Leakage-safe event-anticipation evaluation for one embedding identity."""

    datasets: list[ForecastDataset]
    events: list[ForecastEvent]
    marker: str = "Phase3D"
    feature_key: str = "X_normalized_pca80"
    cohort_intersection_paths: list[str] = Field(default_factory=list)
    key_columns: tuple[str, ...] = ("experiment", "fov_name", "track_id", "t")
    group_columns: tuple[str, ...] = ("experiment", "fov_name", "track_id")
    split_group_columns: tuple[str, ...] = ("experiment", "fov_name")
    time_column: str = "hours_post_perturbation"
    blank_column: str | None = "is_blank"
    window_observations: int = Field(default=5, ge=1)
    require_full_context: bool = True
    horizons_hours: list[float] = Field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0])
    n_folds: int = Field(default=5, ge=2)
    random_seed: int = 42
    models: TemporalModelSettings = Field(default_factory=TemporalModelSettings)

    @model_validator(mode="after")
    def validate_contract(self) -> "TemporalForecastConfig":
        dataset_names = [item.experiment for item in self.datasets]
        event_names = [item.name for item in self.events]
        if not dataset_names or len(dataset_names) != len(set(dataset_names)):
            raise ValueError("datasets must be non-empty and unique by experiment")
        if not event_names or len(event_names) != len(set(event_names)):
            raise ValueError("events must be non-empty and unique by name")
        if not self.horizons_hours or any(value <= 0 for value in self.horizons_hours):
            raise ValueError("horizons_hours must contain positive values")
        if self.horizons_hours != sorted(set(self.horizons_hours)):
            raise ValueError("horizons_hours must be sorted and unique")
        required = {"experiment", "fov_name", "track_id", "t"}
        if not required.issubset(self.key_columns):
            raise ValueError(f"key_columns must include {sorted(required)}")
        if not set(self.group_columns).issubset(self.key_columns):
            raise ValueError("group_columns must be a subset of key_columns")
        return self


def load_temporal_forecast_config(path: str | Path) -> TemporalForecastConfig:
    """Load a standalone or section-wrapped forecast YAML."""

    with Path(path).open() as stream:
        raw = yaml.safe_load(stream) or {}
    if "temporal_forecast" in raw:
        raw = raw["temporal_forecast"]
    return TemporalForecastConfig(**raw)
