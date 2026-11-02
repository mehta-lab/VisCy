"""Configuration schema for the DynaFormer temporal head and its baseline ladder."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from dynaclr.evaluation.temporal.config_forecast import ForecastDataset, ForecastEvent


class DynaFormerHeadSettings(BaseModel):
    """Architecture of the temporal head."""

    d_model: int = Field(default=128, ge=8)
    n_heads: int = Field(default=4, ge=1)
    n_layers: int = Field(default=3, ge=1)
    feedforward_multiplier: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    rope_min_period_hours: float = Field(default=1.0 / 6.0, gt=0)
    rope_max_period_hours: float = Field(default=96.0, gt=0)

    @model_validator(mode="after")
    def validate_head(self) -> "DynaFormerHeadSettings":
        if self.d_model % self.n_heads:
            raise ValueError("head.d_model must be divisible by head.n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE head dimension must be even")
        if self.rope_max_period_hours <= self.rope_min_period_hours:
            raise ValueError("head.rope_max_period_hours must exceed head.rope_min_period_hours")
        return self


class DynaFormerTrainSettings(BaseModel):
    """Self-supervised training settings."""

    steps: int = Field(default=5000, ge=1)
    batch_size: int = Field(default=256, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.05, ge=0)
    temperature: float = Field(default=0.1, gt=0)
    n_negatives: int = Field(default=64, ge=1)
    log_every: int = Field(default=250, ge=1)


class DynaFormerProbeSettings(BaseModel):
    """Linear probe and raw-sequence baseline settings."""

    C: float = Field(default=1.0, gt=0)
    smoothing_sigma_hours: float = Field(default=1.0, gt=0)
    last_k: list[int] = Field(default_factory=lambda: [4, 8])
    hysteresis_frames: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def validate_probe(self) -> "DynaFormerProbeSettings":
        if not self.last_k or self.last_k != sorted(set(self.last_k)) or min(self.last_k) < 2:
            raise ValueError("probe.last_k must be sorted, unique, and >= 2")
        return self


class DynaFormerConfig(BaseModel):
    """Encoder-agnostic temporal head on frozen embeddings with a baseline ladder.

    The dataset, event, and key fields mirror ``TemporalForecastConfig`` so the
    existing ``load_forecast_data`` and ``build_forecast_targets`` are reused
    unchanged.
    """

    datasets: list[ForecastDataset]
    events: list[ForecastEvent]
    primary_event: str = "infection"
    marker: str = "Phase3D"
    feature_key: str = "X"
    cohort_intersection_paths: list[str] = Field(default_factory=list)
    key_columns: tuple[str, ...] = ("experiment", "fov_name", "track_id", "t")
    group_columns: tuple[str, ...] = ("experiment", "fov_name", "track_id")
    split_group_columns: tuple[str, ...] = ("experiment", "fov_name")
    time_column: str = "hours_post_perturbation"
    blank_column: str | None = "is_blank"
    parent_column: str = "parent_track_id"
    k_frames: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16])
    horizons_hours: list[float] = Field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0])
    n_max_frames: int = Field(default=32, ge=1)
    frame_dropout: float = Field(default=0.1, ge=0, lt=1)
    causal_probability: float = Field(default=0.5, ge=0, le=1)
    history_budget: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    future_budget: list[int] = Field(default_factory=lambda: [0, 2, 4, 8])
    n_folds: int = Field(default=5, ge=2)
    random_seed: int = 42
    forecast_reference_dir: str | None = None
    head: DynaFormerHeadSettings = Field(default_factory=DynaFormerHeadSettings)
    train: DynaFormerTrainSettings = Field(default_factory=DynaFormerTrainSettings)
    probe: DynaFormerProbeSettings = Field(default_factory=DynaFormerProbeSettings)

    @model_validator(mode="after")
    def validate_contract(self) -> "DynaFormerConfig":
        dataset_names = [item.experiment for item in self.datasets]
        event_names = [item.name for item in self.events]
        if not dataset_names or len(dataset_names) != len(set(dataset_names)):
            raise ValueError("datasets must be non-empty and unique by experiment")
        if not event_names or len(event_names) != len(set(event_names)):
            raise ValueError("events must be non-empty and unique by name")
        if self.primary_event not in event_names:
            raise ValueError(f"primary_event {self.primary_event!r} is not one of {event_names}")
        if not self.k_frames or self.k_frames != sorted(set(self.k_frames)) or min(self.k_frames) < 1:
            raise ValueError("k_frames must be sorted, unique, positive integers")
        if not self.horizons_hours or self.horizons_hours != sorted(set(self.horizons_hours)):
            raise ValueError("horizons_hours must be sorted and unique")
        if self.history_budget != sorted(set(self.history_budget)) or min(self.history_budget) < 1:
            raise ValueError("history_budget must be sorted, unique, >= 1")
        if self.future_budget != sorted(set(self.future_budget)) or min(self.future_budget) < 0:
            raise ValueError("future_budget must be sorted, unique, >= 0")
        if max(self.history_budget) > self.n_max_frames:
            raise ValueError("history_budget cannot exceed n_max_frames")
        required = {"experiment", "fov_name", "track_id", "t"}
        if not required.issubset(self.key_columns):
            raise ValueError(f"key_columns must include {sorted(required)}")
        return self

    @property
    def primary(self) -> ForecastEvent:
        """The event whose onset drives the probes and timing readouts."""
        return next(event for event in self.events if event.name == self.primary_event)


def load_dynaformer_config(path: str | Path) -> DynaFormerConfig:
    """Load a standalone or ``dynaformer:``-wrapped YAML."""

    with Path(path).open() as stream:
        raw = yaml.safe_load(stream) or {}
    if "dynaformer" in raw:
        raw = raw["dynaformer"]
    return DynaFormerConfig(**raw)
