"""Frozen collection schemas for benchmark data curation."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel


class Provenance(BaseModel):
    """Airtable-derived provenance for a frozen collection."""

    airtable_base_id: str | None = None
    airtable_query: str | None = None
    record_ids: list[str] = []
    created_at: str
    created_by: str


class ChannelEntry(BaseModel):
    """Single channel in a collection experiment."""

    name: str
    marker: str


class CollectionExperiment(BaseModel):
    """One experiment within a benchmark collection."""

    name: str
    data_path: Path
    channels: list[ChannelEntry]
    perturbation_wells: dict[str, list[str]] | None = None
    interval_minutes: float | None = None
    start_hpi: float | None = None
    marker: str | None = None
    organelle: str | None = None
    pixel_size_xy_um: float
    pixel_size_z_um: float | None = None
    exclude_fovs: list[str] = []


class BenchmarkCollection(BaseModel):
    """Frozen collection tying experiments to train/test FOV membership."""

    name: str
    description: str
    provenance: Provenance
    experiments: list[CollectionExperiment]
    train_fovs: list[str] | None = None
    test_fovs: list[str] | None = None


def load_collection(collection_path: Path) -> BenchmarkCollection:
    """Load and validate a frozen benchmark collection.

    Parameters
    ----------
    collection_path : Path
        Path to a collection YAML file.

    Returns
    -------
    BenchmarkCollection
        Validated collection.
    """
    raw = OmegaConf.to_container(OmegaConf.load(collection_path), resolve=True)
    return BenchmarkCollection.model_validate(raw)
