"""Pydantic models for A549 mantis authoring YAMLs.

Two files per plate, both packaged under
``dynacell/_configs/datasets/a549-mantis/authoring/``:

- ``platemaps/<plate>.yaml`` — per-well ``condition``, ``gene_channel_map``,
  plate-level ``hpi_start``, ``native_delta_t_min``.
- ``splits/<plate>.yaml`` — per-target train/test FOV lists (which
  implicitly encode the biological-QC filter).

The pipeline reads only these YAMLs plus the plate zarr. No Airtable or
Confluence access at assembly time — the Airtable fixtures (authoring
reference, not packaged here) live in the dynacell-paper repo.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from dynacell.data._yaml import load_yaml


class WellMetadata(BaseModel):
    """Per-well metadata for one experimental well."""

    condition: str
    """e.g. 'mock', 'DENV', 'ZIKV'."""

    moi: int | None = None
    """Multiplicity of infection (absent for mock)."""

    gene_channel_map: dict[str, str]
    """Native channel name → gene key, e.g. {'GFP EX488 EM525-45': 'sec61b'}."""

    model_config = {"extra": "forbid"}


class Platemap(BaseModel):
    """Authoring platemap for one plate."""

    experiment: str
    run_dir: str
    zarr_filename: str

    native_delta_t_min: float
    hpi_start: float
    native_t: int

    wells: dict[str, WellMetadata]
    """Map well_id (e.g. 'B/1') to WellMetadata."""

    model_config = {"extra": "forbid"}


class TargetSplit(BaseModel):
    """Train/test position lists for one target."""

    train: list[str] = Field(default_factory=list)
    test: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class Splits(BaseModel):
    """Authoring splits for one plate across all targets authored here."""

    experiment: str
    targets: dict[str, TargetSplit]
    """Map target gene key (e.g. 'sec61b') to TargetSplit."""

    model_config = {"extra": "forbid"}


def load_platemap(path: Path | str) -> Platemap:
    """Load + validate a platemap YAML."""
    return load_yaml(Path(path), Platemap)


def load_splits(path: Path | str) -> Splits:
    """Load + validate a splits YAML."""
    return load_yaml(Path(path), Splits)
