"""Dataset manifest schemas and loaders for the DynaCell benchmark.

Pydantic models that parse and validate YAML manifests. Loaders accept
explicit file paths — no import-time registry or hardcoded config roots.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel, field_validator, model_validator


class VoxelSpacing(BaseModel):
    """Physical voxel spacing in micrometers."""

    z: float
    y: float
    x: float

    def as_list(self) -> list[float]:
        """Return spacing as ``[z, y, x]`` list for metric functions."""
        return [self.z, self.y, self.x]


class StoreLocations(BaseModel):
    """Zarr store paths for a single organelle target."""

    train: Path
    test: Path
    cell_segmentation: Path | None = None


class TargetConfig(BaseModel):
    """Configuration for a single organelle prediction target."""

    gene: str
    organelle: str
    display_name: str
    target_channel: str
    stores: StoreLocations
    splits: str


class DatasetManifest(BaseModel):
    """Top-level dataset manifest."""

    name: str
    version: str
    description: str
    cell_type: str
    imaging_modality: str
    spacing: VoxelSpacing
    channels: dict[str, str | list[str]]
    targets: dict[str, TargetConfig]

    @field_validator("targets")
    @classmethod
    def _targets_not_empty(cls, v: dict) -> dict:
        """Validate that at least one target is defined."""
        if not v:
            raise ValueError("Manifest must define at least one target.")
        return v


class SplitDefinition(BaseModel):
    """Train/val/test FOV split for one organelle."""

    split_version: str
    random_seed: int
    source_stores: list[Path] | None = None
    selection_criteria: dict | None = None
    train: dict
    test: dict
    val: dict | None = None

    @model_validator(mode="after")
    def _check_counts(self) -> SplitDefinition:
        """Validate count matches len(fovs) when fovs is non-empty."""
        for split_name in ("train", "val", "test"):
            split = getattr(self, split_name)
            if split is None:
                continue
            fovs = split.get("fovs", [])
            if fovs and "count" in split:
                if len(fovs) != split["count"]:
                    raise ValueError(f"{split_name} declares count={split['count']} but has {len(fovs)} FOVs.")
        return self


def load_manifest(manifest_path: Path) -> DatasetManifest:
    """Load and validate a dataset manifest from a YAML file.

    Parameters
    ----------
    manifest_path : Path
        Path to a dataset manifest YAML file.

    Returns
    -------
    DatasetManifest
        Validated manifest.
    """
    raw = OmegaConf.to_container(OmegaConf.load(manifest_path), resolve=True)
    return DatasetManifest.model_validate(raw)


def load_splits(split_path: Path) -> SplitDefinition:
    """Load and validate a split definition from a YAML file.

    Parameters
    ----------
    split_path : Path
        Path to a split definition YAML file.

    Returns
    -------
    SplitDefinition
        Validated split definition.
    """
    raw = OmegaConf.to_container(OmegaConf.load(split_path), resolve=True)
    return SplitDefinition.model_validate(raw)


def get_target(manifest: DatasetManifest, target_name: str) -> TargetConfig:
    """Get a specific target from a loaded manifest.

    Parameters
    ----------
    manifest : DatasetManifest
        A loaded dataset manifest.
    target_name : str
        Name of the target (e.g., ``"sec61b"``).

    Returns
    -------
    TargetConfig
        Target configuration.

    Raises
    ------
    KeyError
        If ``target_name`` is not in the manifest.
    """
    return manifest.targets[target_name]
