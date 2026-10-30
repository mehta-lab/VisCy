"""Dataset manifest schemas and loaders for the DynaCell benchmark.

Pydantic models that parse and validate YAML manifests. Loaders accept
explicit file paths — no import-time registry or hardcoded config roots.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from dynacell.data._yaml import load_yaml


class DatasetRef(BaseModel):
    """Reference to a dataset target, resolved against a manifest registry.

    Carried under ``benchmark.dataset_ref`` in benchmark leaf configs.
    The composition-time resolver reads this reference and splices
    ``data_path``, ``source_channel``, and ``target_channel`` into the
    composed Lightning config.

    ``source_channel`` overrides the manifest's declared ``channels.source``
    for input-channel ablations (e.g. feeding the raw ``Brightfield`` stack
    instead of the ``Phase3D`` volume reconstructed from it) without minting a
    duplicate dataset. Omit it to use the manifest default.

    Extra keys are **forbidden**: Pydantic's default would silently drop a
    misspelled ``source_channel``, and the resulting run would train on the
    manifest's default channel while every config, log and W&B name claimed
    otherwise — an ablation that is a silent no-op is worse than one that fails.
    """

    model_config = ConfigDict(extra="forbid")

    dataset: str
    target: str
    source_channel: str | None = None


class VoxelSpacing(BaseModel):
    """Physical voxel spacing in micrometers."""

    z: float
    y: float
    x: float

    def as_list(self) -> list[float]:
        """Return spacing as ``[z, y, x]`` list for metric functions."""
        return [self.z, self.y, self.x]


class StoreLocations(BaseModel):
    """Zarr store paths for a single organelle target.

    ``train`` is optional: evaluation-only datasets (e.g. the ``hek-mantis-*``
    third-cell-type probe) ship a test store and no training data. Callers that
    need a train store must check for ``None`` — :func:`resolve_dataset_ref`
    propagates it as ``ResolvedDataset.data_path_train`` and the composition hook
    raises for any non-predict Lightning mode.
    """

    train: Path | None = None
    test: Path
    cell_segmentation: Path | None = None
    gt_cache_dir: Path | None = None


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

    @property
    def source_channel(self) -> str:
        """Return the single source channel name for source-target datasets.

        ``channels["source"]`` may be a string or a single-element list; a
        multi-element list is rejected since downstream ``HCSDataModule``
        takes one channel name.
        """
        source = self.channels["source"]
        if isinstance(source, str):
            return source
        if isinstance(source, list) and len(source) == 1:
            return source[0]
        raise ValueError(f"Manifest source channel must be a string or single-element list, got {source!r}.")

    def resolve_source_channel(self, name: str | None) -> str:
        """Return the model-input channel, validating an explicit override.

        Parameters
        ----------
        name : str or None
            Explicit channel from :attr:`DatasetRef.source_channel`. ``None``
            selects the manifest's declared :attr:`source_channel`.

        Returns
        -------
        str
            Channel name present in this manifest.

        Raises
        ------
        ValueError
            If ``name`` is not one of the manifest's declared channels. A typo
            would otherwise surface as an opaque failure deep inside iohub, or
            worse, silently select the wrong volume.
        """
        if name is None:
            return self.source_channel
        # ``auxiliary`` is optional in the schema (``channels: dict[str, str |
        # list[str]]``), so read it defensively -- absence is legal, not an error.
        auxiliary = self.channels.get("auxiliary", [])
        declared = [auxiliary] if isinstance(auxiliary, str) else list(auxiliary)
        source = self.channels["source"]
        declared += [source] if isinstance(source, str) else list(source)
        if name not in declared:
            raise ValueError(
                f"source_channel {name!r} is not declared in manifest {self.name!r}. "
                f"Available channels: {sorted(declared)}."
            )
        return name


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


@lru_cache(maxsize=64)
def load_manifest(manifest_path: Path) -> DatasetManifest:
    """Load and validate a dataset manifest from a YAML file.

    Cached by resolved path; manifests are treated as immutable within a
    process (same policy as :func:`viscy_utils.compose._load_yaml_cached`).

    Parameters
    ----------
    manifest_path : Path
        Path to a dataset manifest YAML file.

    Returns
    -------
    DatasetManifest
        Validated manifest.
    """
    return load_yaml(manifest_path, DatasetManifest)


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
    return load_yaml(split_path, SplitDefinition)


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
