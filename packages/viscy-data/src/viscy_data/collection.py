"""Collection schema for curated multi-experiment data.

A :class:`Collection` is a git-tracked YAML file that describes which
experiments, channels, and FOV records go into a training run. It is
generated from Airtable at curation time and consumed at training time
with no Airtable dependency.

Data flow::

    Airtable → list[FOVRecord] → collection.yml (git-tracked)
                                       ↓
                 collection.yml + CSVs → cell_index.parquet
                                       ↓
                 parquet + collection → Training
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

from viscy_data.schemas import FOVRecord


class Provenance(BaseModel):
    """Provenance metadata for how a collection was created.

    Parameters
    ----------
    airtable_base_id : str or None
        Airtable base identifier.
    airtable_query : str or None
        Query or formula used to fetch records.
    record_ids : list[str]
        Airtable record IDs included.
    created_at : str or None
        ISO 8601 creation timestamp.
    created_by : str or None
        Author of the collection.
    """

    airtable_base_id: str | None = None
    airtable_query: str | None = None
    record_ids: list[str] = []
    created_at: str | None = None
    created_by: str | None = None


class ChannelEntry(BaseModel):
    """A single channel in an experiment.

    Parameters
    ----------
    name : str
        Zarr channel name (e.g. ``"Phase3D"``, ``"raw GFP EX488 EM525-45"``).
    marker : str
        Protein marker or channel identity (e.g. ``"Phase3D"``, ``"TOMM20"``).
    wells : list[str]
        Wells where this channel is biologically valid (e.g. ``["B/3", "C/2"]``).
        Empty list means the channel is valid in all wells of the experiment.
    """

    name: str
    marker: str
    wells: list[str] = []


class ExperimentEntry(BaseModel):
    """A single experiment within a collection.

    Parameters
    ----------
    name : str
        Unique experiment identifier (typically the zarr plate stem).
    data_path : str
        Path to the HCS OME-Zarr store.
    tracks_path : str
        Root directory for per-FOV tracking CSVs.
    channels : list[ChannelEntry]
        Channels to include, each with zarr name + marker.
    channel_names : list[str]
        All channel names in the zarr store (for channel index lookup).
        If empty, derived from ``channels`` at validation time.
    perturbation_wells : dict[str, list[str]]
        Mapping of perturbation label to well names.
    interval_minutes : float
        Time between frames in minutes.
    start_hpi : float
        Hours post perturbation at frame 0.
    marker : str
        Primary protein marker (deprecated — use per-channel markers).
    organelle : str
        Target organelle or cellular structure.
    microscope : str
        Microscope identifier.
    pixel_size_xy_um : float or None
        Pixel size in XY in micrometers. None means unknown / no rescaling.
    pixel_size_z_um : float or None
        Voxel size in Z in micrometers. None means unknown / no rescaling.
    date : str
        Experiment date string.
    moi : float
        Multiplicity of infection.
    exclude_fovs : list[str]
        FOVs to exclude from this experiment.
    """

    name: str
    data_path: str
    tracks_path: str
    channels: list[ChannelEntry] = []
    channel_names: list[str] = []
    perturbation_wells: dict[str, list[str]] = {}
    interval_minutes: float = 30.0
    start_hpi: float = 0.0
    marker: str = ""
    organelle: str = ""
    microscope: str = ""
    pixel_size_xy_um: float | None = None
    pixel_size_z_um: float | None = None
    date: str = ""
    moi: float = 0.0
    exclude_fovs: list[str] = []

    @model_validator(mode="after")
    def _normalize(self) -> ExperimentEntry:
        # Derive channel_names from channels if not set
        if not self.channel_names and self.channels:
            self.channel_names = [ch.name for ch in self.channels]
        # Derive channels from channel_names if not set
        if not self.channels and self.channel_names:
            self.channels = [ChannelEntry(name=ch, marker=ch) for ch in self.channel_names]
        return self


class Collection(BaseModel):
    """Curated collection of experiments for training.

    The YAML is the complete reproducible recipe for building a flat
    parquet. Channels are defined per-experiment with name + marker.

    Parameters
    ----------
    name : str
        Collection name.
    description : str
        Human-readable description.
    datasets_root : str or None
        Optional path prefix substituted for ``${datasets_root}`` in
        ``data_path`` and ``tracks_path`` at load time.  Paths not
        starting with this root are left unchanged.
    provenance : Provenance
        How the collection was created.
    experiments : list[ExperimentEntry]
        Experiment entries.
    fov_records : list[FOVRecord]
        Raw provenance records from Airtable.
    """

    name: str
    description: str = ""
    datasets_root: str | None = None
    provenance: Provenance = Provenance()
    experiments: list[ExperimentEntry]
    fov_records: list[FOVRecord] = []

    @model_validator(mode="after")
    def _validate_collection(self) -> Collection:
        exp_names = {e.name for e in self.experiments}

        # 1. Experiment names unique
        if len(exp_names) != len(self.experiments):
            seen: set[str] = set()
            for e in self.experiments:
                if e.name in seen:
                    raise ValueError(f"Duplicate experiment name '{e.name}'.")
                seen.add(e.name)

        for exp in self.experiments:
            if exp.interval_minutes < 0:
                raise ValueError(
                    f"Experiment '{exp.name}': interval_minutes must be non-negative, got {exp.interval_minutes}."
                )
            wells = exp.perturbation_wells
            if not wells:
                raise ValueError(f"Experiment '{exp.name}': perturbation_wells must not be empty.")

        return self


_DATASETS_ROOT_VAR = "${datasets_root}"


def _resolve_datasets_root(data: dict) -> None:
    """Replace ``${datasets_root}`` in experiment paths with the root value.

    Mutates *data* in place.
    """
    root = data.get("datasets_root")
    if not root:
        return
    root = root.rstrip("/")
    for exp in data.get("experiments", []):
        for key in ("data_path", "tracks_path"):
            val = exp.get(key, "")
            if _DATASETS_ROOT_VAR in val:
                exp[key] = val.replace(_DATASETS_ROOT_VAR, root)


def _unresolve_datasets_root(data: dict, datasets_root: str) -> None:
    """Replace the resolved root prefix with ``${datasets_root}`` for portable YAML.

    Mutates *data* in place.  Only paths that start with *datasets_root* are
    modified; paths pointing elsewhere are left as absolute strings.
    """
    root = datasets_root.rstrip("/")
    for exp in data.get("experiments", []):
        for key in ("data_path", "tracks_path"):
            val = exp.get(key, "")
            if val.startswith(root + "/"):
                exp[key] = _DATASETS_ROOT_VAR + val[len(root) :]


def load_collection(path: str | Path) -> Collection:
    """Load a collection from a YAML file.

    Parameters
    ----------
    path : str | Path
        Path to the collection YAML.

    Returns
    -------
    Collection
        Validated collection.
    """
    with open(Path(path)) as f:
        data = yaml.safe_load(f)
    _resolve_datasets_root(data)
    return Collection(**data)


def save_collection(collection: Collection, path: str | Path) -> None:
    """Save a collection to a YAML file.

    Parameters
    ----------
    collection : Collection
        Collection to save.
    path : str | Path
        Output YAML path.
    """
    data = collection.model_dump(mode="json")
    if collection.datasets_root:
        _unresolve_datasets_root(data, collection.datasets_root)
    with open(Path(path), "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def _group_records(records: list[FOVRecord]) -> dict[str, list[FOVRecord]]:
    """Group FOV records into experiment entries.

    Records within the same ``dataset`` that have different ``marker``
    values are split into separate groups, with the marker appended to
    the experiment name (e.g. ``"2025_07_24_EXP_TOMM20"``).  Datasets
    where all records share a single marker are grouped under the
    original dataset name.

    Parameters
    ----------
    records : list[FOVRecord]
        FOV-level records.

    Returns
    -------
    dict[str, list[FOVRecord]]
        Mapping of experiment name to records.
    """
    by_dataset: dict[str, list[FOVRecord]] = defaultdict(list)
    for rec in records:
        by_dataset[rec.dataset].append(rec)

    grouped: dict[str, list[FOVRecord]] = {}
    for dataset_name, recs in by_dataset.items():
        markers = {rec.marker for rec in recs}
        if len(markers) <= 1:
            grouped[dataset_name] = recs
        else:
            by_marker: dict[str, list[FOVRecord]] = defaultdict(list)
            for rec in recs:
                by_marker[rec.marker or "unknown"].append(rec)
            for marker, marker_recs in by_marker.items():
                grouped[f"{dataset_name}_{marker}"] = marker_recs
    return grouped


def build_collection(
    records: list[FOVRecord],
    name: str,
    description: str = "",
    channel_markers: dict[str, list[tuple[str, str]]] | None = None,
    datasets_root: str | None = None,
) -> Collection:
    """Build a collection by grouping FOVRecords into experiments.

    Groups records by ``dataset`` to create :class:`ExperimentEntry` instances.
    When a single dataset contains multiple markers (organelles), it is
    automatically split into one experiment entry per marker with a
    ``_{MARKER}`` suffix on the name.

    Parameters
    ----------
    records : list[FOVRecord]
        FOV-level records (typically from Airtable).
    name : str
        Collection name.
    description : str
        Collection description.
    channel_markers : dict[str, list[tuple[str, str]]] or None
        Per-experiment ``{exp_name: [(zarr_channel_name, marker), ...]}`` mapping.
        If None, derives from the first record's ``channel_names`` using
        channel names as markers.
    datasets_root : str or None
        Passed through to :class:`Collection`.  When set, ``save_collection``
        will write ``${datasets_root}`` prefixes instead of absolute paths.

    Returns
    -------
    Collection
        Validated collection.
    """
    grouped = _group_records(records)

    experiments: list[ExperimentEntry] = []
    for exp_name, recs in grouped.items():
        first = recs[0]

        # Derive perturbation_wells from perturbation + well_id
        perturbation_wells: dict[str, list[str]] = defaultdict(list)
        seen_wells: set[tuple[str, str]] = set()
        for rec in recs:
            perturbation = rec.perturbation or rec.cell_state or "unknown"
            if (perturbation, rec.well_id) not in seen_wells:
                perturbation_wells[perturbation].append(rec.well_id)
                seen_wells.add((perturbation, rec.well_id))

        # Derive channels from channel_markers or channel_names
        channels: list[ChannelEntry] = []
        if channel_markers and exp_name in channel_markers:
            channels = [ChannelEntry(name=n, marker=m) for n, m in channel_markers[exp_name]]
        elif first.channel_names:
            channels = [ChannelEntry(name=n, marker=n) for n in first.channel_names]

        # Auto-populate wells per channel from per-record channel_markers.
        # A channel gets a wells restriction if only a subset of wells have
        # a non-None marker for it in Airtable.
        all_wells = sorted({rec.well_id for rec in recs})
        for ch in channels:
            wells_with_marker = sorted({rec.well_id for rec in recs if ch.name in rec.channel_markers})
            if wells_with_marker and wells_with_marker != all_wells:
                ch.wells = wells_with_marker

        experiments.append(
            ExperimentEntry(
                name=exp_name,
                data_path=first.data_path or "",
                tracks_path=first.tracks_path or "",
                channels=channels,
                perturbation_wells=dict(perturbation_wells),
                interval_minutes=first.time_interval_min or 30.0,
                start_hpi=first.hours_post_perturbation or 0.0,
                marker=first.marker or "",
                organelle=first.organelle or "",
                pixel_size_xy_um=getattr(first, "pixel_size_xy_um", None),
                pixel_size_z_um=getattr(first, "pixel_size_z_um", None),
                moi=first.moi or 0.0,
            )
        )

    return Collection(
        name=name,
        description=description,
        datasets_root=datasets_root,
        experiments=experiments,
        fov_records=records,
    )
