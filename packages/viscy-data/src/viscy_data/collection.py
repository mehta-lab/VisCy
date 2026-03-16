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


class SourceChannel(BaseModel):
    """Semantic channel mapping across experiments.

    Parameters
    ----------
    label : str
        Semantic label (e.g. ``"labelfree"``, ``"reporter"``).
    per_experiment : dict[str, str]
        ``{experiment_name: zarr_channel_name}`` mapping.
    """

    label: str
    per_experiment: dict[str, str]


class ExperimentEntry(BaseModel):
    """A single experiment within a collection.

    Parameters
    ----------
    name : str
        Unique experiment identifier.
    data_path : str
        Path to the HCS OME-Zarr store.
    tracks_path : str
        Root directory for per-FOV tracking CSVs.
    channel_names : list[str]
        All channel names in the zarr store.
    condition_wells : dict[str, list[str]]
        Mapping of condition label to well names.
    interval_minutes : float
        Time between frames in minutes.
    start_hpi : float
        Hours post perturbation at frame 0.
    marker : str
        Protein marker or dye name (e.g. ``"TOMM20"``, ``"SEC61B"``).
    organelle : str
        Target organelle or cellular structure (e.g. ``"mitochondria"``).
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
    channel_names: list[str]
    condition_wells: dict[str, list[str]]
    interval_minutes: float
    start_hpi: float = 0.0
    marker: str = ""
    organelle: str = ""
    date: str = ""
    moi: float = 0.0
    exclude_fovs: list[str] = []


class Collection(BaseModel):
    """Curated collection of experiments for training.

    Parameters
    ----------
    name : str
        Collection name.
    description : str
        Human-readable description.
    provenance : Provenance
        How the collection was created.
    source_channels : list[SourceChannel]
        Semantic channel mapping across experiments.
    experiments : list[ExperimentEntry]
        Experiment entries.
    fov_records : list[FOVRecord]
        Raw provenance records from Airtable.
    """

    name: str
    description: str = ""
    provenance: Provenance = Provenance()
    source_channels: list[SourceChannel]
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

        for sc in self.source_channels:
            # 2. Every per_experiment key references a valid experiment
            for key in sc.per_experiment:
                if key not in exp_names:
                    raise ValueError(
                        f"source_channels['{sc.label}'].per_experiment references "
                        f"unknown experiment '{key}'. Valid: {sorted(exp_names)}"
                    )

            # 3. Each mapped channel name exists in that experiment's channel_names
            for exp in self.experiments:
                if exp.name not in sc.per_experiment:
                    continue  # experiment doesn't have this channel — allowed
                mapped_ch = sc.per_experiment[exp.name]
                if mapped_ch not in exp.channel_names:
                    raise ValueError(
                        f"source_channels['{sc.label}'] maps experiment '{exp.name}' "
                        f"to channel '{mapped_ch}', but that experiment's "
                        f"channel_names are {exp.channel_names}."
                    )

        for exp in self.experiments:
            # 5. interval_minutes > 0
            if exp.interval_minutes <= 0:
                raise ValueError(
                    f"Experiment '{exp.name}': interval_minutes must be positive, got {exp.interval_minutes}."
                )
            # 6. condition_wells not empty
            if not exp.condition_wells:
                raise ValueError(f"Experiment '{exp.name}': condition_wells must not be empty.")

        return self


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
    source_channels: list[SourceChannel],
    name: str,
    description: str = "",
) -> Collection:
    """Build a collection by grouping FOVRecords into experiments.

    Groups records by ``dataset`` to create :class:`ExperimentEntry` instances.
    When a single dataset contains multiple markers (organelles), it is
    automatically split into one experiment entry per marker with a
    ``_{MARKER}`` suffix on the name.

    Derives ``condition_wells`` from ``cell_state`` + ``well_id``,
    ``channel_names`` from records' ``channel_names``,
    ``interval_minutes`` from ``time_interval_min``,
    and ``start_hpi`` from ``hours_post_perturbation``.

    Parameters
    ----------
    records : list[FOVRecord]
        FOV-level records (typically from Airtable).
    source_channels : list[SourceChannel]
        Semantic channel mapping.
    name : str
        Collection name.
    description : str
        Collection description.

    Returns
    -------
    Collection
        Validated collection.
    """
    grouped = _group_records(records)

    experiments: list[ExperimentEntry] = []
    for exp_name, recs in grouped.items():
        first = recs[0]

        # Derive condition_wells from cell_state + well_id
        condition_wells: dict[str, list[str]] = defaultdict(list)
        seen_wells: set[tuple[str, str]] = set()
        for rec in recs:
            state = rec.cell_state or "unknown"
            if (state, rec.well_id) not in seen_wells:
                condition_wells[state].append(rec.well_id)
                seen_wells.add((state, rec.well_id))

        # Derive channel_names from first record
        channel_names = first.channel_names if first.channel_names else []

        experiments.append(
            ExperimentEntry(
                name=exp_name,
                data_path=first.data_path or "",
                tracks_path=first.tracks_path or "",
                channel_names=channel_names,
                condition_wells=dict(condition_wells),
                interval_minutes=first.time_interval_min or 30.0,
                start_hpi=first.hours_post_perturbation or 0.0,
                marker=first.marker or "",
                organelle=first.organelle or "",
                moi=first.moi or 0.0,
            )
        )

    return Collection(
        name=name,
        description=description,
        source_channels=source_channels,
        experiments=experiments,
        fov_records=records,
    )
