"""Pydantic models for Airtable Datasets table records and unified zattrs schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from viscy_data.channel_utils import parse_channel_name
from viscy_data.schemas import FOVRecord

MAX_CHANNELS = 8


def parse_position_name(name: str) -> tuple[str, str]:
    """Split an OME-Zarr position name into well path and FOV.

    Parameters
    ----------
    name : str
        Position name, e.g. ``"B/1/000000"``.

    Returns
    -------
    tuple[str, str]
        ``(well_path, fov)`` — e.g. ``("B/1", "000000")``.
    """
    parts = name.split("/")
    well_path = "/".join(parts[:2])
    fov = parts[2] if len(parts) > 2 else ""
    return well_path, fov


class BiologicalAnnotation(BaseModel):
    """Biological meaning of a channel.

    Parameters
    ----------
    organelle : str
        Target organelle (e.g. "endoplasmic_reticulum", "nucleus").
    marker : str
        Marker protein or dye name (e.g. "SEC61B", "H2B").
    marker_type : str
        How the marker is attached to the target.
    fluorophore : str or None
        Fluorophore name if applicable (e.g. "eGFP", "mCherry").
    """

    organelle: str | None = None
    marker: str
    marker_type: Literal["protein_tag", "direct_label", "nuclear_dye", "virtual_stain"] = "protein_tag"
    fluorophore: str | None = None


class ChannelAnnotationEntry(BaseModel):
    """Annotation for a single channel.

    Parameters
    ----------
    channel_type : str
        Modality of the channel.
    biological_annotation : BiologicalAnnotation or None
        Biological meaning; None for label-free channels.
    """

    channel_type: Literal["fluorescence", "labelfree", "virtual_stain"]
    biological_annotation: BiologicalAnnotation | None = None


class Perturbation(BaseModel):
    """A perturbation applied to a well.

    Extra fields (moi, concentration_nm, etc.) are allowed.

    Parameters
    ----------
    name : str
        Perturbation name (e.g. "ZIKV", "DMSO").
    type : str
        Perturbation category (e.g. "virus", "drug", "control").
    hours_post : float
        Hours post-perturbation at imaging time.
    """

    model_config = {"extra": "allow"}

    name: str
    type: str = "unknown"
    hours_post: float


class WellExperimentMetadata(BaseModel):
    """Experiment metadata for a single well.

    Parameters
    ----------
    perturbations : list[Perturbation]
        Perturbations applied to this well.
    time_sampling_minutes : float
        Time interval between frames in minutes.
    """

    perturbations: list[Perturbation] = Field(default_factory=list)
    time_sampling_minutes: float


class DatasetRecord(FOVRecord):
    """A single FOV-level record from the Airtable Datasets table.

    Extends :class:`~viscy_data.schemas.FOVRecord` with Airtable-specific
    raw channel fields (before flattening to ``channel_names``).
    """

    channel_0_name: str | None = None
    channel_0_marker: str | None = None
    channel_1_name: str | None = None
    channel_1_marker: str | None = None
    channel_2_name: str | None = None
    channel_2_marker: str | None = None
    channel_3_name: str | None = None
    channel_3_marker: str | None = None
    channel_4_name: str | None = None
    channel_4_marker: str | None = None
    channel_5_name: str | None = None
    channel_5_marker: str | None = None
    channel_6_name: str | None = None
    channel_6_marker: str | None = None
    channel_7_name: str | None = None
    channel_7_marker: str | None = None
    record_id: str | None = None

    @model_validator(mode="after")
    def _derive_channel_names(self) -> DatasetRecord:
        """Populate ``channel_names`` from ``channel_0..7_name`` fields."""
        if not self.channel_names:
            names = []
            for i in range(MAX_CHANNELS):
                name = getattr(self, f"channel_{i}_name")
                if name is not None:
                    names.append(name)
            self.channel_names = names
        return self

    @classmethod
    def from_airtable_record(cls, record: dict) -> DatasetRecord:
        """Parse from an Airtable API response.

        Parameters
        ----------
        record : dict
            Raw Airtable record with ``"id"`` and ``"fields"`` keys.
        """
        fields = record.get("fields", {})

        # Select fields return dict with "name" key; extract just the name
        def _select_val(v):
            if isinstance(v, dict):
                return v.get("name", v)
            return v

        # multipleSelects return list of dicts
        def _multi_select_val(v):
            if isinstance(v, list):
                return [item.get("name", item) if isinstance(item, dict) else item for item in v]
            return v

        return cls(
            dataset=fields.get("dataset", ""),
            well_id=fields.get("well_id", ""),
            fov=fields.get("fov"),
            cell_type=_select_val(fields.get("cell_type")),
            cell_state=_select_val(fields.get("cell_state")),
            cell_line=_multi_select_val(fields.get("cell_line")),
            marker=_select_val(fields.get("marker")),
            organelle=_select_val(fields.get("organelle")),
            perturbation=_select_val(fields.get("perturbation")),
            hours_post_perturbation=fields.get("hours_post_perturbation"),
            moi=fields.get("moi"),
            time_interval_min=fields.get("time_interval_min"),
            seeding_density=fields.get("seeding_density"),
            treatment_concentration_nm=fields.get("treatment_concentration_nm"),
            **{
                f"channel_{i}_{attr}": (
                    fields.get(f"channel_{i}_{attr}")
                    if attr == "name"
                    else _select_val(fields.get(f"channel_{i}_{attr}"))
                )
                for i in range(MAX_CHANNELS)
                for attr in ("name", "marker")
            },
            data_path=fields.get("data_path"),
            tracks_path=fields.get("tracks_path"),
            fluorescence_modality=_select_val(fields.get("fluorescence_modality")),
            t_shape=fields.get("t_shape"),
            c_shape=fields.get("c_shape"),
            z_shape=fields.get("z_shape"),
            y_shape=fields.get("y_shape"),
            x_shape=fields.get("x_shape"),
            record_id=record.get("id"),
        )

    def to_channels_metadata(self) -> dict[str, dict]:
        """Return dict for writing to ``.zattrs["channels_metadata"]``.

        Maps each channel name to a ``ChannelAnnotationEntry``-compatible dict
        with ``channel_type`` (derived from channel name parsing) and
        ``biological_annotation`` with the marker from Airtable.

        For labelfree channels, ``marker`` defaults to the channel name
        (e.g., Phase3D). For fluorescence channels, ``marker`` comes from
        the ``channel_N_marker`` Airtable field (e.g., TOMM20, SEC61).
        """
        annotation: dict[str, dict] = {}
        for i in range(MAX_CHANNELS):
            name = getattr(self, f"channel_{i}_name")
            if name is None:
                continue
            parsed = parse_channel_name(name)
            ch_type = parsed.get("channel_type", "unknown")
            if ch_type not in ("fluorescence", "labelfree", "virtual_stain"):
                ch_type = "labelfree"

            marker_value = getattr(self, f"channel_{i}_marker")
            bio_dict = None
            if ch_type == "labelfree":
                bio_dict = {"marker": name}
            elif marker_value is not None:
                bio_dict = {
                    "marker": marker_value,
                    "marker_type": "protein_tag",
                    "fluorophore": None,
                }

            annotation[name] = {
                "channel_type": ch_type,
                "biological_annotation": bio_dict,
            }
        return annotation

    def to_experiment_metadata(self) -> dict:
        """Return dict for writing to ``.zattrs["experiment_metadata"]``.

        Produces the unified schema: ``perturbations`` list +
        ``time_sampling_minutes``.
        """
        perturbations: list[dict] = []
        if self.perturbation is not None:
            p: dict = {
                "name": self.perturbation,
                "type": "unknown",
                "hours_post": self.hours_post_perturbation or 0.0,
            }
            if self.moi is not None:
                p["moi"] = self.moi
            if self.treatment_concentration_nm is not None:
                p["concentration_nm"] = self.treatment_concentration_nm
            perturbations.append(p)

        return {
            "perturbations": perturbations,
            "time_sampling_minutes": self.time_interval_min or 0.0,
        }

    def to_airtable_fields(self) -> dict:
        """Return dict for creating/updating an Airtable record.

        Only includes non-None fields. Excludes ``record_id`` and
        ``dataset``/``well_id`` which are typically not updated.
        """
        fields: dict = {}
        exclude = {"record_id", "dataset", "well_id", "fov"}

        for key, val in self.model_dump(exclude_none=True).items():
            if key not in exclude:
                fields[key] = val

        return fields
