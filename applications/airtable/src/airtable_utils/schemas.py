"""Pydantic models for Airtable Datasets table records and unified zattrs schema."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field


def parse_channel_name(name: str) -> dict:
    """Extract channel metadata from a zarr channel label.

    Parameters
    ----------
    name : str
        Channel label from ``omero.channels[].label``,
        e.g. ``"Phase3D"``, ``"raw GFP EX488 EM525-45"``,
        ``"nuclei_prediction"``.

    Returns
    -------
    dict
        Parsed metadata with keys:
        - ``channel_type``: ``"labelfree"`` | ``"fluorescence"`` | ``"virtual_stain"``
        - ``filter_cube``: microscope filter name (e.g. ``"GFP"``) if fluorescence
        - ``excitation_nm``: excitation wavelength if parseable
        - ``emission_nm``: emission center wavelength if parseable
    """
    result: dict = {}
    name_lower = name.lower()

    # Fluorescence pattern: "raw <FILTER> EX<num> EM<num>[-<num>]"
    fl_match = re.match(
        r"raw\s+(\w+)\s+EX(\d+)\s+EM(\d+)(?:-(\d+))?",
        name,
        re.IGNORECASE,
    )
    if fl_match:
        result["channel_type"] = "fluorescence"
        result["filter_cube"] = fl_match.group(1)
        result["excitation_nm"] = int(fl_match.group(2))
        result["emission_nm"] = int(fl_match.group(3))
        return result

    # Virtual stain patterns (check before labelfree to avoid substring collisions)
    vs_keywords = ("prediction", "virtual", "vs_")
    if any(kw in name_lower for kw in vs_keywords):
        result["channel_type"] = "virtual_stain"
        return result

    # Label-free patterns (use word boundaries for short keywords)
    labelfree_substrings = ("phase", "brightfield", "retardance")
    labelfree_word_patterns = (r"\bbf[\b_]", r"\bdic\b", r"\bpol\b")
    if any(kw in name_lower for kw in labelfree_substrings) or any(
        re.search(p, name_lower) for p in labelfree_word_patterns
    ):
        result["channel_type"] = "labelfree"
        return result

    # Fallback: if contains EX/EM pattern without "raw" prefix
    ex_em_match = re.search(r"EX(\d+)\s*EM(\d+)", name, re.IGNORECASE)
    if ex_em_match:
        result["channel_type"] = "fluorescence"
        result["excitation_nm"] = int(ex_em_match.group(1))
        result["emission_nm"] = int(ex_em_match.group(2))
        return result

    result["channel_type"] = "unknown"
    return result


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

    organelle: str
    marker: str
    marker_type: Literal["protein_tag", "direct_label", "nuclear_dye", "virtual_stain"]
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


class DatasetRecord(BaseModel):
    """A single FOV-level record from the Airtable Datasets table.

    Field names match the renamed snake_case Airtable column names 1:1.
    """

    dataset: str
    well_id: str
    fov: str | None = None
    cell_type: str | None = None
    cell_state: str | None = None
    cell_line: list[str] | None = None
    organelle: str | None = None
    perturbation: str | None = None
    hours_post_perturbation: float | None = None
    moi: float | None = None
    time_interval_min: float | None = None
    seeding_density: int | None = None
    treatment_concentration_nm: float | None = None
    channel_0_name: str | None = None
    channel_0_biology: str | None = None
    channel_1_name: str | None = None
    channel_1_biology: str | None = None
    channel_2_name: str | None = None
    channel_2_biology: str | None = None
    channel_3_name: str | None = None
    channel_3_biology: str | None = None
    data_path: str | None = None
    fluorescence_modality: str | None = None
    t_shape: int | None = None
    c_shape: int | None = None
    z_shape: int | None = None
    y_shape: int | None = None
    x_shape: int | None = None
    record_id: str | None = None

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
            organelle=_select_val(fields.get("organelle")),
            perturbation=_select_val(fields.get("perturbation")),
            hours_post_perturbation=fields.get("hours_post_perturbation"),
            moi=fields.get("moi"),
            time_interval_min=fields.get("time_interval_min"),
            seeding_density=fields.get("seeding_density"),
            treatment_concentration_nm=fields.get("treatment_concentration_nm"),
            channel_0_name=fields.get("channel_0_name"),
            channel_0_biology=_select_val(fields.get("channel_0_biology")),
            channel_1_name=fields.get("channel_1_name"),
            channel_1_biology=_select_val(fields.get("channel_1_biology")),
            channel_2_name=fields.get("channel_2_name"),
            channel_2_biology=_select_val(fields.get("channel_2_biology")),
            channel_3_name=fields.get("channel_3_name"),
            channel_3_biology=_select_val(fields.get("channel_3_biology")),
            data_path=fields.get("data_path"),
            fluorescence_modality=_select_val(fields.get("fluorescence_modality")),
            t_shape=fields.get("t_shape"),
            c_shape=fields.get("c_shape"),
            z_shape=fields.get("z_shape"),
            y_shape=fields.get("y_shape"),
            x_shape=fields.get("x_shape"),
            record_id=record.get("id"),
        )

    def to_channel_annotation(self) -> dict[str, dict]:
        """Return dict for writing to ``.zattrs["channel_annotation"]``.

        Maps each channel name to a ``ChannelAnnotationEntry``-compatible dict
        with ``channel_type`` (derived from channel name parsing) and
        ``biological_annotation`` (from the Airtable biology field).
        """
        annotation: dict[str, dict] = {}
        for i in range(4):
            name = getattr(self, f"channel_{i}_name")
            if name is None:
                continue
            parsed = parse_channel_name(name)
            ch_type = parsed.get("channel_type", "unknown")
            # Map "unknown" to a valid literal for the schema
            if ch_type not in ("fluorescence", "labelfree", "virtual_stain"):
                ch_type = "labelfree"

            biology = getattr(self, f"channel_{i}_biology")
            bio_dict = None
            if biology is not None:
                bio_dict = {
                    "organelle": biology.lower().replace(" ", "_"),
                    "marker": "unknown",
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
