"""Pydantic models for Airtable Datasets table records."""

from __future__ import annotations

import re

from pydantic import BaseModel


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

    def to_experiment_metadata(self) -> dict:
        """Return dict for writing to ``.zattrs["experiment_metadata"]``.

        Includes platemap fields and a ``channels`` dict mapping
        channel_name to ``{biology, index}``. Excludes ``None`` values.
        """
        meta: dict = {}

        for key in (
            "cell_type",
            "cell_state",
            "cell_line",
            "organelle",
            "perturbation",
            "hours_post_perturbation",
            "moi",
            "time_interval_min",
            "seeding_density",
            "treatment_concentration_nm",
        ):
            val = getattr(self, key)
            if val is not None:
                meta[key] = val

        # Build channels mapping: channel_name -> {biology, index}
        channels = {}
        for i in range(4):
            name = getattr(self, f"channel_{i}_name")
            biology = getattr(self, f"channel_{i}_biology")
            if name is not None:
                entry: dict = {"index": i}
                if biology is not None:
                    entry["biology"] = biology
                channels[name] = entry
        if channels:
            meta["channels"] = channels

        return meta

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
