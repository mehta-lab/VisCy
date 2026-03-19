"""Shared FOV-level metadata schema for data curation.

Provides :class:`FOVRecord` — the base model for FOV-level metadata
used by both the Airtable app and the collection schema.
"""

from __future__ import annotations

from pydantic import BaseModel


class FOVRecord(BaseModel):
    """FOV-level metadata record.

    Contains data-intrinsic fields shared across Airtable records
    and collection entries. Field names follow the Airtable Datasets
    table naming convention.

    Parameters
    ----------
    dataset : str
        Dataset / experiment name.
    well_id : str
        Well identifier (e.g. ``"A/1"``).
    fov : str or None
        FOV identifier within the well.
    data_path : str or None
        Path to the HCS OME-Zarr store.
    tracks_path : str or None
        Root directory for per-FOV tracking CSVs.
    channel_names : list[str]
        Ordered channel names present in the zarr store.
    time_interval_min : float or None
        Time interval between frames in minutes.
    hours_post_perturbation : float or None
        Hours post perturbation at imaging start.
    moi : float or None
        Multiplicity of infection.
    marker : str or None
        Protein marker or dye name (e.g. ``"TOMM20"``, ``"SEC61B"``).
    organelle : str or None
        Target organelle or cellular structure (e.g. ``"mitochondria"``).
    cell_state : str or None
        Cell state label (e.g. ``"uninfected"``, ``"infected"``).
    cell_type : str or None
        Cell type (e.g. ``"A549"``, ``"HEK293T"``).
    cell_line : list[str] or None
        Cell line(s).
    perturbation : str or None
        Perturbation name.
    seeding_density : int or None
        Cell seeding density.
    treatment_concentration_nm : float or None
        Treatment concentration in nanomolar.
    fluorescence_modality : str or None
        Fluorescence imaging modality.
    t_shape : int or None
        Number of timepoints.
    c_shape : int or None
        Number of channels.
    z_shape : int or None
        Number of Z slices.
    y_shape : int or None
        Image height in pixels.
    x_shape : int or None
        Image width in pixels.
    """

    dataset: str
    well_id: str
    fov: str | None = None
    data_path: str | None = None
    tracks_path: str | None = None
    channel_names: list[str] = []
    time_interval_min: float | None = None
    hours_post_perturbation: float | None = None
    moi: float | None = None
    marker: str | None = None
    organelle: str | None = None
    cell_state: str | None = None
    cell_type: str | None = None
    cell_line: list[str] | None = None
    perturbation: str | None = None
    seeding_density: int | None = None
    treatment_concentration_nm: float | None = None
    fluorescence_modality: str | None = None
    t_shape: int | None = None
    c_shape: int | None = None
    z_shape: int | None = None
    y_shape: int | None = None
    x_shape: int | None = None
