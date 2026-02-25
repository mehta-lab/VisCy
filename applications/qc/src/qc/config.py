"""Pydantic configuration models for QC metrics."""

from typing import Literal

from pydantic import BaseModel, Field


class FocusSliceConfig(BaseModel):
    """Configuration for the FocusSliceMetric.

    Parameters
    ----------
    channel_names : list[str]
        Channel names to compute focus for.
    NA_det : float
        Detection numerical aperture.
    lambda_ill : float
        Illumination wavelength (same units as pixel_size).
    pixel_size : float
        Object-space pixel size (camera pixel size / magnification).
    midband_fractions : tuple[float, float]
        Inner and outer fractions of cutoff frequency.
    device : str
        Torch device for FFT computation (e.g. "cpu", "cuda").
    """

    channel_names: list[str] = Field(..., min_length=1)
    NA_det: float
    lambda_ill: float
    pixel_size: float
    midband_fractions: tuple[float, float] = (0.125, 0.25)
    device: str = "cpu"


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

    Extra fields (moi, concentration_uM, etc.) are allowed.

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
    type: str
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


class AnnotationConfig(BaseModel):
    """Channel annotation and per-well experiment metadata.

    Parameters
    ----------
    channel_annotation : dict[str, ChannelAnnotationEntry]
        Keyed by channel name (must match omero.channels labels).
    experiment_metadata : dict[str, WellExperimentMetadata]
        Keyed by well path (e.g. "A/1").
    """

    channel_annotation: dict[str, ChannelAnnotationEntry]
    experiment_metadata: dict[str, WellExperimentMetadata]


class QCConfig(BaseModel):
    """Top-level QC configuration.

    Parameters
    ----------
    data_path : str
        Path to the HCS OME-Zarr dataset.
    num_workers : int
        Number of workers for data loading.
    focus_slice : FocusSliceConfig or None
        Configuration for focus slice detection. None to skip.
    annotation : AnnotationConfig or None
        Channel and experiment metadata annotation. None to skip.
    """

    data_path: str
    num_workers: int = 4
    focus_slice: FocusSliceConfig | None = None
    annotation: AnnotationConfig | None = None
