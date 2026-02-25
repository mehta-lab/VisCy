"""Pydantic configuration models for QC metrics."""

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
        Torch device for FFT computation.
    """

    channel_names: list[str] = Field(..., min_length=1)
    NA_det: float
    lambda_ill: float
    pixel_size: float
    midband_fractions: tuple[float, float] = (0.125, 0.25)
    device: str = "cpu"


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
    """

    data_path: str
    num_workers: int = 4
    focus_slice: FocusSliceConfig | None = None
